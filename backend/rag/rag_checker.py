"""
RAG Fact-Checking Module

Vérifie les claims en utilisant uniquement les sources récupérées.
Strictement source-grounded - pas d'hallucination.

Contraintes:
- NE DOIT PAS halluciner de faits
- Si sources insuffisantes → NOT_VERIFIABLE
- Temps d'inférence < 1.5 secondes
- Accélération GPU

Architecture:
1. Retrieval: FAISS pour recherche vectorielle
2. Generation: LLM local quantifié (Phi-3/Mistral)
3. Prompt strict pour réponses source-grounded
"""

import re
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import faiss


class RAGVerdict(str, Enum):
    """Verdicts possibles."""
    TRUE = "TRUE"
    FALSE = "FALSE"
    PARTIALLY_TRUE = "PARTIALLY_TRUE"
    NOT_VERIFIABLE = "NOT_VERIFIABLE"


@dataclass
class RetrievedSource:
    """Source récupérée pour vérification."""
    text: str
    title: str
    url: Optional[str] = None
    relevance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text[:500],
            "title": self.title,
            "url": self.url,
            "relevance_score": round(self.relevance_score, 4)
        }


@dataclass
class RAGResult:
    """Résultat de la vérification RAG."""
    verdict: RAGVerdict
    justification: str
    sources: List[RetrievedSource]
    confidence: float
    inference_time_ms: float
    retrieval_time_ms: float
    generation_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "justification": self.justification,
            "sources": [s.to_dict() for s in self.sources],
            "confidence": round(self.confidence, 4),
            "inference_time_ms": round(self.inference_time_ms, 2),
            "retrieval_time_ms": round(self.retrieval_time_ms, 2),
            "generation_time_ms": round(self.generation_time_ms, 2)
        }


# ============================================================================
# PROMPT TEMPLATE - STRICT SOURCE-GROUNDED
# ============================================================================

STRICT_VERIFICATION_PROMPT = """Tu es un moteur de vérification des faits.

RÈGLES STRICTES:
- Tu DOIS utiliser UNIQUEMENT les sources fournies.
- Si les sources ne confirment PAS clairement ou ne réfutent PAS clairement l'affirmation, réponds "NOT_VERIFIABLE".
- Tu ne dois PAS utiliser tes connaissances préalables.
- Tu ne dois PAS deviner.
- Tu ne dois PAS faire d'inférence au-delà des sources.

AFFIRMATION À VÉRIFIER:
{claim}

SOURCES DISPONIBLES:
{sources}

INSTRUCTIONS:
1. Lis attentivement chaque source
2. Cherche des preuves directes qui confirment OU réfutent l'affirmation
3. Si aucune source ne traite directement de l'affirmation → NOT_VERIFIABLE
4. Si les sources se contredisent → PARTIALLY_TRUE avec explication

Réponds STRICTEMENT en JSON (rien d'autre):
{{"verdict": "TRUE|FALSE|PARTIALLY_TRUE|NOT_VERIFIABLE", "justification": "explication courte basée sur les sources", "confidence": 0.0-1.0, "source_ids": [indices des sources utilisées]}}"""


# English version for models that work better in English
STRICT_VERIFICATION_PROMPT_EN = """You are a fact-checking engine.

STRICT RULES:
- You MUST use ONLY the provided sources.
- If the sources do NOT clearly confirm or refute the claim, respond "NOT_VERIFIABLE".
- Do NOT use prior knowledge.
- Do NOT guess.
- Do NOT infer beyond the sources.

CLAIM TO VERIFY:
{claim}

AVAILABLE SOURCES:
{sources}

INSTRUCTIONS:
1. Read each source carefully
2. Look for direct evidence that confirms OR refutes the claim
3. If no source directly addresses the claim → NOT_VERIFIABLE
4. If sources contradict each other → PARTIALLY_TRUE with explanation

Respond STRICTLY in JSON (nothing else):
{{"verdict": "TRUE|FALSE|PARTIALLY_TRUE|NOT_VERIFIABLE", "justification": "short explanation based on sources", "confidence": 0.0-1.0, "source_ids": [indices of sources used]}}"""


class RAGChecker:
    """
    Module RAG pour fact-checking source-grounded.
    
    Pipeline:
    1. Encode le claim
    2. Recherche les sources pertinentes (FAISS)
    3. Construit le prompt avec les sources
    4. Génère le verdict via LLM
    5. Parse et valide la réponse
    
    Le LLM est contraint à ne répondre qu'en se basant sur les sources.
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct",  # Modèle léger et rapide (<3s)
        device: Optional[str] = None,
        top_k: int = 3,  # Réduire pour accélérer
        min_relevance: float = 0.5,
        max_new_tokens: int = 128,  # Réduire pour accélérer
        use_english_prompt: bool = True  # Qwen fonctionne mieux en anglais
    ):
        """
        Initialise le RAG Checker.
        
        Args:
            embedding_model: Modèle pour les embeddings
            llm_model: LLM pour la génération (Qwen2.5-0.5B est rapide)
            device: Device torch
            top_k: Nombre de sources à récupérer
            min_relevance: Score minimum de pertinence
            max_new_tokens: Tokens max pour la génération
            use_english_prompt: Utiliser le prompt anglais
        """
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = top_k
        self.min_relevance = min_relevance
        self.max_new_tokens = max_new_tokens
        self.prompt_template = STRICT_VERIFICATION_PROMPT_EN if use_english_prompt else STRICT_VERIFICATION_PROMPT
        
        self._embedding_model: Optional[SentenceTransformer] = None
        self._llm = None
        self._tokenizer = None
        self._faiss_index: Optional[faiss.Index] = None
        self._source_metadata: List[Dict[str, Any]] = []
        
        self._loaded = False
    
    def load_models(self) -> None:
        """Charge les modèles (embedding + LLM)."""
        if self._loaded:
            return
        
        import logging
        logger = logging.getLogger('factpulse.rag')
        
        # Embedding model
        logger.info(f"Chargement du modèle d'embedding: {self.embedding_model_name}")
        self._embedding_model = SentenceTransformer(
            self.embedding_model_name, 
            device=self.device
        )
        
        # LLM - Qwen2.5-0.5B est assez petit pour tourner sans quantization
        # Utiliser FP16 directement pour plus de rapidité
        logger.info(f"Chargement du LLM: {self.llm_model_name}")
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.llm_model_name,
            trust_remote_code=True
        )
        
        # Pour les petits modèles (<1B), FP16 sans quantization est plus rapide
        self._llm = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info("Modèles chargés avec succès")
        self._loaded = True
    
    def load_sources(self, sources_path: str) -> int:
        """
        Charge les sources pour le retrieval.
        
        Args:
            sources_path: Chemin vers le JSON des sources
            
        Returns:
            Nombre de sources chargées
        """
        path = Path(sources_path)
        if not path.exists():
            raise FileNotFoundError(f"Sources non trouvées: {sources_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._source_metadata = data
        
        # Encoder les sources
        if not self._embedding_model:
            self._embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=self.device
            )
        
        texts = [item.get("fact_text", item.get("text", "")) for item in data]
        embeddings = self._embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype('float32')
        
        # Construire FAISS
        dim = embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)
        
        if "cuda" in self.device and faiss.get_num_gpus() > 0:
            try:
                res = faiss.StandardGpuResources()  # type: ignore
                self._faiss_index = faiss.index_cpu_to_gpu(res, 0, self._faiss_index)  # type: ignore
            except (AttributeError, RuntimeError):
                # GPU FAISS not available, continue with CPU
                pass
        
        self._faiss_index.add(embeddings)  # type: ignore[union-attr]
        
        return len(data)
    
    def _retrieve_sources(self, claim: str) -> Tuple[List[RetrievedSource], float]:
        """
        Récupère les sources pertinentes pour un claim.
        
        Returns:
            (sources, retrieval_time_ms)
        """
        start = time.perf_counter()
        
        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            return [], (time.perf_counter() - start) * 1000
        
        if self._embedding_model is None:
            return [], (time.perf_counter() - start) * 1000
        
        # Encoder le claim
        query_embedding = self._embedding_model.encode(
            [claim],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')
        
        # Recherche
        k = min(self.top_k, self._faiss_index.ntotal)
        similarities, indices = self._faiss_index.search(query_embedding, k)  # type: ignore[union-attr]
        
        sources = []
        for sim, idx in zip(similarities[0], indices[0]):
            if sim < self.min_relevance:
                continue
            
            idx = int(idx)
            if idx < len(self._source_metadata):
                meta = self._source_metadata[idx]
                source_info = meta.get("sources", [{}])[0] if meta.get("sources") else {}
                
                sources.append(RetrievedSource(
                    text=meta.get("fact_text", meta.get("text", "")),
                    title=source_info.get("title", f"Source {idx}"),
                    url=source_info.get("url"),
                    relevance_score=float(sim)
                ))
        
        retrieval_time = (time.perf_counter() - start) * 1000
        return sources, retrieval_time
    
    def _build_prompt(self, claim: str, sources: List[RetrievedSource]) -> str:
        """Construit le prompt avec les sources."""
        if not sources:
            sources_text = "Aucune source pertinente trouvée."
        else:
            parts = []
            for i, src in enumerate(sources):
                parts.append(f"[{i+1}] {src.title}\n{src.text}")
            sources_text = "\n\n".join(parts)
        
        return self.prompt_template.format(
            claim=claim,
            sources=sources_text
        )
    
    @torch.no_grad()
    def _generate_response(self, prompt: str) -> Tuple[str, float]:
        """
        Génère la réponse du LLM.
        
        Returns:
            (response_text, generation_time_ms)
        """
        start = time.perf_counter()
        
        if self._tokenizer is None or self._llm is None:
            return "", (time.perf_counter() - start) * 1000
        
        # Limiter la longueur du prompt pour accélérer
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024  # Réduire pour accélérer
        ).to(self._llm.device)
        
        # Synchroniser GPU avant génération
        if "cuda" in str(self._llm.device):
            torch.cuda.synchronize()
        
        outputs = self._llm.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,  # Greedy decoding (plus rapide)
            pad_token_id=self._tokenizer.eos_token_id or self._tokenizer.pad_token_id or 0,
            use_cache=True  # Activer le cache KV pour accélérer
        )
        
        # Synchroniser GPU après génération
        if "cuda" in str(self._llm.device):
            torch.cuda.synchronize()
        
        # Décoder seulement les nouveaux tokens
        response = self._tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        generation_time = (time.perf_counter() - start) * 1000
        return response, generation_time
    
    def _parse_response(self, response: str, sources: List[RetrievedSource]) -> Tuple[RAGVerdict, str, float, List[RetrievedSource]]:
        """
        Parse la réponse JSON du LLM.
        
        Returns:
            (verdict, justification, confidence, used_sources)
        """
        # Extraire le JSON
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        
        if json_match:
            try:
                data = json.loads(json_match.group())
                
                verdict_str = data.get("verdict", "NOT_VERIFIABLE").upper()
                justification = data.get("justification", "")
                confidence = float(data.get("confidence", 0.5))
                source_ids = data.get("source_ids", [])
                
                # Valider le verdict
                try:
                    verdict = RAGVerdict(verdict_str)
                except ValueError:
                    verdict = RAGVerdict.NOT_VERIFIABLE
                
                # Filtrer les sources utilisées
                used_sources = []
                for idx in source_ids:
                    if isinstance(idx, int) and 0 <= idx - 1 < len(sources):
                        used_sources.append(sources[idx - 1])
                
                if not used_sources:
                    used_sources = sources[:3]  # Fallback
                
                return verdict, justification, min(max(confidence, 0.0), 1.0), used_sources
                
            except (json.JSONDecodeError, ValueError, KeyError):
                pass
        
        # Fallback: parser manuellement
        response_upper = response.upper()
        
        if "NOT_VERIFIABLE" in response_upper or "NOT VERIFIABLE" in response_upper:
            verdict = RAGVerdict.NOT_VERIFIABLE
        elif "FALSE" in response_upper:
            verdict = RAGVerdict.FALSE
        elif "PARTIALLY" in response_upper:
            verdict = RAGVerdict.PARTIALLY_TRUE
        elif "TRUE" in response_upper:
            verdict = RAGVerdict.TRUE
        else:
            verdict = RAGVerdict.NOT_VERIFIABLE
        
        return verdict, response[:200], 0.5, sources[:3]
    
    def verify(self, claim: str) -> RAGResult:
        """
        Vérifie un claim via RAG.
        
        Args:
            claim: Affirmation à vérifier
            
        Returns:
            RAGResult avec verdict, justification, sources
        """
        total_start = time.perf_counter()
        
        # Charger les modèles si nécessaire
        self.load_models()
        
        # Sync GPU
        if "cuda" in self.device:
            torch.cuda.synchronize()
        
        # 1. Retrieval
        sources, retrieval_time = self._retrieve_sources(claim)
        
        # Si pas de sources, retourner NOT_VERIFIABLE
        if not sources:
            return RAGResult(
                verdict=RAGVerdict.NOT_VERIFIABLE,
                justification="Aucune source pertinente trouvée pour vérifier cette affirmation.",
                sources=[],
                confidence=0.0,
                inference_time_ms=(time.perf_counter() - total_start) * 1000,
                retrieval_time_ms=retrieval_time,
                generation_time_ms=0.0
            )
        
        # 2. Build prompt
        prompt = self._build_prompt(claim, sources)
        
        # 3. Generate
        response, generation_time = self._generate_response(prompt)
        
        # Sync GPU
        if "cuda" in self.device:
            torch.cuda.synchronize()
        
        # 4. Parse
        verdict, justification, confidence, used_sources = self._parse_response(response, sources)
        
        total_time = (time.perf_counter() - total_start) * 1000
        
        return RAGResult(
            verdict=verdict,
            justification=justification,
            sources=used_sources,
            confidence=confidence,
            inference_time_ms=total_time,
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time
        )
    
    def verify_batch(self, claims: List[str]) -> List[RAGResult]:
        """Vérifie plusieurs claims (séquentiel pour GPU)."""
        return [self.verify(claim) for claim in claims]


# ============================================================================
# Notes sur la performance
# ============================================================================
"""
PERFORMANCE ATTENDUE (RTX 5060):

| Étape | Temps | Notes |
|-------|-------|-------|
| Retrieval | ~50ms | FAISS GPU |
| Prompt build | ~1ms | String ops |
| LLM generation | ~1000-1400ms | Phi-3 4-bit |
| Parsing | ~1ms | JSON parse |
| **TOTAL** | **< 1500ms** | ✓ Objectif |

CONSIDÉRATIONS:
1. Température basse (0.1) pour réponses déterministes
2. Greedy decoding (pas de sampling)
3. Prompt strict pour éviter hallucinations
4. Si confidence < 0.5, considérer NOT_VERIFIABLE

PRÉVENTION DES HALLUCINATIONS:
- Prompt explicite interdisant les connaissances préalables
- Demande de citer les source_ids utilisées
- Vérification que les sources citées existent
- Fallback vers NOT_VERIFIABLE en cas de doute

AMÉLIORATION POSSIBLE:
- Chain-of-thought pour raisonnement explicite
- Ensemble de vérification (2 passes)
- Confidence calibration post-hoc
"""
