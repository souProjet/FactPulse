"""
Fast Local Fact Store

Module de vérification rapide contre une base de faits connus.
Utilise la similarité sémantique pour un matching robuste.

Performance:
- Lookup time: < 300ms
- Pas de LLM
- Comportement déterministe

Schéma de données:
{
    "fact_id": str,
    "canonical_claim": str,
    "verdict": "TRUE" | "FALSE" | "PARTIALLY_TRUE",
    "confidence": float (0-1),
    "sources": [
        {"title": str, "url": str, "snippet": str}
    ],
    "embedding": [float] (optionnel, pré-calculé)
}

Stratégie de matching:
1. Encoder la requête avec MiniLM
2. Recherche FAISS (cosine similarity)
3. Seuil de match: 0.85 (ajustable)
4. Retourner le match le plus similaire si > seuil
"""

import json
import time
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

import torch
from sentence_transformers import SentenceTransformer
import faiss


class Verdict(str, Enum):
    """Verdicts possibles pour un fait."""
    TRUE = "TRUE"
    FALSE = "FALSE"
    PARTIALLY_TRUE = "PARTIALLY_TRUE"
    NOT_VERIFIABLE = "NOT_VERIFIABLE"


@dataclass
class Source:
    """Source supportant un verdict."""
    title: str
    url: Optional[str] = None
    snippet: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Source":
        return cls(
            title=data.get("title", ""),
            url=data.get("url"),
            snippet=data.get("snippet", "")
        )


@dataclass
class Fact:
    """Un fait vérifié dans la base."""
    fact_id: str
    canonical_claim: str
    verdict: Verdict
    confidence: float
    sources: List[Source] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "canonical_claim": self.canonical_claim,
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "sources": [s.to_dict() for s in self.sources]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Fact":
        return cls(
            fact_id=data.get("fact_id", ""),
            canonical_claim=data.get("canonical_claim", data.get("fact_text", "")),
            verdict=Verdict(data.get("verdict", "NOT_VERIFIABLE")),
            confidence=data.get("confidence", 1.0),
            sources=[Source.from_dict(s) for s in data.get("sources", [])]
        )


@dataclass
class LookupResult:
    """Résultat d'une recherche dans la base."""
    found: bool
    fact: Optional[Fact] = None
    similarity: float = 0.0
    lookup_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "found": self.found,
            "fact": self.fact.to_dict() if self.fact else None,
            "similarity": round(self.similarity, 4),
            "lookup_time_ms": round(self.lookup_time_ms, 2)
        }


class FactStore:
    """
    Base de faits vérifiés avec recherche sémantique.
    
    Supporte deux backends de stockage:
    - JSON: Simple, pour petites bases (< 10k faits)
    - SQLite: Pour bases plus grandes avec indexation
    
    Recherche via FAISS pour performance.
    
    Usage:
        store = FactStore(device="cuda")
        store.load_from_json("data/known_facts.json")
        
        result = store.lookup("La Terre est plate")
        # LookupResult(found=True, fact=Fact(...), similarity=0.95)
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        similarity_threshold: float = 0.85,
        embedding_dim: int = 384
    ):
        """
        Initialise le FactStore.
        
        Args:
            model_name: Modèle d'embedding
            device: Device torch
            similarity_threshold: Seuil de similarité pour match
            embedding_dim: Dimension des embeddings
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.similarity_threshold = similarity_threshold
        self.embedding_dim = embedding_dim
        
        self._encoder: Optional[SentenceTransformer] = None
        self._faiss_index: Optional[faiss.Index] = None
        self._facts: List[Fact] = []
        self._loaded = False
    
    def _load_encoder(self) -> None:
        """Charge le modèle d'embedding."""
        if self._encoder is None:
            self._encoder = SentenceTransformer(self.model_name, device=self.device)
    
    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode une liste de textes en embeddings."""
        self._load_encoder()
        embeddings = self._encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Pour cosine similarity
            show_progress_bar=False
        )
        return embeddings.astype('float32')
    
    def _build_index(self, embeddings: np.ndarray) -> None:
        """Construit l'index FAISS."""
        # IndexFlatIP pour inner product (= cosine sim avec vecteurs normalisés)
        self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Utiliser GPU si disponible
        if "cuda" in self.device and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self._faiss_index = faiss.index_cpu_to_gpu(res, 0, self._faiss_index)
        
        # Ajouter les embeddings
        self._faiss_index.add(embeddings)
    
    def load_from_json(self, json_path: str) -> int:
        """
        Charge les faits depuis un fichier JSON.
        
        Args:
            json_path: Chemin vers le fichier JSON
            
        Returns:
            Nombre de faits chargés
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {json_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Parser les faits
        self._facts = []
        for item in data:
            # Supporter les deux formats (fact_text ou canonical_claim)
            if "canonical_claim" not in item and "fact_text" in item:
                item["canonical_claim"] = item["fact_text"]
            self._facts.append(Fact.from_dict(item))
        
        # Générer les embeddings
        texts = [f.canonical_claim for f in self._facts]
        embeddings = self._encode(texts)
        
        # Construire l'index
        self._build_index(embeddings)
        
        self._loaded = True
        return len(self._facts)
    
    def load_from_sqlite(self, db_path: str) -> int:
        """
        Charge les faits depuis SQLite.
        
        Schema attendu:
        CREATE TABLE facts (
            fact_id TEXT PRIMARY KEY,
            canonical_claim TEXT,
            verdict TEXT,
            confidence REAL,
            sources_json TEXT,
            embedding BLOB
        );
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT fact_id, canonical_claim, verdict, confidence, sources_json, embedding
            FROM facts
        """)
        
        self._facts = []
        embeddings_list = []
        
        for row in cursor.fetchall():
            fact_id, claim, verdict, confidence, sources_json, embedding_blob = row
            
            sources = json.loads(sources_json) if sources_json else []
            
            self._facts.append(Fact(
                fact_id=fact_id,
                canonical_claim=claim,
                verdict=Verdict(verdict),
                confidence=confidence,
                sources=[Source.from_dict(s) for s in sources]
            ))
            
            if embedding_blob:
                embeddings_list.append(np.frombuffer(embedding_blob, dtype=np.float32))
        
        conn.close()
        
        # Embeddings pré-calculés ou à générer
        if embeddings_list and len(embeddings_list) == len(self._facts):
            embeddings = np.vstack(embeddings_list)
        else:
            texts = [f.canonical_claim for f in self._facts]
            embeddings = self._encode(texts)
        
        self._build_index(embeddings)
        self._loaded = True
        
        return len(self._facts)
    
    def save_to_sqlite(self, db_path: str) -> None:
        """Sauvegarde les faits vers SQLite avec embeddings."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Créer la table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                fact_id TEXT PRIMARY KEY,
                canonical_claim TEXT,
                verdict TEXT,
                confidence REAL,
                sources_json TEXT,
                embedding BLOB
            )
        """)
        
        # Générer embeddings
        texts = [f.canonical_claim for f in self._facts]
        embeddings = self._encode(texts)
        
        # Insérer
        for fact, embedding in zip(self._facts, embeddings):
            cursor.execute("""
                INSERT OR REPLACE INTO facts 
                (fact_id, canonical_claim, verdict, confidence, sources_json, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                fact.fact_id,
                fact.canonical_claim,
                fact.verdict.value,
                fact.confidence,
                json.dumps([s.to_dict() for s in fact.sources]),
                embedding.tobytes()
            ))
        
        conn.commit()
        conn.close()
    
    def lookup(self, claim: str, top_k: int = 1) -> LookupResult:
        """
        Recherche un claim dans la base.
        
        Args:
            claim: Texte du claim à vérifier
            top_k: Nombre de résultats à considérer
            
        Returns:
            LookupResult avec le meilleur match si > seuil
        """
        start_time = time.perf_counter()
        
        if not self._loaded or not self._facts:
            return LookupResult(
                found=False,
                lookup_time_ms=(time.perf_counter() - start_time) * 1000
            )
        
        # Encoder la requête
        query_embedding = self._encode([claim])
        
        # Recherche FAISS
        similarities, indices = self._faiss_index.search(query_embedding, top_k)
        
        best_similarity = float(similarities[0][0])
        best_idx = int(indices[0][0])
        
        lookup_time = (time.perf_counter() - start_time) * 1000
        
        # Vérifier le seuil
        if best_similarity >= self.similarity_threshold:
            return LookupResult(
                found=True,
                fact=self._facts[best_idx],
                similarity=best_similarity,
                lookup_time_ms=lookup_time
            )
        
        return LookupResult(
            found=False,
            similarity=best_similarity,
            lookup_time_ms=lookup_time
        )
    
    def lookup_batch(self, claims: List[str], top_k: int = 1) -> List[LookupResult]:
        """
        Recherche batch pour efficacité.
        
        Args:
            claims: Liste de claims
            top_k: Résultats par claim
            
        Returns:
            Liste de LookupResult
        """
        start_time = time.perf_counter()
        
        if not self._loaded or not self._facts:
            return [LookupResult(found=False) for _ in claims]
        
        # Encoder toutes les requêtes
        query_embeddings = self._encode(claims)
        
        # Recherche batch
        similarities, indices = self._faiss_index.search(query_embeddings, top_k)
        
        total_time = (time.perf_counter() - start_time) * 1000
        time_per_query = total_time / len(claims)
        
        results = []
        for sim_row, idx_row in zip(similarities, indices):
            best_sim = float(sim_row[0])
            best_idx = int(idx_row[0])
            
            if best_sim >= self.similarity_threshold:
                results.append(LookupResult(
                    found=True,
                    fact=self._facts[best_idx],
                    similarity=best_sim,
                    lookup_time_ms=time_per_query
                ))
            else:
                results.append(LookupResult(
                    found=False,
                    similarity=best_sim,
                    lookup_time_ms=time_per_query
                ))
        
        return results
    
    def add_fact(self, fact: Fact) -> None:
        """
        Ajoute un fait à la base (mise à jour incrémentale).
        
        Args:
            fact: Fait à ajouter
        """
        self._load_encoder()
        
        # Encoder le nouveau fait
        embedding = self._encode([fact.canonical_claim])
        
        # Ajouter à l'index
        if self._faiss_index is None:
            self._build_index(embedding)
        else:
            self._faiss_index.add(embedding)
        
        self._facts.append(fact)
        self._loaded = True
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de la base."""
        if not self._facts:
            return {"total_facts": 0}
        
        verdicts = {}
        for fact in self._facts:
            v = fact.verdict.value
            verdicts[v] = verdicts.get(v, 0) + 1
        
        return {
            "total_facts": len(self._facts),
            "verdicts": verdicts,
            "similarity_threshold": self.similarity_threshold,
            "index_size": self._faiss_index.ntotal if self._faiss_index else 0
        }


# ============================================================================
# Notes sur la performance
# ============================================================================
"""
PERFORMANCE ATTENDUE:

| Opération | Temps | Notes |
|-----------|-------|-------|
| Load JSON (1k facts) | ~2s | Inclut embedding |
| Load SQLite (pré-calculé) | ~200ms | Embeddings stockés |
| Lookup single | ~15ms | Après warm-up |
| Lookup batch (100) | ~50ms | ~0.5ms/query |

OPTIMISATIONS:
1. Pré-calculer les embeddings (SQLite avec BLOB)
2. FAISS GPU pour grandes bases (>100k)
3. Index HNSW pour bases très grandes (>1M)
4. Cache LRU pour requêtes fréquentes

SEUIL DE SIMILARITÉ:
- 0.90+: Match très strict (quasi identique)
- 0.85: Recommandé pour haute précision
- 0.80: Plus permissif, plus de faux positifs
- 0.75: Matching large, utile pour recall

COMPORTEMENT DÉTERMINISTE:
- Même entrée → même sortie
- Pas de randomness
- Reproductible
"""
