"""
Pipeline Orchestrator avec Fail-Safe

Orchestre le pipeline de fact-checking avec:
- Gestion des timeouts
- Dégradation gracieuse
- Logging des erreurs
- Métriques de performance

Flow:
1. Claim Detection (< 200ms)
2. Check-worthiness filter
3. Fast Lookup (< 500ms)
4. RAG Verification si nécessaire (< 1500ms) - skippable
5. Agrégation des résultats
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from .failsafe import (
    TimeoutManager, 
    GracefulDegradation, 
    ErrorLogger,
    ErrorState,
    PipelineError
)


# Traduction des verdicts en français
VERDICT_TRANSLATIONS = {
    "TRUE": "VRAI",
    "FALSE": "FAUX",
    "PARTIALLY_TRUE": "PARTIELLEMENT_VRAI",
    "NOT_VERIFIABLE": "NON_VÉRIFIABLE",
    "SKIPPED": "IGNORÉ"
}

def translate_verdict(verdict: str) -> str:
    """Traduit un verdict en français."""
    return VERDICT_TRANSLATIONS.get(verdict, verdict)


@dataclass
class PipelineConfig:
    """Configuration du pipeline."""
    # Timeouts (ms)
    claim_detection_timeout: float = 3000
    fast_lookup_timeout: float = 1000
    rag_timeout: float = 8000  # Augmenté pour LLM lent
    total_timeout: float = 15000
    
    # Seuils
    claim_confidence_threshold: float = 0.6
    check_worthiness_threshold: float = 0.5
    fast_lookup_similarity_threshold: float = 0.85
    
    # Fail-safe
    max_consecutive_rag_timeouts: int = 3
    rag_cooldown_seconds: float = 60.0
    enable_graceful_degradation: bool = True
    log_false_positives: bool = True
    
    # Performance
    max_claims_per_request: int = 10
    use_batch_processing: bool = True
    
    # Langue
    translate_verdicts: bool = True  # Traduire les verdicts en français


@dataclass
class PipelineResult:
    """Résultat du pipeline."""
    status: str  # "success", "partial", "error"
    claims_detected: int
    claims_verified: int
    results: List[Dict[str, Any]]
    timing: Dict[str, Any]
    errors: List[Dict[str, Any]] = field(default_factory=list)
    degraded: bool = False
    degradation_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "claims_detected": self.claims_detected,
            "claims_verified": self.claims_verified,
            "results": self.results,
            "timing": self.timing,
            "errors": self.errors,
            "degraded": self.degraded,
            "degradation_reason": self.degradation_reason
        }


class FactCheckPipeline:
    """
    Pipeline de fact-checking avec fail-safe intégré.
    
    Usage:
        config = PipelineConfig()
        pipeline = FactCheckPipeline(config)
        
        result = await pipeline.analyze("La Terre est plate.")
        print(result.to_dict())
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None, device: str = "cuda"):
        self.config = config or PipelineConfig()
        self.device = device
        
        # Fail-safe components
        self.timeout_mgr = TimeoutManager(self.config.total_timeout)
        self.degradation = GracefulDegradation(
            max_consecutive_timeouts=self.config.max_consecutive_rag_timeouts,
            cooldown_seconds=self.config.rag_cooldown_seconds
        )
        self.error_logger = ErrorLogger()
        
        # Pipeline components (lazy loaded)
        self._claim_detector = None
        self._fact_store = None
        self._rag_checker = None
        
        self._initialized = False
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    def _deduplicate_claims(self, claims: List) -> tuple:
        """
        Déduplique les claims similaires pour éviter les analyses redondantes.
        
        Utilise une similarité textuelle simple basée sur les mots clés.
        
        Returns:
            (unique_claims, groups) où groups mappe chaque claim unique vers ses duplicates
        """
        if not claims:
            return [], {}
        
        def normalize_text(text: str) -> str:
            """Normalise le texte pour comparaison."""
            import re
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        def similarity_score(text1: str, text2: str) -> float:
            """Calcule un score de similarité simple basé sur les mots communs."""
            words1 = set(normalize_text(text1).split())
            words2 = set(normalize_text(text2).split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1 & words2
            union = words1 | words2
            
            return len(intersection) / len(union)  # Jaccard similarity
        
        unique_claims = []
        groups = {}  # claim_text -> [similar_claims]
        
        for claim in claims:
            claim_text = claim.text
            is_duplicate = False
            
            # Chercher si ce claim est similaire à un claim existant
            for unique in unique_claims:
                if similarity_score(claim_text, unique.text) > 0.7:  # Seuil de similarité
                    # C'est un duplicate
                    if unique.text not in groups:
                        groups[unique.text] = []
                    groups[unique.text].append(claim)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_claims.append(claim)
                groups[claim_text] = []  # Initialiser le groupe
        
        return unique_claims, groups
    
    def initialize(self) -> None:
        """Initialise les composants du pipeline."""
        if self._initialized:
            return
        
        from ..claim_detector.claim_detector import ClaimDetector
        from ..fact_checker.fact_store import FactStore
        
        # Claim detector
        self._claim_detector = ClaimDetector(
            device=self.device,
            threshold=self.config.claim_confidence_threshold
        )
        
        # Fact store
        self._fact_store = FactStore(
            device=self.device,
            similarity_threshold=self.config.fast_lookup_similarity_threshold
        )
        
        # Load data
        from pathlib import Path
        facts_path = Path(__file__).parent.parent.parent / "data" / "known_facts.json"
        if facts_path.exists():
            self._fact_store.load_from_json(str(facts_path))
        
        self._initialized = True
    
    def _init_rag(self) -> bool:
        """Initialise le RAG checker (lazy, peut échouer)."""
        if self._rag_checker is not None:
            return True
        
        try:
            from ..rag.rag_checker import RAGChecker
            from pathlib import Path
            import logging
            
            logger = logging.getLogger('factpulse.pipeline')
            logger.info("Chargement du module RAG...")
            
            self._rag_checker = RAGChecker(device=self.device)
            
            facts_path = Path(__file__).parent.parent.parent / "data" / "known_facts.json"
            if facts_path.exists():
                self._rag_checker.load_sources(str(facts_path))
            
            # Précharger les modèles pour éviter les timeouts au premier appel
            logger.info("Préchargement des modèles LLM (peut prendre 30-60 secondes)...")
            self._rag_checker.load_models()
            logger.info("Modèles RAG chargés avec succès")
            
            return True
        except Exception as e:
            self.error_logger.log_error(ErrorState(
                error_type=PipelineError.MODEL_ERROR,
                stage="rag_init",
                message=str(e),
                recoverable=False
            ))
            return False
    
    async def analyze(self, text: str) -> PipelineResult:
        """
        Analyse un texte avec fail-safe.
        
        Détecte et vérifie plusieurs claims dans le texte.
        
        Args:
            text: Texte à analyser
            
        Returns:
            PipelineResult avec résultats et métriques
        """
        total_start = time.perf_counter()
        timing = {"stages": []}
        errors = []
        results = []
        
        # Initialize
        self.initialize()
        
        # ====================================================================
        # STAGE 1: Claim Detection (multi-claims)
        # ====================================================================
        stage_start = time.perf_counter()
        
        # Utiliser detect_claims_in_text pour obtenir tous les claims
        detected_claims, error = self.timeout_mgr.run_with_timeout(
            lambda: self._claim_detector.detect_claims_in_text(
                text, 
                max_claims=self.config.max_claims_per_request
            ),
            timeout_ms=self.config.claim_detection_timeout,
            stage="claim_detection",
            default=[]
        )
        
        claim_detection_time = (time.perf_counter() - stage_start) * 1000
        timing["stages"].append({
            "stage": "claim_detection",
            "duration_ms": round(claim_detection_time, 2)
        })
        
        if error:
            errors.append(error.to_dict())
            self.error_logger.log_error(error)
            
            return PipelineResult(
                status="error",
                claims_detected=0,
                claims_verified=0,
                results=[],
                timing=timing,
                errors=errors
            )
        
        # Nombre de claims détectés
        claims_detected = len(detected_claims) if detected_claims else 0
        
        if not detected_claims or claims_detected == 0:
            timing["total_ms"] = round((time.perf_counter() - total_start) * 1000, 2)
            return PipelineResult(
                status="success",
                claims_detected=0,
                claims_verified=0,
                results=[],
                timing=timing
            )
        
        # ====================================================================
        # DEDUPLICATION: Regrouper les claims similaires
        # ====================================================================
        deduplicated_claims, claim_groups = self._deduplicate_claims(detected_claims)
        
        # ====================================================================
        # STAGE 2 & 3: Vérifier chaque claim unique (Fast Lookup puis RAG)
        # ====================================================================
        degraded = False
        degradation_reason = ""
        verified_results = {}  # claim_text -> result (pour réutilisation)
        
        for claim_result in deduplicated_claims:
            claim_text = claim_result.text
            
            # Skip si confiance trop basse
            if claim_result.confidence < self.config.check_worthiness_threshold:
                results.append({
                    "claim_text": claim_text[:200],
                    "verdict": "SKIPPED",
                    "confidence": claim_result.confidence,
                    "explanation": "Confiance insuffisante pour vérification",
                    "sources": [],
                    "verification_method": "none"
                })
                continue
            
            # Fast Lookup d'abord
            stage_start = time.perf_counter()
            
            lookup_result, lookup_error = self.timeout_mgr.run_with_timeout(
                lambda ct=claim_text: self._fact_store.lookup(ct),
                timeout_ms=self.config.fast_lookup_timeout,
                stage="fast_lookup",
                default=None
            )
            
            if lookup_error:
                errors.append(lookup_error.to_dict())
            
            # Fast lookup hit?
            if lookup_result and lookup_result.found:
                verdict = lookup_result.fact.verdict.value
                if self.config.translate_verdicts:
                    verdict = translate_verdict(verdict)
                
                results.append({
                    "claim_text": claim_text[:200],
                    "verdict": verdict,
                    "confidence": lookup_result.similarity,
                    "explanation": f"Match avec fait connu: {lookup_result.fact.canonical_claim}",
                    "sources": [s.to_dict() for s in lookup_result.fact.sources],
                    "verification_method": "fast_lookup"
                })
                continue
            
            # RAG si pas de match fast lookup
            if self.config.enable_graceful_degradation and self.degradation.should_skip_rag():
                state = self.degradation.get_state()
                degraded = True
                degradation_reason = state.get("rag_disable_reason", "RAG désactivé")
                
                results.append({
                    "claim_text": claim_text[:200],
                    "verdict": "NOT_VERIFIABLE",
                    "confidence": 0.0,
                    "explanation": f"Vérification RAG non disponible: {degradation_reason}",
                    "sources": [],
                    "verification_method": "degraded"
                })
                continue
            
            # Essayer RAG
            if not self._init_rag():
                results.append({
                    "claim_text": claim_text[:200],
                    "verdict": "NOT_VERIFIABLE",
                    "confidence": 0.0,
                    "explanation": "Module RAG non disponible",
                    "sources": [],
                    "verification_method": "error"
                })
                continue
            
            # Run RAG
            rag_stage_start = time.perf_counter()
            loop = asyncio.get_event_loop()
            
            rag_result, rag_error = await loop.run_in_executor(
                self._executor,
                lambda ct=claim_text: self.timeout_mgr.run_with_timeout(
                    lambda: self._rag_checker.verify(ct),
                    timeout_ms=self.config.rag_timeout,
                    stage="rag_verification",
                    default=None
                )
            )
            
            rag_time = (time.perf_counter() - rag_stage_start) * 1000
            
            if rag_error:
                errors.append(rag_error.to_dict())
                self.error_logger.log_error(rag_error)
                
                from .failsafe import PipelineError
                if rag_error.error_type == PipelineError.TIMEOUT:
                    self.degradation.report_rag_timeout()
                
                results.append({
                    "claim_text": claim_text[:200],
                    "verdict": "NOT_VERIFIABLE",
                    "confidence": 0.0,
                    "explanation": f"Erreur RAG: {rag_error.message}",
                    "sources": [],
                    "verification_method": "error"
                })
            elif rag_result:
                self.degradation.report_rag_success()
                
                verdict = rag_result.verdict.value
                if self.config.translate_verdicts:
                    verdict = translate_verdict(verdict)
                
                results.append({
                    "claim_text": claim_text[:200],
                    "verdict": verdict,
                    "confidence": rag_result.confidence,
                    "explanation": rag_result.justification,
                    "sources": [s.to_dict() for s in rag_result.sources],
                    "verification_method": "rag"
                })
            else:
                results.append({
                    "claim_text": claim_text[:200],
                    "verdict": "NOT_VERIFIABLE",
                    "confidence": 0.0,
                    "explanation": "Pas de résultat RAG",
                    "sources": [],
                    "verification_method": "error"
                })
            
            # Stocker le résultat pour les duplicates
            if results:
                verified_results[claim_text] = results[-1]
        
        # Appliquer les résultats aux claims similaires (duplicates)
        for original_text, duplicates in claim_groups.items():
            if original_text in verified_results and duplicates:
                original_result = verified_results[original_text]
                for dup_claim in duplicates:
                    # Copier le résultat avec le texte du duplicate
                    dup_result = original_result.copy()
                    dup_result["claim_text"] = dup_claim.text[:200]
                    dup_result["verification_method"] = "deduplicated"
                    results.append(dup_result)
        
        # Ajouter timing pour lookup et RAG (agrégé)
        timing["stages"].append({
            "stage": "verification",
            "duration_ms": round((time.perf_counter() - total_start) * 1000 - claim_detection_time, 2)
        })
        
        # ====================================================================
        # Finalize
        # ====================================================================
        timing["total_ms"] = round((time.perf_counter() - total_start) * 1000, 2)
        
        # Traduire les verdicts en français si configuré
        if self.config.translate_verdicts:
            for result in results:
                if "verdict" in result:
                    result["verdict"] = translate_verdict(result["verdict"])
        
        # Determine status
        status = "success"
        if errors:
            status = "partial" if results else "error"
        
        # Compter les vérifiés (avec verdicts traduits ou non)
        non_verified = ["SKIPPED", "NOT_VERIFIABLE", "IGNORÉ", "NON_VÉRIFIABLE"]
        claims_verified_count = len([r for r in results if r.get("verdict") not in non_verified])
        
        return PipelineResult(
            status=status,
            claims_detected=claims_detected,
            claims_verified=claims_verified_count,
            results=results,
            timing=timing,
            errors=errors,
            degraded=degraded,
            degradation_reason=degradation_reason
        )
    
    def log_false_positive(
        self,
        claim: str,
        predicted: str,
        actual: str,
        confidence: float,
        feedback: Optional[str] = None
    ) -> None:
        """Log un faux positif pour analyse."""
        self.error_logger.log_false_positive(
            claim=claim,
            predicted=predicted,
            actual=actual,
            confidence=confidence,
            user_feedback=feedback
        )
    
    def get_degradation_state(self) -> Dict[str, Any]:
        """Retourne l'état de dégradation."""
        return self.degradation.get_state()
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'erreurs."""
        return self.error_logger.get_error_stats()
    
    def reset_degradation(self) -> None:
        """Reset l'état de dégradation."""
        self.degradation.reset()
    
    def shutdown(self) -> None:
        """Arrête proprement le pipeline."""
        self.timeout_mgr.shutdown()
        self._executor.shutdown(wait=False)
