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


@dataclass
class PipelineConfig:
    """Configuration du pipeline."""
    # Timeouts (ms)
    claim_detection_timeout: float = 200
    fast_lookup_timeout: float = 500
    rag_timeout: float = 1500
    total_timeout: float = 2000
    
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
            
            self._rag_checker = RAGChecker(device=self.device)
            
            facts_path = Path(__file__).parent.parent.parent / "data" / "known_facts.json"
            if facts_path.exists():
                self._rag_checker.load_sources(str(facts_path))
            
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
        # STAGE 1: Claim Detection
        # ====================================================================
        stage_start = time.perf_counter()
        
        detection_result, error = self.timeout_mgr.run_with_timeout(
            lambda: self._claim_detector.detect(text),
            timeout_ms=self.config.claim_detection_timeout,
            stage="claim_detection",
            default=None
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
        
        # Check if claim detected
        claims_detected = 1 if detection_result and detection_result.is_claim else 0
        
        if not detection_result or not detection_result.is_claim:
            timing["total_ms"] = round((time.perf_counter() - total_start) * 1000, 2)
            return PipelineResult(
                status="success",
                claims_detected=claims_detected,
                claims_verified=0,
                results=[],
                timing=timing
            )
        
        # Check if worth verifying
        if detection_result.confidence < self.config.check_worthiness_threshold:
            timing["total_ms"] = round((time.perf_counter() - total_start) * 1000, 2)
            return PipelineResult(
                status="success",
                claims_detected=claims_detected,
                claims_verified=0,
                results=[{
                    "claim_text": text[:200],
                    "verdict": "SKIPPED",
                    "confidence": detection_result.confidence,
                    "explanation": "Confiance insuffisante pour vérification",
                    "sources": [],
                    "verification_method": "none"
                }],
                timing=timing
            )
        
        # ====================================================================
        # STAGE 2: Fast Lookup
        # ====================================================================
        stage_start = time.perf_counter()
        
        lookup_result, error = self.timeout_mgr.run_with_timeout(
            lambda: self._fact_store.lookup(text),
            timeout_ms=self.config.fast_lookup_timeout,
            stage="fast_lookup",
            default=None
        )
        
        fast_lookup_time = (time.perf_counter() - stage_start) * 1000
        timing["stages"].append({
            "stage": "fast_lookup",
            "duration_ms": round(fast_lookup_time, 2)
        })
        
        if error:
            errors.append(error.to_dict())
            self.error_logger.log_error(error)
            # Continue to RAG as fallback
        
        # Fast lookup hit?
        if lookup_result and lookup_result.found:
            results.append({
                "claim_text": text[:200],
                "verdict": lookup_result.fact.verdict.value,
                "confidence": lookup_result.similarity,
                "explanation": f"Match avec fait connu: {lookup_result.fact.canonical_claim}",
                "sources": [s.to_dict() for s in lookup_result.fact.sources],
                "verification_method": "fast_lookup"
            })
            
            timing["total_ms"] = round((time.perf_counter() - total_start) * 1000, 2)
            return PipelineResult(
                status="success",
                claims_detected=claims_detected,
                claims_verified=1,
                results=results,
                timing=timing,
                errors=errors
            )
        
        # ====================================================================
        # STAGE 3: RAG Verification (with graceful degradation)
        # ====================================================================
        degraded = False
        degradation_reason = ""
        
        # Check if RAG should be skipped
        if self.config.enable_graceful_degradation and self.degradation.should_skip_rag():
            state = self.degradation.get_state()
            degraded = True
            degradation_reason = state.get("rag_disable_reason", "RAG désactivé")
            
            results.append({
                "claim_text": text[:200],
                "verdict": "NOT_VERIFIABLE",
                "confidence": 0.0,
                "explanation": f"Vérification RAG non disponible: {degradation_reason}",
                "sources": [],
                "verification_method": "degraded"
            })
        else:
            # Try RAG
            if not self._init_rag():
                results.append({
                    "claim_text": text[:200],
                    "verdict": "NOT_VERIFIABLE",
                    "confidence": 0.0,
                    "explanation": "Module RAG non disponible",
                    "sources": [],
                    "verification_method": "error"
                })
            else:
                stage_start = time.perf_counter()
                
                # Run RAG in executor to not block
                loop = asyncio.get_event_loop()
                # Capture text dans une closure
                _text = text
                rag_result, error = await loop.run_in_executor(
                    self._executor,
                    lambda: self.timeout_mgr.run_with_timeout(
                        lambda: self._rag_checker.verify(_text),
                        timeout_ms=self.config.rag_timeout,
                        stage="rag_verification",
                        default=None
                    )
                )
                
                rag_time = (time.perf_counter() - stage_start) * 1000
                timing["stages"].append({
                    "stage": "rag_verification",
                    "duration_ms": round(rag_time, 2)
                })
                
                if error:
                    errors.append(error.to_dict())
                    self.error_logger.log_error(error)
                    
                    if error.error_type == PipelineError.TIMEOUT:
                        self.degradation.report_rag_timeout()
                    
                    results.append({
                        "claim_text": text[:200],
                        "verdict": "NOT_VERIFIABLE",
                        "confidence": 0.0,
                        "explanation": f"Erreur RAG: {error.message}",
                        "sources": [],
                        "verification_method": "error"
                    })
                elif rag_result:
                    self.degradation.report_rag_success()
                    
                    results.append({
                        "claim_text": text[:200],
                        "verdict": rag_result.verdict.value,
                        "confidence": rag_result.confidence,
                        "explanation": rag_result.justification,
                        "sources": [s.to_dict() for s in rag_result.sources],
                        "verification_method": "rag"
                    })
        
        # ====================================================================
        # Finalize
        # ====================================================================
        timing["total_ms"] = round((time.perf_counter() - total_start) * 1000, 2)
        
        # Determine status
        status = "success"
        if errors:
            status = "partial" if results else "error"
        
        return PipelineResult(
            status=status,
            claims_detected=claims_detected,
            claims_verified=len([r for r in results if r.get("verdict") not in ["SKIPPED", "NOT_VERIFIABLE"]]),
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
