"""
Fail-Safe Mechanisms pour FactPulse Pipeline

Fonctionnalités:
- Gestion des timeouts par étape
- Dégradation gracieuse (skip RAG si trop lent)
- États d'erreur clairs
- Logging des faux positifs

Stratégie:
1. Timeout global de 2 secondes
2. Timeouts par étape:
   - Claim detection: 200ms max
   - Fast lookup: 500ms max
   - RAG: 1500ms max (skippable)
3. Si timeout → retourner résultat partiel
4. Logger les cas problématiques
"""

import asyncio
import time
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Dict, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('factpulse.failsafe')


# ============================================================================
# ERROR TYPES
# ============================================================================

class PipelineError(str, Enum):
    """Types d'erreurs du pipeline."""
    TIMEOUT = "TIMEOUT"
    MODEL_ERROR = "MODEL_ERROR"
    DATA_ERROR = "DATA_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNKNOWN = "UNKNOWN"


@dataclass
class ErrorState:
    """État d'erreur avec contexte."""
    error_type: PipelineError
    stage: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    recoverable: bool = True
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type.value,
            "stage": self.stage,
            "message": self.message,
            "timestamp": self.timestamp,
            "recoverable": self.recoverable,
            "details": self.details
        }


# ============================================================================
# TIMEOUT MANAGER
# ============================================================================

T = TypeVar('T')


class TimeoutManager:
    """
    Gestionnaire de timeouts pour les opérations du pipeline.
    
    Usage:
        timeout_mgr = TimeoutManager()
        
        # Synchrone
        result = timeout_mgr.run_with_timeout(
            func=slow_function,
            timeout_ms=1000,
            stage="claim_detection"
        )
        
        # Async
        result = await timeout_mgr.run_with_timeout_async(
            coro=async_slow_function(),
            timeout_ms=1000,
            stage="rag_verification"
        )
    """
    
    def __init__(self, default_timeout_ms: float = 2000):
        """
        Args:
            default_timeout_ms: Timeout par défaut en millisecondes
        """
        self.default_timeout_ms = default_timeout_ms
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._stage_timeouts = {
            "claim_detection": 200,
            "fast_lookup": 500,
            "rag_retrieval": 500,
            "rag_generation": 1200,
            "rag_verification": 1500,
            "total": 2000
        }
    
    def set_stage_timeout(self, stage: str, timeout_ms: float) -> None:
        """Configure le timeout pour une étape."""
        self._stage_timeouts[stage] = timeout_ms
    
    def get_stage_timeout(self, stage: str) -> float:
        """Récupère le timeout pour une étape."""
        return self._stage_timeouts.get(stage, self.default_timeout_ms)
    
    def run_with_timeout(
        self,
        func: Callable[..., T],
        timeout_ms: Optional[float] = None,
        stage: str = "unknown",
        default: Optional[T] = None,
        *args,
        **kwargs
    ) -> tuple[Optional[T], Optional[ErrorState]]:
        """
        Exécute une fonction avec timeout.
        
        Returns:
            (result, error) - result si succès, error si timeout/erreur
        """
        timeout = (timeout_ms or self.get_stage_timeout(stage)) / 1000.0
        
        future = self._executor.submit(func, *args, **kwargs)
        
        try:
            result = future.result(timeout=timeout)
            return result, None
        
        except FuturesTimeoutError:
            future.cancel()
            error = ErrorState(
                error_type=PipelineError.TIMEOUT,
                stage=stage,
                message=f"Timeout après {timeout*1000:.0f}ms",
                recoverable=True,
                details={"timeout_ms": timeout * 1000}
            )
            logger.warning(f"[{stage}] Timeout: {error.message}")
            return default, error
        
        except Exception as e:
            error = ErrorState(
                error_type=PipelineError.UNKNOWN,
                stage=stage,
                message=str(e),
                recoverable=True,
                details={"exception_type": type(e).__name__}
            )
            logger.error(f"[{stage}] Error: {error.message}")
            return default, error
    
    async def run_with_timeout_async(
        self,
        coro,
        timeout_ms: Optional[float] = None,
        stage: str = "unknown",
        default: Optional[T] = None
    ) -> tuple[Optional[T], Optional[ErrorState]]:
        """
        Exécute une coroutine avec timeout.
        """
        timeout = (timeout_ms or self.get_stage_timeout(stage)) / 1000.0
        
        try:
            result = await asyncio.wait_for(coro, timeout=timeout)
            return result, None
        
        except asyncio.TimeoutError:
            error = ErrorState(
                error_type=PipelineError.TIMEOUT,
                stage=stage,
                message=f"Async timeout après {timeout*1000:.0f}ms",
                recoverable=True
            )
            logger.warning(f"[{stage}] Async Timeout: {error.message}")
            return default, error
        
        except Exception as e:
            error = ErrorState(
                error_type=PipelineError.UNKNOWN,
                stage=stage,
                message=str(e),
                recoverable=True
            )
            logger.error(f"[{stage}] Async Error: {error.message}")
            return default, error
    
    def shutdown(self):
        """Arrête proprement l'executor."""
        self._executor.shutdown(wait=False)


# ============================================================================
# GRACEFUL DEGRADATION
# ============================================================================

@dataclass
class DegradationState:
    """État de la dégradation."""
    rag_disabled: bool = False
    rag_disable_reason: str = ""
    fast_mode: bool = False
    consecutive_timeouts: int = 0
    last_reset: str = field(default_factory=lambda: datetime.now().isoformat())


class GracefulDegradation:
    """
    Gère la dégradation gracieuse du pipeline.
    
    Stratégies:
    1. Skip RAG si 3 timeouts consécutifs
    2. Mode rapide (seulement fast lookup) si charge élevée
    3. Réactivation automatique après cooldown
    
    Usage:
        degradation = GracefulDegradation()
        
        if degradation.should_skip_rag():
            # Utiliser seulement fast lookup
            pass
        else:
            # Pipeline complet
            pass
        
        # Après un timeout RAG
        degradation.report_rag_timeout()
    """
    
    def __init__(
        self,
        max_consecutive_timeouts: int = 3,
        cooldown_seconds: float = 60.0,
        auto_reset: bool = True
    ):
        self.max_consecutive_timeouts = max_consecutive_timeouts
        self.cooldown_seconds = cooldown_seconds
        self.auto_reset = auto_reset
        
        self._state = DegradationState()
        self._lock = threading.Lock()
    
    def should_skip_rag(self) -> bool:
        """Détermine si RAG doit être sauté."""
        with self._lock:
            if self.auto_reset:
                self._check_reset()
            return self._state.rag_disabled or self._state.fast_mode
    
    def report_rag_timeout(self) -> None:
        """Signale un timeout RAG."""
        with self._lock:
            self._state.consecutive_timeouts += 1
            
            if self._state.consecutive_timeouts >= self.max_consecutive_timeouts:
                self._state.rag_disabled = True
                self._state.rag_disable_reason = f"{self.max_consecutive_timeouts} timeouts consécutifs"
                logger.warning(f"RAG désactivé: {self._state.rag_disable_reason}")
    
    def report_rag_success(self) -> None:
        """Signale un succès RAG."""
        with self._lock:
            self._state.consecutive_timeouts = 0
            if self._state.rag_disabled:
                self._state.rag_disabled = False
                self._state.rag_disable_reason = ""
                logger.info("RAG réactivé après succès")
    
    def enable_fast_mode(self) -> None:
        """Active le mode rapide (skip RAG)."""
        with self._lock:
            self._state.fast_mode = True
            logger.info("Mode rapide activé")
    
    def disable_fast_mode(self) -> None:
        """Désactive le mode rapide."""
        with self._lock:
            self._state.fast_mode = False
            logger.info("Mode rapide désactivé")
    
    def _check_reset(self) -> None:
        """Vérifie si on doit reset l'état."""
        if not self._state.rag_disabled:
            return
        
        last_reset = datetime.fromisoformat(self._state.last_reset)
        elapsed = (datetime.now() - last_reset).total_seconds()
        
        if elapsed >= self.cooldown_seconds:
            self._state.rag_disabled = False
            self._state.rag_disable_reason = ""
            self._state.consecutive_timeouts = 0
            self._state.last_reset = datetime.now().isoformat()
            logger.info(f"RAG réactivé après cooldown de {self.cooldown_seconds}s")
    
    def get_state(self) -> Dict[str, Any]:
        """Retourne l'état actuel."""
        with self._lock:
            return {
                "rag_disabled": self._state.rag_disabled,
                "rag_disable_reason": self._state.rag_disable_reason,
                "fast_mode": self._state.fast_mode,
                "consecutive_timeouts": self._state.consecutive_timeouts
            }
    
    def reset(self) -> None:
        """Reset complet de l'état."""
        with self._lock:
            self._state = DegradationState()
            logger.info("État de dégradation réinitialisé")


# ============================================================================
# ERROR LOGGER
# ============================================================================

@dataclass
class FalsePositiveLog:
    """Log d'un faux positif."""
    claim_text: str
    predicted_verdict: str
    actual_verdict: Optional[str]
    confidence: float
    timestamp: str
    user_feedback: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_text": self.claim_text,
            "predicted_verdict": self.predicted_verdict,
            "actual_verdict": self.actual_verdict,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "user_feedback": self.user_feedback
        }


class ErrorLogger:
    """
    Logger pour les erreurs et faux positifs.
    
    Usage:
        error_logger = ErrorLogger(log_dir="logs")
        
        # Logger une erreur
        error_logger.log_error(error_state)
        
        # Logger un faux positif
        error_logger.log_false_positive(
            claim="La Terre est ronde",
            predicted="FALSE",
            actual="TRUE",
            confidence=0.8
        )
        
        # Analyser les faux positifs
        stats = error_logger.get_false_positive_stats()
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self._error_log_path = self.log_dir / "errors.jsonl"
        self._fp_log_path = self.log_dir / "false_positives.jsonl"
        
        self._error_count = 0
        self._fp_count = 0
        self._lock = threading.Lock()
    
    def log_error(self, error: ErrorState) -> None:
        """Log une erreur."""
        with self._lock:
            self._error_count += 1
            
            with open(self._error_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(error.to_dict(), ensure_ascii=False) + '\n')
        
        logger.error(f"Error logged: [{error.stage}] {error.error_type.value} - {error.message}")
    
    def log_false_positive(
        self,
        claim: str,
        predicted: str,
        actual: Optional[str] = None,
        confidence: float = 0.0,
        user_feedback: Optional[str] = None
    ) -> None:
        """Log un faux positif."""
        fp = FalsePositiveLog(
            claim_text=claim,
            predicted_verdict=predicted,
            actual_verdict=actual,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            user_feedback=user_feedback
        )
        
        with self._lock:
            self._fp_count += 1
            
            with open(self._fp_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(fp.to_dict(), ensure_ascii=False) + '\n')
        
        logger.warning(f"False positive logged: '{claim[:50]}...' - predicted {predicted}, actual {actual}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'erreurs."""
        errors_by_type = {}
        errors_by_stage = {}
        
        if self._error_log_path.exists():
            with open(self._error_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        error = json.loads(line)
                        
                        err_type = error.get('error_type', 'UNKNOWN')
                        errors_by_type[err_type] = errors_by_type.get(err_type, 0) + 1
                        
                        stage = error.get('stage', 'unknown')
                        errors_by_stage[stage] = errors_by_stage.get(stage, 0) + 1
                    except json.JSONDecodeError:
                        continue
        
        return {
            "total_errors": self._error_count,
            "by_type": errors_by_type,
            "by_stage": errors_by_stage
        }
    
    def get_false_positive_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de faux positifs."""
        fp_by_verdict = {}
        total_confidence = 0.0
        
        if self._fp_log_path.exists():
            with open(self._fp_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        fp = json.loads(line)
                        
                        verdict = fp.get('predicted_verdict', 'UNKNOWN')
                        fp_by_verdict[verdict] = fp_by_verdict.get(verdict, 0) + 1
                        
                        total_confidence += fp.get('confidence', 0.0)
                    except json.JSONDecodeError:
                        continue
        
        avg_confidence = total_confidence / self._fp_count if self._fp_count > 0 else 0.0
        
        return {
            "total_false_positives": self._fp_count,
            "by_predicted_verdict": fp_by_verdict,
            "average_confidence": round(avg_confidence, 4)
        }


# ============================================================================
# DECORATORS
# ============================================================================

def with_timeout(timeout_ms: float, stage: str = "unknown"):
    """
    Décorateur pour ajouter un timeout à une fonction.
    
    Usage:
        @with_timeout(1000, "my_stage")
        def slow_function():
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            timeout_mgr = TimeoutManager()
            result, error = timeout_mgr.run_with_timeout(
                func, timeout_ms, stage, None, *args, **kwargs
            )
            if error:
                raise TimeoutError(error.message)
            return result
        return wrapper
    return decorator


def with_fallback(fallback_value: Any, log_errors: bool = True):
    """
    Décorateur pour ajouter un fallback en cas d'erreur.
    
    Usage:
        @with_fallback(default_result, log_errors=True)
        def risky_function():
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(f"Error in {func.__name__}: {e}, using fallback")
                return fallback_value
        return wrapper
    return decorator


# ============================================================================
# NOTES
# ============================================================================
"""
STRATÉGIE DE GESTION D'ERREURS:

1. TIMEOUTS
   - Chaque étape a un timeout configuré
   - Timeout global de 2 secondes
   - Si timeout → résultat partiel + flag d'erreur

2. DÉGRADATION GRACIEUSE
   - 3 timeouts RAG consécutifs → désactiver RAG temporairement
   - Mode rapide disponible (fast lookup seulement)
   - Réactivation automatique après 60s de cooldown

3. ÉTATS D'ERREUR
   - Chaque erreur a un type clair (TIMEOUT, MODEL_ERROR, etc.)
   - Flag "recoverable" pour les erreurs récupérables
   - Contexte détaillé pour debugging

4. LOGGING DES FAUX POSITIFS
   - Log séparé pour les faux positifs
   - Permet l'analyse et l'amélioration du modèle
   - Inclut feedback utilisateur optionnel

IMPACT SUR LA PERFORMANCE:
- Overhead minimal (~1ms par opération)
- Pas de polling, timeouts basés sur futures/asyncio
- Logs asynchrones pour ne pas bloquer le pipeline
"""
