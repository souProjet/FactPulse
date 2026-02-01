"""
FactPulse FastAPI Backend

API principale avec fail-safe intégré.
Endpoint: POST /analyze

Performance cible: < 2 secondes
"""

import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import logging

from .utils.timing import reset_timings, get_timing_stats
from .utils.config import load_config, get_config
from .pipeline.orchestrator import FactCheckPipeline, PipelineConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("factpulse.api")


# ============================================================================
# Pydantic Models
# ============================================================================

class AnalyzeRequest(BaseModel):
    """Requête d'analyse de texte."""
    text: str = Field(..., min_length=1, max_length=50000, description="Texte à analyser")
    fast_mode: bool = Field(False, description="Mode rapide (sans RAG)")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "La Terre est plate. L'eau bout à 100°C.", "fast_mode": False}
            ]
        }
    }


class ClaimResult(BaseModel):
    """Résultat pour un claim."""
    claim_text: str
    verdict: str
    confidence: float
    explanation: str
    sources: List[Dict[str, Any]]
    verification_method: str


class AnalyzeResponse(BaseModel):
    """Réponse de l'analyse."""
    status: str
    claims_detected: int
    claims_verified: int
    results: List[ClaimResult]
    timing: Dict[str, Any]
    errors: List[Dict[str, Any]] = []
    degraded: bool = False
    degradation_reason: str = ""


class HealthResponse(BaseModel):
    """Réponse du health check."""
    status: str
    version: str
    gpu_available: bool
    models_loaded: bool


class FeedbackRequest(BaseModel):
    """Requête de feedback utilisateur."""
    claim_text: str
    predicted_verdict: str
    actual_verdict: str
    confidence: float
    feedback: Optional[str] = None


# ============================================================================
# Pipeline Instance
# ============================================================================

pipeline: Optional[FactCheckPipeline] = None


def get_pipeline() -> FactCheckPipeline:
    """Récupère l'instance du pipeline."""
    global pipeline
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline non initialisé"
        )
    return pipeline


# ============================================================================
# Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application."""
    global pipeline
    
    # Startup
    logger.info("Démarrage de FactPulse...")
    
    try:
        load_config()
        
        config = PipelineConfig(
            claim_detection_timeout=5000,
            fast_lookup_timeout=1000,
            rag_timeout=8000,  # Qwen2.5-0.5B est rapide
            total_timeout=15000,
            claim_confidence_threshold=0.4,
            check_worthiness_threshold=0.3,
            fast_lookup_similarity_threshold=0.6,
            enable_graceful_degradation=True,
            log_false_positives=True,
            max_claims_per_request=10
        )
        
        pipeline = FactCheckPipeline(config=config, device="cuda")
        pipeline.initialize()
        
        # Précharger le module RAG (modèles LLM) pour éviter les timeouts
        logger.info("Préchargement du module RAG (cela peut prendre 30-60 secondes)...")
        pipeline._init_rag()
        logger.info("Module RAG préchargé avec succès")
        
        logger.info("FactPulse démarré avec succès - prêt à analyser")
        
    except Exception as e:
        logger.error(f"Erreur au démarrage: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Arrêt de FactPulse...")
    if pipeline:
        pipeline.shutdown()
    logger.info("FactPulse arrêté")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="FactPulse API",
    description="API de fact-checking en temps réel",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Gestion globale des exceptions."""
    logger.error(f"Erreur non gérée: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Erreur interne du serveur",
            "detail": str(exc) if app.debug else None
        }
    )


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Vérifie l'état du serveur.
    
    Returns:
        Status du serveur et des composants
    """
    import torch
    
    global pipeline
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        gpu_available=torch.cuda.is_available(),
        models_loaded=pipeline is not None and pipeline._initialized
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest):
    """
    Analyse un texte pour détecter et vérifier les claims.
    
    Pipeline:
    1. Détection de claims (< 200ms)
    2. Lookup rapide local (< 500ms)
    3. Vérification RAG si nécessaire (< 1500ms)
    
    Total: < 2 secondes
    
    Args:
        request: Texte à analyser
        
    Returns:
        Résultats de vérification avec timing
    """
    pipe = get_pipeline()
    
    # Mode rapide (sans RAG)
    if request.fast_mode:
        pipe.degradation.enable_fast_mode()
    
    try:
        result = await pipe.analyze(request.text)
        
        return AnalyzeResponse(
            status=result.status,
            claims_detected=result.claims_detected,
            claims_verified=result.claims_verified,
            results=[ClaimResult(**r) for r in result.results],
            timing=result.timing,
            errors=result.errors,
            degraded=result.degraded,
            degradation_reason=result.degradation_reason
        )
        
    except Exception as e:
        logger.error(f"Erreur d'analyse: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur d'analyse: {str(e)}"
        )
    finally:
        # Désactiver le mode rapide après la requête
        if request.fast_mode:
            pipe.degradation.disable_fast_mode()


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Soumet un feedback sur un faux positif.
    
    Utilisé pour améliorer le modèle.
    """
    pipe = get_pipeline()
    
    pipe.log_false_positive(
        claim=request.claim_text,
        predicted=request.predicted_verdict,
        actual=request.actual_verdict,
        confidence=request.confidence,
        feedback=request.feedback
    )
    
    return {"status": "logged", "message": "Feedback enregistré"}


@app.get("/stats")
async def get_stats():
    """
    Récupère les statistiques du pipeline.
    
    Returns:
        Stats GPU, erreurs, dégradation
    """
    import torch
    
    pipe = get_pipeline()
    
    return {
        "gpu": {
            "available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "memory_allocated_mb": torch.cuda.memory_allocated(0) / 1024**2 if torch.cuda.is_available() else 0
        },
        "pipeline": {
            "initialized": pipe._initialized,
            "degradation": pipe.get_degradation_state()
        },
        "errors": pipe.get_error_stats()
    }


@app.post("/reset-degradation")
async def reset_degradation():
    """
    Réinitialise l'état de dégradation.
    
    Réactive RAG si désactivé.
    """
    pipe = get_pipeline()
    pipe.reset_degradation()
    
    return {
        "status": "reset",
        "degradation": pipe.get_degradation_state()
    }


@app.get("/")
async def root():
    """Page d'accueil."""
    return {
        "name": "FactPulse API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "backend.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1  # Single worker pour GPU
    )
