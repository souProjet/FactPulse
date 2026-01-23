"""
Claim Detection Module

Détecte si une phrase contient une affirmation factuelle vérifiable.
Utilise un modèle transformer léger (MiniLM) pour une inférence < 100ms.

Performance:
- Latence cible: < 100ms par phrase
- GPU compatible avec batching
- Haute précision privilégiée (faux positifs coûteux)

Évaluation de la précision:
1. Dataset annoté manuellement (claims vs non-claims)
2. Matrice de confusion: TP, FP, TN, FN
3. Précision = TP / (TP + FP) - métrique principale
4. Recall = TP / (TP + FN) - métrique secondaire
5. F1 = 2 * (Precision * Recall) / (Precision + Recall)

Pour améliorer la précision:
- Augmenter le seuil de confiance (default: 0.7)
- Fine-tuner sur un dataset de claims en français
- Utiliser un ensemble de modèles
"""

import re
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel


@dataclass
class ClaimDetectionResult:
    """Résultat de détection de claim."""
    text: str
    is_claim: bool
    confidence: float
    processing_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "is_claim": self.is_claim,
            "confidence": round(self.confidence, 4),
            "processing_time_ms": round(self.processing_time_ms, 2)
        }


class ClaimClassificationHead(nn.Module):
    """
    Tête de classification pour la détection de claims.
    Architecture simple: Linear -> ReLU -> Dropout -> Linear -> Sigmoid
    """
    
    def __init__(self, input_dim: int = 384, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class ClaimDetector:
    """
    Détecteur de claims factuels utilisant MiniLM.
    
    Caractéristiques:
    - Modèle léger (22M params) pour inférence rapide
    - Support GPU avec FP16
    - Batching pour efficacité
    - Seuil ajustable pour précision/recall
    
    Usage:
        detector = ClaimDetector(device="cuda")
        result = detector.detect("La Terre est plate.")
        # {"is_claim": True, "confidence": 0.92}
    """
    
    # Indicateurs heuristiques de claims factuels
    CLAIM_INDICATORS = [
        r'\d+%',                          # Pourcentages
        r'\d+\s*(millions?|milliards?)',  # Grands nombres
        r'\b(est|sont|a|ont|fait|font)\b',  # Verbes d'état
        r'\b(toujours|jamais|tous|aucun)\b',  # Absolus
        r'\b(cause|provoque|entraîne)\b',  # Causalité
        r'\b(prouvé|démontré|confirmé)\b',  # Certitude
        r'\b(selon|d\'après)\b',          # Citations
    ]
    
    # Indicateurs de NON-claims
    NON_CLAIM_INDICATORS = [
        r'\?$',                           # Questions
        r'\b(je pense|à mon avis|peut-être)\b',  # Opinions
        r'\b(devrait|pourrait|serait)\b',  # Conditionnel
        r'^(bonjour|salut|merci|svp)\b',  # Salutations
    ]
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        threshold: float = 0.5,  # MVP: seuil bas car modèle non entraîné
        max_length: int = 128,
        batch_size: int = 16
    ):
        """
        Initialise le détecteur de claims.
        
        Args:
            model_name: Nom du modèle HuggingFace
            device: Device torch ('cuda' ou 'cpu')
            threshold: Seuil de décision (plus haut = plus de précision)
            max_length: Longueur max des tokens
            batch_size: Taille des batches
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.max_length = max_length
        self.batch_size = batch_size
        
        self._tokenizer = None
        self._encoder = None
        self._classifier = None
        self._loaded = False
        
        # Compiler les patterns
        self._claim_patterns = [re.compile(p, re.IGNORECASE) for p in self.CLAIM_INDICATORS]
        self._non_claim_patterns = [re.compile(p, re.IGNORECASE) for p in self.NON_CLAIM_INDICATORS]
    
    def load(self) -> None:
        """Charge le modèle et le tokenizer."""
        if self._loaded:
            return
        
        # Tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Encodeur (MiniLM)
        self._encoder = AutoModel.from_pretrained(self.model_name)
        self._encoder.to(self.device)
        self._encoder.eval()
        
        # Utiliser FP16 sur GPU pour performance
        if "cuda" in self.device:
            self._encoder = self._encoder.half()
        
        # Tête de classification
        hidden_size = self._encoder.config.hidden_size
        self._classifier = ClaimClassificationHead(input_dim=hidden_size)
        self._classifier.to(self.device)
        self._classifier.eval()
        
        if "cuda" in self.device:
            self._classifier = self._classifier.half()
        
        self._loaded = True
    
    def _preprocess(self, text: str) -> str:
        """
        Prétraitement du texte.
        
        - Normalisation des espaces
        - Suppression des URLs
        - Nettoyage des caractères spéciaux
        """
        # Supprimer URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Normaliser les espaces
        text = re.sub(r'\s+', ' ', text)
        
        # Supprimer les emojis (garder le texte)
        text = re.sub(r'[^\w\s\'\"\-\.,!?;:()%°€$]', '', text)
        
        return text.strip()
    
    def _compute_heuristic_score(self, text: str) -> float:
        """
        Score heuristique basé sur des patterns.
        Utilisé pour ajuster la confiance du modèle.
        
        Returns:
            Score entre -1 (non-claim) et +1 (claim)
        """
        score = 0.0
        
        # Indicateurs positifs
        for pattern in self._claim_patterns:
            if pattern.search(text):
                score += 0.15
        
        # Indicateurs négatifs
        for pattern in self._non_claim_patterns:
            if pattern.search(text):
                score -= 0.3
        
        # Longueur (phrases très courtes souvent non-claims)
        word_count = len(text.split())
        if word_count < 4:
            score -= 0.2
        elif word_count > 8:
            score += 0.1
        
        return max(-1.0, min(1.0, score))
    
    def _mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling des embeddings."""
        token_embeddings = model_output[0]
        # Garder le même dtype que les embeddings (FP16 sur GPU)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    @torch.no_grad()
    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Génère les embeddings pour une liste de textes."""
        # Tokenization
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Encoder
        outputs = self._encoder(**encoded)
        
        # Mean pooling
        embeddings = self._mean_pooling(outputs, encoded['attention_mask'])
        
        # Normalisation L2
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    @torch.no_grad()
    def detect(self, text: str) -> ClaimDetectionResult:
        """
        Détecte si un texte contient un claim factuel.
        
        Args:
            text: Texte à analyser
            
        Returns:
            ClaimDetectionResult avec is_claim, confidence, et temps
        """
        start_time = time.perf_counter()
        
        # Charger si nécessaire
        self.load()
        
        # Prétraitement
        clean_text = self._preprocess(text)
        
        if not clean_text or len(clean_text) < 3:
            return ClaimDetectionResult(
                text=text,
                is_claim=False,
                confidence=0.0,
                processing_time_ms=(time.perf_counter() - start_time) * 1000
            )
        
        # Sync GPU pour timing précis
        if "cuda" in self.device:
            torch.cuda.synchronize()
        
        # Embedding
        embedding = self._get_embeddings([clean_text])
        
        # Classification (modèle non-entraîné pour le MVP)
        model_score = self._classifier(embedding).item()
        
        # Score heuristique
        heuristic_score = self._compute_heuristic_score(clean_text)
        
        # MVP: privilégier l'heuristique (modèle non entraîné)
        # Quand le modèle sera entraîné, revenir à 70/30
        # combined_score = 0.7 * model_score + 0.3 * ((heuristic_score + 1) / 2)
        combined_score = 0.3 * model_score + 0.7 * ((heuristic_score + 1) / 2)
        
        # Sync GPU
        if "cuda" in self.device:
            torch.cuda.synchronize()
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return ClaimDetectionResult(
            text=text,
            is_claim=combined_score >= self.threshold,
            confidence=combined_score,
            processing_time_ms=processing_time
        )
    
    @torch.no_grad()
    def detect_batch(self, texts: List[str]) -> List[ClaimDetectionResult]:
        """
        Détection par batch pour efficacité.
        
        Args:
            texts: Liste de textes à analyser
            
        Returns:
            Liste de ClaimDetectionResult
        """
        start_time = time.perf_counter()
        
        self.load()
        
        results = []
        
        # Prétraiter tous les textes
        clean_texts = [self._preprocess(t) for t in texts]
        
        # Traiter par batches
        for i in range(0, len(clean_texts), self.batch_size):
            batch_start = time.perf_counter()
            batch_texts = clean_texts[i:i + self.batch_size]
            original_texts = texts[i:i + self.batch_size]
            
            # Filtrer les textes vides
            valid_indices = [j for j, t in enumerate(batch_texts) if t and len(t) >= 3]
            valid_texts = [batch_texts[j] for j in valid_indices]
            
            if not valid_texts:
                for orig in original_texts:
                    results.append(ClaimDetectionResult(
                        text=orig,
                        is_claim=False,
                        confidence=0.0,
                        processing_time_ms=0.0
                    ))
                continue
            
            # Sync GPU
            if "cuda" in self.device:
                torch.cuda.synchronize()
            
            # Embeddings batch
            embeddings = self._get_embeddings(valid_texts)
            
            # Classification batch
            model_scores = self._classifier(embeddings).squeeze(-1).cpu().numpy()
            
            # Sync GPU
            if "cuda" in self.device:
                torch.cuda.synchronize()
            
            batch_time = (time.perf_counter() - batch_start) * 1000
            time_per_item = batch_time / len(batch_texts)
            
            # Construire résultats
            valid_idx = 0
            for j, (orig, clean) in enumerate(zip(original_texts, batch_texts)):
                if j in valid_indices:
                    model_score = float(model_scores[valid_idx]) if model_scores.ndim > 0 else float(model_scores)
                    heuristic_score = self._compute_heuristic_score(clean)
                    combined = 0.7 * model_score + 0.3 * ((heuristic_score + 1) / 2)
                    valid_idx += 1
                    
                    results.append(ClaimDetectionResult(
                        text=orig,
                        is_claim=combined >= self.threshold,
                        confidence=combined,
                        processing_time_ms=time_per_item
                    ))
                else:
                    results.append(ClaimDetectionResult(
                        text=orig,
                        is_claim=False,
                        confidence=0.0,
                        processing_time_ms=time_per_item
                    ))
        
        return results
    
    def set_threshold(self, threshold: float) -> None:
        """
        Ajuste le seuil de décision.
        
        - Plus haut (0.8-0.9): Meilleure précision, moins de détections
        - Plus bas (0.5-0.6): Meilleur recall, plus de faux positifs
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1")
        self.threshold = threshold


# ============================================================================
# Notes sur la performance attendue
# ============================================================================
"""
PERFORMANCE ATTENDUE (RTX 5060):

| Métrique | Valeur cible | Notes |
|----------|--------------|-------|
| Latence single | < 50ms | Avec warm-up |
| Latence batch (16) | < 100ms | ~6ms/item |
| Mémoire GPU | ~500MB | MiniLM FP16 |
| Précision | > 85% | Sur dataset annoté |
| Recall | > 70% | Secondaire |

COMMENT ÉVALUER LA PRÉCISION:

1. Créer un dataset de test annoté:
   - 500+ phrases avec labels (claim / non-claim)
   - Équilibré entre catégories
   - Inclure cas difficiles (opinions, questions rhétoriques)

2. Calculer métriques:
   ```python
   from sklearn.metrics import precision_score, recall_score, f1_score
   
   y_true = [...]  # Labels réels
   y_pred = [detector.detect(text).is_claim for text in texts]
   
   precision = precision_score(y_true, y_pred)
   recall = recall_score(y_true, y_pred)
   f1 = f1_score(y_true, y_pred)
   ```

3. Ajuster le seuil pour maximiser précision:
   ```python
   thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
   for t in thresholds:
       detector.set_threshold(t)
       # Recalculer et comparer
   ```

AMÉLIORATIONS POSSIBLES:
- Fine-tuning sur ClaimBuster dataset
- Ensemble avec règles linguistiques
- Modèle multilingue (XLM-RoBERTa)
"""
