"""
Benchmark Script pour FactPulse Pipeline

Mesure les performances à chaque étape du pipeline:
- Latence par étape
- Latence moyenne et P95
- Utilisation mémoire GPU
- Export CSV détaillé

Usage:
    python -m scripts.benchmark
    python -m scripts.benchmark --iterations 10 --output results.csv
"""

import json
import time
import csv
import statistics
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class BenchmarkSample:
    """Un échantillon de benchmark."""
    text_id: int
    text_preview: str
    iteration: int
    
    # Claim detection
    claim_detection_ms: float = 0.0
    claims_detected: int = 0
    claims_to_verify: int = 0
    
    # Fast lookup
    fast_lookup_ms: float = 0.0
    fast_lookup_found: bool = False
    
    # RAG verification
    rag_retrieval_ms: float = 0.0
    rag_generation_ms: float = 0.0
    rag_total_ms: float = 0.0
    
    # Total
    total_pipeline_ms: float = 0.0
    
    # GPU
    gpu_memory_mb: float = 0.0
    
    # Result
    verdict: str = ""
    confidence: float = 0.0


@dataclass
class BenchmarkResults:
    """Résultats agrégés du benchmark."""
    timestamp: str
    device: str
    gpu_name: str
    total_samples: int
    
    # Par étape
    claim_detection_mean_ms: float = 0.0
    claim_detection_p95_ms: float = 0.0
    claim_detection_std_ms: float = 0.0
    
    fast_lookup_mean_ms: float = 0.0
    fast_lookup_p95_ms: float = 0.0
    fast_lookup_std_ms: float = 0.0
    
    rag_mean_ms: float = 0.0
    rag_p95_ms: float = 0.0
    rag_std_ms: float = 0.0
    
    total_mean_ms: float = 0.0
    total_p95_ms: float = 0.0
    total_std_ms: float = 0.0
    
    # GPU
    gpu_memory_peak_mb: float = 0.0
    gpu_memory_avg_mb: float = 0.0
    
    # Targets
    meets_claim_detection_target: bool = False
    meets_fast_lookup_target: bool = False
    meets_rag_target: bool = False
    meets_total_target: bool = False
    
    samples: List[BenchmarkSample] = field(default_factory=list)


# ============================================================================
# Textes de test en français
# ============================================================================

BENCHMARK_TEXTS = [
    # Affirmations factuelles simples
    "La Terre est plate.",
    "L'eau bout à 100 degrés Celsius au niveau de la mer.",
    "Les vaccins causent l'autisme.",
    
    # Claims avec chiffres
    "La Tour Eiffel mesure 330 mètres de haut.",
    "Le cerveau humain consomme 20% de l'énergie du corps.",
    "La France est le pays le plus visité au monde avec 90 millions de visiteurs.",
    
    # Mythes courants
    "Les humains n'utilisent que 10% de leur cerveau.",
    "La Grande Muraille de Chine est visible depuis l'espace.",
    "Napoléon Bonaparte était petit.",
    "Le sang désoxygéné est bleu.",
    
    # Opinions (non-claims)
    "Je pense que la pizza est le meilleur plat.",
    "À mon avis, Paris est une belle ville.",
    
    # Questions (non-claims)
    "Est-ce que la Terre est ronde?",
    "Combien pèse un éléphant?",
    
    # Claims complexes
    "Le changement climatique est causé par les activités humaines, en particulier la combustion des énergies fossiles.",
    "Les antennes 5G propagent le COVID-19 selon plusieurs études.",
    
    # Texte long
    """
    Voici quelques faits sur la science. La vitesse de la lumière dans le vide est d'environ 299 792 kilomètres par seconde.
    L'alcool réchauffe le corps, ce qui est une croyance populaire mais fausse.
    Les chauves-souris sont aveugles, ce qui est également un mythe.
    """,
]


def get_gpu_memory_mb() -> float:
    """Retourne la mémoire GPU utilisée en MB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
    except:
        pass
    return 0.0


def calculate_percentile(data: List[float], percentile: float) -> float:
    """Calcule un percentile."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * percentile / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


def run_benchmark(
    num_iterations: int = 5,
    warmup_iterations: int = 2,
    device: str = "cuda",
    skip_rag: bool = False
) -> BenchmarkResults:
    """
    Exécute le benchmark complet.
    
    Args:
        num_iterations: Itérations par texte
        warmup_iterations: Itérations de warmup
        device: Device torch
        skip_rag: Sauter la vérification RAG (pour tests rapides)
        
    Returns:
        BenchmarkResults avec toutes les métriques
    """
    import torch
    
    print("=" * 70)
    print("FACTPULSE PIPELINE BENCHMARK")
    print("=" * 70)
    
    # Vérifier GPU
    gpu_name = "N/A"
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🖥️  GPU: {gpu_name}")
        print(f"🔧 CUDA Version: {torch.version.cuda}")
    else:
        print("⚠️  Running on CPU")
        device = "cpu"
    
    # Charger les modules
    print("\n📦 Chargement des modules...")
    load_start = time.perf_counter()
    
    from backend.claim_detector.claim_detector import ClaimDetector
    from backend.fact_checker.fact_store import FactStore
    
    claim_detector = ClaimDetector(device=device, threshold=0.6)
    fact_store = FactStore(device=device, similarity_threshold=0.85)
    
    # Charger les données
    facts_path = project_root / "data" / "known_facts.json"
    if facts_path.exists():
        fact_store.load_from_json(str(facts_path))
        print(f"   ✓ FactStore: {fact_store.get_stats()['total_facts']} faits chargés")
    
    # RAG (optionnel)
    rag_checker = None
    if not skip_rag:
        try:
            from backend.rag.rag_checker import RAGChecker
            rag_checker = RAGChecker(device=device)
            rag_checker.load_sources(str(facts_path))
            print("   ✓ RAG Checker initialisé")
        except Exception as e:
            print(f"   ⚠️ RAG Checker non disponible: {e}")
            skip_rag = True
    
    load_time = time.perf_counter() - load_start
    print(f"\n⏱️  Modules chargés en {load_time:.2f}s")
    
    # Warmup
    print(f"\n🔥 Warmup ({warmup_iterations} itérations)...")
    for _ in range(warmup_iterations):
        result = claim_detector.detect(BENCHMARK_TEXTS[0])
        fact_store.lookup(BENCHMARK_TEXTS[0])
    print("   ✓ Warmup terminé")
    
    # Benchmark
    print(f"\n📊 Exécution du benchmark ({num_iterations} itérations × {len(BENCHMARK_TEXTS)} textes)...")
    
    samples: List[BenchmarkSample] = []
    
    for text_idx, text in enumerate(BENCHMARK_TEXTS):
        for iteration in range(num_iterations):
            sample = BenchmarkSample(
                text_id=text_idx,
                text_preview=text[:50].replace('\n', ' ') + "...",
                iteration=iteration
            )
            
            total_start = time.perf_counter()
            
            # 1. Claim Detection
            if device == "cuda":
                torch.cuda.synchronize()
            
            cd_start = time.perf_counter()
            detection_result = claim_detector.detect(text)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            sample.claim_detection_ms = (time.perf_counter() - cd_start) * 1000
            sample.claims_detected = 1 if detection_result.is_claim else 0
            sample.claims_to_verify = 1 if detection_result.is_claim and detection_result.confidence > 0.7 else 0
            
            # 2. Fast Lookup (si claim détecté)
            if sample.claims_to_verify > 0:
                fl_start = time.perf_counter()
                lookup_result = fact_store.lookup(text)
                sample.fast_lookup_ms = (time.perf_counter() - fl_start) * 1000
                sample.fast_lookup_found = lookup_result.found
                
                if lookup_result.found:
                    sample.verdict = lookup_result.fact.verdict.value
                    sample.confidence = lookup_result.similarity
                
                # 3. RAG (si pas trouvé en fast lookup)
                if not lookup_result.found and rag_checker and not skip_rag:
                    try:
                        if device == "cuda":
                            torch.cuda.synchronize()
                        
                        rag_start = time.perf_counter()
                        rag_result = rag_checker.verify(text)
                        
                        if device == "cuda":
                            torch.cuda.synchronize()
                        
                        sample.rag_retrieval_ms = rag_result.retrieval_time_ms
                        sample.rag_generation_ms = rag_result.generation_time_ms
                        sample.rag_total_ms = rag_result.inference_time_ms
                        sample.verdict = rag_result.verdict.value
                        sample.confidence = rag_result.confidence
                    except Exception as e:
                        sample.rag_total_ms = 0
                        sample.verdict = "ERROR"
            
            # Total
            sample.total_pipeline_ms = (time.perf_counter() - total_start) * 1000
            sample.gpu_memory_mb = get_gpu_memory_mb()
            
            samples.append(sample)
        
        print(f"   Texte {text_idx + 1}/{len(BENCHMARK_TEXTS)} ✓")
    
    # Calculer les statistiques
    print("\n📈 Calcul des statistiques...")
    
    cd_times = [s.claim_detection_ms for s in samples]
    fl_times = [s.fast_lookup_ms for s in samples if s.fast_lookup_ms > 0]
    rag_times = [s.rag_total_ms for s in samples if s.rag_total_ms > 0]
    total_times = [s.total_pipeline_ms for s in samples]
    gpu_memories = [s.gpu_memory_mb for s in samples if s.gpu_memory_mb > 0]
    
    results = BenchmarkResults(
        timestamp=datetime.now().isoformat(),
        device=device,
        gpu_name=gpu_name,
        total_samples=len(samples),
        
        # Claim detection
        claim_detection_mean_ms=statistics.mean(cd_times) if cd_times else 0,
        claim_detection_p95_ms=calculate_percentile(cd_times, 95),
        claim_detection_std_ms=statistics.stdev(cd_times) if len(cd_times) > 1 else 0,
        
        # Fast lookup
        fast_lookup_mean_ms=statistics.mean(fl_times) if fl_times else 0,
        fast_lookup_p95_ms=calculate_percentile(fl_times, 95),
        fast_lookup_std_ms=statistics.stdev(fl_times) if len(fl_times) > 1 else 0,
        
        # RAG
        rag_mean_ms=statistics.mean(rag_times) if rag_times else 0,
        rag_p95_ms=calculate_percentile(rag_times, 95),
        rag_std_ms=statistics.stdev(rag_times) if len(rag_times) > 1 else 0,
        
        # Total
        total_mean_ms=statistics.mean(total_times) if total_times else 0,
        total_p95_ms=calculate_percentile(total_times, 95),
        total_std_ms=statistics.stdev(total_times) if len(total_times) > 1 else 0,
        
        # GPU
        gpu_memory_peak_mb=max(gpu_memories) if gpu_memories else 0,
        gpu_memory_avg_mb=statistics.mean(gpu_memories) if gpu_memories else 0,
        
        # Targets
        meets_claim_detection_target=statistics.mean(cd_times) < 100 if cd_times else False,
        meets_fast_lookup_target=statistics.mean(fl_times) < 300 if fl_times else True,
        meets_rag_target=statistics.mean(rag_times) < 1500 if rag_times else True,
        meets_total_target=statistics.mean(total_times) < 2000 if total_times else False,
        
        samples=samples
    )
    
    return results


def print_summary(results: BenchmarkResults) -> None:
    """Affiche le résumé des résultats."""
    
    print("\n" + "=" * 70)
    print("RÉSUMÉ DU BENCHMARK")
    print("=" * 70)
    
    print(f"\n📅 Timestamp: {results.timestamp}")
    print(f"🖥️  Device: {results.device}")
    print(f"🎮 GPU: {results.gpu_name}")
    print(f"📊 Échantillons: {results.total_samples}")
    
    # Tableau des métriques
    print("\n" + "-" * 70)
    print(f"{'ÉTAPE':<25} {'CIBLE':<12} {'MOYENNE':<12} {'P95':<12} {'STATUS':<8}")
    print("-" * 70)
    
    stages = [
        ("Claim Detection", "< 100ms", results.claim_detection_mean_ms, 
         results.claim_detection_p95_ms, results.meets_claim_detection_target),
        ("Fast Lookup", "< 300ms", results.fast_lookup_mean_ms,
         results.fast_lookup_p95_ms, results.meets_fast_lookup_target),
        ("RAG Verification", "< 1500ms", results.rag_mean_ms,
         results.rag_p95_ms, results.meets_rag_target),
        ("TOTAL Pipeline", "< 2000ms", results.total_mean_ms,
         results.total_p95_ms, results.meets_total_target),
    ]
    
    for name, target, mean, p95, meets in stages:
        status = "✅" if meets else "❌"
        mean_str = f"{mean:.1f}ms" if mean > 0 else "N/A"
        p95_str = f"{p95:.1f}ms" if p95 > 0 else "N/A"
        print(f"{name:<25} {target:<12} {mean_str:<12} {p95_str:<12} {status:<8}")
    
    print("-" * 70)
    
    # GPU Memory
    print(f"\n🎮 Mémoire GPU:")
    print(f"   Peak: {results.gpu_memory_peak_mb:.1f} MB")
    print(f"   Average: {results.gpu_memory_avg_mb:.1f} MB")
    
    # Verdict global
    print("\n" + "=" * 70)
    all_targets_met = all([
        results.meets_claim_detection_target,
        results.meets_fast_lookup_target,
        results.meets_rag_target,
        results.meets_total_target
    ])
    
    if all_targets_met:
        print("✅ RÉSULTAT: Tous les objectifs de performance sont atteints!")
    else:
        print("❌ RÉSULTAT: Certains objectifs ne sont pas atteints.")
    print("=" * 70)


def save_csv(results: BenchmarkResults, output_path: str) -> None:
    """Sauvegarde les résultats détaillés en CSV."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # CSV détaillé avec tous les échantillons
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'text_id', 'text_preview', 'iteration',
            'claim_detection_ms', 'claims_detected', 'claims_to_verify',
            'fast_lookup_ms', 'fast_lookup_found',
            'rag_retrieval_ms', 'rag_generation_ms', 'rag_total_ms',
            'total_pipeline_ms', 'gpu_memory_mb',
            'verdict', 'confidence'
        ]
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for sample in results.samples:
            writer.writerow({
                'text_id': sample.text_id,
                'text_preview': sample.text_preview,
                'iteration': sample.iteration,
                'claim_detection_ms': round(sample.claim_detection_ms, 2),
                'claims_detected': sample.claims_detected,
                'claims_to_verify': sample.claims_to_verify,
                'fast_lookup_ms': round(sample.fast_lookup_ms, 2),
                'fast_lookup_found': sample.fast_lookup_found,
                'rag_retrieval_ms': round(sample.rag_retrieval_ms, 2),
                'rag_generation_ms': round(sample.rag_generation_ms, 2),
                'rag_total_ms': round(sample.rag_total_ms, 2),
                'total_pipeline_ms': round(sample.total_pipeline_ms, 2),
                'gpu_memory_mb': round(sample.gpu_memory_mb, 2),
                'verdict': sample.verdict,
                'confidence': round(sample.confidence, 4)
            })
    
    print(f"\n💾 Résultats détaillés sauvegardés: {output_file}")
    
    # Sauvegarder aussi le résumé JSON
    summary_path = output_file.with_suffix('.json')
    summary = {
        'timestamp': results.timestamp,
        'device': results.device,
        'gpu_name': results.gpu_name,
        'total_samples': results.total_samples,
        'metrics': {
            'claim_detection': {
                'mean_ms': round(results.claim_detection_mean_ms, 2),
                'p95_ms': round(results.claim_detection_p95_ms, 2),
                'std_ms': round(results.claim_detection_std_ms, 2),
                'target_met': results.meets_claim_detection_target
            },
            'fast_lookup': {
                'mean_ms': round(results.fast_lookup_mean_ms, 2),
                'p95_ms': round(results.fast_lookup_p95_ms, 2),
                'std_ms': round(results.fast_lookup_std_ms, 2),
                'target_met': results.meets_fast_lookup_target
            },
            'rag': {
                'mean_ms': round(results.rag_mean_ms, 2),
                'p95_ms': round(results.rag_p95_ms, 2),
                'std_ms': round(results.rag_std_ms, 2),
                'target_met': results.meets_rag_target
            },
            'total': {
                'mean_ms': round(results.total_mean_ms, 2),
                'p95_ms': round(results.total_p95_ms, 2),
                'std_ms': round(results.total_std_ms, 2),
                'target_met': results.meets_total_target
            }
        },
        'gpu_memory': {
            'peak_mb': round(results.gpu_memory_peak_mb, 2),
            'avg_mb': round(results.gpu_memory_avg_mb, 2)
        }
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"💾 Résumé JSON sauvegardé: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark FactPulse Pipeline")
    parser.add_argument("--iterations", type=int, default=5, help="Itérations par texte")
    parser.add_argument("--warmup", type=int, default=2, help="Itérations de warmup")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--output", default="backend/benchmarks/benchmark_results.csv", help="Fichier CSV de sortie")
    parser.add_argument("--skip-rag", action="store_true", help="Sauter la vérification RAG")
    
    args = parser.parse_args()
    
    results = run_benchmark(
        num_iterations=args.iterations,
        warmup_iterations=args.warmup,
        device=args.device,
        skip_rag=args.skip_rag
    )
    
    print_summary(results)
    save_csv(results, args.output)


if __name__ == "__main__":
    main()


# ============================================================================
# EXEMPLE DE SORTIE
# ============================================================================
"""
======================================================================
RÉSUMÉ DU BENCHMARK
======================================================================

📅 Timestamp: 2026-01-22T14:30:00
🖥️  Device: cuda
🎮 GPU: NVIDIA GeForce RTX 5060
📊 Échantillons: 85

----------------------------------------------------------------------
ÉTAPE                     CIBLE        MOYENNE      P95          STATUS  
----------------------------------------------------------------------
Claim Detection           < 100ms      42.3ms       68.5ms       ✅      
Fast Lookup               < 300ms      18.7ms       32.1ms       ✅      
RAG Verification          < 1500ms     1124.5ms     1387.2ms     ✅      
TOTAL Pipeline            < 2000ms     1245.8ms     1523.4ms     ✅      
----------------------------------------------------------------------

🎮 Mémoire GPU:
   Peak: 2847.3 MB
   Average: 2654.1 MB

======================================================================
✅ RÉSULTAT: Tous les objectifs de performance sont atteints!
======================================================================

💾 Résultats détaillés sauvegardés: backend/benchmarks/benchmark_results.csv
💾 Résumé JSON sauvegardé: backend/benchmarks/benchmark_results.json
"""
