"""
Build FAISS index from known facts.

This script:
1. Loads known facts from JSON
2. Generates embeddings using sentence-transformers
3. Builds and saves FAISS index
4. Saves metadata for retrieval

Run: python -m scripts.build_index
"""

import json
import time
import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np


def build_index(
    facts_path: str = "data/known_facts.json",
    output_index_path: str = "data/vector_index.faiss",
    output_metadata_path: str = "data/facts_metadata.json",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cuda"
):
    """
    Build FAISS index from facts.
    
    Args:
        facts_path: Path to known facts JSON
        output_index_path: Path to save FAISS index
        output_metadata_path: Path to save metadata
        model_name: Embedding model name
        device: Torch device
    """
    import torch
    from sentence_transformers import SentenceTransformer
    import faiss
    
    print(f"🔧 Building FAISS index...")
    print(f"   Device: {device}")
    print(f"   Model: {model_name}")
    
    # Check GPU
    if device == "cuda" and torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    elif device == "cuda":
        print("   ⚠️ CUDA requested but not available, falling back to CPU")
        device = "cpu"
    
    # Load facts
    facts_file = project_root / facts_path
    if not facts_file.exists():
        print(f"❌ Facts file not found: {facts_file}")
        return
    
    with open(facts_file, 'r', encoding='utf-8') as f:
        facts = json.load(f)
    
    print(f"   Loaded {len(facts)} facts")
    
    # Load embedding model
    print("📦 Loading embedding model...")
    start = time.perf_counter()
    model = SentenceTransformer(model_name, device=device)
    print(f"   Model loaded in {(time.perf_counter() - start)*1000:.0f}ms")
    
    # Generate embeddings
    print("🔢 Generating embeddings...")
    start = time.perf_counter()
    
    texts = [f["fact_text"] for f in facts]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    
    print(f"   Generated {len(embeddings)} embeddings in {(time.perf_counter() - start)*1000:.0f}ms")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    
    # Build FAISS index
    print("🏗️ Building FAISS index...")
    start = time.perf_counter()
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    index.add(embeddings.astype('float32'))
    
    print(f"   Index built in {(time.perf_counter() - start)*1000:.0f}ms")
    print(f"   Index size: {index.ntotal} vectors")
    
    # Save index
    index_file = project_root / output_index_path
    index_file.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_file))
    print(f"💾 Index saved to: {index_file}")
    
    # Save metadata (already in correct format)
    metadata_file = project_root / output_metadata_path
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(facts, f, indent=2, ensure_ascii=False)
    print(f"💾 Metadata saved to: {metadata_file}")
    
    # Verify with test search
    print("\n🧪 Verification test...")
    test_query = "Is the Earth round?"
    query_emb = model.encode([test_query], normalize_embeddings=True)
    
    start = time.perf_counter()
    D, I = index.search(query_emb.astype('float32'), 3)
    search_time = (time.perf_counter() - start) * 1000
    
    print(f"   Query: '{test_query}'")
    print(f"   Search time: {search_time:.2f}ms")
    print(f"   Top 3 results:")
    for i, (sim, idx) in enumerate(zip(D[0], I[0])):
        fact = facts[idx]
        print(f"      {i+1}. [{sim:.3f}] {fact['fact_text'][:60]}... ({fact['verdict']})")
    
    print("\n✅ Index build complete!")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from known facts")
    parser.add_argument("--facts", default="data/known_facts.json", help="Path to facts JSON")
    parser.add_argument("--output-index", default="data/vector_index.faiss", help="Output FAISS index path")
    parser.add_argument("--output-metadata", default="data/facts_metadata.json", help="Output metadata path")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device")
    
    args = parser.parse_args()
    
    build_index(
        facts_path=args.facts,
        output_index_path=args.output_index,
        output_metadata_path=args.output_metadata,
        model_name=args.model,
        device=args.device
    )


if __name__ == "__main__":
    main()
