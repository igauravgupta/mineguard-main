"""
Build FAISS index from chunks using SentenceTransformer embeddings.

Output:
- models/mining_laws.index: FAISS index
- models/index_meta.json: Metadata (model name, chunk count, etc.)
"""

import pickle
import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


def main():
    print("Building FAISS index...")
    
    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    models_dir = project_root / 'models'
    models_dir.mkdir(exist_ok=True)
    
    chunks_file = data_dir / 'chunks.pkl'
    index_file = models_dir / 'mining_laws.index'
    meta_file = models_dir / 'index_meta.json'
    
    # Check if chunks exist
    if not chunks_file.exists():
        print(f"ERROR: {chunks_file} not found. Run scripts/ingest.py first.")
        return
    
    # Load chunks
    print(f"Loading chunks from {chunks_file}...")
    with open(chunks_file, 'rb') as f:
        chunks = pickle.load(f)
    print(f"Loaded {len(chunks)} chunks")
    
    # Load embedding model
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # L2 normalize for cosine similarity
    print("Normalizing embeddings...")
    faiss.normalize_L2(embeddings)
    
    # Build FAISS index (inner product = cosine similarity after normalization)
    print("Building FAISS IndexFlatIP...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Save index
    print(f"Saving index to {index_file}...")
    faiss.write_index(index, str(index_file))
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'dimension': dimension,
        'num_chunks': len(chunks),
        'index_type': 'IndexFlatIP',
        'normalized': True
    }
    print(f"Saving metadata to {meta_file}...")
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✓ Index building complete!")
    print(f"  - Index: {index_file}")
    print(f"  - Metadata: {meta_file}")
    print(f"  - Chunks indexed: {len(chunks)}")
    print(f"  - Embedding dimension: {dimension}")


if __name__ == '__main__':
    main()
