"""
Step 5: Build Initial Vector Index (Base Model Only)
Quick script to create embeddings without fine-tuning
"""

import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def l2_normalize(embeddings):
    """L2-normalize embeddings for cosine similarity"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def build_base_index(model_name='all-mpnet-base-v2'):
    """Build vector index with base model"""
    
    print("="*80)
    print("BUILDING INITIAL VECTOR INDEX")
    print("="*80 + "\n")
    
    # Setup paths
    data_dir = Path("data")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    chunks_file = data_dir / "mining_laws_chunks.json"
    index_file = models_dir / "mining_laws_base.index"
    metadata_file = models_dir / "chunks_metadata.json"
    
    # Load model
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"✓ Model loaded")
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
    print(f"  Max sequence length: {model.max_seq_length}\n")
    
    # Load chunks
    print("Loading chunks...")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    chunks = data['chunks']
    print(f"✓ Loaded {len(chunks)} chunks\n")
    
    # Extract texts
    texts = [chunk['text'] for chunk in chunks]
    
    # Generate embeddings
    print(f"Generating embeddings for {len(texts)} chunks...")
    print("This may take 5-10 minutes on CPU...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=8,  # Smaller batch size for better progress visibility
        convert_to_numpy=True,
        device='cpu'  # Explicitly use CPU
    )
    
    # L2-normalize
    print("\nNormalizing embeddings...")
    embeddings = l2_normalize(embeddings)
    
    # Create FAISS index
    print("Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
    index.add(embeddings.astype('float32'))
    
    # Save index
    print(f"Saving index to {index_file}...")
    faiss.write_index(index, str(index_file))
    
    # Save metadata
    print(f"Saving metadata to {metadata_file}...")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    # Summary
    print(f"\n{'='*80}")
    print("INDEX BUILD COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Index created successfully!")
    print(f"  Index type: FAISS IndexFlatIP (cosine similarity)")
    print(f"  Total vectors: {index.ntotal}")
    print(f"  Dimension: {dimension}")
    print(f"  Index size: {index_file.stat().st_size / 1024:.2f} KB")
    print(f"  Metadata size: {metadata_file.stat().st_size / 1024:.2f} KB")
    print(f"\nFiles created:")
    print(f"  - {index_file}")
    print(f"  - {metadata_file}")
    print("="*80)
    
    return index, chunks


def test_search(query, k=5):
    """Test search with a sample query"""
    
    models_dir = Path("models")
    index_file = models_dir / "mining_laws_base.index"
    metadata_file = models_dir / "chunks_metadata.json"
    
    print(f"\n{'='*80}")
    print("TESTING SEARCH")
    print(f"{'='*80}\n")
    
    # Load model, index, and metadata
    print("Loading model and index...")
    model = SentenceTransformer('all-mpnet-base-v2')
    index = faiss.read_index(str(index_file))
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Encode query
    print(f"Query: '{query}'\n")
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = l2_normalize(query_embedding)
    
    # Search
    scores, indices = index.search(query_embedding.astype('float32'), k)
    
    # Display results
    print(f"Top {k} results:")
    print("="*80)
    
    for i, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        chunk = chunks[idx]
        print(f"\n{i}. Score: {score:.4f}")
        print(f"   Act: {chunk['metadata']['act']}")
        print(f"   Section: {chunk['metadata']['section']}")
        print(f"   Title: {chunk['metadata']['section_title'][:60]}...")
        print(f"   Text: {chunk['text'][:150]}...")
    
    print("="*80)


if __name__ == "__main__":
    # Build index
    build_base_index(model_name='all-mpnet-base-v2')
    
    # Test with sample queries
    print("\n")
    test_search("What is the definition of a mine?", k=3)
    
    print("\n")
    test_search("What are safety requirements for explosives?", k=3)
