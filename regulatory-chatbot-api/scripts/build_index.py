"""
Enhanced Index Building with Sentence-T5-Large Embeddings
Replaces generic all-MiniLM-L6-v2 with larger, more capable model
"""

import pickle
import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedIndexBuilder:
    """
    Build FAISS index with improved embedding model
    """
    
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2'):
        """
        Initialize with better embedding model
        
        Options (in order of quality vs speed):
        - 'sentence-transformers/all-MiniLM-L6-v2': 384 dim, fast (baseline)
        - 'sentence-transformers/all-mpnet-base-v2': 768 dim, better quality
        - 'sentence-transformers/sentence-t5-large': 768 dim, best quality
        """
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dim}")
    
    def build_index(self, chunks, batch_size=32):
        """Build FAISS index from chunks"""
        logger.info(f"Building index for {len(chunks)} chunks...")
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings with progress bar
        logger.info("Generating embeddings...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )
        
        logger.info(f"Embeddings shape: {embeddings.shape}")
        
        # Create FAISS index (Inner Product = Cosine Similarity with normalized vectors)
        logger.info("Creating FAISS index...")
        index = faiss.IndexFlatIP(self.dim)
        
        # Add embeddings
        index.add(embeddings.astype('float32'))
        
        logger.info(f"Index built successfully with {index.ntotal} vectors")
        
        return index, embeddings
    
    def save_index(self, index, embeddings, chunks, output_dir):
        """Save index, embeddings, and metadata"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save FAISS index
        index_path = output_dir / 'mining_laws.index'
        logger.info(f"Saving FAISS index to {index_path}")
        faiss.write_index(index, str(index_path))
        
        # Save embeddings separately (for hybrid search)
        embeddings_path = output_dir / 'embeddings.npy'
        logger.info(f"Saving embeddings to {embeddings_path}")
        np.save(embeddings_path, embeddings)
        
        # Save metadata
        meta_path = output_dir / 'index_meta.json'
        metadata = {
            'model': self.model_name,
            'dim': self.dim,
            'num_chunks': len(chunks),
            'index_type': 'IndexFlatIP',
            'normalized': True,
            'version': '2.0',
            'improvements': [
                'Semantic chunking',
                'Better embedding model',
                'Normalized embeddings'
            ]
        }
        
        logger.info(f"Saving metadata to {meta_path}")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("✓ Index saved successfully!")


def main():
    # Paths
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / 'data'
    models_dir = project_dir / 'models'
    
    # Load chunks
    chunks_file = data_dir / 'chunks.pkl'
    if not chunks_file.exists():
        logger.error(f"chunks.pkl not found in {data_dir}")
        logger.error("Run scripts/ingest.py first to generate chunks")
        return
    
    logger.info(f"Loading chunks from {chunks_file}")
    with open(chunks_file, 'rb') as f:
        chunks = pickle.load(f)
    
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # Choose embedding model
    # Options: 'sentence-transformers/all-MiniLM-L6-v2' (fast, 384d)
    #          'sentence-transformers/all-mpnet-base-v2' (better, 768d)
    #          'sentence-transformers/sentence-t5-large' (best, 768d)
    
    model_name = 'sentence-transformers/all-mpnet-base-v2'
    
    # Build index
    builder = EnhancedIndexBuilder(model_name=model_name)
    index, embeddings = builder.build_index(chunks, batch_size=32)
    
    # Save
    builder.save_index(index, embeddings, chunks, models_dir)
    
    # Test search
    logger.info("\n" + "="*60)
    logger.info("TEST SEARCH")
    logger.info("="*60)
    
    test_query = "What are the wage regulations for mine workers?"
    logger.info(f"Query: {test_query}")
    
    query_embedding = builder.model.encode(
        test_query,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    
    distances, indices = index.search(
        query_embedding.reshape(1, -1).astype('float32'),
        k=3
    )
    
    logger.info(f"\nTop 3 results:")
    for i, (idx, score) in enumerate(zip(indices[0], distances[0])):
        chunk = chunks[idx]
        logger.info(f"\n{i+1}. Score: {score:.4f}")
        logger.info(f"   Section: {chunk['metadata'].get('section', 'N/A')}")
        logger.info(f"   Act: {chunk['metadata'].get('act', 'N/A')}")
        logger.info(f"   Preview: {chunk['text'][:150]}...")
    
    logger.info("\n✓ Enhanced index building complete!")


if __name__ == '__main__':
    main()
