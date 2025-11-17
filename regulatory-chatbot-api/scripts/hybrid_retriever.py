"""
Hybrid Retrieval System: FAISS (Dense) + BM25 (Sparse)
Combines semantic understanding with keyword matching
"""

import pickle
import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Combines dense (FAISS) and sparse (BM25) retrieval for better results
    """
    
    def __init__(
        self,
        chunks: List[Dict],
        faiss_index: faiss.Index,
        embedding_model: SentenceTransformer,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ):
        """
        Args:
            chunks: List of document chunks
            faiss_index: Pre-built FAISS index
            embedding_model: SentenceTransformer model for query encoding
            dense_weight: Weight for dense retrieval scores
            sparse_weight: Weight for sparse retrieval scores
        """
        self.chunks = chunks
        self.faiss_index = faiss_index
        self.embedding_model = embedding_model
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        # Build BM25 index
        logger.info("Building BM25 index...")
        self.corpus_texts = [chunk['text'] for chunk in chunks]
        tokenized_corpus = [text.lower().split() for text in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"✓ BM25 index built with {len(self.corpus_texts)} documents")
    
    def retrieve_dense(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Dense retrieval using FAISS"""
        # Encode query
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # Search
        distances, indices = self.faiss_index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            k=top_k
        )
        
        # Return (index, score) tuples
        results = [(int(idx), float(score)) for idx, score in zip(indices[0], distances[0])]
        return results
    
    def retrieve_sparse(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Sparse retrieval using BM25"""
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Return (index, score) tuples
        results = [(int(idx), float(scores[idx])) for idx in top_indices]
        return results
    
    def hybrid_retrieve(
        self,
        query: str,
        top_k: int = 10,
        retrieval_k: int = 20
    ) -> List[Dict]:
        """
        Hybrid retrieval combining dense and sparse methods
        
        Args:
            query: User query
            top_k: Number of final results to return
            retrieval_k: Number of candidates from each method
        
        Returns:
            List of chunks with combined scores
        """
        # Get candidates from both methods
        dense_results = self.retrieve_dense(query, top_k=retrieval_k)
        sparse_results = self.retrieve_sparse(query, top_k=retrieval_k)
        
        # Normalize scores to [0, 1]
        dense_scores = self._normalize_scores([score for _, score in dense_results])
        sparse_scores = self._normalize_scores([score for _, score in sparse_results])
        
        # Create score dictionaries
        dense_dict = {idx: score for (idx, _), score in zip(dense_results, dense_scores)}
        sparse_dict = {idx: score for (idx, _), score in zip(sparse_results, sparse_scores)}
        
        # Combine scores
        all_indices = set(dense_dict.keys()) | set(sparse_dict.keys())
        combined_scores = {}
        
        for idx in all_indices:
            dense_score = dense_dict.get(idx, 0.0)
            sparse_score = sparse_dict.get(idx, 0.0)
            
            # Weighted combination
            combined_scores[idx] = (
                self.dense_weight * dense_score +
                self.sparse_weight * sparse_score
            )
        
        # Sort by combined score
        ranked_indices = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Prepare results
        results = []
        for idx, score in ranked_indices:
            chunk = self.chunks[idx].copy()
            chunk['score'] = float(score)
            chunk['dense_score'] = dense_dict.get(idx, 0.0)
            chunk['sparse_score'] = sparse_dict.get(idx, 0.0)
            results.append(chunk)
        
        return results
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range"""
        if not scores:
            return []
        
        scores = np.array(scores)
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score - min_score == 0:
            return [1.0] * len(scores)
        
        normalized = (scores - min_score) / (max_score - min_score)
        return normalized.tolist()


def test_hybrid_retriever():
    """Test the hybrid retriever"""
    # Paths
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / 'data'
    models_dir = project_dir / 'models'
    
    # Load chunks
    chunks_file = data_dir / 'chunks_v2.pkl'
    if not chunks_file.exists():
        chunks_file = data_dir / 'chunks.pkl'
    
    logger.info(f"Loading chunks from {chunks_file}")
    with open(chunks_file, 'rb') as f:
        chunks = pickle.load(f)
    
    # Load FAISS index
    index_file = models_dir / 'mining_laws_v2.index'
    if not index_file.exists():
        index_file = models_dir / 'mining_laws.index'
    
    logger.info(f"Loading FAISS index from {index_file}")
    faiss_index = faiss.read_index(str(index_file))
    
    # Load metadata to get model name
    meta_file = models_dir / 'index_meta_v2.json'
    if not meta_file.exists():
        meta_file = models_dir / 'index_meta.json'
    
    with open(meta_file, 'r') as f:
        metadata = json.load(f)
    
    model_name = metadata.get('model', 'sentence-transformers/all-MiniLM-L6-v2')
    
    # Load embedding model
    logger.info(f"Loading embedding model: {model_name}")
    embedding_model = SentenceTransformer(model_name)
    
    # Create hybrid retriever
    retriever = HybridRetriever(
        chunks=chunks,
        faiss_index=faiss_index,
        embedding_model=embedding_model,
        dense_weight=0.7,
        sparse_weight=0.3
    )
    
    # Test queries
    test_queries = [
        "What are the penalties for illegal mining?",
        "How are explosives regulated in mines?",
        "What are worker safety requirements?"
    ]
    
    logger.info("\n" + "="*80)
    logger.info("HYBRID RETRIEVAL TEST")
    logger.info("="*80)
    
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        logger.info("-" * 80)
        
        results = retriever.hybrid_retrieve(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            logger.info(f"\n{i}. Combined Score: {result['score']:.4f}")
            logger.info(f"   Dense: {result['dense_score']:.4f} | Sparse: {result['sparse_score']:.4f}")
            logger.info(f"   Section: {result['metadata'].get('section', 'N/A')}")
            logger.info(f"   Preview: {result['text'][:150]}...")
    
    logger.info("\n✓ Hybrid retrieval test complete!")


if __name__ == '__main__':
    test_hybrid_retriever()
