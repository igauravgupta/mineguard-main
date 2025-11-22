"""
Phase 3: Retrieval System
Steps 8, 9, 10 - Dense, Sparse, and Hybrid Retrieval
"""

import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple
import re


class RetrievalSystem:
    """
    Complete retrieval system with:
    - Step 8: Dense retrieval (FAISS)
    - Step 9: Sparse retrieval (BM25)
    - Step 10: Hybrid retrieval (RRF)
    """
    
    def __init__(self, use_finetuned=True):
        """
        Initialize retrieval system
        
        Args:
            use_finetuned: Use fine-tuned model if True, else base model
        """
        self.models_dir = Path("models")
        self.use_finetuned = use_finetuned
        
        # Model and index paths
        if use_finetuned:
            self.model_path = str(self.models_dir / "bi_encoder_finetuned")
            self.index_path = self.models_dir / "mining_laws_finetuned.index"
        else:
            self.model_path = "all-mpnet-base-v2"
            self.index_path = self.models_dir / "mining_laws_base.index"
        
        self.metadata_path = self.models_dir / "chunks_metadata.json"
        
        # Initialize components
        self.model = None
        self.faiss_index = None
        self.chunks = None
        self.bm25 = None
        self.tokenized_chunks = None
        
        # Load all components
        self._load_components()
    
    def _load_components(self):
        """Load model, FAISS index, and metadata"""
        print(f"{'='*80}")
        print("INITIALIZING RETRIEVAL SYSTEM")
        print(f"{'='*80}\n")
        
        # Load sentence transformer model
        print(f"Loading model: {self.model_path}")
        self.model = SentenceTransformer(self.model_path)
        print(f"✓ Model loaded (dim: {self.model.get_sentence_embedding_dimension()})")
        
        # Load FAISS index
        print(f"\nLoading FAISS index: {self.index_path}")
        self.faiss_index = faiss.read_index(str(self.index_path))
        print(f"✓ FAISS index loaded ({self.faiss_index.ntotal} vectors)")
        
        # Load chunk metadata
        print(f"\nLoading chunk metadata: {self.metadata_path}")
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        print(f"✓ Loaded {len(self.chunks)} chunks")
        
        # Build BM25 index
        print("\nBuilding BM25 index...")
        self._build_bm25_index()
        print(f"✓ BM25 index built")
        
        print(f"\n{'='*80}")
        print("RETRIEVAL SYSTEM READY")
        print(f"{'='*80}\n")
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text for BM25
        Simple tokenization: lowercase, split on non-alphanumeric
        """
        # Lowercase and split
        text = text.lower()
        # Keep alphanumeric and spaces, split on whitespace
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _build_bm25_index(self):
        """Build BM25 index from chunks"""
        # Tokenize all chunks
        self.tokenized_chunks = [
            self._tokenize_text(chunk['text']) 
            for chunk in self.chunks
        ]
        
        # Create BM25 index with standard parameters
        # k1 = 1.5, b = 0.75 (default values)
        self.bm25 = BM25Okapi(self.tokenized_chunks)
    
    def _l2_normalize(self, embeddings):
        """L2-normalize embeddings for cosine similarity"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms
    
    # ========================================================================
    # STEP 8: Dense Retrieval (FAISS)
    # ========================================================================
    
    def dense_retrieval(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """
        Dense retrieval using FAISS semantic search
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (chunk_id, score) tuples
        """
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = self._l2_normalize(query_embedding)
        
        # Search FAISS index
        scores, indices = self.faiss_index.search(
            query_embedding.astype('float32'), 
            top_k
        )
        
        # Return list of (chunk_id, score)
        results = [
            (int(idx), float(score)) 
            for idx, score in zip(indices[0], scores[0])
        ]
        
        return results
    
    # ========================================================================
    # STEP 9: Sparse Retrieval (BM25)
    # ========================================================================
    
    def sparse_retrieval(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """
        Sparse retrieval using BM25 keyword search
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (chunk_id, score) tuples
        """
        # Tokenize query
        query_tokens = self._tokenize_text(query)
        
        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Return list of (chunk_id, score)
        results = [
            (int(idx), float(scores[idx]))
            for idx in top_indices
        ]
        
        return results
    
    # ========================================================================
    # STEP 10: Hybrid Retrieval (Reciprocal Rank Fusion)
    # ========================================================================
    
    def reciprocal_rank_fusion(
        self, 
        dense_results: List[Tuple[int, float]], 
        sparse_results: List[Tuple[int, float]], 
        k: int = 60
    ) -> List[Tuple[int, float]]:
        """
        Combine dense and sparse results using Reciprocal Rank Fusion
        
        RRF Formula: score = 1/(rank_dense + k) + 1/(rank_sparse + k)
        
        Args:
            dense_results: Results from dense retrieval [(chunk_id, score), ...]
            sparse_results: Results from sparse retrieval [(chunk_id, score), ...]
            k: RRF constant (typically 60)
            
        Returns:
            Combined ranked results [(chunk_id, rrf_score), ...]
        """
        rrf_scores = {}
        
        # Add dense retrieval scores
        for rank, (chunk_id, _) in enumerate(dense_results, 1):
            rrf_scores[chunk_id] = 1.0 / (rank + k)
        
        # Add sparse retrieval scores
        for rank, (chunk_id, _) in enumerate(sparse_results, 1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (rank + k)
        
        # Sort by combined RRF score
        combined_results = sorted(
            rrf_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return combined_results
    
    def hybrid_retrieval(
        self, 
        query: str, 
        top_k: int = 10,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        rrf_k: int = 60
    ) -> List[Dict]:
        """
        Hybrid retrieval combining dense and sparse methods
        
        Args:
            query: Search query
            top_k: Final number of results to return
            dense_top_k: Number of results from dense retrieval
            sparse_top_k: Number of results from sparse retrieval
            rrf_k: RRF fusion constant
            
        Returns:
            List of chunk dictionaries with scores
        """
        # Step 1: Dense retrieval
        dense_results = self.dense_retrieval(query, dense_top_k)
        
        # Step 2: Sparse retrieval
        sparse_results = self.sparse_retrieval(query, sparse_top_k)
        
        # Step 3: Reciprocal Rank Fusion
        combined_results = self.reciprocal_rank_fusion(
            dense_results, 
            sparse_results, 
            k=rrf_k
        )
        
        # Step 4: Get top-k final results with full chunk data
        final_results = []
        for chunk_id, rrf_score in combined_results[:top_k]:
            chunk = self.chunks[chunk_id].copy()
            chunk['retrieval_score'] = rrf_score
            chunk['chunk_id'] = chunk_id
            final_results.append(chunk)
        
        return final_results
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def search(self, query: str, method: str = 'hybrid', top_k: int = 10) -> List[Dict]:
        """
        Main search interface
        
        Args:
            query: Search query
            method: 'dense', 'sparse', or 'hybrid'
            top_k: Number of results to return
            
        Returns:
            List of chunk dictionaries with scores
        """
        if method == 'dense':
            results = self.dense_retrieval(query, top_k)
            return [
                {**self.chunks[idx], 'retrieval_score': score, 'chunk_id': idx}
                for idx, score in results
            ]
        elif method == 'sparse':
            results = self.sparse_retrieval(query, top_k)
            return [
                {**self.chunks[idx], 'retrieval_score': score, 'chunk_id': idx}
                for idx, score in results
            ]
        elif method == 'hybrid':
            return self.hybrid_retrieval(query, top_k)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def print_results(self, results: List[Dict], query: str = None):
        """Pretty print search results"""
        if query:
            print(f"\n{'='*80}")
            print(f"QUERY: {query}")
            print(f"{'='*80}\n")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result['retrieval_score']:.4f}")
            print(f"   Act: {result['metadata']['act']}")
            print(f"   Section: {result['metadata']['section']}")
            print(f"   Title: {result['metadata']['section_title'][:60]}...")
            print(f"   Text: {result['text'][:150]}...")
            print(f"{'-'*80}\n")
    
    def compare_methods(self, query: str, top_k: int = 5):
        """Compare all three retrieval methods side by side"""
        print(f"\n{'='*80}")
        print(f"COMPARING RETRIEVAL METHODS")
        print(f"{'='*80}")
        print(f"Query: {query}\n")
        
        methods = ['dense', 'sparse', 'hybrid']
        
        for method in methods:
            print(f"\n{'='*80}")
            print(f"{method.upper()} RETRIEVAL")
            print(f"{'='*80}")
            
            results = self.search(query, method=method, top_k=top_k)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Score: {result['retrieval_score']:.4f}")
                print(f"   Section: {result['metadata']['section']} - {result['metadata']['section_title'][:50]}...")
                print(f"   Text: {result['text'][:100]}...")


def main():
    """Demo and testing"""
    print("="*80)
    print("PHASE 3: RETRIEVAL SYSTEM")
    print("="*80 + "\n")
    
    # Initialize retrieval system
    # Try fine-tuned model first, fall back to base if not available
    try:
        retrieval = RetrievalSystem(use_finetuned=True)
        print("Using fine-tuned model\n")
    except:
        print("Fine-tuned model not found, using base model\n")
        retrieval = RetrievalSystem(use_finetuned=False)
    
    # Test queries
    test_queries = [
        "What is the definition of a mine?",
        "Who can be appointed as mine inspector?",
        "What are safety requirements for explosives?",
        "Section 40 provisions",
        "Competent person qualifications"
    ]
    
    print(f"\n{'='*80}")
    print("TESTING HYBRID RETRIEVAL (RRF)")
    print(f"{'='*80}\n")
    
    for query in test_queries:
        results = retrieval.search(query, method='hybrid', top_k=3)
        retrieval.print_results(results, query)
    
    # Compare methods for one query
    print(f"\n{'='*80}")
    print("METHOD COMPARISON")
    print(f"{'='*80}\n")
    
    retrieval.compare_methods(
        "What are the safety requirements for explosives?",
        top_k=3
    )
    
    print(f"\n{'='*80}")
    print("PHASE 3 COMPLETE")
    print(f"{'='*80}")
    print("\nRetrieval system ready for use!")
    print("\nUsage:")
    print("  retrieval = RetrievalSystem()")
    print("  results = retrieval.search('your query', method='hybrid', top_k=10)")
    print("  retrieval.print_results(results)")


if __name__ == "__main__":
    main()
