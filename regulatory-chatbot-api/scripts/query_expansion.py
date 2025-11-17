"""
Query Expansion using HyDE (Hypothetical Document Embeddings)
Improves retrieval by generating hypothetical answers that better match document style
"""

from typing import List, Tuple
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryExpander:
    """
    Expands queries using HyDE technique for better retrieval
    """
    
    def __init__(
        self,
        embedding_model: SentenceTransformer,
        generator_model: str = "google/flan-t5-base"
    ):
        """
        Args:
            embedding_model: Model for encoding queries and hypothetical docs
            generator_model: Model for generating hypothetical documents
        """
        self.embedding_model = embedding_model
        
        logger.info(f"Loading generator model: {generator_model}")
        self.generator = pipeline(
            "text2text-generation",
            model=generator_model,
            device=-1  # CPU
        )
        logger.info("✓ Query expander initialized")
    
    def generate_hypothetical_document(
        self,
        query: str,
        max_length: int = 200
    ) -> str:
        """
        Generate a hypothetical answer to the query
        This answer should match the style of actual documents
        """
        prompt = f"""You are a legal expert on Indian mining laws. Generate a detailed, formal answer to this question as it would appear in official mining regulations:

Question: {query}

Answer (use formal legal language with section references):"""
        
        result = self.generator(
            prompt,
            max_length=max_length,
            min_length=50,
            num_beams=4,
            temperature=0.7,
            do_sample=False,
            early_stopping=True
        )
        
        hypothetical_doc = result[0]['generated_text']
        return hypothetical_doc
    
    def expand_query_simple(self, query: str) -> np.ndarray:
        """
        Simple query expansion: encode both query and hypothetical document,
        then average their embeddings
        """
        # Encode original query
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # Generate hypothetical document
        hypo_doc = self.generate_hypothetical_document(query)
        
        # Encode hypothetical document
        hypo_embedding = self.embedding_model.encode(
            hypo_doc,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # Average embeddings
        expanded_embedding = (query_embedding + hypo_embedding) / 2
        
        # Normalize again
        expanded_embedding = expanded_embedding / np.linalg.norm(expanded_embedding)
        
        return expanded_embedding
    
    def expand_query_multi(
        self,
        query: str,
        num_hypotheses: int = 3
    ) -> np.ndarray:
        """
        Advanced query expansion: generate multiple hypothetical documents
        and combine their embeddings with the original query
        """
        embeddings = []
        
        # Original query embedding
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        embeddings.append(query_embedding)
        
        # Generate multiple hypothetical documents
        for i in range(num_hypotheses):
            hypo_doc = self.generate_hypothetical_document(query)
            hypo_embedding = self.embedding_model.encode(
                hypo_doc,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            embeddings.append(hypo_embedding)
        
        # Average all embeddings
        expanded_embedding = np.mean(embeddings, axis=0)
        
        # Normalize
        expanded_embedding = expanded_embedding / np.linalg.norm(expanded_embedding)
        
        return expanded_embedding
    
    def expand_with_context(self, query: str) -> Tuple[np.ndarray, str]:
        """
        Expand query and return both embedding and hypothetical document
        for transparency/debugging
        """
        # Generate hypothetical document
        hypo_doc = self.generate_hypothetical_document(query)
        
        logger.info(f"Original query: {query}")
        logger.info(f"Hypothetical doc: {hypo_doc[:200]}...")
        
        # Encode both
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        hypo_embedding = self.embedding_model.encode(
            hypo_doc,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # Combine
        expanded_embedding = (query_embedding + hypo_embedding) / 2
        expanded_embedding = expanded_embedding / np.linalg.norm(expanded_embedding)
        
        return expanded_embedding, hypo_doc


class MultiQueryExpander:
    """
    Breaks complex queries into multiple simpler sub-queries
    """
    
    def __init__(self, generator_model: str = "google/flan-t5-base"):
        logger.info(f"Loading generator model: {generator_model}")
        self.generator = pipeline(
            "text2text-generation",
            model=generator_model,
            device=-1
        )
        logger.info("✓ Multi-query expander initialized")
    
    def generate_subqueries(
        self,
        query: str,
        num_queries: int = 3
    ) -> List[str]:
        """
        Break down a complex query into multiple focused sub-queries
        """
        prompt = f"""Break down this complex mining law query into {num_queries} specific, focused questions that together answer the original question:

Original Query: {query}

Generate {num_queries} sub-queries (one per line):"""
        
        result = self.generator(
            prompt,
            max_length=150,
            num_beams=4,
            temperature=0.7,
            do_sample=False
        )
        
        response = result[0]['generated_text']
        
        # Parse sub-queries (split by newlines, filter empty)
        subqueries = [
            q.strip().lstrip('123456789.-) ')
            for q in response.split('\n')
            if q.strip()
        ]
        
        # Ensure we have the requested number
        subqueries = subqueries[:num_queries]
        
        # If we didn't get enough, add the original query
        while len(subqueries) < num_queries:
            subqueries.append(query)
        
        return subqueries


def test_query_expansion():
    """Test query expansion"""
    from sentence_transformers import SentenceTransformer
    
    # Load embedding model
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    logger.info(f"Loading embedding model: {model_name}")
    embedding_model = SentenceTransformer(model_name)
    
    # Create expander
    expander = QueryExpander(embedding_model)
    
    # Test queries
    test_queries = [
        "What are the penalties for illegal mining?",
        "How are worker wages regulated in mines?"
    ]
    
    logger.info("\n" + "="*80)
    logger.info("QUERY EXPANSION TEST")
    logger.info("="*80)
    
    for query in test_queries:
        logger.info(f"\nOriginal Query: {query}")
        logger.info("-" * 80)
        
        # Generate hypothetical document
        hypo_doc = expander.generate_hypothetical_document(query)
        logger.info(f"Hypothetical Document:\n{hypo_doc}")
        
        # Get expanded embedding
        expanded_emb = expander.expand_query_simple(query)
        logger.info(f"Expanded embedding shape: {expanded_emb.shape}")
    
    # Test multi-query
    logger.info("\n" + "="*80)
    logger.info("MULTI-QUERY EXPANSION TEST")
    logger.info("="*80)
    
    multi_expander = MultiQueryExpander()
    complex_query = "What are the complete regulations for mining safety including worker protection and equipment requirements?"
    
    logger.info(f"\nComplex Query: {complex_query}")
    logger.info("-" * 80)
    
    subqueries = multi_expander.generate_subqueries(complex_query, num_queries=3)
    for i, sq in enumerate(subqueries, 1):
        logger.info(f"{i}. {sq}")
    
    logger.info("\n✓ Query expansion test complete!")


if __name__ == '__main__':
    test_query_expansion()
