"""
Enhanced RAG Server with Hybrid Retrieval, Query Expansion, and Better Generation
Version 2.0 - Production-ready implementation with Online LLM Support
"""

import pickle
import json
import numpy as np
import faiss
import os
from pathlib import Path
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from rank_bm25 import BM25Okapi
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# LLM Configuration - Set environment variables to use online LLMs
USE_ONLINE_LLM = os.getenv("USE_ONLINE_LLM", "false").lower() == "true"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # "groq", "openai", "anthropic"
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")  # For Groq

# Initialize FastAPI
app = FastAPI(
    title="Mining Laws RAG API v2.0",
    description="Enhanced RAG system with hybrid retrieval and query expansion",
    version="2.0.0"
)

# Global variables for loaded models
chunks = None
faiss_index = None
embedding_model = None
generator = None
bm25_index = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    use_hybrid: bool = True
    use_query_expansion: bool = False
    dense_weight: float = 0.7
    sparse_weight: float = 0.3


class ChatRequest(BaseModel):
    query: str
    top_k: int = 3
    generate_answer: bool = True
    use_hybrid: bool = True
    use_query_expansion: bool = False
    answer_style: str = "detailed"  # "detailed" or "concise"
    dense_weight: float = 0.7
    sparse_weight: float = 0.3


class HybridRetriever:
    """Hybrid retrieval combining FAISS and BM25"""
    
    def __init__(
        self,
        chunks: List[Dict],
        faiss_index: faiss.Index,
        embedding_model: SentenceTransformer,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ):
        self.chunks = chunks
        self.faiss_index = faiss_index
        self.embedding_model = embedding_model
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        # Build BM25 index
        corpus_texts = [chunk['text'] for chunk in chunks]
        tokenized_corpus = [text.lower().split() for text in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_hybrid: bool = True,
        query_embedding: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """Retrieve relevant chunks"""
        
        if not use_hybrid:
            # Dense only (FAISS)
            if query_embedding is None:
                query_embedding = self.embedding_model.encode(
                    query,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )
            
            distances, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1).astype('float32'),
                k=top_k
            )
            
            results = []
            for idx, score in zip(indices[0], distances[0]):
                chunk = self.chunks[idx].copy()
                chunk['score'] = float(score)
                results.append(chunk)
            
            return results
        
        # Hybrid retrieval
        retrieval_k = top_k * 2
        
        # Dense retrieval
        if query_embedding is None:
            query_embedding = self.embedding_model.encode(
                query,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
        
        distances, indices = self.faiss_index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            k=retrieval_k
        )
        
        dense_dict = {int(idx): float(score) for idx, score in zip(indices[0], distances[0])}
        
        # Sparse retrieval (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:retrieval_k]
        sparse_dict = {int(idx): float(bm25_scores[idx]) for idx in top_bm25_indices}
        
        # Normalize and combine
        dense_scores = self._normalize_scores(list(dense_dict.values()))
        sparse_scores = self._normalize_scores(list(sparse_dict.values()))
        
        dense_dict = {idx: score for idx, score in zip(dense_dict.keys(), dense_scores)}
        sparse_dict = {idx: score for idx, score in zip(sparse_dict.keys(), sparse_scores)}
        
        # Combine scores
        all_indices = set(dense_dict.keys()) | set(sparse_dict.keys())
        combined_scores = {}
        
        for idx in all_indices:
            combined_scores[idx] = (
                self.dense_weight * dense_dict.get(idx, 0.0) +
                self.sparse_weight * sparse_dict.get(idx, 0.0)
            )
        
        # Sort and return top-k
        ranked_indices = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        results = []
        for idx, score in ranked_indices:
            chunk = self.chunks[idx].copy()
            chunk['score'] = float(score)
            chunk['dense_score'] = dense_dict.get(idx, 0.0)
            chunk['sparse_score'] = sparse_dict.get(idx, 0.0)
            results.append(chunk)
        
        return results
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1]"""
        if not scores:
            return []
        scores_arr = np.array(scores)
        min_s = scores_arr.min()
        max_s = scores_arr.max()
        if max_s - min_s == 0:
            return [1.0] * len(scores)
        return ((scores_arr - min_s) / (max_s - min_s)).tolist()


class QueryExpander:
    """Query expansion using HyDE"""
    
    def __init__(self, embedding_model: SentenceTransformer, generator):
        self.embedding_model = embedding_model
        self.generator = generator
    
    def expand_query(self, query: str) -> np.ndarray:
        """Generate hypothetical document and return expanded embedding"""
        
        # Generate hypothetical answer
        prompt = f"""Generate a detailed formal answer to this mining law question:

Question: {query}

Answer (use legal language):"""
        
        result = self.generator(
            prompt,
            max_new_tokens=100,
            min_length=50,
            num_beams=4,
            do_sample=False  # Deterministic for caching
        )
        
        hypo_doc = result[0]['generated_text']
        
        # Encode both query and hypothetical document
        query_emb = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        hypo_emb = self.embedding_model.encode(
            hypo_doc,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # Average and normalize
        expanded_emb = (query_emb + hypo_emb) / 2
        expanded_emb = expanded_emb / np.linalg.norm(expanded_emb)
        
        return expanded_emb


def generate_answer_with_online_llm(prompt: str, provider: str = "groq") -> str:
    """
    Generate answer using online LLM APIs (Groq, OpenAI, etc.)
    Much better quality than local FLAN-T5
    """
    try:
        if provider == "groq":
            # Groq API (Fast and Free!)
            import requests
            
            headers = {
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": LLM_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert legal assistant specializing in Indian mining laws and regulations. Provide clear, accurate answers based on the legal context provided."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 500,
                "top_p": 0.9
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                logger.error(f"Groq API error: {response.status_code} - {response.text}")
                return "Error: Could not generate answer with online LLM"
                
        elif provider == "openai":
            # OpenAI API
            import openai
            openai.api_key = LLM_API_KEY
            
            response = openai.ChatCompletion.create(
                model=LLM_MODEL or "gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert legal assistant specializing in Indian mining laws."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
            
        else:
            return "Error: Unsupported LLM provider"
            
    except Exception as e:
        logger.error(f"Online LLM error: {e}")
        return f"Error generating answer: {str(e)}"


@app.on_event("startup")
async def load_resources():
    """Load all models and indices on startup"""
    global chunks, faiss_index, embedding_model, generator, bm25_index
    
    try:
        project_dir = Path(__file__).parent
        data_dir = project_dir / 'data'
        models_dir = project_dir / 'models'
        
        # Load chunks
        chunks_file = data_dir / 'chunks.pkl'
        logger.info(f"Loading chunks from {chunks_file}")
        with open(chunks_file, 'rb') as f:
            chunks = pickle.load(f)
        logger.info(f"✓ Loaded {len(chunks)} chunks")
        
        # Load FAISS index
        index_file = models_dir / 'mining_laws.index'
        logger.info(f"Loading FAISS index from {index_file}")
        faiss_index = faiss.read_index(str(index_file))
        logger.info(f"✓ Loaded FAISS index ({faiss_index.ntotal} vectors)")
        
        # Load metadata
        meta_file = models_dir / 'index_meta.json'
        
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
        
        model_name = metadata.get('model', 'sentence-transformers/all-MiniLM-L6-v2')
        
        # Load embedding model
        logger.info(f"Loading embedding model: {model_name}")
        embedding_model = SentenceTransformer(model_name)
        logger.info(f"✓ Loaded embedding model (dim: {embedding_model.get_sentence_embedding_dimension()})")
        
        # Load generator (use FLAN-T5-base for efficiency)
        # For better quality, change to "google/flan-t5-large" (requires 4GB+ RAM and download time)
        generator_model = "google/flan-t5-base"
        logger.info(f"Loading generator: {generator_model}")
        generator = pipeline(
            "text2text-generation",
            model=generator_model,
            device=-1
        )
        logger.info(f"✓ Loaded {generator_model}")
        
        logger.info("="*60)
        logger.info("✓ All resources loaded successfully!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Failed to load resources: {e}")
        raise


@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "chunks": len(chunks) if chunks else 0,
        "index_size": faiss_index.ntotal if faiss_index else 0,
        "embedding_model": embedding_model.model_card_data.model_id if embedding_model else "not loaded",
        "features": [
            "Hybrid retrieval (FAISS + BM25)",
            "Query expansion (HyDE)",
            "Semantic chunking",
            "Enhanced generation with optimized prompts"
        ]
    }


@app.post("/search")
async def search(request: SearchRequest):
    """
    Search endpoint with hybrid retrieval
    """
    try:
        # Create retriever
        retriever = HybridRetriever(
            chunks=chunks,
            faiss_index=faiss_index,
            embedding_model=embedding_model,
            dense_weight=request.dense_weight,
            sparse_weight=request.sparse_weight
        )
        
        # Query expansion if requested
        query_embedding = None
        if request.use_query_expansion:
            expander = QueryExpander(embedding_model, generator)
            query_embedding = expander.expand_query(request.query)
        
        # Retrieve
        results = retriever.retrieve(
            request.query,
            top_k=request.top_k,
            use_hybrid=request.use_hybrid,
            query_embedding=query_embedding
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "chunk_id": result['id'],
                "text": result['text'],
                "score": result['score'],
                "metadata": result.get('metadata', {}),
                "dense_score": result.get('dense_score'),
                "sparse_score": result.get('sparse_score')
            })
        
        return {
            "query": request.query,
            "top_k": request.top_k,
            "use_hybrid": request.use_hybrid,
            "use_query_expansion": request.use_query_expansion,
            "results": formatted_results
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint with answer generation
    """
    try:
        # Create retriever
        retriever = HybridRetriever(
            chunks=chunks,
            faiss_index=faiss_index,
            embedding_model=embedding_model,
            dense_weight=request.dense_weight,
            sparse_weight=request.sparse_weight
        )
        
        # Query expansion if requested
        query_embedding = None
        if request.use_query_expansion:
            expander = QueryExpander(embedding_model, generator)
            query_embedding = expander.expand_query(request.query)
        
        # Retrieve - get top_k for ranking, but use only best for generation
        results = retriever.retrieve(
            request.query,
            top_k=request.top_k,
            use_hybrid=request.use_hybrid,
            query_embedding=query_embedding
        )
        
        # Get only the BEST matching chunk (highest score)
        best_chunk = results[0] if results else None
        
        response = {
            "query": request.query,
            "best_chunk": {
                "chunk_id": best_chunk['id'],
                "text": best_chunk['text'],
                "score": best_chunk['score'],
                "metadata": best_chunk.get('metadata', {})
            } if best_chunk else None,
            "all_chunks": [
                {
                    "chunk_id": r['id'],
                    "text": r['text'],
                    "score": r['score'],
                    "metadata": r.get('metadata', {})
                }
                for r in results
            ],
            "use_hybrid": request.use_hybrid,
            "use_query_expansion": request.use_query_expansion
        }
        
        # Generate answer if requested
        if request.generate_answer and best_chunk:
            # Use ONLY the best matching chunk for generation
            metadata = best_chunk.get('metadata', {})
            section = metadata.get('section', 'N/A')
            act = metadata.get('act', 'N/A').replace('\n', ' ').strip()
            score = best_chunk.get('score', 0.0)
            
            # Simple, focused context from the BEST chunk only
            context = f"""Source: Section {section}, {act}
Relevance Score: {score:.2f}

Legal Text:
{best_chunk['text']}"""
            
            # Build prompt for answer generation
            if request.answer_style == "detailed":
                prompt = f"""Based on the following legal provision from Indian mining laws, provide a comprehensive answer to the question.

Question: {request.query}

Legal Context:
{context}

Provide a detailed answer that:
1. Directly addresses the question
2. Cites the specific section and act
3. Explains key legal provisions clearly
4. Mentions requirements, procedures, or penalties if applicable
5. Uses clear, professional language

Answer:"""
            else:  # concise
                prompt = f"""Based on the legal provision below, provide a concise answer (2-3 sentences).

Question: {request.query}

Legal Context:
{context}

Concise Answer:"""
            
            # Check if we should use online LLM or local model
            if USE_ONLINE_LLM and LLM_API_KEY:
                # Use online LLM (Groq, OpenAI, etc.) - MUCH BETTER QUALITY!
                logger.info(f"Generating answer with {LLM_PROVIDER} ({LLM_MODEL})")
                answer = generate_answer_with_online_llm(prompt, provider=LLM_PROVIDER)
                response['model'] = f"{LLM_PROVIDER}/{LLM_MODEL}"
                response['generation_method'] = "online_llm"
                
            else:
                # Use local FLAN-T5 model
                logger.info("Generating answer with local FLAN-T5")
                
                # Enhanced generation parameters - optimized for CPU performance
                if request.answer_style == "detailed":
                    max_new_tokens = 300
                    min_length = 80
                    num_beams = 4
                    length_penalty = 2.0
                else:
                    max_new_tokens = 120
                    min_length = 30
                    num_beams = 3
                    length_penalty = 1.5
                
                generated = generator(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    min_length=min_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    do_sample=False
                )
                
                answer = generated[0]['generated_text'].strip()
                
                # Fallback if answer is too short
                if len(answer) < 30:
                    answer = f"Based on Section {section} of {act}: {best_chunk['text'][:300]}..."
                
                response['model'] = "google/flan-t5-base (local)"
                response['generation_method'] = "local_model"
            
            response['answer'] = answer
        
        return response
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
