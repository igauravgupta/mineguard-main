"""
FastAPI Server for Mining Laws Chatbot
Exposes the complete RAG pipeline as REST API endpoints
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
import os
import uvicorn
from datetime import datetime

# Import the RAG pipeline
from src.complete_rag_pipeline import MiningLawsChatbot

# Global chatbot instance (initialized on startup)
chatbot = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    # Startup
    global chatbot
    
    print("\n" + "="*70)
    print("Starting Mining Laws Chatbot API Server")
    print("="*70)
    
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("\n⚠ WARNING: GROQ_API_KEY not set!")
        print("Set it using: export GROQ_API_KEY='your-key-here'")
        print("API will not work without it.\n")
    else:
        try:
            # Initialize chatbot
            chatbot = MiningLawsChatbot(
                retrieval_method="hybrid",
                top_k=5,
                min_answer_length=100
            )
            print("\n✓ Chatbot initialized successfully!")
            print("API server ready on http://localhost:8000")
            print("Docs available at http://localhost:8000/docs")
            print("="*70 + "\n")
        except Exception as e:
            print(f"\n✗ Failed to initialize chatbot: {e}")
            print("\nMake sure you've run:")
            print("  python build_base_index.py")
            print("="*70 + "\n")
    
    yield
    
    # Shutdown
    print("\nShutting down API server...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Mining Laws Chatbot API",
    description="RAG-based API for Indian mining and explosives laws",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global chatbot instance (initialized on startup)
chatbot = None


# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for asking questions"""
    query: str = Field(..., description="User's question", min_length=1, max_length=500)
    retrieval_method: Optional[str] = Field("hybrid", description="Retrieval method: dense, sparse, or hybrid")
    top_k: Optional[int] = Field(5, description="Number of chunks to retrieve", ge=1, le=20)
    min_answer_length: Optional[int] = Field(100, description="Minimum answer length", ge=50, le=500)

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What licenses are required to manufacture explosives?",
                "retrieval_method": "hybrid",
                "top_k": 5,
                "min_answer_length": 100
            }
        }
    }


class BatchQueryRequest(BaseModel):
    """Request model for batch queries"""
    queries: List[str] = Field(..., description="List of questions", min_length=1, max_length=10)
    retrieval_method: Optional[str] = Field("hybrid", description="Retrieval method")
    top_k: Optional[int] = Field(5, description="Number of chunks to retrieve", ge=1, le=20)

    model_config = {
        "json_schema_extra": {
            "example": {
                "queries": [
                    "What are the safety requirements for explosives?",
                    "Who can inspect mining operations?"
                ],
                "retrieval_method": "hybrid",
                "top_k": 5
            }
        }
    }


class RetrieveRequest(BaseModel):
    """Request model for chunk retrieval"""
    query: str = Field(..., description="Search query", min_length=1, max_length=500)
    method: str = Field("hybrid", description="Retrieval method: dense, sparse, or hybrid")
    top_k: int = Field(5, description="Number of chunks to retrieve", ge=1, le=20)

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "explosives storage requirements",
                "method": "hybrid",
                "top_k": 3
            }
        }
    }


class AnswerResponse(BaseModel):
    """Response model for answers"""
    answer: str
    source: str
    sources: List[str]
    validation: Dict
    query: str
    tokens_used: Dict
    retrieval_method: str
    top_k: int
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    model: str
    retrieval_methods: List[str]


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: str


# API Endpoints

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Mining Laws Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "ask": "/ask",
            "batch": "/batch",
            "retrieve": "/retrieve"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    if chatbot is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chatbot not initialized. Check GROQ_API_KEY and FAISS index."
        )
    
    return HealthResponse(
        status="healthy",
        message="Chatbot is ready",
        model="Groq Llama 3.1 70B",
        retrieval_methods=["dense", "sparse", "hybrid"]
    )


@app.post("/ask", response_model=AnswerResponse, tags=["Query"])
async def ask_question(request: QueryRequest):
    """
    Ask a question and get an answer with citations
    
    This endpoint implements the complete Step 14 RAG pipeline:
    1. Receives user query
    2. Runs hybrid retrieval (FAISS + BM25 + RRF)
    3. Generates answer with LLM
    4. Post-processes (clean, cite, validate)
    5. Returns structured response
    """
    if chatbot is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chatbot not initialized. Check server logs."
        )
    
    try:
        # Temporarily update chatbot settings for this request
        original_method = chatbot.retrieval_method
        original_top_k = chatbot.top_k
        original_min_length = chatbot.min_answer_length
        
        chatbot.retrieval_method = request.retrieval_method
        chatbot.top_k = request.top_k
        chatbot.min_answer_length = request.min_answer_length
        
        # Get answer from RAG pipeline
        response = chatbot.ask(request.query, verbose=False)
        
        # Restore original settings
        chatbot.retrieval_method = original_method
        chatbot.top_k = original_top_k
        chatbot.min_answer_length = original_min_length
        
        # Add timestamp
        response['timestamp'] = datetime.now().isoformat()
        
        return AnswerResponse(**response)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@app.post("/batch", tags=["Query"])
async def batch_queries(request: BatchQueryRequest):
    """
    Process multiple queries in batch
    
    Returns a list of answers for all queries.
    Maximum 10 queries per request.
    """
    if chatbot is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chatbot not initialized. Check server logs."
        )
    
    if len(request.queries) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10 queries per batch request"
        )
    
    try:
        # Temporarily update settings
        original_method = chatbot.retrieval_method
        original_top_k = chatbot.top_k
        
        chatbot.retrieval_method = request.retrieval_method
        chatbot.top_k = request.top_k
        
        # Process all queries
        results = []
        for query in request.queries:
            response = chatbot.ask(query, verbose=False)
            response['timestamp'] = datetime.now().isoformat()
            results.append(response)
        
        # Restore settings
        chatbot.retrieval_method = original_method
        chatbot.top_k = original_top_k
        
        return {
            "count": len(results),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing batch queries: {str(e)}"
        )


@app.post("/retrieve", tags=["Retrieval"])
async def retrieve_chunks(request: RetrieveRequest):
    """
    Retrieve relevant chunks without generating an answer
    
    Useful for testing retrieval quality or building custom pipelines.
    """
    if chatbot is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chatbot not initialized"
        )
    
    if request.method not in ["dense", "sparse", "hybrid"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Method must be 'dense', 'sparse', or 'hybrid'"
        )
    
    try:
        # Get chunks from retrieval system
        chunks = chatbot.retrieval.search(
            query=request.query,
            method=request.method,
            top_k=request.top_k
        )
        
        return {
            "query": request.query,
            "method": request.method,
            "count": len(chunks),
            "chunks": chunks
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving chunks: {str(e)}"
        )


@app.get("/methods", tags=["Info"])
async def get_methods():
    """Get available retrieval methods and their descriptions"""
    return {
        "methods": {
            "dense": {
                "name": "Dense Retrieval (Semantic)",
                "description": "Uses FAISS with semantic embeddings",
                "best_for": "Conceptual queries, meaning-based search"
            },
            "sparse": {
                "name": "Sparse Retrieval (Keyword)",
                "description": "Uses BM25 for keyword matching",
                "best_for": "Specific terms, section numbers"
            },
            "hybrid": {
                "name": "Hybrid Retrieval (Recommended)",
                "description": "Combines dense + sparse with RRF fusion",
                "best_for": "Best overall performance"
            }
        },
        "default": "hybrid"
    }


@app.get("/stats", tags=["Info"])
async def get_stats():
    """Get system statistics"""
    if chatbot is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chatbot not initialized"
        )
    
    return {
        "total_chunks": len(chatbot.retrieval.chunks),
        "embedding_dimension": 768,
        "model": "all-mpnet-base-v2",
        "llm": "Groq Llama 3.1 70B",
        "index_type": "FAISS IndexFlatIP",
        "current_settings": {
            "retrieval_method": chatbot.retrieval_method,
            "top_k": chatbot.top_k,
            "min_answer_length": chatbot.min_answer_length
        }
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "status_code": 500
        }
    )


# Run server
if __name__ == "__main__":
    print("\n" + "="*70)
    print("Mining Laws Chatbot API Server")
    print("="*70)
    print("\nStarting server on http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Alternative docs: http://localhost:8000/redoc")
    print("\nPress Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
