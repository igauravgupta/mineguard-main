"""
FastAPI server for semantic search and QA over mining laws.

Endpoints:
- GET /: Health check
- POST /search: Top-K retrieval
- POST /chat: Retrieval + answer generation
"""

import pickle
import faiss
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration


app = FastAPI(title="Mining Laws QA API", version="1.0.0")

# Global state
chunks = None
encoder = None
index = None
tokenizer = None
generator = None


def load_resources():
    """Load all models and data at startup."""
    global chunks, encoder, index, tokenizer, generator
    
    print("Loading resources...")
    project_root = Path(__file__).parent
    data_dir = project_root / 'data'
    models_dir = project_root / 'models'
    
    # Load chunks
    chunks_file = data_dir / 'chunks.pkl'
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks not found: {chunks_file}")
    
    with open(chunks_file, 'rb') as f:
        chunks = pickle.load(f)
    print(f"✓ Loaded {len(chunks)} chunks")
    
    # Load encoder (use fine-tuned if available)
    finetuned_dir = models_dir / 'bi_encoder_finetuned'
    if finetuned_dir.exists():
        encoder = SentenceTransformer(str(finetuned_dir))
        print(f"✓ Loaded fine-tuned encoder")
    else:
        encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print(f"✓ Loaded base encoder")
    
    # Load FAISS index
    index_file = models_dir / 'mining_laws.index'
    if not index_file.exists():
        raise FileNotFoundError(f"Index not found: {index_file}")
    
    index = faiss.read_index(str(index_file))
    print(f"✓ Loaded FAISS index ({index.ntotal} vectors)")
    
    # Load T5 generator (use base for better quality, small for faster inference)
    model_name = 'google/flan-t5-base'  # Options: flan-t5-small, flan-t5-base, flan-t5-large
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    generator = T5ForConditionalGeneration.from_pretrained(model_name)
    print(f"✓ Loaded T5 generator ({model_name})")
    
    print("All resources loaded successfully!\n")


# Load on startup
@app.on_event("startup")
async def startup_event():
    load_resources()


# Request models
class SearchRequest(BaseModel):
    query: str
    top_k: int = 1


class ChatRequest(BaseModel):
    query: str
    generate_answer: bool = True
    top_k: int = 3
    answer_style: str = "detailed"  # "detailed" or "extractive"


# Endpoints
@app.get("/")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Mining Laws QA API",
        "chunks": len(chunks) if chunks else 0,
        "index_size": index.ntotal if index else 0
    }


@app.post("/search")
def search(request: SearchRequest):
    """
    Search for relevant chunks.
    
    Returns top-k most similar chunks to the query.
    """
    if not chunks or not encoder or not index:
        raise HTTPException(status_code=503, detail="Resources not loaded")
    
    # Encode query
    query_vec = encoder.encode([request.query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    
    # Search
    distances, indices = index.search(query_vec, request.top_k)
    
    # Format results
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            'chunk_id': int(idx),
            'text': chunks[idx],
            'score': float(dist)
        })
    
    return {
        'query': request.query,
        'top_k': request.top_k,
        'results': results
    }


@app.post("/chat")
def chat(request: ChatRequest):
    """
    Answer question using retrieval + generation.
    
    Retrieves relevant chunks and optionally generates an answer.
    """
    if not chunks or not encoder or not index:
        raise HTTPException(status_code=503, detail="Resources not loaded")
    
    # Encode query
    query_vec = encoder.encode([request.query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    
    # Search
    distances, indices = index.search(query_vec, request.top_k)
    
    # Format chunks
    retrieved_chunks = []
    for dist, idx in zip(distances[0], indices[0]):
        retrieved_chunks.append({
            'chunk_id': int(idx),
            'text': chunks[idx],
            'score': float(dist)
        })
    
    response = {
        'query': request.query,
        'chunks': retrieved_chunks
    }
    
    # Generate answer if requested
    if request.generate_answer and tokenizer and generator:
        # Combine multiple chunks for richer context
        top_chunks_for_context = retrieved_chunks[:min(request.top_k, len(retrieved_chunks))]
        context = "\n\n".join([f"[Section {i+1}]: {chunk['text']}" for i, chunk in enumerate(top_chunks_for_context)])
        
        if request.answer_style == "detailed":
            # Generate comprehensive, structured answer
            prompt = f"""You are a knowledgeable legal assistant specializing in Indian mining laws and regulations. 
Your task is to provide a clear, comprehensive, and well-structured answer to help users understand mining regulations.

Question: {request.query}

Relevant Legal Provisions:
{context}

Instructions for your answer:
1. Start with a direct answer to the question
2. Explain the key provisions and requirements in detail
3. Include specific details like time periods, responsibilities, procedures, or penalties if mentioned
4. Break down complex regulations into easy-to-understand points
5. Use professional but accessible language
6. Provide a complete answer that covers all relevant aspects from the legal provisions
7. Structure your answer with clear paragraphs

Please provide a detailed, professional answer:"""

            inputs = tokenizer(prompt, return_tensors='pt', max_length=1200, truncation=True)
            outputs = generator.generate(
                inputs.input_ids,
                max_length=400,
                min_length=80,
                num_beams=6,
                temperature=0.9,
                top_p=0.95,
                early_stopping=True,
                no_repeat_ngram_size=4,
                length_penalty=1.5,
                repetition_penalty=1.2,
                do_sample=False
            )
            
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        else:
            # Extractive style - simpler, more direct
            prompt = f"""Based on the legal context provided, answer the following question clearly and concisely.

Question: {request.query}

Context: {context}

Answer:"""
            
            inputs = tokenizer(prompt, return_tensors='pt', max_length=800, truncation=True)
            outputs = generator.generate(
                inputs.input_ids,
                max_length=200,
                min_length=30,
                num_beams=4,
                early_stopping=True
            )
            
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean and format the answer
        answer = answer.strip()
        
        # Remove any prompt remnants
        if answer.lower().startswith(('answer:', 'detailed answer:', 'professional answer:')):
            answer = answer.split(':', 1)[1].strip()
        
        # Ensure proper capitalization
        if answer:
            answer = answer[0].upper() + answer[1:]
        
        # Add punctuation if missing
        if answer and not answer.endswith(('.', '!', '?')):
            answer += '.'
        
        # If answer is still poor quality, create a structured extractive summary
        if len(answer) < 50 or not answer:
            # Fallback: Create structured answer from chunks
            key_points = []
            for chunk in top_chunks_for_context:
                # Extract sentences mentioning key terms from the query
                sentences = chunk['text'].split('.')
                for sentence in sentences[:3]:  # Take up to 3 sentences per chunk
                    if len(sentence.strip()) > 20:
                        key_points.append(sentence.strip())
            
            if key_points:
                answer = f"According to the mining regulations: {'. '.join(key_points[:3])}."
        
        response['answer'] = answer
        response['context_used'] = len(top_chunks_for_context)
        response['model'] = 'google/flan-t5-base'
        response['answer_style'] = request.answer_style
    
    return response


if __name__ == '__main__':
    import uvicorn
    print("Starting Mining Laws QA API server...")
    print("Visit http://localhost:8000/docs for API documentation")
    uvicorn.run(app, host="0.0.0.0", port=8000)
