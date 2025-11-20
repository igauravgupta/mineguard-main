"""
Generate answers using RAG (Retrieval-Augmented Generation).

Uses flan-t5-small to generate answers based on retrieved context.
"""

import pickle
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration


class AnswerGenerator:
    def __init__(self, model_path='sentence-transformers/all-MiniLM-L6-v2', use_finetuned=True):
        """Initialize retrieval and generation models."""
        project_root = Path(__file__).parent.parent
        data_dir = project_root / 'data'
        models_dir = project_root / 'models'
        
        # Load chunks
        chunks_file = data_dir / 'chunks.pkl'
        with open(chunks_file, 'rb') as f:
            self.chunks = pickle.load(f)
        
        # Load retrieval model
        finetuned_dir = models_dir / 'bi_encoder_finetuned'
        if use_finetuned and finetuned_dir.exists():
            print(f"Loading fine-tuned model from {finetuned_dir}")
            self.encoder = SentenceTransformer(str(finetuned_dir))
        else:
            print(f"Loading base model: {model_path}")
            self.encoder = SentenceTransformer(model_path)
        
        # Load FAISS index
        index_file = models_dir / 'mining_laws.index'
        self.index = faiss.read_index(str(index_file))
        
        # Load generation model
        print("Loading T5 generation model...")
        self.tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')
        self.generator = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')
        
        print("✓ AnswerGenerator initialized")
    
    def retrieve(self, query, top_k=3):
        """Retrieve top-k relevant chunks."""
        query_vec = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)
        
        distances, indices = self.index.search(query_vec, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({
                'chunk': self.chunks[idx],
                'score': float(dist)
            })
        return results
    
    def generate(self, query, context):
        """Generate answer from query and context."""
        prompt = f"Question: {query}\n\nContext: {context}\n\nAnswer:"
        
        inputs = self.tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)
        outputs = self.generator.generate(
            inputs.input_ids,
            max_length=150,
            num_beams=4,
            early_stopping=True
        )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    
    def answer(self, query, top_k=3, generate=True):
        """Retrieve and optionally generate answer."""
        # Retrieve relevant chunks
        results = self.retrieve(query, top_k=top_k)
        
        if not generate:
            return {
                'query': query,
                'chunks': results
            }
        
        # Generate answer from top chunk
        context = results[0]['chunk']
        answer = self.generate(query, context)
        
        return {
            'query': query,
            'answer': answer,
            'chunks': results
        }


def main():
    """Demo answer generation."""
    print("Initializing Answer Generator...\n")
    
    generator = AnswerGenerator(use_finetuned=True)
    
    # Test queries
    test_queries = [
        "What is the penalty for illegal mining?",
        "What is a mining lease?",
        "What are the safety requirements for explosives?"
    ]
    
    print("\n" + "=" * 60)
    print("ANSWER GENERATION DEMO")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        
        result = generator.answer(query, top_k=1, generate=True)
        
        print(f"Answer: {result['answer']}")
        print(f"\nSource (score={result['chunks'][0]['score']:.4f}):")
        print(f"{result['chunks'][0]['chunk'][:200]}...")
        print("=" * 60)


if __name__ == '__main__':
    main()
