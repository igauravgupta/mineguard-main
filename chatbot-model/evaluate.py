"""
Simple Evaluation Script for Mining Laws RAG Chatbot
Tests the model on test.jsonl and shows metrics

Dual Evaluation Metrics:
1. Chunk Retrieval Accuracy (partial scoring)
2. Semantic Answer Similarity (using embeddings)
Final Score = Average of both metrics
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.complete_rag_pipeline import MiningLawsChatbot
from sentence_transformers import SentenceTransformer


def load_env():
    """Load API key from .env file"""
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    if line.startswith('export '):
                        line = line[7:]
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")


def load_test_data(path: str = "data/test.jsonl") -> List[Dict]:
    """Load test data from JSONL file"""
    test_data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                test_data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️ Warning: Invalid JSON at line {line_num}: {e}")
                continue
    return test_data


def calculate_chunk_accuracy(retrieved_ids: List[int], relevant_ids: List[int], k: int = 5) -> float:
    """
    Partial scoring for chunk retrieval accuracy.
    Score = (number of relevant chunks found in top-k) / (total relevant chunks)
    
    Args:
        retrieved_ids: List of retrieved chunk IDs
        relevant_ids: List of ground truth relevant chunk IDs
        k: Number of top results to consider
        
    Returns:
        Float between 0.0 and 1.0
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0
    
    # Convert to sets for intersection
    retrieved_set = set(map(int, retrieved_ids[:k]))
    relevant_set = set(map(int, relevant_ids))
    
    # Count matches
    matches = len(retrieved_set & relevant_set)
    
    # Partial score: matches / total relevant
    score = matches / len(relevant_set)
    
    return score


def calculate_semantic_similarity(generated: str, reference: str, model: SentenceTransformer) -> float:
    """
    Calculate semantic similarity between generated and reference answers
    using sentence embeddings (cosine similarity).
    
    Args:
        generated: Generated answer text
        reference: Reference answer text
        model: SentenceTransformer model for encoding
        
    Returns:
        Cosine similarity score between 0.0 and 1.0
    """
    if not generated or not reference:
        return 0.0
    
    # Encode both texts
    embeddings = model.encode([generated, reference], convert_to_numpy=True)
    
    # Calculate cosine similarity
    # Normalize embeddings
    gen_emb = embeddings[0] / np.linalg.norm(embeddings[0])
    ref_emb = embeddings[1] / np.linalg.norm(embeddings[1])
    
    # Cosine similarity
    similarity = np.dot(gen_emb, ref_emb)
    
    # Ensure in [0, 1] range (cosine is in [-1, 1], but semantic text is usually positive)
    similarity = max(0.0, min(1.0, similarity))
    
    return float(similarity)


def main():
    """Run evaluation"""
    print("="*70)
    print("MINING LAWS CHATBOT - EVALUATION")
    print("="*70)
    
    # Load environment
    load_env()
    if not os.getenv('GROQ_API_KEY'):
        print("\n❌ Error: GROQ_API_KEY not found!")
        print("Please create .env file with:")
        print("export GROQ_API_KEY='your-key-here'")
        return
    
    print(f"✓ API Key loaded: {os.getenv('GROQ_API_KEY')[:20]}...")
    
    # Load test data
    print("\nLoading test data...")
    try:
        test_data = load_test_data()
        print(f"✓ Loaded {len(test_data)} test samples")
    except FileNotFoundError:
        print("❌ Error: data/test.jsonl not found!")
        return
    
    # Initialize chatbot
    print("\nInitializing chatbot...")
    chatbot = MiningLawsChatbot(
        retrieval_method="hybrid",
        top_k=5
    )
    print("✓ Chatbot ready")
    
    # Load embedding model for semantic similarity
    print("\nLoading embedding model for semantic evaluation...")
    try:
        semantic_model = SentenceTransformer('models/bi_encoder_finetuned')
        print("✓ Using fine-tuned model for semantic evaluation")
    except:
        semantic_model = SentenceTransformer('all-mpnet-base-v2')
        print("✓ Using base model for semantic evaluation")
    
    # Run evaluation
    print("\n" + "="*70)
    print("RUNNING DUAL-METRIC EVALUATION")
    print("="*70)
    
    results = []
    total_chunk_accuracy = 0
    total_semantic_similarity = 0
    total_final_accuracy = 0
    num_completed = 0
    num_failed = 0
    
    for i, sample in enumerate(test_data, 1):
        query = sample.get("question") or sample.get("query", "")
        reference = sample.get("answer", "")
        relevant_chunks = sample.get("relevant_chunks", [])
        
        print(f"\n[{i}/{len(test_data)}] {query[:60]}...")
        
        # Get answer
        try:
            response = chatbot.ask(query, verbose=False)
            generated = response.get("answer", "")
            chunks = response.get("chunks", [])
            
            # Extract retrieved chunk IDs
            retrieved_ids = [c.get('chunk_id', c['metadata'].get('chunk_id', '')) for c in chunks]
            
            # Calculate chunk retrieval accuracy (partial scoring)
            chunk_acc = 0.0
            if relevant_chunks:
                chunk_acc = calculate_chunk_accuracy(retrieved_ids, relevant_chunks, k=5)
            
            # Calculate semantic similarity
            semantic_sim = calculate_semantic_similarity(generated, reference, semantic_model)
            
            # Calculate final accuracy (average of both metrics)
            final_acc = (chunk_acc + semantic_sim) / 2.0
            
            # Update totals
            total_chunk_accuracy += chunk_acc
            total_semantic_similarity += semantic_sim
            total_final_accuracy += final_acc
            num_completed += 1
            
            print(f"  ✓ Chunk Accuracy: {chunk_acc:.3f} | Semantic Similarity: {semantic_sim:.3f} | Final: {final_acc:.3f}")
            
            # Check if answer has citations
            has_citations = "Section" in generated or "Act" in generated
            
            results.append({
                "query": query,
                "reference": reference,
                "generated": generated,
                "has_citations": has_citations,
                "chunk_accuracy": chunk_acc,
                "semantic_similarity": semantic_sim,
                "final_accuracy": final_acc,
                "status": "completed"
            })
            
        except Exception as e:
            print(f"  ❌ FAILED: {str(e)[:100]}")
            results.append({
                "query": query,
                "reference": reference,
                "generated": None,
                "has_citations": False,
                "chunk_accuracy": None,
                "semantic_similarity": None,
                "final_accuracy": None,
                "status": "failed",
                "error": str(e)
            })
            num_failed += 1
            continue
    
    # Calculate averages
    print("\n" + "="*70)
    print("EVALUATION RESULTS - DUAL METRICS")
    print("="*70)
    
    print(f"\n📊 SUMMARY:")
    print("-"*70)
    print(f"  Total queries:     {len(test_data)}")
    print(f"  Completed:         {num_completed} ✓")
    print(f"  Failed:            {num_failed} ✗")
    
    if num_completed > 0:
        avg_chunk_acc = total_chunk_accuracy / num_completed
        avg_semantic_sim = total_semantic_similarity / num_completed
        avg_final_acc = total_final_accuracy / num_completed
        
        print(f"\n📊 DUAL-METRIC EVALUATION SCORES:")
        print("-"*70)
        print(f"  Chunk Retrieval Accuracy:    {avg_chunk_acc:.2%}")
        print(f"  Semantic Answer Similarity:  {avg_semantic_sim:.2%}")
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  FINAL ACCURACY (Average):    {avg_final_acc:.2%}")
        
        # Check targets
        print(f"\n🎯 PERFORMANCE EVALUATION:")
        print("-"*70)
        if avg_final_acc >= 0.85:
            print(f"  ✅ EXCELLENT - Final Accuracy {avg_final_acc:.1%} >= 85%")
        elif avg_final_acc >= 0.70:
            print(f"  ✓ GOOD - Final Accuracy {avg_final_acc:.1%} >= 70%")
        elif avg_final_acc >= 0.50:
            print(f"  ⚠️  FAIR - Final Accuracy {avg_final_acc:.1%} >= 50%")
        else:
            print(f"  ❌ NEEDS IMPROVEMENT - Final Accuracy {avg_final_acc:.1%} < 50%")
        
        print(f"\n  Breakdown:")
        print(f"    • Chunk retrieval: {avg_chunk_acc:.1%}")
        print(f"    • Semantic quality: {avg_semantic_sim:.1%}")
    else:
        print("\n⚠️  No completed samples to calculate metrics")
    
    # Show sample answers from completed queries only
    completed_results = [r for r in results if r.get('status') == 'completed']
    
    if completed_results:
        print("\n" + "="*70)
        print("SAMPLE PREDICTIONS (First 3 completed)")
        print("="*70)
        
        for i, result in enumerate(completed_results[:3], 1):
            print(f"\n{'='*70}")
            print(f"SAMPLE {i}")
            print(f"{'='*70}")
            print(f"\n❓ Query:")
            print(f"   {result['query']}")
            print(f"\n🤖 Generated Answer:")
            answer_preview = result['generated'][:300] if result['generated'] else "N/A"
            print(f"   {answer_preview}...")
            print(f"\n✓ Has Citations: {result['has_citations']}")
            print(f"✓ Chunk Accuracy: {result.get('chunk_accuracy', 0):.2%}")
            print(f"✓ Semantic Similarity: {result.get('semantic_similarity', 0):.2%}")
            print(f"✓ Final Accuracy: {result.get('final_accuracy', 0):.2%}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation_results_{timestamp}.json"
    
    output_data = {
        "timestamp": timestamp,
        "total_queries": len(test_data),
        "completed": num_completed,
        "failed": num_failed,
        "results": results
    }
    
    # Add averages if we have completed queries
    if num_completed > 0:
        output_data["avg_chunk_accuracy"] = total_chunk_accuracy / num_completed
        output_data["avg_semantic_similarity"] = total_semantic_similarity / num_completed
        output_data["avg_final_accuracy"] = total_final_accuracy / num_completed
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    if num_failed > 0:
        print(f"\n⚠️  {num_failed} queries failed (rate limit or API errors)")
        print(f"   But calculated metrics from {num_completed} successful queries!")
    
    print("\n" + "="*70)
    print("✅ DUAL-METRIC EVALUATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
