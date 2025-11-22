"""
Simple Evaluation Script for Mining Laws RAG Chatbot
Tests the model on test.jsonl and shows metrics
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.complete_rag_pipeline import MiningLawsChatbot


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
        for line in f:
            test_data.append(json.loads(line.strip()))
    return test_data


def calculate_accuracy(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
    """Check if at least one relevant doc in top-K"""
    if not retrieved_ids or not relevant_ids:
        return 0.0
    return 1.0 if len(set(retrieved_ids[:k]) & set(relevant_ids)) > 0 else 0.0


def calculate_precision(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
    """Precision@K"""
    if k == 0:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    return len(set(retrieved_k) & set(relevant_ids)) / k


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
    
    # Run evaluation
    print("\n" + "="*70)
    print("RUNNING EVALUATION")
    print("="*70)
    
    results = []
    total_accuracy_5 = 0
    total_precision_1 = 0
    num_with_relevant = 0
    num_completed = 0
    num_failed = 0
    
    for i, sample in enumerate(test_data, 1):
        query = sample.get("question", "")
        reference = sample.get("answer", "")
        relevant_chunks = sample.get("relevant_chunks", [])
        
        print(f"\n[{i}/{len(test_data)}] {query[:60]}...")
        
        # Get answer
        try:
            response = chatbot.ask(query, verbose=False)
            generated = response.get("answer", "")
            chunks = response.get("chunks", [])
            
            # Extract retrieved chunk IDs
            retrieved_ids = [str(c['metadata'].get('chunk_id', '')) for c in chunks]
            
            # Calculate metrics
            acc_5 = None
            prec_1 = None
            if relevant_chunks:
                acc_5 = calculate_accuracy(retrieved_ids, relevant_chunks, k=5)
                prec_1 = calculate_precision(retrieved_ids, relevant_chunks, k=1)
                total_accuracy_5 += acc_5
                total_precision_1 += prec_1
                num_with_relevant += 1
                
                print(f"  ✓ Accuracy@5: {acc_5:.2f} | Precision@1: {prec_1:.2f}")
            else:
                print(f"  ✓ Completed (no relevant chunks to measure)")
            
            # Check if answer has citations
            has_citations = "Section" in generated or "Act" in generated
            
            results.append({
                "query": query,
                "reference": reference,
                "generated": generated,
                "has_citations": has_citations,
                "accuracy_5": acc_5,
                "precision_1": prec_1,
                "status": "completed"
            })
            
            num_completed += 1
            
        except Exception as e:
            print(f"  ❌ FAILED: {str(e)[:100]}")
            results.append({
                "query": query,
                "reference": reference,
                "generated": None,
                "has_citations": False,
                "accuracy_5": None,
                "precision_1": None,
                "status": "failed",
                "error": str(e)
            })
            num_failed += 1
            continue
    
    # Calculate averages
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print(f"\n📊 SUMMARY:")
    print("-"*70)
    print(f"  Total queries:     {len(test_data)}")
    print(f"  Completed:         {num_completed} ✓")
    print(f"  Failed:            {num_failed} ✗")
    
    avg_accuracy_5 = None
    avg_precision_1 = None
    
    if num_with_relevant > 0:
        avg_accuracy_5 = total_accuracy_5 / num_with_relevant
        avg_precision_1 = total_precision_1 / num_with_relevant
        
        print(f"\n📊 METRICS (averaged over {num_with_relevant} completed samples with ground truth):")
        print("-"*70)
        print(f"  Accuracy@5:    {avg_accuracy_5:.2%}  (Target: >90%)")
        print(f"  Precision@1:   {avg_precision_1:.2%}  (Target: >85%)")
        
        # Check targets
        print("\n🎯 TARGET CHECK:")
        print("-"*70)
        if avg_accuracy_5 > 0.90:
            print("  ✅ PASS - Accuracy@5 > 90%")
        else:
            print("  ❌ FAIL - Accuracy@5 <= 90%")
        
        if avg_precision_1 > 0.85:
            print("  ✅ PASS - Precision@1 > 85%")
        else:
            print("  ❌ FAIL - Precision@1 <= 85%")
    else:
        print("\n⚠️  No samples with ground truth to calculate metrics")
    
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
            if result['accuracy_5'] is not None:
                print(f"✓ Accuracy@5: {result['accuracy_5']:.2f}")
                print(f"✓ Precision@1: {result['precision_1']:.2f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation_results_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp,
            "total_queries": len(test_data),
            "completed": num_completed,
            "failed": num_failed,
            "avg_accuracy_5": avg_accuracy_5,
            "avg_precision_1": avg_precision_1,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    if num_failed > 0:
        print(f"\n⚠️  {num_failed} queries failed (rate limit or API errors)")
        print(f"   But calculated metrics from {num_completed} successful queries!")
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
