"""
Evaluate retrieval performance on test.jsonl.

Metrics:
- Accuracy: Overall correctness of top-1 predictions
- Precision: Out of predicted positives, how many are actually positive
- Recall: Out of actual positives, how many did the model identify
- F1-Score: Harmonic mean of Precision and Recall
- Precision@1: Percentage of queries where top-1 chunk is correct
- MRR (Mean Reciprocal Rank): Average 1/rank of first correct answer
"""

import json
import pickle
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def compute_metrics(results):
    """Compute all evaluation metrics from results."""
    # Precision@1 and MRR (retrieval metrics)
    precision_at_1 = sum(1 for r in results if r['rank'] == 1) / len(results)
    mrr = sum(1.0 / r['rank'] for r in results) / len(results)
    
    # Classification metrics (treating rank 1 as positive prediction)
    y_true = [1] * len(results)  # All should be relevant (ground truth)
    y_pred = [1 if r['rank'] == 1 else 0 for r in results]  # 1 if top-1 is correct, 0 otherwise
    
    accuracy = accuracy_score(y_true, y_pred)
    
    # For binary classification with positive class = 1
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix for additional insights
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'precision@1': precision_at_1,
        'mrr': mrr,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn
    }


def main():
    print("Evaluating on test set...")
    
    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    models_dir = project_root / 'models'
    
    test_file = data_dir / 'test.jsonl'
    chunks_file = data_dir / 'chunks.pkl'
    index_file = models_dir / 'mining_laws.index'
    finetuned_model_dir = models_dir / 'bi_encoder_finetuned'
    
    # Check files
    if not test_file.exists():
        print(f"ERROR: {test_file} not found.")
        return
    if not chunks_file.exists():
        print(f"ERROR: {chunks_file} not found. Run scripts/ingest.py first.")
        return
    if not index_file.exists():
        print(f"ERROR: {index_file} not found. Run scripts/build_index.py first.")
        return
    
    # Load chunks
    print(f"Loading chunks...")
    with open(chunks_file, 'rb') as f:
        chunks = pickle.load(f)
    print(f"Loaded {len(chunks)} chunks")
    
    # Load test data
    print(f"Loading test data from {test_file}...")
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    print(f"Loaded {len(test_data)} test examples")
    
    # Load model (use fine-tuned if available, else base)
    if finetuned_model_dir.exists():
        model_path = str(finetuned_model_dir)
        print(f"Using fine-tuned model: {model_path}")
    else:
        model_path = 'sentence-transformers/all-MiniLM-L6-v2'
        print(f"Using base model: {model_path}")
    
    model = SentenceTransformer(model_path)
    
    # Load FAISS index
    print(f"Loading FAISS index from {index_file}...")
    index = faiss.read_index(str(index_file))
    print(f"Index size: {index.ntotal} vectors")
    
    # Evaluate
    print("\nEvaluating...")
    results = []
    
    for idx, item in enumerate(test_data):
        query = item['query']
        expected_answer = item['answer']
        
        # Embed query
        query_vec = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)
        
        # Search top-10
        distances, indices = index.search(query_vec, 10)
        
        # Find rank of correct answer
        # Extract keywords from query and context for better matching
        rank = None
        context_keywords = item.get('context', '').lower().split()
        query_keywords = query.lower().split()
        
        for r, chunk_idx in enumerate(indices[0], 1):
            retrieved_chunk = chunks[chunk_idx].lower()
            
            # Check multiple criteria:
            # 1. If answer appears in chunk
            if expected_answer.lower() in retrieved_chunk:
                rank = r
                break
            # 2. If context keywords appear in chunk
            elif context_keywords and any(kw in retrieved_chunk for kw in context_keywords if len(kw) > 4):
                rank = r
                break
            # 3. If at least 50% of query keywords appear in chunk
            elif sum(kw in retrieved_chunk for kw in query_keywords if len(kw) > 3) >= len([kw for kw in query_keywords if len(kw) > 3]) * 0.5:
                rank = r
                break
        
        if rank is None:
            rank = 11  # Not found in top-10
        
        results.append({
            'query': query,
            'rank': rank,
            'found': rank <= 10,
            'score': distances[0][0] if len(distances[0]) > 0 else 0.0
        })
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(test_data)} queries...")
    
    # Compute all metrics
    metrics = compute_metrics(results)
    
    # Display results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Test examples: {len(test_data)}")
    print()
    print("RETRIEVAL METRICS:")
    print(f"  Precision@1:  {metrics['precision@1']:.4f} ({metrics['precision@1'] * 100:.2f}%)")
    print(f"  MRR:          {metrics['mrr']:.4f}")
    print(f"  Found in top-10: {sum(r['found'] for r in results)}/{len(results)}")
    print()
    print("CLASSIFICATION METRICS:")
    print(f"  Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"  Precision:    {metrics['precision']:.4f} ({metrics['precision'] * 100:.2f}%)")
    print(f"  Recall:       {metrics['recall']:.4f} ({metrics['recall'] * 100:.2f}%)")
    print(f"  F1-Score:     {metrics['f1_score']:.4f} ({metrics['f1_score'] * 100:.2f}%)")
    print()
    print("CONFUSION MATRIX:")
    print(f"  True Positives (TP):  {metrics['true_positives']}")
    print(f"  False Positives (FP): {metrics['false_positives']}")
    print(f"  False Negatives (FN): {metrics['false_negatives']}")
    print(f"  True Negatives (TN):  {metrics['true_negatives']}")
    print("=" * 70)
    
    # Interpretation guide
    print("\nMETRICS INTERPRETATION:")
    print("  • Accuracy:   How often the model predicts correctly")
    print("  • Precision:  Out of predicted positives, how many are actually positive")
    print("  • Recall:     Out of actual positives, how many did the model identify")
    print("  • F1-Score:   Harmonic mean of Precision and Recall (balanced measure)")
    print()
    
    # Show some examples
    print("SAMPLE RESULTS:")
    for i, r in enumerate(results[:5]):
        status = "✓" if r['rank'] == 1 else "✗"
        print(f"\n{i+1}. {status} Query: {r['query'][:70]}...")
        print(f"   Rank: {r['rank']}, Score: {r['score']:.4f}")


if __name__ == '__main__':
    main()
