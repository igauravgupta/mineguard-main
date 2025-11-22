"""
Step 6 & 7: Fine-tune model and rebuild index
Separate script for fine-tuning on domain data
"""

import json
import random
from pathlib import Path
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import numpy as np
import faiss


def l2_normalize(embeddings):
    """L2-normalize embeddings for cosine similarity"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def load_qa_pairs(file_path):
    """Load Q&A pairs from JSONL file"""
    qa_pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                qa_pairs.append(json.loads(line))
    return qa_pairs


def prepare_training_data():
    """Prepare training data for fine-tuning with diverse evaluation examples"""
    
    data_dir = Path("data")
    train_file = data_dir / "train.jsonl"
    
    print("\nPreparing training data...")
    
    # Load all Q&A pairs from train.jsonl
    all_pairs = load_qa_pairs(train_file)
    
    # Shuffle data for random split
    random.shuffle(all_pairs)
    
    # Split: 80% train, 20% eval
    split_idx = int(len(all_pairs) * 0.8)
    train_pairs = all_pairs[:split_idx]
    eval_pairs = all_pairs[split_idx:]
    
    # Create training examples (no scores needed for training)
    train_examples = []
    for pair in train_pairs:
        # Handle both 'query' and 'question' fields
        query = pair.get('query') or pair.get('question')
        answer = pair['answer']
        # Use query-answer pairs for contrastive learning
        train_examples.append(
            InputExample(texts=[query, answer])
        )
    
    # Create evaluation examples WITH VARIED SCORES (fixes NaN issue)
    eval_examples = []
    half_point = len(eval_pairs) // 2
    
    for i, pair in enumerate(eval_pairs):
        # Handle both 'query' and 'question' fields
        query = pair.get('query') or pair.get('question')
        answer = pair['answer']
        
        if i < half_point:
            # First half: Positive pairs (query with correct answer) → score 1.0
            eval_examples.append(
                InputExample(texts=[query, answer], label=1.0)
            )
        else:
            # Second half: Negative pairs (query with wrong answer) → score 0.0
            # Get wrong answer from next item
            wrong_answer = eval_pairs[(i + 1) % len(eval_pairs)]['answer']
            eval_examples.append(
                InputExample(texts=[query, wrong_answer], label=0.0)
            )
    
    print(f"✓ Training examples: {len(train_examples)}")
    print(f"✓ Evaluation examples: {len(eval_examples)} ({half_point} positive, {len(eval_examples) - half_point} negative)")
    
    return train_examples, eval_examples


def fine_tune_model(epochs=3, batch_size=16, learning_rate=2e-5):
    """Fine-tune bi-encoder on mining law domain"""
    
    print("="*80)
    print("STEP 6: FINE-TUNING BI-ENCODER")
    print("="*80 + "\n")
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    finetuned_model_dir = models_dir / "bi_encoder_finetuned"
    
    # Load base model
    print("Loading base model: all-mpnet-base-v2")
    model = SentenceTransformer('all-mpnet-base-v2')
    print(f"✓ Model loaded (dim: {model.get_sentence_embedding_dimension()})")
    
    # Prepare training data
    train_examples, eval_examples = prepare_training_data()
    
    # Create DataLoader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size
    )
    
    # Define loss function
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Calculate warmup steps
    num_epochs = epochs
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Total training steps: {len(train_dataloader) * num_epochs}")
    print(f"  Loss function: MultipleNegativesRankingLoss")
    print(f"\nStarting fine-tuning...\n")
    
    # Create evaluator
    evaluator = None
    if eval_examples:
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            eval_examples,
            name='mining-laws-eval'
        )
    
    # Fine-tune
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': learning_rate},
        evaluator=evaluator,
        evaluation_steps=len(train_dataloader) // 2 if len(train_dataloader) > 2 else 1,
        output_path=str(finetuned_model_dir),
        save_best_model=True,
        show_progress_bar=True
    )
    
    print(f"\n{'='*80}")
    print("FINE-TUNING COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Fine-tuning completed!")
    print(f"  Model saved to: {finetuned_model_dir}")
    print("="*80)
    
    return model


def rebuild_index_with_finetuned_model():
    """Rebuild index with fine-tuned model"""
    
    print(f"\n{'='*80}")
    print("STEP 7: REBUILDING INDEX WITH FINE-TUNED MODEL")
    print(f"{'='*80}\n")
    
    data_dir = Path("data")
    models_dir = Path("models")
    
    chunks_file = data_dir / "mining_laws_chunks.json"
    finetuned_model_dir = models_dir / "bi_encoder_finetuned"
    finetuned_index_file = models_dir / "mining_laws_finetuned.index"
    metadata_file = models_dir / "chunks_metadata.json"
    
    # Load fine-tuned model
    print(f"Loading fine-tuned model from {finetuned_model_dir}...")
    model = SentenceTransformer(str(finetuned_model_dir))
    print("✓ Fine-tuned model loaded")
    
    # Load chunks
    print("\nLoading chunks...")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    chunks = data['chunks']
    texts = [chunk['text'] for chunk in chunks]
    print(f"✓ Loaded {len(texts)} chunks")
    
    # Re-encode with fine-tuned model
    print(f"\nRe-encoding chunks with fine-tuned model...")
    print("This may take 5-10 minutes on CPU...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=8,  # Smaller batch for better progress
        convert_to_numpy=True,
        device='cpu'
    )
    
    # L2-normalize
    print("Normalizing embeddings...")
    embeddings = l2_normalize(embeddings)
    
    # Create new FAISS index
    print("Creating new FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))
    
    # Save new index
    print(f"Saving fine-tuned index to {finetuned_index_file}...")
    faiss.write_index(index, str(finetuned_index_file))
    
    print(f"\n{'='*80}")
    print("INDEX REBUILD COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Index rebuilt successfully!")
    print(f"  Index type: FAISS IndexFlatIP (cosine similarity)")
    print(f"  Total vectors: {index.ntotal}")
    print(f"  Dimension: {dimension}")
    print(f"  Index size: {finetuned_index_file.stat().st_size / 1024:.2f} KB")
    print("="*80)
    
    return index


def test_comparison(query, k=3):
    """Compare base vs fine-tuned retrieval"""
    
    models_dir = Path("models")
    base_index_file = models_dir / "mining_laws_base.index"
    finetuned_index_file = models_dir / "mining_laws_finetuned.index"
    metadata_file = models_dir / "chunks_metadata.json"
    finetuned_model_dir = models_dir / "bi_encoder_finetuned"
    
    print(f"\n{'='*80}")
    print("COMPARING BASE VS FINE-TUNED RETRIEVAL")
    print(f"{'='*80}\n")
    print(f"Query: '{query}'\n")
    
    # Load metadata
    with open(metadata_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Test base model
    print("BASE MODEL RESULTS:")
    print("-"*80)
    base_model = SentenceTransformer('all-mpnet-base-v2')
    base_index = faiss.read_index(str(base_index_file))
    
    query_emb = base_model.encode([query], convert_to_numpy=True)
    query_emb = l2_normalize(query_emb)
    scores, indices = base_index.search(query_emb.astype('float32'), k)
    
    for i, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        chunk = chunks[idx]
        print(f"{i}. Score: {score:.4f} | Section: {chunk['metadata']['section']}")
        print(f"   {chunk['text'][:100]}...")
    
    # Test fine-tuned model
    print(f"\n{'='*80}")
    print("FINE-TUNED MODEL RESULTS:")
    print("-"*80)
    ft_model = SentenceTransformer(str(finetuned_model_dir))
    ft_index = faiss.read_index(str(finetuned_index_file))
    
    query_emb = ft_model.encode([query], convert_to_numpy=True)
    query_emb = l2_normalize(query_emb)
    scores, indices = ft_index.search(query_emb.astype('float32'), k)
    
    for i, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        chunk = chunks[idx]
        print(f"{i}. Score: {score:.4f} | Section: {chunk['metadata']['section']}")
        print(f"   {chunk['text'][:100]}...")
    
    print("="*80)


if __name__ == "__main__":
    # Fine-tune model
    fine_tune_model(epochs=3, batch_size=16, learning_rate=2e-5)
    
    # Rebuild index
    rebuild_index_with_finetuned_model()
    
    # Test comparison
    test_queries = [
        "What is the definition of a mine?",
        "What are the safety requirements for explosives?",
    ]
    
    for query in test_queries:
        test_comparison(query, k=3)
