"""
Fine-tune bi-encoder on train.jsonl using MultipleNegativesRankingLoss.

Output:
- models/bi_encoder_finetuned/: Fine-tuned SentenceTransformer model
"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


def main():
    print("Fine-tuning bi-encoder...")
    
    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    models_dir = project_root / 'models'
    
    train_file = data_dir / 'train.jsonl'
    output_dir = models_dir / 'bi_encoder_finetuned'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if train file exists
    if not train_file.exists():
        print(f"ERROR: {train_file} not found.")
        return
    
    # Load training data
    print(f"Loading training data from {train_file}...")
    train_examples = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            # Create (query, positive_answer) pairs
            train_examples.append(
                InputExample(texts=[obj['query'], obj['answer']])
            )
    print(f"Loaded {len(train_examples)} training examples")
    
    # Load base model
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    print(f"Loading base model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    # Create DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    
    # Define loss function
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Train
    print("Starting fine-tuning...")
    print(f"  - Epochs: 3")
    print(f"  - Batch size: 16")
    print(f"  - Loss: MultipleNegativesRankingLoss")
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=100,
        output_path=str(output_dir),
        show_progress_bar=True
    )
    
    print(f"✓ Fine-tuning complete!")
    print(f"  - Model saved to: {output_dir}")
    print(f"  - Training examples: {len(train_examples)}")


if __name__ == '__main__':
    main()
