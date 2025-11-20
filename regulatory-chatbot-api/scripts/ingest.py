"""
Ingest regulatory-chatbot-api.txt and create overlapping chunks.

Output:
- data/chunks.pkl: List of text chunks
- data/chunks.jsonl: Chunks with metadata
"""

import pickle
import json
from pathlib import Path


def simple_chunk(text, chunk_size=220, overlap=40):
    """
    Chunk text into overlapping segments.
    
    Args:
        text: Input text string
        chunk_size: Number of words per chunk
        overlap: Number of overlapping words between chunks
    
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if len(chunk_words) >= 50:  # Min chunk size
            chunks.append(' '.join(chunk_words))
        
        # Stop if we've reached the end
        if i + chunk_size >= len(words):
            break
    
    return chunks


def main():
    print("Starting document ingestion...")
    
    # Paths
    data_dir = Path(__file__).parent.parent / 'data'
    input_file = data_dir / 'regulatory-chatbot-api.txt'
    output_pkl = data_dir / 'chunks.pkl'
    output_jsonl = data_dir / 'chunks.jsonl'
    
    # Check if source file exists
    if not input_file.exists():
        print(f"ERROR: Source file not found: {input_file}")
        print("Please copy Mining Laws.txt to data/regulatory-chatbot-api.txt first:")
        print(f"  cp ../regulatory-chatbot-api/data/Mining\\ Laws.txt {data_dir / 'regulatory-chatbot-api.txt'}")
        return
    
    # Read source document
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Document length: {len(text)} characters, {len(text.split())} words")
    
    # Create chunks
    print("Creating chunks (220 words, 40 overlap)...")
    chunks = simple_chunk(text, chunk_size=220, overlap=40)
    print(f"Created {len(chunks)} chunks")
    
    # Save as pickle
    print(f"Saving to {output_pkl}...")
    with open(output_pkl, 'wb') as f:
        pickle.dump(chunks, f)
    
    # Save as JSONL
    print(f"Saving to {output_jsonl}...")
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for idx, chunk in enumerate(chunks):
            obj = {
                'chunk_id': idx,
                'text': chunk,
                'word_count': len(chunk.split())
            }
            f.write(json.dumps(obj) + '\n')
    
    print("✓ Ingestion complete!")
    print(f"  - Chunks: {len(chunks)}")
    print(f"  - Output files: chunks.pkl, chunks.jsonl")


if __name__ == '__main__':
    main()
