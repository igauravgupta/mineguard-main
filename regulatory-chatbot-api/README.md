# Semantic Information Retrieval for Mining Laws

Complete semantic search and QA system for regulatory mining documents with fine-tuned bi-encoder, evaluation metrics, and answer generation.

## Features

- **Document Ingestion**: Chunks mining laws into overlapping segments
- **Synthetic QA Generation**: 100 question-answer pairs (70 train / 30 test)
- **FAISS Indexing**: Fast similarity search with normalized embeddings
- **Bi-Encoder Fine-tuning**: Domain-specific training with MultipleNegativesRankingLoss
- **Evaluation Metrics**: Precision@1 and Mean Reciprocal Rank (MRR)
- **Answer Generation**: RAG-style responses using flan-t5-small
- **FastAPI Server**: Top-1 retrieval with optional answer generation

## Project Structure

```
semantic-ir-project/
├── data/
│   ├── regulatory-chatbot-api.txt  # Source legal text
│   ├── train.jsonl                  # 70 Q&A pairs for training
│   ├── test.jsonl                   # 30 Q&A pairs for testing
│   ├── chunks.pkl                   # Processed text chunks
│   └── chunks.jsonl                 # Chunks in JSON format
├── models/
│   ├── mining_laws.index            # FAISS index
│   ├── index_meta.json              # Index metadata
│   └── bi_encoder_finetuned/        # Fine-tuned model (optional)
├── scripts/
│   ├── ingest.py                    # Chunk documents
│   ├── build_index.py               # Build FAISS index
│   ├── train_bi_encoder.py          # Fine-tune bi-encoder
│   ├── eval.py                      # Evaluate on test set
│   └── answer_generator.py          # Generate answers
├── serve.py                         # FastAPI server
├── requirements.txt
└── README.md
```

## Setup

1. **Install Dependencies**:

```bash
pip install -r requirements.txt
```

2. **Copy Source Document**:

```bash
cp ../regulatory-chatbot-api/data/Mining\ Laws.txt data/regulatory-chatbot-api.txt
```

## Usage

### 1. Ingest Documents

```bash
python scripts/ingest.py
```

Creates `data/chunks.pkl` and `data/chunks.jsonl` with 220-word chunks and 40-word overlap.

### 2. Build FAISS Index

```bash
python scripts/build_index.py
```

Embeds chunks using `all-MiniLM-L6-v2` and creates `models/mining_laws.index`.

### 3. (Optional) Fine-tune Bi-Encoder

```bash
python scripts/train_bi_encoder.py
```

Fine-tunes on `data/train.jsonl` with MultipleNegativesRankingLoss. Saves to `models/bi_encoder_finetuned/`.

### 4. Evaluate

```bash
python scripts/eval.py
```

Computes **Precision@1** and **MRR** on `data/test.jsonl`.

### 5. Serve API

```bash
uvicorn serve:app --reload --port 8000
```

**Endpoints**:

- `GET /` - Health check
- `POST /search` - Top-1 retrieval
  ```json
  {
    "query": "What is the penalty for illegal mining?",
    "top_k": 1
  }
  ```
- `POST /chat` - Retrieval + Answer generation
  ```json
  {
    "query": "What is a mining lease?",
    "generate_answer": true
  }
  ```

## Dataset Details

- **Total Q&A Pairs**: 100
- **Training Set**: 70 pairs (`data/train.jsonl`)
- **Test Set**: 30 pairs (`data/test.jsonl`)
- **Coverage**: Explosives Act 1884, Mines Act 1952, Coal Bearing Areas Act 1957, Mines and Minerals Act 1957

## Model Details

- **Base Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Loss Function**: MultipleNegativesRankingLoss
- **Training Epochs**: 3
- **Batch Size**: 16
- **Evaluation**: Cosine similarity with L2 normalization

## Performance

After fine-tuning and evaluation, expect:

- **Precision@1**: ~0.7-0.9 (70-90% correct top-1 answers)
- **MRR**: ~0.75-0.95 (high reciprocal rank)

## API Examples

**Search**:

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is dead rent?", "top_k": 1}'
```

**Chat**:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain mining lease procedure", "generate_answer": true}'
```

## Notes

- Chunks are created with 220-word windows and 40-word overlap
- All vectors are L2-normalized for cosine similarity via inner product
- FAISS uses IndexFlatIP for exact similarity search
- Answer generation uses `google/flan-t5-small` with retrieved context

## License

MIT License - Educational and research purposes
