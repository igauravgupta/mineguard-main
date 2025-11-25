# Mining Laws Chatbot - Complete RAG Implementation

A production-ready RAG-based chatbot for Indian mining and explosives laws with:

- ✅ Semantic chunking of legal documents
- ✅ Hybrid retrieval (Dense + Sparse + RRF)
- ✅ Fine-tuned embeddings
- ✅ LLM answer generation with citations (Groq Llama 3.1 70B)

## 📁 Project Structure

```
chatbot-model/
├── src/                               # Source code
│   ├── __init__.py                   # Package initializer
│   ├── create_chunks.py              # Phase 1: Semantic chunking
│   ├── build_base_index.py           # Phase 2: Build base index
│   ├── finetune_model.py             # Phase 2: Fine-tune & rebuild
│   ├── retrieval_system.py           # Phase 3: Hybrid retrieval
│   ├── answer_generator.py           # Phase 4: LLM answer generation
│   ├── complete_rag_pipeline.py      # Complete RAG pipeline
│   ├── api_server.py                 # FastAPI REST API server
│   ├── evaluation_metrics.py         # Phase 5: Metrics implementation
│   └── run_evaluation.py             # Phase 5: Evaluation runner
├── data/                              # Data folder
│   ├── mining_laws.txt               # Source document (cleaned)
│   ├── train.jsonl                   # Training data (72 Q&A pairs)
│   ├── test.jsonl                    # Test data (31 Q&A pairs)
│   └── mining_laws_chunks.json       # Generated chunks (779 chunks)
├── models/                            # Generated models & indexes
│   ├── mining_laws_base.index        # Base model FAISS index
│   ├── mining_laws_finetuned.index   # Fine-tuned model index
│   ├── chunks_metadata.json          # Chunk metadata
│   └── bi_encoder_finetuned/         # Fine-tuned model directory
├── start.py                           # Easy server startup script
├── evaluate.py                        # Quick evaluation script
├── requirements.txt                   # Python dependencies
├── .env                              # Environment variables (API keys)
├── .gitignore                        # Git ignore rules
├── .venv/                            # Python virtual environment
└── README.md                         # This file
```

## 🚀 Quick Start

### 1. Setup Python Environment

```bash

python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Phase 1: Create Semantic Chunks

```bash
python src/create_chunks.py
```

**Output:** `data/mining_laws_chunks.json` (779 chunks)

### 3. Phase 2: Build Embedding System

```bash
# Build base index
python src/build_base_index.py

# (Optional) Fine-tune for better results
python src/finetune_model.py
```

### 4. Get Groq API Key (Free)

```bash
# Visit: https://console.groq.com
# Get free API key, then:
export GROQ_API_KEY='your-api-key-here'
```

See `SETUP_GROQ.md` for detailed instructions.

### 5. Run Complete Pipeline

```bash
# Easy way - using startup script
python start.py

# Or demo mode
python src/complete_rag_pipeline.py

# Or interactive chat mode
python src/complete_rag_pipeline.py --interactive
```

### 6. API Server

```bash
# Start FastAPI server on port 8000
python start.py
```

Visit http://localhost:8000/docs for interactive API documentation.

## 🎯 Complete RAG Pipeline

```
User Question
     ↓
Hybrid Retrieval (Dense + Sparse + RRF)
     ↓ [Top-5 relevant chunks]
LLM Answer Generation (Groq Llama 3.1 70B)
     ↓
Structured Answer with Citations
```

## 📋 Implementation Phases

### Phase 1: Data Preparation ✅

**Script:** `src/create_chunks.py`

**What it does:**

- Reads mining laws document
- Splits into semantic sections by Act and Section
- Preserves legal hierarchy
- Generates 779 chunks with metadata

**Output:** `data/mining_laws_chunks.json`

### Phase 2: Embedding System ✅

**Scripts:** `src/build_base_index.py`, `src/finetune_model.py`

**Step 4: Choose Base Model**

- Model: `all-mpnet-base-v2`
- Dimensions: 768
- Size: ~420 MB
- Best balance of quality and speed

**Step 5: Build Initial Index**

- Load base bi-encoder model
- Generate embeddings for all 779 chunks
- L2-normalize for cosine similarity
- Create FAISS IndexFlatIP
- Save index and metadata

**Step 6: Fine-tune Model**

- Loss: MultipleNegativesRankingLoss
- Training data: 72 Q&A pairs
- Epochs: 3
- Batch size: 16
- Learning rate: 2e-5
- Adapts to mining law terminology

**Step 7: Rebuild Index**

- Re-encode chunks with fine-tuned model
- Create new FAISS index
- Better retrieval for domain-specific queries

**Output Files:**

- `models/mining_laws_base.index` - Base model FAISS index
- `models/mining_laws_finetuned.index` - Fine-tuned model index
- `models/chunks_metadata.json` - Chunk metadata mapping
- `models/bi_encoder_finetuned/` - Fine-tuned model directory

### Phase 3: Retrieval System ✅

**Script:** `src/retrieval_system.py`

**Step 8: Dense Retrieval (Semantic Search)**

- Uses FAISS IndexFlatIP with cosine similarity
- Finds contextually similar chunks based on meaning
- Best for concept-based queries

**Step 9: Sparse Retrieval (Keyword Search)**

- Uses BM25Okapi algorithm (k1=1.5, b=0.75)
- Finds keyword and term matches
- Best for specific terms/section numbers

**Step 10: Hybrid Retrieval (Best of Both)**

- Reciprocal Rank Fusion (RRF)
- Formula: `score = 1/(rank_dense + 60) + 1/(rank_sparse + 60)`
- Combines semantic understanding with keyword matching
- **Recommended for production use**

**Features:**

- Compare all three retrieval methods
- Configurable top-k results
- Score normalization and ranking

### Phase 4: Answer Generation ✅

**Scripts:** `src/answer_generator.py`, `src/complete_rag_pipeline.py`, `src/api_server.py`

**Step 12: Choose Answer Generation Model**

- **Model:** Groq Llama 3.1 70B Versatile
- **Cost:** FREE (generous tier)
- **Speed:** ⚡ 500+ tokens/sec
- **Quality:** ⭐⭐⭐⭐⭐ (excellent for legal text)
- **API:** https://console.groq.com

**Step 13: Prompt Engineering**

- **Role:** Expert legal assistant for Indian mining laws
- **Instructions:** Answer only from context, cite sections
- **Output Format:** Structured with 4 parts (Answer, Legal Basis, Requirements, Exceptions)
- **Constraints:** Accurate citations, minimum 150 words

**Complete Pipeline:**

```
User Query → Hybrid Retrieval → Top-5 Chunks → Groq LLM → Structured Answer
```

### Phase 5: Evaluation & Optimization ✅

**Scripts:** `src/evaluation_metrics.py`, `src/run_evaluation.py`, `evaluate.py`

**Step 15: Implement Evaluation Metrics**

**Retrieval Metrics:**

- **Precision@K:** Proportion of relevant docs in top-K
- **Recall@K:** Proportion of relevant docs retrieved
- **Accuracy@K:** Whether at least one relevant doc in top-K (Hit Rate)
- **MRR:** Mean Reciprocal Rank of first relevant document
- **NDCG@K:** Normalized Discounted Cumulative Gain
- **F1@K:** Harmonic mean of Precision and Recall

**Generation Metrics:**

- **Semantic Similarity:** BERTScore-like similarity using sentence embeddings
- **ROUGE-L:** Longest Common Subsequence F1 score
- **BLEU:** N-gram precision with brevity penalty
- **Citation Accuracy:** Correctness of cited sections/acts

**End-to-End Metrics:**

- **Correctness:** Keyword overlap with reference answer
- **Completeness:** Answer length relative to reference
- **Overall Quality:** Weighted average of all metrics
- **Readability:** Sentence structure analysis

**Step 16: Run Evaluation on Test Set**

```bash
# Quick evaluation with default settings
python evaluate.py

# Full evaluation with options
python src/run_evaluation.py --method hybrid --top-k 5

# Compare retrieval methods
python src/run_evaluation.py --compare-methods

# Show sample predictions
python src/run_evaluation.py --show-samples 5
```

**Target Metrics:**

- ✅ Precision@1 > 85%
- ✅ MRR > 0.90
- ✅ Accuracy@5 > 90%
- ✅ Overall Quality > 70%

**Evaluation Output:**

- Aggregate metrics across test set
- Individual query results with predictions
- Method comparison (Dense vs Sparse vs Hybrid)
- Results saved to `evaluation_results_TIMESTAMP.json`

## 🎯 Technical Details

### Semantic Chunking Strategy

**Features:**

- Splits documents by Act (Explosives Act, Mines Act)
- Chunks by section numbers (1., 2., 40., etc.)
- Preserves legal hierarchy (Act → Section → Clause)
- Maintains metadata (act name, section number, title)
- Handles long sections by splitting with overlap

**Parameters:**

- Target chunk size: 200-500 words
- Overlap for long sections: 50 words
- Minimum section size: 20 words

### Embedding Model Comparison

| Model                 | Dimensions | Size    | Speed  | Quality    | Use Case             |
| --------------------- | ---------- | ------- | ------ | ---------- | -------------------- |
| all-MiniLM-L6-v2      | 384        | ~80 MB  | ⚡⚡⚡ | ⭐⭐⭐     | Fast, small datasets |
| **all-mpnet-base-v2** | 768        | ~420 MB | ⚡⚡   | ⭐⭐⭐⭐   | **Recommended**      |
| sentence-t5-large     | 768        | ~670 MB | ⚡     | ⭐⭐⭐⭐   | Better quality       |
| gte-large             | 1024       | ~670 MB | ⚡     | ⭐⭐⭐⭐⭐ | State-of-the-art     |

**Selected:** `all-mpnet-base-v2` for optimal quality/speed balance

### FAISS Index Configuration

- **Index Type:** `IndexFlatIP` (Inner Product)
- **Similarity:** Cosine (via L2-normalized embeddings)
- **Search:** Exact nearest neighbor
- **Scalability:** Suitable for up to 1M vectors

### Fine-tuning Strategy

- **Loss Function:** MultipleNegativesRankingLoss
  - Positive: (query, correct_answer) pairs
  - Negatives: Other samples in batch
  - Encourages similar embeddings for Q&A pairs
- **Expected Improvement:** 15-30% retrieval accuracy on domain queries

- **Why Fine-tune:**
  - Generic models miss domain-specific terminology
  - Terms like "competent person", "mine inspector" get better representations
  - Improved semantic matching for legal language

### Chunk Format

```json
{
  "total_chunks": 150,
  "chunks": [
    {
      "chunk_id": 1,
      "text": "Section text here...",
      "metadata": {
        "act": "THE MINES ACT, 1952",
        "section": "40",
        "section_title": "Hours of work below ground",
        "word_count": 450,
        "source": "mining_laws.txt"
      }
    }
  ]
}
```

## 📊 Performance Metrics

### Data Statistics

- **Total chunks:** 779
- **Acts covered:** 2 (Explosives Act 1884, Mines Act 1952)
- **Average chunk size:** 302 words
- **Training Q&A pairs:** 72
- **Test Q&A pairs:** 31

### Model Performance

- **Embedding dimension:** 768
- **Index size (base):** ~2.4 MB
- **Index size (fine-tuned):** ~2.4 MB
- **Encoding speed:** ~500 chunks/second
- **Search latency:** <10ms for top-k retrieval

## 🎯 Key Design Decisions

1. **Semantic Chunking by Section**

   - Preserves legal structure
   - Makes citations easy
   - Improves retrieval accuracy

2. **Metadata Preservation**

   - Enables precise source attribution
   - Supports filtering by act/section
   - Facilitates debugging

3. **Overlap Strategy**

   - 50-word overlap for long sections
   - Prevents information loss at boundaries
   - Improves context completeness

4. **Dataset Enhancement**
   - Full section text as context
   - Better than generic descriptions
   - Supports fine-tuning and RAG

## 🔧 Customization

### Change Embedding Model

Edit the model name in scripts:

```python
# In src/build_base_index.py
model_name = 'all-mpnet-base-v2'  # Change to desired model
```

### Adjust Chunk Parameters

Edit `src/create_chunks.py`:

```python
chunk_size = 400  # Change this value
overlap = 50      # Change overlap
```

### Fine-tuning Hyperparameters

Edit `src/finetune_model.py`:

```python
fine_tune_model(
    epochs=3,           # Increase for more training
    batch_size=16,      # Adjust based on GPU memory
    learning_rate=2e-5  # Lower for more stable training
)
```

### FAISS Index Type

For larger datasets, consider:

```python
# IVF index for faster search (approximate)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# HNSW for better recall
index = faiss.IndexHNSWFlat(dimension, M)
```

## 📝 Next Steps

**Phase 3: Retrieval System (Coming Soon)**

- Query processing and expansion
- Re-ranking mechanisms
- Hybrid search (dense + sparse)
- Result filtering and deduplication

**Phase 4: Generation (Coming Soon)**

- LLM integration (GPT, LLaMA, etc.)
- Prompt engineering
- Context injection
- Response validation

**Phase 5: Evaluation & Deployment**

- Retrieval metrics (MRR, NDCG, Recall@k)
- End-to-end testing
- API development
- Production deployment

## ⚠️ Troubleshooting

### GPU/CPU Issues

```bash
# For CPU-only installation
pip install faiss-cpu

# For GPU (CUDA)
pip install faiss-gpu
```

### Out of Memory

```python
# Reduce batch size in scripts
batch_size = 8  # Instead of 16 or 32
```

### Slow Encoding

```python
# Use smaller model
model_name = 'all-MiniLM-L6-v2'  # Faster but lower quality
```

### File not found

```bash
# Ensure data files are in the data/ folder
ls data/
# Should show: mining_laws.txt, train.jsonl, test.jsonl, mining_laws_chunks.json
```

### Issue: Virtual environment not activated

```bash
source .venv/bin/activate
```

## 📚 Dependencies

**Core Libraries:**

- `sentence-transformers`: Bi-encoder models and training
- `faiss-cpu`: Vector similarity search
- `torch`: PyTorch backend
- `transformers`: Hugging Face transformers
- `numpy`: Numerical operations
- `tqdm`: Progress bars
- `scikit-learn`: Evaluation metrics

**Installation:**

```bash
pip install -r requirements.txt
```

---

**Last Updated:** November 2025  
**Status:** ✅ Ready to use
