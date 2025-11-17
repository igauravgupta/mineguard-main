# Complete RAG Implementation — From Scratch (Theory)

A concise, step-by-step guide to build a Retrieval-Augmented Generation (RAG) system for mining laws. This README covers Phase 1: Data Preparation and provides practical recommendations for document processing, chunking, and dataset creation.

## Table of Contents

- [Overview](#overview)
- [Phase 1: Data Preparation](#phase-1-data-preparation)
  - [Step 1 — Document Collection](#step-1---document-collection)
  - [Step 2 — Intelligent Chunking](#step-2---intelligent-chunking)
  - [Step 3 — Training & Test Datasets](#step-3---training--test-datasets)
- [Example Chunk (JSON)](#example-chunk-json)

- [Phase 2: Embedding System](#phase-2-embedding-system)
  - [Step 4 — Choose Base Bi-Encoder Model](#step-4---choose-base-bi-encoder-model)
  - [Step 5 — Build Initial Vector Index (Before Fine-tuning)](#step-5---build-initial-vector-index-before-fine-tuning)
  - [Step 6 — Fine-Tune Bi-Encoder on Domain Data](#step-6---fine-tune-bi-encoder-on-domain-data)
  - [Step 7 — Rebuild Index with Fine-Tuned Model](#step-7---rebuild-index-with-fine-tuned-model)
- [Phase 3: Retrieval System](#phase-3-retrieval-system)
  - [Step 8 — Implement Dense Retrieval (FAISS)](#step-8---implement-dense-retrieval-faiss)
  - [Step 9 — Implement Sparse Retrieval (BM25)](#step-9---implement-sparse-retrieval-bm25)
  - [Step 10 — Implement Hybrid Retrieval (FAISS + BM25)](#step-10---implement-hybrid-retrieval-faiss--bm25)
  - [Step 11 — Implement Query Expansion (HyDE)](#step-11---implement-query-expansion-hyde---optional)
- [Phase 4: Answer Generation](#phase-4-answer-generation)
  - [Step 12 — Choose Answer Generation Model](#step-12---choose-answer-generation-model)
  - [Step 13 — Design Prompt Engineering](#step-13---design-prompt-engineering)
  - [Step 14 — Implement Answer Generation Pipeline](#step-14---implement-answer-generation-pipeline)
- [Phase 5: Evaluation & Optimization](#phase-5-evaluation--optimization)
  - [Step 15 — Implement Evaluation Metrics](#step-15---implement-evaluation-metrics)
  - [Step 16 — Run Evaluation on Test Set](#step-16---run-evaluation-on-test-set)
  - [Step 17 — Iterative Improvement](#step-17---iterative-improvement)

---

## Overview

This document focuses on preparing legal documents (mining laws) so they can be ingested into a vector store and used in a RAG pipeline. Emphasis is on preserving legal structure and creating reliable training/evaluation sets.

## Phase 1: Data Preparation

### Step 1 — Document Collection

**Goal:** Gather all relevant mining law documents and convert them to clean, plain text.

What to do

- Collect source files: PDFs, Word docs, text files, etc.
- Convert to plain text (OCR where necessary).
- Clean the text:
  - Remove headers/footers and page numbers
  - Fix encoding issues
  - Normalize whitespace
  - Remove OCR artifacts
- Output: one or more cleaned text files (e.g., `mining_laws.txt`)

### Step 2 — Intelligent Chunking

**Goal:** Split documents into meaningful, retrievable pieces while preserving legal structure.

Strategies

- Option A — Semantic Chunking (recommended for legal docs)
  - Chunk by section/article
  - Preserve hierarchy: Act → Section → Clause
  - Keep section titles and numbers in metadata
- Option B — Fixed-size with overlap
  - 300–500 words per chunk
  - 50–100 words overlap
  - Simple but less ideal for complex legal structure
- Option C — Hybrid
  - Primary: by section
  - Secondary: split long sections into fixed-size chunks

Key decisions

- Chunk size: 200–500 words (balance context vs. precision)
- Overlap: 10–20% of chunk size
- Metadata: store act name, section number, dates, source filename
- Output: list of chunks with metadata ready for embedding and indexing

### Step 3 — Create Training & Test Datasets

**Goal:** Prepare Q&A pairs and evaluation data for fine-tuning and testing.

What to create

- Q&A pairs (recommended 200–500 pairs minimum)
  - Example pair format:
    {"query": "What are wage regulations for miners?", "answer": "Section 54 states...", "context": "Section 54 text"}
- Approaches to generate pairs:
  - Manual labeling (most accurate)
  - Semi-automated: use a large language model to propose questions from sections, then human review
  - Extract real user queries from logs (if available) and label answers
- Dataset split:
  - Training: ~70% (e.g., 200 pairs)
  - Test: ~30% (e.g., 100 pairs)

Best practices

- Ensure each Q&A references the exact section(s) that justify the answer
- Keep evaluation set strictly unseen during training
- Include edge cases and ambiguous queries for robust evaluation

## Example Chunk (JSON)

```json
{
  "chunk_id": 1,
  "text": "Section 40. Any person employed...",
  "metadata": {
    "act": "Mines Act 1952",
    "section": "40",
    "word_count": 450,
    "source": "Mines_Act_1952.txt"
  }
}
```

## Phase 2: Embedding System

### Step 4 — Choose Base Bi-Encoder Model

Goal: Select a pre-trained model for embeddings.

Models

| Model             | Dimensions |    Size | Use Case                      |
| ----------------- | ---------: | ------: | ----------------------------- |
| all-MiniLM-L6-v2  |        384 |  ~80 MB | Fast, good for small datasets |
| all-mpnet-base-v2 |        768 | ~420 MB | Best balance (Recommended)    |
| sentence-t5-large |        768 | ~670 MB | Better quality, slower        |
| gte-large         |       1024 | ~670 MB | State-of-the-art quality      |

Recommendation: all-mpnet-base-v2 for quality/speed trade-off.

---

### Step 5 — Build Initial Vector Index (Before Fine-tuning)

Goal: Create a searchable index of all chunks.

Process:

1. Load base bi-encoder model.
2. For each chunk:
   - Generate embedding (e.g., 768-dim).
   - L2-normalize embedding (cosine similarity via inner product).
3. Store embeddings in a vector DB (FAISS).
4. Save metadata mapping (chunk_id → text + metadata).

Pseudo-code (Python-style):

```python
model = load_bi_encoder('all-mpnet-base-v2')
chunks = load_chunks()  # list of dicts with 'text' and metadata

embeddings = []
for chunk in chunks:
        emb = model.encode(chunk['text'])
        emb = l2_normalize(emb)
        embeddings.append(emb)

faiss_index = create_faiss_index(embeddings)  # e.g., IndexFlatIP
save_index(faiss_index, 'mining_laws.index')
save_json(chunks, 'chunks_metadata.json')
```

Output files:

- mining_laws.index — FAISS vector database
- chunks_metadata.json — chunk texts and metadata

---

### Step 6 — Fine-Tune Bi-Encoder on Domain Data

Goal: Adapt the model to mining law terminology to improve semantic matching.

Why fine-tune:

- Generic embeddings miss domain-specific senses (e.g., "competent person").
- Expect 15–30% retrieval accuracy improvement on domain queries.

Strategy:

- Loss: MultipleNegativesRankingLoss
- Positive example: (query, correct_chunk)
- Negatives: other chunks within the batch

Hyperparameters (example):

- epochs = 2–5 (more if >500 cases)
- batch_size = 16–32
- learning_rate = 2e-5
- warmup_steps = 0.1 \* total_steps

Data format:

```python
train_data = [
        InputExample(texts=['What are wage regulations for miners?', 'Section 54 text ...']),
        InputExample(texts=['Who is a competent person?', 'Definition clause ...']),
        ...
]
```

Training flow:

1. Load base model.
2. Create DataLoader from Q&A pairs.
3. Use MultipleNegativesRankingLoss.
4. Train 2–5 epochs.
5. Save model to bi_encoder_finetuned/.

Output: bi_encoder_finetuned/ directory.

---

### Step 7 — Rebuild Index with Fine-Tuned Model

Goal: Re-encode all chunks using the fine-tuned model and replace the old index.

Process:

1. Load fine-tuned bi-encoder.
2. Recompute embeddings for all chunks.
3. Create a new FAISS index and save.
4. Replace mining_laws.index with the new index.

Reason: Index must be consistent with the encoder used for query vectors.

---

## Phase 3: Retrieval System

### Step 8 — Implement Dense Retrieval (FAISS)

Goal: Fast semantic search using vector similarity.

Flow:
Query → Bi-Encoder → Query Vector → FAISS search → Top-K chunk vectors → Retrieve chunk texts

FAISS index types:

- IndexFlatIP — Exact inner-product (use for <100k chunks)
- IndexHNSWFlat — Approximate (good for >100k chunks)

Similarity metric: cosine similarity (use normalized vectors and inner product).

---

### Step 9 — Implement Sparse Retrieval (BM25)

Goal: Keyword-based search to complement semantic search.

Why BM25:

- Catches exact term matches (e.g., "Section 40")
- Complements semantic retrieval for different strengths

How BM25 works (brief):

1. Tokenize chunks and build inverted index.
2. For each query term, find documents containing it.
3. Score by term frequency, inverse document frequency, and length normalization.

Common parameters:

- k1 = 1.5
- b = 0.75

---

### Step 10 — Implement Hybrid Retrieval (FAISS + BM25)

Goal: Combine semantic and lexical search to improve coverage.

Fusion strategies:

Option A — Score Fusion:
final_score = alpha _ dense_score + (1 - alpha) _ bm25_score
(alpha ≈ 0.7 typical)

Option B — Reciprocal Rank Fusion:
score = 1 / (rank_dense + k) + 1 / (rank_bm25 + k)
(k ≈ 60 typical)

Process:

1. Run FAISS → top-50 dense results (with scores/ranks).
2. Run BM25 → top-50 lexical results.
3. Combine and deduplicate results.
4. Re-rank based on chosen fusion strategy.
5. Return top-K results.

---

### Step 11 — Implement Query Expansion (HyDE) — Optional

Goal: Improve retrieval for vague or colloquial queries.

HyDE workflow:

1. Use an LLM to generate a hypothetical detailed answer for the user query.
2. Embed both the original query and the hypothetical answer.
3. Average or combine embeddings and use for retrieval.

Example:
User query: "How much do miners get paid?"
LLM hypothetical answer: "According to Section 54, miners shall be paid wages based on daily averages..."
Embed query and hypothetical answer → improved retrieval with legal terminology.

Trade-offs: extra LLM call and latency but 15–30% retrieval improvement in many domains.

---

## Phase 4: Answer Generation

### Step 12 — Choose Answer Generation Model

Goal: Select an LLM to generate natural, cited answers.

Model options (summary):

| Model              | Type  |    Quality |    Speed | Cost             |
| ------------------ | ----- | ---------: | -------: | ---------------- |
| Groq Llama 3.1 70B | API   | ⭐⭐⭐⭐⭐ |     Fast | Free (as listed) |
| OpenAI GPT-4       | API   | ⭐⭐⭐⭐⭐ |     Fast | Paid             |
| OpenAI GPT-3.5     | API   |   ⭐⭐⭐⭐ |     Fast | Low cost         |
| FLAN-T5-XL (3B)    | Local |     ⭐⭐⭐ | Moderate | Free (local)     |
| Mistral-7B         | Local |   ⭐⭐⭐⭐ | Moderate | Free (local)     |

Recommendation:

- Production: Groq (if available and suitable)
- Development/local: FLAN-T5 variants or Mistral

---

### Step 13 — Design Prompt Engineering

Goal: Craft prompts to produce accurate, cited answers.

Prompt components:

1. Role: "You are an expert legal assistant specializing in Indian mining regulations."
2. Context injection: include retrieved chunk text with section number.
3. Instructions:
   - Answer the user query based only on the provided context.
   - Cite specific section numbers.
   - Include relevant details and exceptions.
   - Minimum 150 words (optional).
4. Output format (structured):
   1. Direct Answer
   2. Legal Basis (sections/acts)
   3. Key Requirements
   4. Exceptions (if any)

---

### Step 14 — Implement Answer Generation Pipeline

Goal: Orchestrate retrieval → generation → post-processing.

Pipeline:

1. Receive user query.
2. Optionally run HyDE for query expansion.
3. Run hybrid retrieval (FAISS + BM25).
4. Select best chunk(s) (e.g., top-1 or top-k for augmentation).
5. Construct prompt:
   - System role
   - Retrieved context (best chunk(s))
   - User query and instructions
6. Send to LLM and receive answer.
7. Post-process:
   - Clean formatting
   - Add citations
   - Validate length
8. Return structured response, e.g.:

```json
{
    "answer": "...",
    "source": "Section 40, Mines Act 1952",
    "chunk_used": { "chunk_id": 1, "metadata": {...} }
}
```

Notes:

- Consider few-shot examples in the prompt for improved format consistency.
- If using multiple chunks, instruct the model to explicitly cite which chunks/sections support each statement.

---

## Phase 5: Evaluation & Optimization

### Step 15 — Implement Evaluation Metrics

Goal: Measure system performance objectively.

Retrieval metrics:

- Precision@K
- Recall@K
- MRR (Mean Reciprocal Rank)
- NDCG@K

Generation metrics:

- BLEU, ROUGE (n-gram overlaps)
- BERTScore (semantic)
- Human evaluation (accuracy, completeness, clarity)

End-to-end metrics:

- Correctness, Completeness, Citation accuracy, Readability

---

### Step 16 — Run Evaluation on Test Set

Goal: Measure baseline performance and validate changes.

Example evaluation loop (pseudo-code):

```python
for query, ground_truth in test_set:
        retrieved = retrieve(query, top_k=10)
        precision = precision_at_k(retrieved, ground_truth, k=10)

        generated = generate_answer(query, retrieved[0])
        quality_score = evaluate_generated(generated, ground_truth_answer)

        log_metrics(query, precision, quality_score)
```

Target metrics (example goals):

- Precision@1 > 85%
- MRR > 0.90
- Human-rated answer quality > 4/5

---

### Step 17 — Iterative Improvement

Goal: Continuously improve system performance.

Actions:

- Expand training data (more Q&A pairs).
- Improve chunking strategy and metadata quality.
- Fine-tune bi-encoder with more domain pairs.
- Tune hybrid fusion weights and BM25 parameters.
- Add reranking or cross-encoder for final re-rank.
- Monitor production logs and add real user queries to dataset (with labeling).

---

End of additional README content for PHASE 2 — PHASE 5.
