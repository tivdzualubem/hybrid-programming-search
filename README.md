---
title: Hybrid Programming Search Engine
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Hybrid Programming Search Engine

A hybrid information retrieval system for programming questions using:

- BM25 lexical retrieval
- Dense semantic retrieval
- Reciprocal Rank Fusion (RRF)
- Cross-Encoder reranking

The project is built on the **MTEB StackOverflowDupQuestions** benchmark and packaged as a working web application.

## Project Links

- **GitHub Repository:** https://github.com/tivdzualubem/hybrid-programming-search
- **Live Demo:** https://huggingface.co/spaces/lubem/hybrid-programming-search

## Project Goal

Programming search suffers from a **vocabulary gap**.

A user may describe a problem informally, such as:

- `website is white after running javascript`

while the relevant technical discussion may use very different terms, such as:

- `Uncaught ReferenceError`
- `JavaScript not working on website`

To address this, this project combines lexical and semantic retrieval in one pipeline.

## Final Retrieval Systems

The project compares four systems:

1. **BM25**
2. **Dense Retrieval**
3. **Hybrid (RRF)**
4. **Hybrid + Rerank (top-20)**

## Final Results

| System | Queries Evaluated | Precision@10 | Recall@10 | MAP | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| BM25 | 2992 | 0.663068 | 0.221738 | 0.309820 | 0.692774 |
| Dense | 2992 | 0.665441 | 0.222582 | 0.306795 | 0.697388 |
| Hybrid (RRF) | 2992 | **0.721491** | **0.241327** | **0.350161** | **0.747357** |
| Hybrid + Rerank (top-20, BEST) | 2992 | 0.704813 | 0.235739 | 0.343802 | 0.729364 |

## Main Finding

**Hybrid (RRF)** is the best overall system.

It consistently outperforms BM25, Dense Retrieval, and the reranked system on the final evaluation table.

## Query Examples

The UI can be tested with queries such as:

- `python list index error`
- `website is white after running javascript`
- `java null pointer exception in arraylist`

## Tech Stack

### Backend
- FastAPI
- Uvicorn

### Retrieval
- Pyserini (BM25 / Lucene)
- FAISS
- Sentence-Transformers
- Cross-Encoder reranking

### Models
- `all-MiniLM-L6-v2`
- `cross-encoder/ms-marco-MiniLM-L-6-v2`

## Project Structure

    app/
      main.py
      config.py
      schemas.py
      search_service.py
      retrievers/
        bm25.py
        dense.py
        fusion.py
        rerank.py
      data/
        bm25_index/
        dense_index/
        documents.pkl
        query_dict.pkl
        qrels_dict.pkl
        doc_ids.pkl
        evaluation_results.csv
        evaluation_results_final.csv
        significance_tests.json
        qualitative_analysis.txt
        per_query_winner_ndcg10.csv
      static/
        index.html
        style.css
        script.js
        assets/
    Dockerfile
    requirements.txt
    README.md

## Run Locally

Create and activate a virtual environment, then install dependencies.

    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2+cpu torchvision==0.17.2+cpu torchaudio==2.2.2+cpu
    pip install -r requirements.txt

Then run:

    export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
    uvicorn app.main:app --reload

Open:

    http://127.0.0.1:8000

## Public Demo

This repository is prepared for deployment as a **Hugging Face Docker Space** so that the application can be accessed publicly by instructors, teammates, and reviewers.

## Notes

- The dataset was evaluated on the full test split because no reliable language metadata field was available for strict Python / Java / JavaScript filtering.
- The best reranking depth was **top-20**.
- Larger rerank depths degraded performance, likely due to domain mismatch between MS MARCO training data and Stack Overflow technical content.
