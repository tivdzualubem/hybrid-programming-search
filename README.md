# Hybrid Programming Search Engine

A hybrid information retrieval system for programming questions that combines lexical retrieval, dense retrieval, reciprocal rank fusion, and cross-encoder reranking.

The project addresses the vocabulary gap in programming-question search. For example, a user may search with informal wording such as:

    website is white after running javascript

while the relevant Stack Overflow duplicate question may use more technical language such as an exception name, API issue, or implementation detail.

This repository contains two parts:

1. A public web application for interactive search.
2. A reproducible experimental pipeline for rebuilding the dataset artifacts, indexes, evaluation results, and qualitative analysis.

---

## Project Overview

The system compares four retrieval configurations:

1. BM25 — lexical retrieval using Pyserini/Lucene.
2. Dense Retrieval — semantic retrieval using all-MiniLM-L6-v2 and FAISS.
3. Hybrid RRF — BM25 and Dense results combined using Reciprocal Rank Fusion.
4. Hybrid + Rerank — Hybrid RRF candidates reranked with a cross-encoder.

The evaluation uses the MTEB StackOverflowDupQuestions benchmark.

Main evaluation metrics:

- Precision@10
- Recall@10
- MAP
- nDCG@10
- Wilcoxon signed-rank statistical tests

---

## Repository Structure

    hybrid-programming-search/
    ├── app/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── main.py
    │   ├── schemas.py
    │   ├── search_service.py
    │   └── utils.py
    │
    ├── scripts/
    │   ├── config.py
    │   ├── export_artifacts.py
    │   ├── build_indexes.py
    │   ├── retrieval_pipeline.py
    │   ├── evaluate.py
    │   ├── qualitative_analysis.py
    │   └── run_all_experiments.sh
    │
    ├── report/
    │   └── final_report.pdf
    │
    ├── Dockerfile
    ├── requirements.txt
    ├── README.md
    ├── .gitignore
    ├── .dockerignore
    └── .gitattributes

---

## Web Application

The web application exposes the retrieval system through a FastAPI interface.

The deployed version allows users to:

- Enter a programming search query.
- Select the retrieval system.
- Compare BM25, Dense Retrieval, Hybrid RRF, and Hybrid + Rerank.
- Return top-k ranked results.

Live demo:

    https://huggingface.co/spaces/lubem/hybrid-programming-search

---

## Reproducing the IR Experiments

This repository includes the full experimental pipeline needed to reproduce the results from scratch.

The pipeline does the following:

1. Loads the MTEB StackOverflowDupQuestions dataset.
2. Processes the corpus, queries, and qrels.
3. Builds a Pyserini BM25 index.
4. Builds a FAISS dense index.
5. Runs BM25, Dense, Hybrid RRF, and Hybrid + Rerank retrieval.
6. Computes Precision@10, Recall@10, MAP, and nDCG@10.
7. Runs Wilcoxon signed-rank statistical tests.
8. Saves qualitative retrieval examples.

---

## Setup

Install Python dependencies:

    pip install -r requirements.txt

Pyserini requires Java. On Ubuntu:

    sudo apt-get update
    sudo apt-get install -y openjdk-11-jdk
    export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

You may also need:

    export JVM_OPTS="--add-modules=jdk.incubator.vector"

---

## Run the Full Experiment Pipeline

From the repository root:

    bash scripts/run_all_experiments.sh

This runs:

    python scripts/export_artifacts.py
    python scripts/build_indexes.py
    python scripts/retrieval_pipeline.py
    python scripts/evaluate.py
    python scripts/qualitative_analysis.py

---

## Run Each Step Manually

### 1. Export Dataset Artifacts

    python scripts/export_artifacts.py

This loads and preprocesses the MTEB benchmark.

Generated files include:

    artifacts/documents.pkl
    artifacts/query_dict.pkl
    artifacts/qrels_dict.pkl
    artifacts/run_config.json
    artifacts/environment_info.json

---

### 2. Build BM25 and FAISS Indexes

    python scripts/build_indexes.py

This builds:

    artifacts/bm25_corpus/
    artifacts/bm25_index/
    artifacts/dense_index/stackoverflow_faiss.index
    artifacts/dense_index/doc_ids.pkl

---

### 3. Test the Retrieval Pipeline

    python scripts/retrieval_pipeline.py

This runs a small sanity check query through:

- BM25
- Dense Retrieval
- Hybrid RRF
- Hybrid + Rerank

---

### 4. Run Evaluation

    python scripts/evaluate.py

This generates:

    results/evaluation_results_final.csv
    results/significance_tests.json

The evaluation includes:

- Precision@10
- Recall@10
- MAP
- nDCG@10
- Wilcoxon signed-rank tests

---

### 5. Run Qualitative Analysis

    python scripts/qualitative_analysis.py

This generates:

    results/qualitative_analysis.txt

The qualitative analysis compares the four retrieval systems on example programming queries.

---

## Script Descriptions

| Script | Purpose |
|---|---|
| scripts/config.py | Stores experiment configuration, model names, candidate sizes, and paths. |
| scripts/export_artifacts.py | Loads the MTEB dataset, preprocesses corpus/query/qrels data, and saves reproducibility artifacts. |
| scripts/build_indexes.py | Builds the Pyserini BM25 index and FAISS dense index. |
| scripts/retrieval_pipeline.py | Implements BM25, dense retrieval, RRF hybrid fusion, and cross-encoder reranking. |
| scripts/evaluate.py | Computes Precision@10, Recall@10, MAP, nDCG@10, and Wilcoxon signed-rank tests. |
| scripts/qualitative_analysis.py | Generates qualitative examples comparing the retrieval systems. |
| scripts/run_all_experiments.sh | Runs the full experiment pipeline in the correct order. |

---

## Experiment Configuration

The main experiment settings are stored in:

    scripts/config.py

Important settings:

    Seed: 42
    Dataset: mteb/stackoverflowdupquestions
    Corpus split: test
    Query split: test
    Qrels split: test
    Bi-encoder: all-MiniLM-L6-v2
    Cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2
    BM25 candidate pool: 100
    Dense candidate pool: 100
    RRF constant: 60
    Rerank depth: 20
    Final cutoff: 10
    Common MAP ranking depth: 20

---

## Generated Files Not Committed

The following files and directories are generated locally and intentionally ignored by Git:

    artifacts/
    results/
    *.pkl
    *.index
    evaluation_results*.csv
    significance_tests.json
    qualitative_analysis.txt
    plot_*.png

These files are not committed because they can be regenerated from the scripts.

This keeps the repository lightweight while still making the experiments reproducible.

---

## Main Methodology

### BM25

BM25 is implemented with Pyserini/Lucene. The corpus is serialized into JSONL format and indexed using Lucene.

### Dense Retrieval

Dense retrieval uses all-MiniLM-L6-v2 to encode Stack Overflow questions into dense vectors. FAISS is used for nearest-neighbor search.

### Reciprocal Rank Fusion

BM25 and Dense ranked lists are combined using Reciprocal Rank Fusion:

    RRF(d) = sum(1 / (k + rank(d)))

where k = 60.

### Cross-Encoder Reranking

The top fused candidates are reranked using:

    cross-encoder/ms-marco-MiniLM-L-6-v2

The final best reranking setting uses top-20 fused candidates.

---

## Report

The final project report is included here:

    report/final_report.pdf

The report explains the dataset, methodology, results, statistical testing, query-type analysis, runtime analysis, and public demo.

---

## Notes on Reproducibility

The project was designed so another researcher can regenerate the experimental results from source scripts rather than relying only on final CSV or PKL outputs.

To reproduce the experiments, run:

    bash scripts/run_all_experiments.sh

The generated artifacts and results will appear in:

    artifacts/
    results/

---

## Authors

Tivdzua Lubem Noah  
Sikeh Gisele Wiykiynyuy  
Venera Bikbulatova
