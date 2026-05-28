#!/usr/bin/env bash
set -e

python scripts/export_artifacts.py
python scripts/build_indexes.py
python scripts/retrieval_pipeline.py
python scripts/evaluate.py
python scripts/qualitative_analysis.py
