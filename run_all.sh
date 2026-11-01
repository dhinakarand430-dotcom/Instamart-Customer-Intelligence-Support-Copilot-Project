#!/usr/bin/env bash
export PYTHONIOENCODING=utf-8

python generate_data.py
python src/data_pipeline.py
python src/customer_models.py
python src/triage.py
python src/policy_rag.py
python -m pytest
