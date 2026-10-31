#!/usr/bin/env bash

# Master orchestration script for the Swiggy Instamart Support Copilot
# Executes Parts A → B → C → D sequentially and ensures all expected artefacts are generated.

set -euo pipefail

echo "--- Part A: Data Foundation ---"
python generate_data.py
python data_pipeline.py

echo "--- Part B: Customer Segments & Lapse Classification ---"
python rfm_clustering.py

echo "--- Part C: Support Ticket Triage ---"
python triage_pipeline.py

echo "--- Part D: Policy RAG & Refund ReAct Agent ---"
python policy_rag_refund_agent.py

echo "All parts completed. Expected outputs:"
echo "  data/instamart.db"
echo "  outputs/customer_segments.csv"
echo "  outputs/lapse_model.joblib"
echo "  outputs/retention_list.csv"
echo "  outputs/triage_budget_demo.csv"
echo "  outputs/triage_results.csv"
echo "  outputs/agent_runs.jsonl"
echo "  outputs/refund_ledger.jsonl"

# End of script
