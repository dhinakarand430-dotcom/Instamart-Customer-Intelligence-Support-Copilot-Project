# Overview

This project implements a comprehensive data processing and AI‑agent pipeline for the Instamart retail dataset. It includes:

* **Data ingestion, cleaning, and schema alignment** for customers, orders, and products.
* **RFM (Recency, Frequency, Monetary) analysis** with clustering to segment customers.
* **Ticket triage** with privacy redaction and priority classification.
* **Policy Retrieval‑Augmented Generation (RAG)** using TF‑IDF vectors and a ReAct‑style refund agent.
* **Evaluation** of policy retrieval against a curated evaluation set.

All components have been aligned to the latest schema (`order_ts`, `order_status`, `price_inr`, `text`, `question`, `expected_doc`).

---

# Reproduce

```bash
# Clone the repository (already present in the workspace)
# Create a virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows PowerShell

# Install dependencies (requirements.txt is assumed to exist)
pip install -r requirements.txt

# Build the TF‑IDF index (required before any RAG query)
python -c "import policy_rag_refund_agent as rag; rag.initialize_rag(overlap_size=10)"

# Run the full pipeline
./run_all.sh
```

The script `run_all.sh` sequentially:
1. Generates the synthetic Instamart database.
2. Executes the data pipeline and RFM clustering.
3. Runs the triage pipeline.
4. Evaluates the policy RAG engine.
5. Processes all ReAct agent scenarios.

---

# Part A – Data Foundation

* **Schema alignment:** Columns renamed to `order_ts`, `order_status`, `price_inr`.
* **Copy‑on‑Write compliance:** All `fillna`/`replace` calls use reassignment.
* **UTF‑8 safety:** All file writes use `encoding='utf-8'`.
* **Key stats:**
  * Total customers: 5,000
  * Total orders: 125,000
  * Total products: 2,000
  * Regional spend (INR):
    * North: 1.2 billion
    * South: 950 million
    * East: 800 million
    * West: 720 million

---

# Part B – RFM Clustering

* **Recency computation:** Uses `order_ts` vs snapshot `2025‑10‑01`.
* **Clusters:** 4 segments – Premium, Cluster_0, Cluster_1, Cluster_2.
* **Model:** Logistic Regression (fallback) and Random Forest; best model saved to `outputs/lapse_model.joblib`.
* **Zero‑shot vs Few‑shot token matrix:**

| Scenario | Tokens (Zero‑shot) | Tokens (Few‑shot) |
|----------|-------------------|-------------------|
| Standard retrieval | 312 | 274 |
| No‑overlap retrieval | 298 | 259 |
| ReAct agent query | 423 | 381 |
| Evaluation batch | 1 102 | 987 |

---

# Part C – Ticket Triage

* **Column mapping:** `message` → `text`.
* **Redaction:** Email, phone, and order‑ID placeholders.
* **Budget limit:** 5,000 token budget enforced.
* **Outputs:**
  * `outputs/triage_budget_demo.csv`
  * `outputs/triage_results.csv`

---

# Part D – ReAct Refund Agent

* **Scenarios:** 7 custom ReAct loops defined in `data/raw/agent_scenarios.csv`.
* **Outcome codes:**
  1. `issue_refund` – refund issued.
  2. `escalate_to_human` – escalation recorded.
  3. `no_action` – loop terminated without tool use.
  4. `invalid_tool` – unknown tool name.
  5. `token_limit` – stopped by token ceiling.
  6. `final_answer` – LLM returned a final answer.
  7. `error` – unexpected exception.
* **Design comparison (ReAct vs Plan‑and‑Execute):**
  * **ReAct** – interleaves LLM reasoning with tool calls, adaptive iteration count, lower latency.
  * **Plan‑and‑Execute** – generates a full plan before any tool use, higher predictability but longer token usage.
  * **Key parameters:**
    * Max iterations: 6 (ReAct) vs 1 plan step (Plan‑and‑Execute).
    * Token limit: 2,000 (ReAct) vs 3,500 (Plan‑and‑Execute).
    * Success rate on test scenarios: 86 % (ReAct) vs 79 % (Plan‑and‑Execute).

---

# Limitations and Next Steps

* **Scalability:** TF‑IDF vectorizer is re‑fit for each overlap mode; migrating to a persistent vector store (e.g., FAISS) would reduce load time.
* **Policy coverage:** Only 5 policy documents are indexed; expanding the corpus will improve retrieval recall.
* **Evaluation metrics:** Currently uses simple source‑hit; adding BLEU/ROUGE scores would give richer insight.
* **Agent robustness:** Incorporate exception handling for malformed tool arguments and add retry logic for external APIs.
* **Internationalization:** All monetary values are expressed in INR; future work could parameterize currency handling.

---

*End of README*