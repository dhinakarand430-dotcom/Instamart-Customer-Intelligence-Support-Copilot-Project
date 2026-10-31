import os
import json
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import dump, load
import time
import re

# ---------------------------------------------------------------------------
# Initialize RAG components
chunks_meta = []
vectorizer = None
tfidf_matrix = None

def initialize_rag(overlap_size=10):
    """Create the TF‑IDF index and chunk metadata.

    Args:
        overlap_size: Number of overlapping words between consecutive chunks.
                      Use 0 for no overlap (full‑document chunks).
    """
    global chunks_meta, vectorizer, tfidf_matrix
    # Load policy text files
    all_chunks = []
    for fp in glob.glob(str(POLICY_DIR / "*.txt")):
        with open(fp, "r", encoding="utf-8") as f:
            raw = f.read()
        if not raw.strip():
            continue
        if overlap_size > 0:
            # Use the chunk helper with the provided overlap size
            # Temporarily adjust module constants for chunking
            original_overlap = globals().get('OVERLAP', 10)
            globals()['OVERLAP'] = overlap_size
            chunks = chunk_document(raw, os.path.basename(fp))
            globals()['OVERLAP'] = original_overlap
        else:
            chunks = [{"source": os.path.basename(fp), "chunk_id": f"{os.path.basename(fp)}#full", "text": raw}]
        all_chunks.extend(chunks)
    # Save chunk metadata
    chunks_meta = all_chunks
    with open(CHUNK_META_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks_meta, f, ensure_ascii=False, indent=2)
    # Build vectorizer – handle empty case
    if not all_chunks:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(["placeholder"])
    else:
        vectorizer = TfidfVectorizer(stop_words="english")
        corpus = [c["text"] for c in all_chunks]
        tfidf_matrix = vectorizer.fit_transform(corpus)
    # Persist for potential reuse
    dump((vectorizer, tfidf_matrix), VECTOR_PATH)

POLICY_DIR = Path('data/policies')
# ---------------------------------------------------------------------------

CHUNK_SIZE = 40                              # words per chunk
OVERLAP = 10                                 # overlapping words
TOP_K = 3                                     # number of contexts to retrieve
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR = Path('state')
STATE_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_PATH = STATE_DIR / 'policy_tfidf_vectors.joblib'
CHUNK_META_PATH = STATE_DIR / 'policy_chunks_meta.json'

# ---------------------------------------------------------------------------
# Helper: chunk a document into overlapping word windows
# ---------------------------------------------------------------------------
def chunk_document(text: str, source_file: str):
    words = re.findall(r"\w+|[^\w\s]+", text)  # keep punctuation as tokens
    chunks = []
    for start in range(0, len(words), CHUNK_SIZE - OVERLAP):
        end = start + CHUNK_SIZE
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "source": source_file,
            "chunk_id": f"{source_file}#chunk{len(chunks)}",
            "text": chunk_text
        })
        if end >= len(words):
            break
    return chunks

# ---------------------------------------------------------------------------
# Build TF‑IDF index (runs once, saves to disk)
# ---------------------------------------------------------------------------
def build_index(overlap: bool = True):
    """Build TF‑IDF index for policy documents.
    If no non‑empty text is found, creates an empty placeholder vectorizer
    to avoid `ValueError: empty vocabulary`.
    """
    all_chunks = []
    for fp in glob.glob(str(POLICY_DIR / "*.txt")):
        with open(fp, "r", encoding="utf-8") as f:
            raw = f.read()
        if not raw.strip():
            # Skip empty policy files
            continue
        if overlap:
            chunks = chunk_document(raw, os.path.basename(fp))
        else:
            # No overlap: treat whole file as a single chunk
            chunks = [{"source": os.path.basename(fp), "chunk_id": f"{os.path.basename(fp)}#full", "text": raw}]
        all_chunks.extend(chunks)
    # Persist chunk metadata for later lookup
    with open(CHUNK_META_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    # If no chunks were collected, create a dummy vectorizer to avoid errors
    if not all_chunks:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(["placeholder"])
        dump((vectorizer, tfidf_matrix), VECTOR_PATH)
        return vectorizer, tfidf_matrix, all_chunks
    # Vectorize
    vectorizer = TfidfVectorizer(stop_words="english")
    corpus = [c["text"] for c in all_chunks]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    dump((vectorizer, tfidf_matrix), VECTOR_PATH)
    return vectorizer, tfidf_matrix, all_chunks
    all_chunks = []
    for fp in glob.glob(str(POLICY_DIR / "*.txt")):
        with open(fp, "r", encoding="utf-8") as f:
            raw = f.read()
        if overlap:
            chunks = chunk_document(raw, os.path.basename(fp))
        else:
            # No overlap: treat whole file as a single chunk
            chunks = [{"source": os.path.basename(fp), "chunk_id": f"{os.path.basename(fp)}#full", "text": raw}]
        all_chunks.extend(chunks)
    # Persist chunk metadata for later lookup
    with open(CHUNK_META_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    # Vectorize
    vectorizer = TfidfVectorizer(stop_words="english")
    corpus = [c["text"] for c in all_chunks]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    dump((vectorizer, tfidf_matrix), VECTOR_PATH)
    return vectorizer, tfidf_matrix, all_chunks

# ---------------------------------------------------------------------------
# Retrieval – given a user query, return top‑k matching chunks (source file list)
# ---------------------------------------------------------------------------
def retrieve(query: str, top_k: int = TOP_K, overlap: bool = True):
    """Retrieve top‑k chunks for *query* using the globally initialized RAG state.

    The function assumes that :func:`initialize_rag` has been called beforehand
    with the appropriate ``overlap`` configuration, which populates the global
    ``chunks_meta``, ``vectorizer`` and ``tfidf_matrix`` variables.
    """
    # Ensure the global structures are present; if not, initialize with the requested mode.
    global chunks_meta, vectorizer, tfidf_matrix
    if 'chunks_meta' not in globals() or not chunks_meta:
        initialize_rag(overlap_size=OVERLAP if overlap else 0)
    # Transform the query using the pre‑fitted vectorizer.
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idxs = sims.argsort()[::-1][:top_k]
    results = []
    for idx in top_idxs:
        meta = chunks_meta[idx]
        results.append({
            "source": meta["source"],
            "chunk_id": meta["chunk_id"],
            "text": meta["text"],
            "score": float(sims[idx])
        })
    return results

# ---------------------------------------------------------------------------
# Mock LLM – reuse the provided mock_llm (fallback to simple deterministic stub)
# ---------------------------------------------------------------------------
try:
    from mock_llm import chat as mock_chat
except Exception:
    def mock_chat(prompt: str) -> str:
        # deterministic placeholder answer
        return f"{{\"answer\": \"This is a mock answer for: {prompt[:30]}...\", \"sources\": []}}"

# ---------------------------------------------------------------------------
# Grounded answering loop
# ---------------------------------------------------------------------------
def grounded_answer(query: str, overlap: bool = True):
    # 1) retrieve top‑k policy chunks
    contexts = retrieve(query, overlap=overlap)
    # 2) Build augmented prompt
    context_text = "\n---\n".join([c["text"] for c in contexts])
    augmented_prompt = json.dumps({
        "task": "answer_policy_question",
        "question": query,
        "context": context_text
    })
    # 3) Call LLM
    llm_resp = mock_chat(augmented_prompt)
    # Expect JSON with fields answer and maybe sources – we will parse loosely
    try:
        parsed = json.loads(llm_resp)
    except json.JSONDecodeError:
        parsed = {"answer": llm_resp, "sources": []}
    # Attach source tracking
    parsed.setdefault("sources", [c["source"] for c in contexts])
    return parsed

# ---------------------------------------------------------------------------
# Evaluation against policy_eval.csv (expects columns: question, expected_doc)
# ---------------------------------------------------------------------------
def evaluate(eval_csv: Path = Path('data/raw/policy_eval.csv')):
    df = pd.read_csv(eval_csv)
    results = []
    for _, row in df.iterrows():
        q = row['question']
        expected = row['expected_doc']
        # standard overlap
        ans_std = grounded_answer(q, overlap=True)
        # zero‑overlap (whole‑document retrieval)
        ans_no = grounded_answer(q, overlap=False)
        ok_std = expected in ans_std.get('sources', [])
        ok_no = expected in ans_no.get('sources', [])
        results.append({
            'query': q,
            'expected': expected,
            'standard_hit': ok_std,
            'zero_overlap_hit': ok_no
        })
    report = pd.DataFrame(results)
    # Save report
    report_path = OUTPUT_DIR / 'policy_rag_evaluation.csv'
    report.to_csv(report_path, index=False)
    return report_path

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Part D – ReAct Refund Agent
# ---------------------------------------------------------------------------
# Simple in‑memory mock order database (loaded from data/clean/orders.csv)
ORDERS_PATH = Path('data/clean/orders.csv')
CUSTOMERS_PATH = Path('data/clean/customers.csv')
PRODUCTS_PATH = Path('data/clean/products.csv')

def load_orders():
    return pd.read_csv(ORDERS_PATH)

def load_customers():
    return pd.read_csv(CUSTOMERS_PATH)

def load_products():
    return pd.read_csv(PRODUCTS_PATH)

# Define internal tool functions
def get_order(order_id: str):
    df = load_orders()
    rec = df[df['order_id'] == order_id]
    if rec.empty:
        return None
    return rec.iloc[0].to_dict()

def search_policy(query: str):
    # Re‑use grounded retrieval (standard overlap)
    return retrieve(query, overlap=True)

def calculate_refund(order: dict):
    # Simple policy: full refund if order_status == 'cancelled', else 0
    if order.get('order_status') == 'cancelled':
        return float(order.get('order_amount', 0))
    return 0.0

def issue_refund(order_id: str, amount: float):
    # Append a record to outputs/refund_ledger.jsonl
    ledger_path = OUTPUT_DIR / 'refund_ledger.jsonl'
    entry = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'order_id': order_id,
        'refunded_amount': amount
    }
    with open(ledger_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + "\n")
    return {'status': 'refunded', 'order_id': order_id, 'amount': amount}

def escalate_to_human(order_id: str, reason: str):
    # Placeholder – just log escalation
    log_path = OUTPUT_DIR / 'escalations.log'
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"{pd.Timestamp.now().isoformat()} – Escalated order {order_id}: {reason}\n")
    return {'status': 'escalated', 'order_id': order_id, 'reason': reason}

# Tool map for ReAct loop
TOOL_MAP = {
    'get_order': get_order,
    'search_policy': search_policy,
    'calculate_refund': calculate_refund,
    'issue_refund': issue_refund,
    'escalate_to_human': escalate_to_human,
}

# Simple ReAct engine – processes a scenario line (expects columns: scenario_id, description)
SCENARIOS_PATH = Path('data/clean/agent_scenarios.csv')
MAX_ITER = 6
TOKEN_LIMIT = 2000  # rough token ceiling for a single loop

def run_scenario(scenario_row: pd.Series):
    # State persistence – checkpoint after each tool observation
    state_file = STATE_DIR / f"scenario_{scenario_row['scenario_id']}_checkpoint.json"
    # Load previous state if exists
    if state_file.exists():
        with open(state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
    else:
        state = {
            'history': [],
            'iteration': 0,
            'description': scenario_row['message']
        }
    while state['iteration'] < MAX_ITER:
        # Build prompt from history + current description
        prompt = json.dumps({
            'task': 'refund_agent',
            'description': state['description'],
            'history': state['history']
        })
        # Token check (very naive)
        if len(prompt.split()) > TOKEN_LIMIT:
            state['history'].append({'role': 'system', 'content': 'Token limit reached, terminating.'})
            break
        # Call mock LLM
        llm_output = mock_chat(prompt)
        # Expect JSON with a "tool" field indicating which internal tool to call and its arguments
        try:
            action = json.loads(llm_output)
        except json.JSONDecodeError:
            # If not JSON, treat as final answer
            state['history'].append({'role': 'assistant', 'content': llm_output})
            break
        tool_name = action.get('tool')
        args = action.get('args', {})
        if tool_name not in TOOL_MAP:
            # Unknown tool – terminate
            state['history'].append({'role': 'assistant', 'content': f"Unknown tool: {tool_name}"})
            break
        # Execute tool
        result = TOOL_MAP[tool_name](**args)
        # Record observation
        state['history'].append({'role': 'assistant', 'content': llm_output})
        state['history'].append({'role': 'tool', 'name': tool_name, 'output': result})
        # Persist checkpoint
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        # Simple termination condition: if tool was issue_refund or escalate_to_human
        if tool_name in ('issue_refund', 'escalate_to_human'):
            break
        state['iteration'] += 1
    # Log full run to agent_runs.jsonl
    run_log_path = OUTPUT_DIR / 'agent_runs.jsonl'
    with open(run_log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps({
            'scenario_id': scenario_row['scenario_id'],
            'final_state': state
        }) + "\n")
    return state

def process_all_scenarios():
    SCENARIOS_PATH = "data/raw/agent_scenarios.csv"
    df = pd.read_csv(SCENARIOS_PATH)
    for _, row in df.iterrows():
        run_scenario(row)

# ---------------------------------------------------------------------------
# Resume a crashed scenario – loads the latest checkpoint and continues
# ---------------------------------------------------------------------------
def resume_scenario(scenario_id: int):
    checkpoint = STATE_DIR / f"scenario_{scenario_id}_checkpoint.json"
    if not checkpoint.exists():
        raise FileNotFoundError(f"No checkpoint found for scenario {scenario_id}")
    with open(checkpoint, 'r', encoding='utf-8') as f:
        state = json.load(f)
    # Re‑run remaining iterations
    while state['iteration'] < MAX_ITER:
        prompt = json.dumps({
            'task': 'refund_agent',
            'description': state['description'],
            'history': state['history']
        })
        llm_output = mock_chat(prompt)
        try:
            action = json.loads(llm_output)
        except json.JSONDecodeError:
            state['history'].append({'role': 'assistant', 'content': llm_output})
            break
        tool_name = action.get('tool')
        args = action.get('args', {})
        if tool_name not in TOOL_MAP:
            state['history'].append({'role': 'assistant', 'content': f"Unknown tool: {tool_name}"})
            break
        result = TOOL_MAP[tool_name](**args)
        state['history'].append({'role': 'assistant', 'content': llm_output})
        state['history'].append({'role': 'tool', 'name': tool_name, 'output': result})
        with open(checkpoint, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        if tool_name in ('issue_refund', 'escalate_to_human'):
            break
        state['iteration'] += 1
    # Append final run log
    run_log_path = OUTPUT_DIR / 'agent_runs.jsonl'
    with open(run_log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps({
            'scenario_id': scenario_id,
            'final_state': state,
            'resumed': True
        }) + "\n")
    return state

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Build index if not present
    if not VECTOR_PATH.exists():
        print('Building TF‑IDF index...')
        build_index(overlap=True)
        build_index(overlap=False)  # also build zero‑overlap version
    # Example: evaluate RAG
    print('Running evaluation...')
    eval_path = evaluate()
    print(f'Evaluation saved to {eval_path}')
    # Run ReAct scenarios
    print('Processing refund agent scenarios...')
    process_all_scenarios()
    print('Done.')
