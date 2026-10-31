import re
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import time
import random

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
TOKEN_ESTIMATOR = lambda text: len(text.split())  # simple whitespace token estimate

# Regex patterns for redaction
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
ORDER_ID_RE = re.compile(r"\bORD[0-9]{6}\b", re.IGNORECASE)

def redact_text(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = ORDER_ID_RE.sub("[ORDER_ID]", text)
    return text

# ---------------------------------------------------------------------------
# Mock LLM interface (import from provided mock_llm module)
# ---------------------------------------------------------------------------
try:
    from mock_llm import chat as mock_chat
except Exception:
    # Fallback simple deterministic mock if the real module is missing
    def mock_chat(prompt: str) -> str:
        # Very naive mock that returns a JSON with a random priority
        priority = random.choice(["high", "medium", "low"])
        return json.dumps({"priority": priority, "category": "general"})

# ---------------------------------------------------------------------------
# Retry wrapper with exponential backoff (max 3 attempts)
# ---------------------------------------------------------------------------
def call_llm_with_retry(prompt: str, max_attempts: int = 3) -> str:
    delay = 0.5
    for attempt in range(1, max_attempts + 1):
        try:
            response = mock_chat(prompt)
            # Basic sanity: must be non‑empty string
            if not response:
                raise ValueError("Empty response")
            return response
        except Exception as e:
            if attempt == max_attempts:
                raise
            time.sleep(delay)
            delay *= 2  # exponential backoff
    return ""  # unreachable

# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------
def parse_json_response(text: str) -> Dict[str, Any] | None:
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return None
    return None

def fallback_keyword_parse(text: str) -> Dict[str, Any]:
    # Very simple heuristic: look for words "urgent" or "complaint"
    low = {"priority": "low", "category": "general"}
    if re.search(r"\burgent\b", text, re.I):
        low["priority"] = "high"
    elif re.search(r"\bcomplaint\b", text, re.I):
        low["priority"] = "medium"
    return low

# ---------------------------------------------------------------------------
# Main triage pipeline
# ---------------------------------------------------------------------------
RAW_TICKETS_PATH = Path('data/raw/tickets.csv')  # updated input
CLEAN_TICKETS_PATH = Path('data/clean/triage_input.csv')
DEMO_OUTPUT = Path('outputs/triage_budget_demo.csv')
FULL_OUTPUT = Path('outputs/triage_results.csv')
SEGMENTS_PATH = Path('outputs/customer_segments.csv')

def load_segments() -> pd.DataFrame:
    return pd.read_csv(SEGMENTS_PATH)

def assign_priority_and_queue(row: pd.Series, segments_df: pd.DataFrame) -> Dict[str, Any]:
    # Use segment info for queue mapping – simplistic example
    segment = segments_df.loc[segments_df['customer_id'] == row['customer_id'], 'segment']
    queue = "premium_queue" if not segment.empty and segment.iloc[0] == "Premium" else "regular_queue"
    return {"priority": row.get('priority', 'low'), "queue": queue}

def process_tickets():
    df = pd.read_csv(RAW_TICKETS_PATH)
    # Input filtering
    df = df.dropna(subset=['text']).loc[df['text'].str.strip().astype(bool)]
    df['token_count'] = df['text'].apply(TOKEN_ESTIMATOR)
    df['flagged_long'] = df['token_count'] > 400

    # Privacy redaction
    df['redacted_text'] = df['text'].apply(redact_text)

    segments = load_segments()
    results: List[Dict[str, Any]] = []
    budget_used = 0
    BUDGET_LIMIT = 5000

    for _, row in df.iterrows():
        prompt = json.dumps({"task": "classify_ticket", "message": row['redacted_text']})
        try:
            raw_resp = call_llm_with_retry(prompt)
        except Exception:
            raw_resp = ""
        parsed = parse_json_response(raw_resp)
        if parsed is None:
            parsed = fallback_keyword_parse(row['redacted_text'])
        priority = parsed.get('priority', 'low')
        row_result = {
            "ticket_id": row['ticket_id'],
            "customer_id": row['customer_id'],
            "priority": priority,
            "category": parsed.get('category', 'general'),
            "token_used": row['token_count']
        }
        # Merge queue info
        row_result.update(assign_priority_and_queue(row_result, segments))
        results.append(row_result)
        budget_used += row['token_count']
        if budget_used >= BUDGET_LIMIT:
            break

    demo_df = pd.DataFrame(results)
    demo_df.to_csv(DEMO_OUTPUT, index=False)

    # Full run (no budget constraint)
    full_results: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        prompt = json.dumps({"task": "classify_ticket", "message": row['redacted_text']})
        try:
            raw_resp = call_llm_with_retry(prompt)
        except Exception:
            raw_resp = ""
        parsed = parse_json_response(raw_resp) or fallback_keyword_parse(row['redacted_text'])
        priority = parsed.get('priority', 'low')
        full_row = {
            "ticket_id": row['ticket_id'],
            "customer_id": row['customer_id'],
            "priority": priority,
            "category": parsed.get('category', 'general'),
            "token_used": row['token_count']
        }
        full_row.update(assign_priority_and_queue(full_row, segments))
        full_results.append(full_row)
    pd.DataFrame(full_results).to_csv(FULL_OUTPUT, index=False)

if __name__ == '__main__':
    process_tickets()
