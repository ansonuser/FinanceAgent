"""Revenue-by-segment extractor powered by an Ollama-compatible HTTP API."""

from __future__ import annotations
from textwrap import dedent
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple, List
import re
from schema.data import RevenueData
from screw.agents import Extrator
from dataclasses import asdict
# OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5-coder:14b"



DEFAULT_CHUNK_SIZE = 5000
DEFAULT_OVERLAP = 200
CHUNK_RECORDS = []

def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> Iterator[Tuple[int, str]]:
    """Yield (index, slice) pairs with the desired overlap."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + chunk_size)
        yield start, text[start:end]
        if end == length:
            break
        start = end - overlap

# def reconcile_value(name: str, existing: Optional[str], incoming: Optional[str]) -> Optional[str]:
#     """Keep the first non-empty value unless a later chunk agrees; log conflicts."""
#     if not incoming:
#         return existing
#     if not existing:
#         return incoming
#     if existing == incoming:
#         return existing
#     logging.warning("Conflicting %s values ignored: '%s' vs '%s'", name, existing, incoming)
#     return existing


class Manager:
    """Track the process and operate meta"""

    def __init__(self) -> None:
        self.history = []


    def write_history(self, filename : str = "history.json"):
        with open(filename, "w") as f:
            json.dump(self.history, f)

def load_text(path: Optional[Path]) -> str:
    """Read input text from a file or stdin."""
    if path:
        text = path.read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    if path and path.suffix.lower() == ".json":
        return json.loads(text)
    return text

def generator(
    chunks: List[Dict[str, str]],
) -> Iterator[Dict[str, str]]:
    """
    Table-aware global splitter for plain text that may still include <table>...</table> blocks.

    - Precomputes ALL cut points once (global).
    - Avoids cutting inside tables unless the table itself > chunk_size.
    - When forced to split a large table, prefers </tr> boundaries, then \n\n, \n, '. '.
    - Computes per-chunk anchor labels based on overlapping source anchors.
    """

    # ------------- 1) Merge text and keep interval→anchor mapping -------------
    full_text_parts: List[str] = []
    intervals: List[Tuple[int, int, str]] = []  # (start, end, anchor)
    cursor = 0
    for entry in chunks:
        t = (entry.get("text") or "").strip()
        if not t:
            continue
        full_text_parts.append(t)
        start = cursor
        cursor += len(t)
        intervals.append((start, cursor, entry.get("anchor", "")))
        # join with a single newline between entries (not after last)
        full_text_parts.append("\n")
        cursor += 1
    if full_text_parts:
        full_text_parts.pop()  # remove trailing join newline
        cursor -= 1
    full_text = "".join(full_text_parts)
    n = len(full_text)
    if n <= 0:
        return

    # ------------- 2) Identify table spans -------------
    tables: List[str] = [m.group(1) for m in re.finditer(r"<table.*?>(.*?)</table>", full_text, re.DOTALL) if m and len(m.group(1)) >= 200]
    
  
    for table in tables:
        yield {
        "text": "<table>" + table + "</table>",
        "anchor": "placeholder",
        }
  


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract revenue by segment via Ollama HTTP API.")
    parser.add_argument("--base-url", default=OLLAMA_URL, help="Base URL of the Ollama service.")
    parser.add_argument("--model", default=MODEL_NAME, help="Model name to request.")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds.")
    parser.add_argument("--api-token", help="Optional bearer token for authenticated deployments.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("path", nargs="?", type=Path, help="Input file (defaults to stdin).")
    args = parser.parse_args()
    # args.path = Path(r"D:\Side_projects\llm_cache_test\out_test.md")
    args.path = Path(r"D:\Side_projects\llm_cache_test\chunked_test_aapl.json")
    # year = 2024
    # q = 1
    debug = False
    company_name = "AMZN" # id=22
    company_name = "GOOGL" # id=44, 51
    company_name = "AAPL"
    # Path(r"D:\Side_projects\llm_coder\preprocessed\META\2015-01-01K\fb-12312014x10k.html")
    logging.basicConfig(level=getattr(logging, args.log_level))
    data = load_text(args.path)
    
    
    repeat = 1
    for k in range(repeat):
        manager = Manager()
        extractor = Extrator()
        for i, chunk in enumerate(generator(data), 1):
            if i < 20 or i > 36:
            # if i < 50 or i > 52:
                continue
            # assert 22 <= i <= 24
            print(f"\n=== Chunk {i} ===")
            assert len(chunk["text"]) != 0
            print(chunk["text"])
            print(f"*** Key :{chunk['anchor']} ***")
            print("len of chunk =", len(chunk["text"]))
            # target_period = f"Target Period = {year}_Q{q}\n"
            # print(target_period)
            input_ = chunk["text"]
            messages = [{"role" : "user", "content" : input_}]
            result = extractor.generate(messages, schema = RevenueData)
            if debug:
                result = extractor.generate(messages)
                messages.append({"role": "assistant", "content": result})
                messages.append({"role": "user", "content": "Explain why is not and show reasoning step by step."})
                extractor.generate(messages)
            
            if not isinstance(result, RevenueData):
                continue
        
            if result:
                print(f"# chunk id:{i}\n", result)
                if result.product_segments:
                    CHUNK_RECORDS.append(messages[0]["content"])
                    manager.history.append( json.dumps(asdict(result)) )
        
    
        manager.write_history(f"history_{k}.json")


    messages = [{"role" : "user", "content" : extractor.send_validation(company_name) + '\n' + json.dumps(manager.history)}]
    result = extractor.generate(messages, schema = RevenueData, use_sys = False)
    assert isinstance(result, RevenueData)

    
    with open("full_process.json", "w") as f:
        json.dump(CHUNK_RECORDS, f)

    target = result
    selected_chunk, best_score = "", 0
    for doc_chunk, respond in zip(CHUNK_RECORDS, manager.history):
        respond = RevenueData(**json.loads(respond))
        res_keys = respond.product_segments.keys()
        d = len(set(target.product_segments.keys()).intersection(res_keys)) / len(res_keys) + (target.reasoning == respond.reasoning)
        if d > best_score:
            best_score = d
            selected_chunk = doc_chunk
    user_input = {
        "task": "Identify the revenue segments for the products given document.",
        "document_chunk": selected_chunk,
        "result": json.dumps(asdict(target))
    }
    prompt = extractor.send_refinement() + "\n" + json.dumps(user_input)

    messages = [{"role" : "user", "content" : dedent(prompt)}]
    result = extractor.generate(messages, schema = RevenueData, use_sys = False)
    assert isinstance(result, RevenueData)
    with open("final_result.json", "w") as f:
        json.dump(asdict(result), f, indent = 4)

    
if __name__ == "__main__":
    main()
