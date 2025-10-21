"""Revenue-by-segment extractor """

from __future__ import annotations
from textwrap import dedent
import math
from security.post_scan import get_error_table
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple, List
import re
from schema.data import RevenueData
from screw.agents import Extractor
from dataclasses import asdict
import os
from collections import defaultdict
import asyncio
import aiohttp
from utils.utils import build_mapping_table

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5-coder:14b"
MAX_CONCURRENT = 2
THROTTLE_DELAY = 0.5
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_CHUNK_SIZE = 5000
DEFAULT_OVERLAP = 200
META_MODEL = "openai/gpt-oss-20b:free"


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


class Manager:
    """Track the process and operate meta"""

    def __init__(self) -> None:
        self.history = []

    def write_history(self, filename : str = "history.json"):
        with open(filename, "w") as f:
            json.dump(self.history, f)

    def clear_history(self):
        self.history.clear()


def to_candidated_chunk_path(input_path: str) -> str:
    """
    Convert a preprocessed JSON path like:
        preprocessed\TSLA\2015-01-01K\tsla-10k_20141231.json
    to:
        result\TSLA\candidates\2015-01-01K\chunks\tsla-10k_20141231_candidated_chunks.json
    """
    # Normalize the path
    input_path = os.path.normpath(input_path)
    
    # Split components
    parts = input_path.split(os.sep)
    
    # Expect structure: preprocessed / COMPANY / PERIOD / FILENAME
    if len(parts) < 4 or parts[0].lower() != "preprocessed":
        raise ValueError(f"Unexpected input path structure: {input_path}")
    
    company = parts[1]
    period = parts[2]
    filename = parts[3]
    
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_candidated_chunks{ext}"
    
    # Build the new output path
    output_path = os.path.join("result", company, "candidates", period, "chunks", new_filename)
    return output_path


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
    
    cur_unit = None
    for chunk in chunks:
        chunk_text = chunk["text"]
        matches = re.search(r"in\s+(million|billion|thousand)s?", chunk["text"], re.IGNORECASE)
        if matches:
            cur_unit = matches.group() 
        elif cur_unit is not None:
            chunk_text = cur_unit + '/n' + chunk["text"]
        yield {
        "text": chunk_text,
        "anchor": chunk["anchor"],
        }


async def predict(file_path: str, output_dir: str, is_k: bool = False, session = None, meta = False, extractor = None):
    """Fully async version with bounded concurrency, throttle, and gather batching."""
    chunks_record = []
    quality_check = True
    manager = Manager()

    # path, folder set-up 
    splits = file_path.split(os.sep)
    filename = os.path.splitext(splits[-1])[0]
    if meta:
        # ex: result/AAPL/candidates/2025-01-01/chunks/filename
        company_name = splits[1]
        date_str = splits[3]
    else:
        # ex: preprocessd/AAPL/2015-01-01/filename
        company_name = splits[1]
        date_str = splits[2]

    output_dir = os.path.join(output_dir, company_name)
    candidate_path = os.path.join(output_dir, "candidates", date_str)
    prediction_path = os.path.join(output_dir, "predictions")
    prediction_file_path = os.path.join(prediction_path, filename + "_prediction.json")

    if not meta:
        os.makedirs(os.path.join(candidate_path, "chunks"), exist_ok=True)
        os.makedirs(os.path.join(candidate_path, "predictions"), exist_ok=True)
        os.makedirs(prediction_path, exist_ok=True)
    

    if not meta and os.path.exists(prediction_file_path):
        print(prediction_file_path + " existed !")
        return True
    
    found_candidate = False
    
    if not meta:
        candidated_chunk_path = to_candidated_chunk_path(file_path)
        if os.path.exists(candidated_chunk_path):
            print(f"Found candidate {candidated_chunk_path} existing!!!")
            found_candidate = True
            content = await asyncio.to_thread(load_text, Path(candidated_chunk_path))
    else:
        content = await asyncio.to_thread(load_text, Path(file_path))
        if len(content) > 0:
            found_candidate = True

    chunks = []
    print("loading data...")
    """
    # first round prediction
    # found candidate -> load valid
    #                 ->  not valid on meta -> return False
    #                 ->      valid on -> empty -> fallback if meta else return False
    #                                  -> non-empty -> select  
    """
    if meta or found_candidate:
        chunks = [{"text" : json.dumps(content)}]
        if meta and not found_candidate:
            return False
        if found_candidate and chunks[0]['text'] == '[]':
            if meta:
                return False
            chunks = []
        
    if not chunks:
        content = await asyncio.to_thread(load_text, Path(file_path))
        chunks = list(generator(content))
        print(f"Total chunks: {len(chunks)}")

    assert session is not None
    extractor.session = session 
    extractor.set_sys(mode="K" if is_k else "Q")

    # --- Concurrency control ---
    
    
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async def process_chunk(i, chunk):
        async with sem:
            messages = [{"role": "user", "content": chunk["text"]}]
            try:
                result = await extractor.a_generate(messages, schema=RevenueData)
            except Exception as e:
                print(f"Chunk {i} failed: {e}")
                return
            if isinstance(result, RevenueData) and len(result.product_segments) > 0:
                chunks_record.append(messages[0]["content"])
                manager.history.append(json.dumps(asdict(result)))
                print(f"Chunk {i} OK ({len(result.product_segments)} segments)")
        

    # --- Launch chunk tasks with throttle ---
    tasks = []
    print(f"Number of chunks is: {len(chunks)}")
    for i, chunk in enumerate(chunks, 1):
        tasks.append(asyncio.create_task(process_chunk(i, chunk)))
        await asyncio.sleep(THROTTLE_DELAY)
    await asyncio.gather(*tasks)
    print(f"All {len(tasks)} chunks processed.")

    # --- Record candidates for first run ---
    if not meta or not found_candidate:
        history_path = os.path.join(candidate_path, "predictions", f"{filename}_prediction_history.json")
        chunks_path = os.path.join(candidate_path, "chunks", f"{filename}_candidated_chunks.json")
        if chunks_record:
            write_tasks = [
                asyncio.to_thread(manager.write_history, history_path),
                asyncio.to_thread(json.dump, chunks_record, open(chunks_path, "w"))
            ]
            await asyncio.gather(*write_tasks)

    
    if is_k:
        mode = "K"
    else:
        mode = "Q"

    # --- Select best matching chunk (async offload since it's CPU-ish) ---
    def select_best_chunk():
        best_score, selected = math.inf, ""
        for doc_chunk, respond in zip(chunks_record, manager.history):
            respond = RevenueData(**json.loads(respond))
           
            res_values = [float(v) for v in respond.product_segments.values() if v]
            target_values = [float(v) for v in target.product_segments.values() if v]
            d = (
                abs(respond.total_revenue - target.total_revenue)
                + abs(sum(res_values) - sum(target_values))
            )
            if d < best_score:
                best_score, selected = d, doc_chunk
        return selected

    target = None
    if manager.history:
        messages = [{
            "role": "user",
            "content": extractor.send_validation(company_name, mode = mode) + '\n\n' + json.dumps(manager.history)
        }]

        result = await extractor.a_generate(messages, schema=RevenueData, use_sys=False)
        if not isinstance(result, RevenueData):
            print("Got result:", result)
            print("Fail to predict filename:", filename)
            return

        target = result
    elif not meta:
        return False


    if not meta or not found_candidate:
        selected_chunk = await asyncio.to_thread(select_best_chunk)
    
        # --- Refinement phase ---
        user_input = {
            "task": "Identify the revenue segments for the products given document.",
            "document_chunk": selected_chunk,
            "result": json.dumps(asdict(target))
        }
        prompt = extractor.send_refinement(mode = mode) + "\n\n" + json.dumps(user_input)
        messages = [{"role": "user", "content": dedent(prompt)}]
        result = await extractor.a_generate(messages, schema=RevenueData, use_sys=False)
        assert isinstance(result, RevenueData)
    
    else:
        prediction_file_path = prediction_file_path.replace("_candidated_chunks", "")
    # --- Save prediction ---
    await asyncio.to_thread(
        json.dump, asdict(result), open(prediction_file_path, "w"), indent=4
    )

    # --- Quality check & refinement ---
    
    total_sum = sum(v for v in result.product_segments.values() if v)
    if abs(total_sum - result.total_revenue) / (result.total_revenue + 1e-5) >= 0.01:
        quality_check = False
        await clean_segment(extractor, result)
        await asyncio.to_thread(
            json.dump, asdict(result), open(prediction_file_path, "w"), indent=4
        )

    print(f"Prediction finished for {filename}")
    return quality_check

async def clean_segment(extractor : Extractor, result : RevenueData) -> RevenueData:
    prompt = extractor.checksum() + "\n\n" + json.dumps(asdict(result), indent=4)
    messages = [{"role" : "user", "content" : dedent(prompt)}]
    validation = await extractor.a_generate(messages, use_sys = False)
    if validation and validation['is_valid']:
        for seg in validation['aggregate_segments']:
            del result.product_segments[seg]
  

async def run_single_company(src: str, company: str, output_dir: str, session = None, extractor = None):
    """
    Run predictions for all available filings of a single company (10-Q / 10-K).
    Processes each date subfolder asynchronously but sequentially throttled.
    """
    company_path = Path(src) / company
    date_folders = sorted(os.listdir(company_path))
    error_list = []

    
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async def process_date_folder(date_folder: str):
        async with sem:
            try:
                parent = company_path / date_folder
                is_k = date_folder.endswith("K")
                files = os.listdir(parent)
                if not files:
                    print(f"No file in {parent}")
                    return
                file_path = parent / files[0]

                success = await predict(str(file_path), output_dir, is_k, session, extractor=extractor)
                if success:
                    print(f"Done {file_path}")
                else:
                    error_list.append(str(file_path))
                    print(f"Failed {file_path}")
                    
            except Exception as e:
                print(f"Exception in {date_folder}: {e}")
                error_list.append(str(parent))
            await asyncio.sleep(THROTTLE_DELAY)

    # Launch all date-folder tasks
    tasks = [asyncio.create_task(process_date_folder(d)) for d in date_folders]
    await asyncio.gather(*tasks)
    
    print(f"Company {company}: completed {len(date_folders)} filings.")
    return error_list


async def run_prediction(src: str, output_dir: str, ready: set, extractor = None):
    """
    Run async predictions for multiple companies concurrently.
    """
    error_files = defaultdict(list)

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    session = aiohttp.ClientSession()
    async def process_company(company: str, session = None):
        async with sem:
            try:
                errs = await run_single_company(src, company, output_dir, session, extractor=extractor)
                error_files[company].extend(errs)
            except Exception as e:
                print(f"Error in {company}: {e}")
            await asyncio.sleep(THROTTLE_DELAY)

    # Launch all company-level tasks
    tasks = [asyncio.create_task(process_company(c, session)) for c in ready]
    await asyncio.gather(*tasks)
    await session.close()
    # Write aggregated error logs asynchronously
    await asyncio.to_thread(json.dump, error_files, open("error_records.json", "w"), indent=2)
    print("Error records saved to error_records.json")

    return error_files




async def main(ready:set, extractor: Extractor):
    """
    Entry point for async batch run.
    """
    src = r"preprocessed"
    output_dir = r"result"
    os.makedirs(output_dir, exist_ok=True)
    await run_prediction(src, output_dir, ready, extractor=extractor)
    
    error_table = get_error_table(ready)

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    extractor.update_source(META_MODEL, OPENROUTER)
    extractor.set_ollama(False)
    tasks = []
    

    session = aiohttp.ClientSession()
    async def run_pred(is_k : bool, file : str, output_dir : str, session = None):
        async with sem:
            try:
                isfixed = await predict(file, output_dir, is_k, session = session, meta = True, extractor = extractor)
                if isfixed:
                    print(f"Fixed chunks {file} from {company}")
             
            except Exception as e:
                print(f"Error in {company}: got error message at fix stag---------------{e}")
            await asyncio.sleep(THROTTLE_DELAY)

    mapping_table = build_mapping_table("result")
    for company, files in error_table.items():
        print(f"Fix error in {company}") 
        for file in files:
            file = mapping_table[file.split(os.sep)[-1]]
            print("file=", file)
            is_k = file.split(os.sep)[-3].endswith("K")   
            tasks.append(asyncio.create_task(run_pred(is_k, file, output_dir, session = session)))
    await asyncio.gather(*tasks)
    await session.close()

    


if __name__ == "__main__":
    ready = {'AMZN', 'MSFT', "NVDA", "AAPL", "GOOGL"}
    
    extractor = Extractor()
    extractor.set_ollama(True)
    try:
        asyncio.run(main(ready, extractor))
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")

