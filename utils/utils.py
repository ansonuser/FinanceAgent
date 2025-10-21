from typing import Optional
import requests
from typing import Dict, List
import json
import json5 # for cases with comments
import re 
from arelle import Cntlr
import logging
import aiohttp
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("api_token")

def convert(src:str = "aapl-20240630.htm", dest:str = 'aapl-20240630.htm'):
    controller = Cntlr.Cntlr()
    controller.run([
        "--file", src,
        "--plugin", "xbrl.view.xhtml.ViewerCmd",
        "--viewerFile", dest
    ])
    logging.info(f"[Data Transformation] Convert {src} to {dest}")
    
def extract_from_obj(obj: Dict) -> str:
        if "message" in obj and obj["message"]:
            content = obj["message"].get("content")
            if content:
                return content
        if "response" in obj:
            return obj.get("response", "")
        raise ValueError(f"Unexpected response payload: {obj}")


async def a_call_api(
    session: aiohttp.ClientSession,
    messages: List[Dict],
    base_url: str,
    model: str,
    timeout: Optional[float] = None,
    is_ollama: bool = True
) -> str:
    """Send one chunk to the remote Ollama-compatible API asynchronously and return its response."""
    
    if is_ollama:
        headers = {"Content-Type": "application/json"}
    else:
        headers = {"Content-Type": "application/json"}
        headers["Authorization"] = f"Bearer {TOKEN}"


    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1},
        "provider": {
            "order" : [
                "atlas-cloud/fp8",
                "chutes/bf16",
                "deepinfra/fp4"
            ]
        }
    }

    try:
        async with session.post(base_url, json=payload, headers=headers, timeout=timeout) as response:
            if is_ollama:
                raw = (await response.text()).strip()
                if response.status != 200:
                    raise ValueError(f"HTTP {response.status}: {raw[:300]}")
                if not raw:
                    raise ValueError("Empty response body from Ollama API.")
            else:
                response.raise_for_status()
                result = await response.json()
                content = extract_json(result["choices"][0]["message"]["content"])

    except asyncio.TimeoutError:
        raise TimeoutError("Ollama API request timed out.")
    except aiohttp.ClientError as e:
        raise ConnectionError(f"Network error: {e}")

    # ---- Parse and extract ----
    try:
        if is_ollama:
            lines = [line for line in raw.splitlines() if line.strip()]
            data = extract_json(lines[-1]) if len(lines) > 1 else extract_json(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from API: {raw[:300]}...") from e

    # ---- Extract the text field ----
    def extract_from_obj(obj: Dict) -> str:
        if "message" in obj and obj["message"]:
            content = obj["message"].get("content")
            if content:
                return content
        if "response" in obj:
            return obj.get("response", "")
        raise ValueError(f"Unexpected response payload: {obj}")
    
    if is_ollama:
        content = extract_from_obj(data).strip()
        content = extract_json(content)
    if not content:
        raise ValueError(f"No content returned from API. Payload: {data}")
    return content 

def call_api(
    messages: List[str],
    base_url: str,
    model: str,
    timeout: Optional[float] = None,
    api_token: Optional[str] = None,
) -> str:
    """Send one chunk to the remote Ollama-compatible API and return its response."""
    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    payload = {
        "model": model,
        "messages": messages ,
        "stream": False,
        "options": {"temperature": 0.1}
    }

    response = requests.post(base_url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()

    raw = response.text.strip()
    if not raw:
        raise ValueError("Empty response body from Ollama API.")
    
    try:
        lines = [line for line in raw.splitlines() if line.strip()]
        if len(lines) > 1:
            # Ollama often streams multiple objects — take the last one (final completion)
            data = json.loads(lines[-1])
        else:
            data = json.loads(raw)
    except json.JSONDecodeError as e:
        # Log or handle decoding issues cleanly
        raise ValueError(f"Invalid JSON from Ollama API: {raw[:300]}...") from e

    content = extract_from_obj(data).strip()
    if not content:
        raise ValueError(f"No content returned from Ollama API. Payload: {data}")
    return content


def extract_json(payload: str) -> Dict:
    """Extract JSON from the model output, tolerating stray text if possible."""

    pattern = r"```json\s*(\{[\s\S]*?\})\s*```"
    matches = re.findall(pattern, payload, flags=re.MULTILINE)
    if matches:
        payload = matches[-1]

    start = payload.find("{")
    end = payload.rfind("}") + 1
    if end == 0 and start != -1:
        payload = payload + "}"
        end = len(payload)
    jsonStr = payload[start:end] if start != -1 and end != 0 else ""
    jsonStr = re.sub(r",\s*([\]}])", r"\1", jsonStr)
    try:
        return json5.loads(jsonStr)
    except (json.JSONDecodeError, ValueError):
        error_str = "LLM outputted an invalid JSON. Please use a better structured model."
        print(error_str)
        print(payload)
        raise  
    except Exception as e:
        print(payload)
        raise Exception(f"An unexpected error occurred: {str(e)}")
        

def build_mapping_table(src : str)->Dict[str, str]:
    # prediction -> candidate
    companys = os.listdir(src)
    table = {}
    for company in companys:
        c_folder = os.path.join(src, company, "candidates")
        for date_folder in os.listdir(c_folder):
            full_path = os.path.join( c_folder, date_folder)
      
            candidates =  os.listdir(os.path.join(full_path, "chunks"))
            if candidates:
                candidate = candidates[0]
                key = candidate.replace("candidated_chunks", "prediction")
                table[key] = os.path.join(full_path, "chunks", candidate)
    return table