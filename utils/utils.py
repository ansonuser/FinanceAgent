from typing import Optional
import requests
from typing import Dict, List
import json
import json5 # for cases with comments
import re 

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
        # Some Ollama responses contain multiple JSON objects separated by newlines.
        # Try parsing the last valid JSON object.
        lines = [line for line in raw.splitlines() if line.strip()]
        if len(lines) > 1:
            # Ollama often streams multiple objects — take the last one (final completion)
            data = json.loads(lines[-1])
        else:
            data = json.loads(raw)
    except json.JSONDecodeError as e:
        # Log or handle decoding issues cleanly
        raise ValueError(f"Invalid JSON from Ollama API: {raw[:300]}...") from e

    def extract_from_obj(obj: Dict) -> str:
        if "message" in obj and obj["message"]:
            content = obj["message"].get("content")
            if content:
                return content
        if "response" in obj:
            return obj.get("response", "")
        raise ValueError(f"Unexpected response payload: {obj}")

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
        