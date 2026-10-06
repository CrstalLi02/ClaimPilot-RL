from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Dict, Optional, Tuple


BASE_URL = os.environ.get("AIGC_BASE_URL", "")
API_KEY = os.environ.get("AIGC_APP_ID", "")
MODEL = os.environ.get("FILL_THOUGHT_MODEL", "")


def chat_json(system_prompt: str, user_prompt: str, temperature: float = 0.2, timeout: int = 60) -> Optional[Dict[str, Any]]:
    result = chat_json_detailed(system_prompt, user_prompt, temperature=temperature, timeout=timeout)
    data = result.get("data")
    return data if isinstance(data, dict) else None


def chat_json_detailed(system_prompt: str, user_prompt: str, temperature: float = 0.2, timeout: int = 60) -> Dict[str, Any]:
    payload = {
        "model": MODEL,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    content, error = _curl_chat_detailed(payload, timeout=timeout)
    if error:
        return {"ok": False, "data": None, "raw_text": content, "error": error}
    if not content:
        return {"ok": False, "data": None, "raw_text": content, "error": "empty_response"}
    try:
        data = json.loads(content)
    except Exception:
        return {"ok": False, "data": None, "raw_text": content, "error": "invalid_json_text"}
    if not isinstance(data, dict):
        return {"ok": False, "data": None, "raw_text": content, "error": "non_object_json"}
    return {"ok": True, "data": data, "raw_text": content, "error": ""}


def chat_text(system_prompt: str, user_prompt: str, temperature: float = 0.4, timeout: int = 60) -> Optional[str]:
    payload = {
        "model": MODEL,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    return _curl_chat(payload, timeout=timeout)


def _curl_chat(payload: Dict[str, Any], timeout: int) -> Optional[str]:
    content, error = _curl_chat_detailed(payload, timeout=timeout)
    if error:
        return None
    return content


def _curl_chat_detailed(payload: Dict[str, Any], timeout: int) -> Tuple[Optional[str], str]:
    url = BASE_URL.rstrip("/") + "/chat/completions"
    cmd = [
        "curl",
        "-sS",
        "--max-time",
        str(max(1, int(timeout))),
        "-H",
        f"Authorization: Bearer {API_KEY}",
        "-H",
        "Content-Type: application/json",
        "-X",
        "POST",
        url,
        "-d",
        json.dumps(payload, ensure_ascii=False),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=max(2, int(timeout) + 2))
        if proc.returncode != 0:
            raw_error = (proc.stderr or proc.stdout or "").strip()
            return (raw_error or None), "request_failed"
        stdout = (proc.stdout or "").strip()
        if not stdout:
            return None, "empty_http_body"
        try:
            data = json.loads(stdout)
        except Exception:
            return stdout, "gateway_non_json"
        if isinstance(data, dict) and data.get("error"):
            return json.dumps(data.get("error"), ensure_ascii=False), "api_error"
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception:
            return stdout, "invalid_response_shape"
        if isinstance(content, str):
            return content, ""
        if isinstance(content, (dict, list)):
            return json.dumps(content, ensure_ascii=False), ""
        return str(content), ""
    except subprocess.TimeoutExpired:
        return None, "client_timeout"
    except Exception:
        return None, "client_exception"
