import os
from typing import Any, Dict, List

import requests

from .base import ChatProvider


def _load_api_key_from_env_or_dotenv(env_var: str = "OPENAI_API_KEY", env_path: str | None = None) -> str:
    # 1) direct env
    key = os.getenv(env_var, "")
    if key:
        return key
    # 2) python-dotenv if available
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(dotenv_path=env_path or ".env")
        key = os.getenv(env_var, "")
        if key:
            return key
    except Exception:
        pass
    # 3) manual parse of .env in CWD
    path = env_path or os.path.join(os.getcwd(), ".env")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith(env_var + "="):
                        _, val = line.split("=", 1)
                        val = val.strip().strip('"').strip("'")
                        if val:
                            return val
        except Exception:
            pass
    return ""


class OpenAIProvider(ChatProvider):
    """Thin HTTP adapter for OpenAI Chat Completions API (>= 2024-xx schema).

    Uses environment variable OPENAI_API_KEY. Keeps deps minimal (requests only).
    """

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None, env_path: str | None = None):
        self.api_key = api_key or _load_api_key_from_env_or_dotenv("OPENAI_API_KEY", env_path)
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

    def send(self, messages: List[Dict[str, str]], *, model: str, max_tokens: int = 1024, temperature: float = 0.7) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        content = (choice.get("message") or {}).get("content", "")
        finish_reason = choice.get("finish_reason", "")
        usage = data.get("usage", {})
        return {"content": content, "finish_reason": finish_reason, "usage": usage, "raw": data}


