import json
import logging
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class OpenAICompatClient:
    # Small OpenAI-compatible chat completions client with retry support.

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_sec: float = 30.0,
        extra_headers: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        initial_delay_sec: float = 1.0,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self.extra_headers = extra_headers or {}
        self.max_retries = max_retries
        self.initial_delay_sec = initial_delay_sec

    @classmethod
    def from_env(
        cls,
        prefix: str,
        default_base_url: str,
        default_model: str,
    ) -> "OpenAICompatClient":
        extra_headers_raw = cls._env(f"{prefix}_EXTRA_HEADERS", "")
        extra_headers: Dict[str, str] = {}
        if extra_headers_raw:
            try:
                parsed = json.loads(extra_headers_raw)
                if isinstance(parsed, dict):
                    extra_headers = {str(k): str(v) for k, v in parsed.items()}
            except json.JSONDecodeError:
                logger.warning("Ignoring invalid %s_EXTRA_HEADERS JSON", prefix)

        return cls(
            base_url=cls._env(f"{prefix}_API_URL", default_base_url),
            model=cls._env(f"{prefix}_MODEL", default_model),
            api_key=cls._env(f"{prefix}_API_KEY", ""),
            timeout_sec=float(cls._env(f"{prefix}_TIMEOUT_SEC", "30")),
            extra_headers=extra_headers,
            max_retries=int(cls._env(f"{prefix}_MAX_RETRIES", "3")),
            initial_delay_sec=float(cls._env(f"{prefix}_INITIAL_DELAY_SEC", "1")),
        )

    @staticmethod
    def _env(name: str, default: str) -> str:
        import os

        return os.getenv(name, default)

    def _headers(self) -> Optional[Dict[str, str]]:
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers or None

    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> Dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout_sec,
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.ConnectionError:
                if attempt < self.max_retries - 1:
                    delay = self.initial_delay_sec * (2 ** attempt)
                    logger.warning(
                        "Judge connection failed. Retrying in %.1fs (attempt %d/%d)",
                        delay,
                        attempt + 1,
                        self.max_retries,
                    )
                    time.sleep(delay)
                else:
                    logger.error("Could not connect to judge endpoint after retries")
                    raise
            except Exception as exc:
                logger.error("Judge API error: %s", exc)
                raise
