"""CFBD API helpers shared across models."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests

BASE_URL = "https://api.collegefootballdata.com"


class CFBDClient:
    """Minimal helper around CFBD HTTP endpoints."""

    def __init__(self, api_key: Optional[str] = None, *, timeout: int = 30) -> None:
        api_key = api_key or os.environ.get("CFBD_API_KEY")
        if not api_key:
            raise RuntimeError(
                "CFBD API key not provided. Set CFBD_API_KEY env var or pass api_key."  # noqa: EM101
            )
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.timeout = timeout

    def get(self, path: str, **params: Any) -> list[dict]:
        response = requests.get(
            BASE_URL + path,
            headers=self.headers,
            params=params,
            timeout=self.timeout,
        )
        if response.status_code == 401:
            raise RuntimeError(
                "CFBD API rejected the key (401). Double-check the token and Bearer prefix."  # noqa: EM101
            )
        response.raise_for_status()
        return response.json()
