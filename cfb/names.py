"""Utility helpers for normalising team/player names across data sources."""

from __future__ import annotations

import unicodedata


def normalize_ascii(text: str | None) -> str:
    """Return an upper-case ASCII string (empty string if input is falsy)."""

    if not text:
        return ""
    normalized = (
        unicodedata.normalize("NFKD", str(text))
        .encode("ascii", "ignore")
        .decode("ascii")
        .upper()
        .strip()
    )
    return " ".join(normalized.split())


def normalize_player(text: str | None) -> str:
    return normalize_ascii(text)


def normalize_team(text: str | None) -> str:
    return normalize_ascii(text)
