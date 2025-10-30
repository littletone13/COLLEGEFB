"""Shared injury parsing utilities for model adjustments."""

from __future__ import annotations

from typing import Tuple

OFFENSE_POSITIONS = {
    "QB",
    "RB",
    "TB",
    "HB",
    "FB",
    "WR",
    "TE",
    "LT",
    "LG",
    "C",
    "RG",
    "RT",
    "OL",
}

DEFENSE_POSITIONS = {
    "DL",
    "DE",
    "DT",
    "NT",
    "EDGE",
    "LB",
    "ILB",
    "OLB",
    "MLB",
    "CB",
    "DB",
    "S",
    "FS",
    "SS",
    "STAR",
    "NB",
}

STATUS_PENALTIES = {
    "out": 0.12,
    "suspended": 0.10,
    "doubtful": 0.08,
    "questionable": 0.04,
    "probable": 0.02,
}


def normalize_status(status: str, *, custom_text: str = "") -> str | None:
    """Normalise raw injury status strings into our canonical keys."""

    if not status:
        return None
    status_key = status.strip().lower()
    if status_key in STATUS_PENALTIES:
        return status_key

    text = custom_text.lower()
    if "ruled out" in text or "out for" in text:
        return "out"
    if "doubtful" in text:
        return "doubtful"
    if "questionable" in text:
        return "questionable"
    if "probable" in text or "available" in text or "upgraded to available" in text:
        return "probable"
    if "suspended" in text:
        return "suspended"
    return None


def penalties_for_player(status: str, position: str, *, custom_text: str = "") -> Tuple[float, float]:
    """Return offense/defense penalties for a player status + position."""

    normalized = normalize_status(status, custom_text=custom_text)
    if not normalized:
        return (0.0, 0.0)
    penalty = STATUS_PENALTIES.get(normalized, 0.0)
    if penalty <= 0.0:
        return (0.0, 0.0)
    pos = (position or "").strip().upper()
    offense_pen = penalty if pos in OFFENSE_POSITIONS else 0.0
    defense_pen = penalty if pos in DEFENSE_POSITIONS else 0.0
    return offense_pen, defense_pen


__all__ = [
    "OFFENSE_POSITIONS",
    "DEFENSE_POSITIONS",
    "STATUS_PENALTIES",
    "normalize_status",
    "penalties_for_player",
]
