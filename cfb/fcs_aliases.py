"""Reusable team alias helpers for FCS data sources."""

from __future__ import annotations

import re
from typing import Optional

RAW_TEAM_NAME_ALIASES = {
    "LONG ISLAND": "LIUSHAR",
    "LONG ISLAND UNIVERSITY": "LIUSHAR",
    "LONG ISLAND UNIVERSITY SHARKS": "LIUSHAR",
    "LIU": "LIUSHAR",
    "LIU SHARKS": "LIUSHAR",
    "STONEHILL": "STHLSK",
    "STONEHILL SKYHAWKS": "STHLSK",
    "STONEHILL COLLEGE": "STHLSK",
    "STONE HILL": "STHLSK",
    "NEW HAVEN": "NEWHVN",
    "NEW HAVEN CHARGERS": "NEWHVN",
    "UNIVERSITY OF NEW HAVEN": "NEWHVN",
    "RHODE ISLAND": "RHODE ISLD",
    "RHODE ISLAND RAMS": "RHODE ISLD",
    "NORTH DAKOTA": "N DAKOTA",
    "NORTH DAKOTA STATE": "N DAK ST",
    "NORTH DAKOTA ST": "N DAK ST",
    "NDSU": "N DAK ST",
    "SOUTH DAKOTA": "S DAKOTA",
    "SOUTH DAKOTA STATE": "S DAK ST",
    "SOUTH DAKOTA ST": "S DAK ST",
    "SDSU": "S DAK ST",
    "EAST TEXAS A&M": "TXAMCO",
    "EAST TEXAS A AND M": "TXAMCO",
    "E TEXAS A&M": "TXAMCO",
    "NORTH CAROLINA CENTRAL": "NC CENT",
    "NORTH CAROLINA A&T": "NC A&T",
    "NORTH CAROLINA A AND T": "NC A&T",
    "NORTH CAROLINA A AND T STATE": "NC A&T",
    "SOUTHEASTERN LOUISIANA": "SE LA",
    "HOUSTON CHRISTIAN": "HOUCHR",
    "HOUSTON BAPTIST": "HOUCHR",
    "VALPARAISO": "VALPO",
    "MOREHEAD STATE": "MOREHEAD",
    "WILLIAM & MARY": "WM & MARY",
    "WILLIAM AND MARY": "WM & MARY",
    "TENNESSEE TECH": "TENN TECH",
    "SOUTH CAROLINA STATE": "SCAR STATE",
    "SE MISSOURI STATE": "SE MO ST",
    "SE MISSOURI": "SE MO ST",
    "SAINT FRANCIS": "ST FRANCIS",
    "ST FRANCIS (PA)": "ST FRANCIS",
    "ROBERT MORRIS": "ROB MORRIS",
    "PRAIRIE VIEW A&M": "PRVIEW A&M",
    "PRAIRIE VIEW A AND M": "PRVIEW A&M",
    "BETHUNE COOKMAN": "BETH COOK",
    "ALABAMA A&M": "ALAB A&M",
    "ALABAMA STATE": "ALABAMA ST",
    "PORTLAND STATE": "PORTLAND",
    "NICHOLLS STATE": "NICHOLLS",
    "MCNEESE STATE": "MCNEESE",
    "GRAMBLING STATE": "GRAMBLING",
    "JACKSON STATE": "JACKSON ST",
    "STEPHEN F. AUSTIN": "STF AUSTIN",
    "STEPHEN F AUSTIN": "STF AUSTIN",
    "UTAH TECH": "UTAHTC",
    "UT RIO GRANDE VALLEY": "TXGV",
    "INCARNATE WORD": "INCAR WORD",
}

TEAM_NAME_ALIASES = {re.sub(r"[^A-Z0-9 ]", "", key.upper()).strip(): value for key, value in RAW_TEAM_NAME_ALIASES.items()}

DISPLAY_NAME_OVERRIDES = {
    "TXAMCO": "East Texas A&M",
    "STHLSK": "Stonehill",
}


def normalize_label(label: str) -> str:
    return re.sub(r"[^A-Z0-9 ]", "", label.upper()).strip()


def map_team(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    normalized = normalize_label(label)
    return TEAM_NAME_ALIASES.get(normalized)


__all__ = [
    "RAW_TEAM_NAME_ALIASES",
    "TEAM_NAME_ALIASES",
    "DISPLAY_NAME_OVERRIDES",
    "normalize_label",
    "map_team",
]
