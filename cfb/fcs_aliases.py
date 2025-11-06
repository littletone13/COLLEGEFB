"""Reusable team alias helpers for FCS data sources."""

from __future__ import annotations

import re
from typing import Optional

RAW_TEAM_NAME_ALIASES = {
    "ABILENE CHRISTIAN": "ABILENE CH",
    "ARKANSAS PINE BLUFF": "ARKAPB",
    "CENTRAL ARKANSAS": "CENT ARK",
    "CENTRAL CONNECTICUT STATE": "CENT CT ST",
    "CHARLESTON SOUTHERN": "CHARLES SO",
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
    "MOREHEAD": "MOREHEADSTATE",
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
    "EASTERN ILLINOIS": "E ILLINOIS",
    "EAST TENNESSEE STATE": "E TENN ST",
    "FLORIDA A&M": "FL A&M",
    "MISSISSIPPI VALLEY STATE": "MS VLY ST",
    "NORTHERN IOWA": "N IOWA",
    "NORTHWESTERN STATE": "NWSTATE",
    "PENNSYLVANIA": "PENN",
    "SE LOUISIANA": "SE LA",
    "SOUTHEAST LOUISIANA": "SE LA",
    "SOUTHEAST MISSOURI STATE": "SE MO ST",
    "SOUTHERN UTAH": "SO UTAH",
    "ST THOMAS": "STTHOM",
    "ST. THOMAS": "STTHOM",
    "TARLETON STATE": "TARLETON",
    "UT MARTIN": "TN MARTIN",
    "TENNESSEE-MARTIN": "TN MARTIN",
    "TENNESSEE MARTIN": "TN MARTIN",
    "VIRGINIA MILITARY INSTITUTE": "VA MILT IN",
    "VMI": "VA MILT IN",
    "CENTRAL CONNECTICUT": "CENT CT ST",
    "CENTRAL CONN STATE": "CENT CT ST",
    "CENT CONN STATE": "CENT CT ST",
    "ST THOMAS MN": "STTHOM",
    "ST THOMAS (MN)": "STTHOM",
    "WEBER STATE": "WEBER ST",
    "MONTANA STATE": "MONTANA ST",
    "SOUTHERN ILLINOIS": "S ILLINOIS",
    "NORTHERN ARIZONA": "N ARIZONA",
    "CAL POLY": "CAL POLY",
    "CAL. POLY - SLO": "CAL POLY",
    "CAL POLY SLO": "CAL POLY",
    "CAL POLY  SLO": "CAL POLY",
    "SOUTHERN UNIVERSITY": "SOUTHERN",
    "TENNESSEE STATE": "TENN ST",
    "EASTERN WASHINGTON": "E WASH",
    "NORTHERN COLORADO": "N COLORADO",
}

TEAM_NAME_ALIASES = {re.sub(r"[^A-Z0-9 ]", "", key.upper()).strip(): value for key, value in RAW_TEAM_NAME_ALIASES.items()}

DISPLAY_NAME_OVERRIDES = {
    "TXAMCO": "East Texas A&M",
    "STHLSK": "Stonehill",
}


def _to_oddslogic_key(label: str) -> str:
    """Collapse a label into the OddsLogic key format (alnum only)."""
    return "".join(ch for ch in label.upper() if ch.isalnum())


_ODDSLOGIC_ALIAS = {}
for long_label, short_label in RAW_TEAM_NAME_ALIASES.items():
    short_key = _to_oddslogic_key(short_label)
    long_key = _to_oddslogic_key(long_label)
    # Preserve the first mapping encountered; later aliases shouldn't overwrite.
    _ODDSLOGIC_ALIAS.setdefault(short_key, long_key)


def normalize_label(label: str) -> str:
    cleaned = label.upper().replace("-", " ")
    cleaned = re.sub(r"[^A-Z0-9 ]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def map_team(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    normalized = normalize_label(label)
    return TEAM_NAME_ALIASES.get(normalized)


def oddslogic_key(label: str) -> str:
    """
    Return the OddsLogic-normalised key for the given team label.

    This maps common abbreviations (e.g. ``PENN``) back to the identifiers
    used by OddsLogic (``PENNSYLVANIA``) so live market lookups can resolve.
    """
    base = _to_oddslogic_key(label)
    return _ODDSLOGIC_ALIAS.get(base, base)


__all__ = [
    "RAW_TEAM_NAME_ALIASES",
    "TEAM_NAME_ALIASES",
    "DISPLAY_NAME_OVERRIDES",
    "normalize_label",
    "map_team",
    "oddslogic_key",
]
