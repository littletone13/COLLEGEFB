#!/usr/bin/env python3
"""Enhance baseline player-prop inputs with team context and variance estimates.

The script augments the simple baselines produced by
``scripts/generate_player_prop_baselines.py`` with richer distributional
information:

- Team-level pace (plays per game) and pass/rush rates derived from CFBD
  play-by-play, including per-game variance for use in Negative Binomial volume
  draws.
- Estimated player target / rush shares (and their kappas) derived from the
  baseline usage numbers and the team rates above.
- Optional caching and backoff logic so we can re-run the script rapidly without
  hammering the CFBD API.

Example usage:

    python scripts/enhance_player_prop_inputs.py \
        --year 2025 \
        --week 10 \
        --in out/player_prop_baselines_week10.csv \
        --out out/player_prop_inputs_week10_enhanced.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cfb.names import normalize_ascii, normalize_player, normalize_team
from cfb.props_enhancements import estimate_overdispersion_k

CFBD_BASE = "https://api.collegefootballdata.com"
DEFAULT_CACHE_DIR = Path("data/cache/cfbd_plays")
PASS_KEYWORDS = ("pass", "sack", "interception")
RUSH_KEYWORDS = ("rush", "run")
EXCLUDE_KEYWORDS = ("punt", "kick", "kneel", "spike")
TEAM_NAME_OVERRIDES: Dict[str, str] = {
    "SAN JOSÉ STATE": "San Jose State",
    "SAN JOSE ST": "San Jose State",
    "SAN JOSE STATE": "San Jose State",
    "SMU": "SMU",
    "BYU": "BYU",
    "UTEP": "UTEP",
    "UTSA": "UTSA",
    "UNLV": "UNLV",
    "UAB": "UAB",
    "FIU": "FIU",
    "FAU": "FAU",
    "UMASS": "UMass",
    "UCONN": "UConn",
    "MIAMI (OH)": "Miami (OH)",
    "MIAMI (FL)": "Miami",
    "MIAMI FL": "Miami",
    "MIAMI OH": "Miami (OH)",
    "PITT": "Pittsburgh",
    "PENN STATE": "Penn State",
    "NC STATE": "NC State",
    "NORTH CAROLINA": "North Carolina",
    "OLE MISS": "Ole Miss",
    "LA TECH": "Louisiana Tech",
    "LA.-MONROE": "Louisiana Monroe",
    "LA MONROE": "Louisiana Monroe",
    "LA.-LAFAYETTE": "Louisiana",
    "LOUISIANA-LAFAYETTE": "Louisiana",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enhance baseline CFB player prop projections with CFBD-derived context."
    )
    parser.add_argument("--year", type=int, required=True, help="Season year (e.g., 2025).")
    parser.add_argument("--week", type=int, required=True, help="Target regular-season week (1+).")
    parser.add_argument("--in", dest="in_path", type=Path, required=True, help="Baseline CSV input.")
    parser.add_argument("--out", dest="out_path", type=Path, required=True, help="Destination CSV path.")
    parser.add_argument(
        "--season-type",
        choices=["regular", "postseason"],
        default="regular",
        help="Season type for CFBD pulls (default regular).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory for cached CFBD play JSON payloads.",
    )
    parser.add_argument(
        "--max-week-lookback",
        type=int,
        default=None,
        help="Optional limit on number of historical weeks to consider (most recent first).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch data and compute summaries but skip writing the output CSV.",
    )
    return parser.parse_args()


def _require_api_key() -> str:
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        raise RuntimeError("Set CFBD_API_KEY in the environment before running the enhancer.")
    return key


def _slugify_team(name: str) -> str:
    ascii_upper = normalize_ascii(name)
    return ascii_upper.lower().replace(" ", "_").replace("&", "and")


def _canonical_cfbd_team(name: str) -> str:
    key = normalize_ascii(name)
    if key in TEAM_NAME_OVERRIDES:
        return TEAM_NAME_OVERRIDES[key]
    if key.endswith(" ST"):
        return key[:-3].title() + " State"
    if key.endswith("ST"):
        return key[:-2].title() + " State"
    return key.title()


def _request_cfbd(path: str, params: Dict[str, Any], api_key: str, *, attempts: int = 5) -> list[dict]:
    headers = {"Authorization": f"Bearer {api_key}"}
    for attempt in range(attempts):
        resp = requests.get(
            CFBD_BASE + path,
            headers=headers,
            params=params,
            timeout=30,
        )
        if resp.status_code == 429 and attempt < attempts - 1:
            sleep_for = 1.5 * (attempt + 1) + random.random()
            time.sleep(sleep_for)
            continue
        if resp.status_code == 401:
            raise RuntimeError("CFBD API rejected the provided key (401).")
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Exceeded retry budget fetching {path} with params {params}.")


def _load_cached_payload(path: Path) -> Optional[list[dict]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_cache(path: Path, payload: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def fetch_team_week_plays(
    *,
    api_key: str,
    cache_dir: Path,
    team: str,
    year: int,
    week: int,
    season_type: str,
) -> list[dict]:
    slug = _slugify_team(team)
    cache_path = cache_dir / f"plays_{year}_{season_type}_{slug}_wk{week}.json"
    payload = _load_cached_payload(cache_path)
    if payload is not None:
        return payload
    params = {"year": year, "week": week, "team": team, "seasonType": season_type}
    payload = _request_cfbd("/plays", params, api_key=api_key)
    _save_cache(cache_path, payload)
    return payload


def _classify_play(play_type: Optional[str]) -> Optional[str]:
    if not play_type:
        return None
    lower = play_type.lower()
    if any(keyword in lower for keyword in EXCLUDE_KEYWORDS):
        return None
    if any(keyword in lower for keyword in PASS_KEYWORDS):
        return "pass"
    if any(keyword in lower for keyword in RUSH_KEYWORDS):
        return "rush"
    return None


def summarise_team_history(
    *,
    team: str,
    cfbd_team: str,
    year: int,
    week: int,
    api_key: str,
    cache_dir: Path,
    season_type: str,
    max_week_lookback: Optional[int],
) -> dict[str, float]:
    if week <= 1:
        # No prior data; return NaNs so callers can fall back to neutral priors.
        return {
            "team_plays_mean": float("nan"),
            "team_plays_var": float("nan"),
            "team_pass_rate": float("nan"),
            "team_pass_rate_var": float("nan"),
            "team_pass_attempts_mean": float("nan"),
            "team_rush_attempts_mean": float("nan"),
            "team_sack_rate": float("nan"),
            "team_throwaway_rate": float("nan"),
            "team_scramble_rate": float("nan"),
            "team_pass_k": float("inf"),
            "team_rush_k": float("inf"),
        }

    plays_per_game: list[float] = []
    pass_rate_per_game: list[float] = []
    pass_attempts_per_game: list[float] = []
    rush_attempts_per_game: list[float] = []

    start_week = max(1, week - (max_week_lookback or week - 1))
    for wk in range(start_week, week):
        payload = fetch_team_week_plays(
            api_key=api_key,
            cache_dir=cache_dir,
            team=cfbd_team,
            year=year,
            week=wk,
            season_type=season_type,
        )
        # Guard against look-ahead: CFBD sometimes returns future games if week overlaps postseason.
        if wk >= week:
            raise AssertionError("Preventing look-ahead leakage from CFBD plays payload.")

        per_game_counts: dict[int, dict[str, int]] = defaultdict(
            lambda: {"pass": 0, "rush": 0, "sack": 0, "throwaway": 0, "scramble": 0}
        )
        for play in payload:
            if play.get("offense") != cfbd_team:
                continue
            play_type = str(play.get("playType") or "")
            play_text = str(play.get("playText") or "").lower()
            game_id = int(play.get("gameId"))

            lower_type = play_type.lower()
            if "sack" in lower_type:
                per_game_counts[game_id]["sack"] += 1
                continue

            classification = _classify_play(play_type)
            if classification is None:
                continue

            is_throwaway = "throw away" in play_text or "threw the ball away" in play_text
            is_scramble = "scramble" in lower_type or "scramble" in play_text

            if classification == "pass":
                per_game_counts[game_id]["pass"] += 1
                if is_throwaway:
                    per_game_counts[game_id]["throwaway"] += 1
            elif classification == "rush":
                per_game_counts[game_id]["rush"] += 1
                if is_scramble:
                    per_game_counts[game_id]["scramble"] += 1

        sacks_per_game: list[float] = []
        throwaways_per_game: list[float] = []
        scrambles_per_game: list[float] = []
        dropbacks_per_game: list[float] = []

        for counts in per_game_counts.values():
            total = counts["pass"] + counts["rush"] + counts["sack"]
            if total <= 0:
                continue
            plays_per_game.append(float(total))
            pass_attempts_per_game.append(float(counts["pass"]))
            rush_attempts_per_game.append(float(counts["rush"]))
            pass_rate_per_game.append(float(counts["pass"]) / float(total))
            sacks_per_game.append(float(counts["sack"]))
            throwaways_per_game.append(float(counts["throwaway"]))
            scrambles_per_game.append(float(counts["scramble"]))
            dropbacks_per_game.append(float(counts["pass"] + counts["sack"] + counts["scramble"]))

    if not plays_per_game:
        return {
            "team_plays_mean": float("nan"),
            "team_plays_var": float("nan"),
            "team_pass_rate": float("nan"),
            "team_pass_rate_var": float("nan"),
            "team_pass_attempts_mean": float("nan"),
            "team_rush_attempts_mean": float("nan"),
            "team_sack_rate": float("nan"),
            "team_throwaway_rate": float("nan"),
            "team_scramble_rate": float("nan"),
            "team_pass_k": float("inf"),
            "team_rush_k": float("inf"),
        }

    def _variance(values: Iterable[float]) -> float:
        if not values:
            return float("nan")
        if len(values) == 1:
            return 0.0
        return float(np.var(np.array(values, dtype=float), ddof=1))

    plays_arr = np.array(plays_per_game, dtype=float)
    pass_rate_arr = np.array(pass_rate_per_game, dtype=float)
    pass_attempts_arr = np.array(pass_attempts_per_game, dtype=float)
    rush_attempts_arr = np.array(rush_attempts_per_game, dtype=float)
    dropbacks_arr = np.array(dropbacks_per_game, dtype=float)
    sacks_arr = np.array(sacks_per_game, dtype=float)
    throwaways_arr = np.array(throwaways_per_game, dtype=float)
    scrambles_arr = np.array(scrambles_per_game, dtype=float)

    total_dropbacks = dropbacks_arr.sum()
    sack_rate = float(sacks_arr.sum() / total_dropbacks) if total_dropbacks > 0 else float("nan")
    throwaway_rate = float(throwaways_arr.sum() / total_dropbacks) if total_dropbacks > 0 else float("nan")
    scramble_rate = float(scrambles_arr.sum() / total_dropbacks) if total_dropbacks > 0 else float("nan")

    pass_k = estimate_overdispersion_k(pass_attempts_arr)
    rush_k = estimate_overdispersion_k(rush_attempts_arr)

    return {
        "team_plays_mean": float(np.mean(plays_arr)),
        "team_plays_var": _variance(plays_per_game),
        "team_pass_rate": float(np.mean(pass_rate_arr)),
        "team_pass_rate_var": _variance(pass_rate_per_game),
        "team_pass_attempts_mean": float(np.mean(pass_attempts_arr)),
        "team_rush_attempts_mean": float(np.mean(rush_attempts_arr)),
        "team_sack_rate": sack_rate,
        "team_throwaway_rate": throwaway_rate,
        "team_scramble_rate": scramble_rate,
        "team_pass_k": pass_k,
        "team_rush_k": rush_k,
    }


def _per_game_usage(row: pd.Series) -> float:
    usage = row.get("usage")
    games = row.get("games_played")
    if usage is None or games in (None, 0) or pd.isna(usage) or pd.isna(games):
        return float("nan")
    return float(usage) / float(games)


def _neg_bin_variance(mean: float, overdispersion: float = 0.25) -> float:
    if mean is None or pd.isna(mean):
        return float("nan")
    mean = max(float(mean), 0.0)
    return mean + overdispersion * (mean**2)


def _spread_to_win_prob(spread: Any) -> float:
    try:
        val = float(spread)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(val):
        return float("nan")
    # Simple logistic transform; cap to avoid degenerate 0/1
    win_prob = 1.0 / (1.0 + math.exp(val / 7.0))
    return float(min(max(win_prob, 0.01), 0.99))


def _normalise_player_name(name: str) -> str:
    return normalize_player(name)


def kappa_for_role(depth: Any, injury_q: Any, alpha_target: Any) -> float:
    base = 60.0
    if bool(injury_q):
        base -= 20.0
    try:
        depth_val = float(depth) if depth is not None and not pd.isna(depth) else 1.0
    except Exception:
        depth_val = 1.0
    if depth_val >= 4:
        base -= 15.0
    if bool(alpha_target):
        base += 20.0
    return float(max(15.0, min(120.0, base)))


METRIC_DEFAULTS: dict[str, float] = {
    "target_share_mean": float("nan"),
    "target_share_kappa": float("nan"),
    "tgt_mean": float("nan"),
    "tgt_var": float("nan"),
    "rush_share_mean": float("nan"),
    "rush_share_kappa": float("nan"),
    "rush_mean": float("nan"),
    "rush_var": float("nan"),
    "att_mean": float("nan"),
    "att_var": float("nan"),
    "catch_rate": float("nan"),
    "yds_per_rec_mu": float("nan"),
    "yds_per_rec_sd": float("nan"),
    "yds_per_rush_mu": float("nan"),
    "yds_per_rush_sd": float("nan"),
    "comp_rate": float("nan"),
    "yds_per_comp_mu": float("nan"),
    "yds_per_comp_sd": float("nan"),
    "receiving_zero_inflation": float("nan"),
    "win_prob_q4": float("nan"),
}

LINE_COLUMNS = [
    "line_receptions",
    "line_rec_yds",
    "line_rush_att",
    "line_rush_yds",
    "line_pass_att",
    "line_pass_comp",
    "line_pass_yds",
]

PRICE_COLUMNS = [
    "price_over_receptions",
    "price_over_rec_yds",
    "price_over_rush_att",
    "price_over_rush_yds",
    "price_over_pass_att",
    "price_over_pass_comp",
    "price_over_pass_yds",
]


def main() -> None:
    args = parse_args()
    api_key = _require_api_key()

    baseline = pd.read_csv(args.in_path)
    if baseline.empty:
        raise RuntimeError(f"Baseline input {args.in_path} is empty.")

    baseline["player"] = baseline["player"].map(normalize_player)
    baseline["team"] = baseline["team"].map(normalize_team)
    if "opponent" in baseline.columns:
        baseline["opponent"] = baseline["opponent"].map(normalize_team)
    baseline["kickoff_iso"] = baseline["start_dt"] if "start_dt" in baseline.columns else np.nan
    if "book" not in baseline.columns:
        baseline["book"] = ""
    else:
        baseline["book"] = baseline["book"].fillna("")
    if "notes" not in baseline.columns:
        baseline["notes"] = ""
    else:
        baseline["notes"] = baseline["notes"].fillna("")
    if "injury_q" not in baseline.columns:
        baseline["injury_q"] = False
    if "depth" not in baseline.columns:
        baseline["depth"] = 1
    if "alpha_target" not in baseline.columns:
        baseline["alpha_target"] = False
    baseline["team_cfbd"] = baseline["team"].astype(str).apply(_canonical_cfbd_team)

    team_context: dict[str, dict[str, float]] = {}
    for team_cfbd in sorted(baseline["team_cfbd"].unique()):
        stats = summarise_team_history(
            team=team_cfbd,
            cfbd_team=team_cfbd,
            year=args.year,
            week=args.week,
            api_key=api_key,
            cache_dir=args.cache_dir,
            season_type=args.season_type,
            max_week_lookback=args.max_week_lookback,
        )
        team_context[team_cfbd] = stats

    group_cols = ["game_id", "team", "player"]
    metric_lookup: dict[tuple, dict[str, dict[str, Any]]] = {}
    for key, group in baseline.groupby(group_cols):
        metric_lookup[key] = {rec["metric"]: rec for rec in group.to_dict("records")}

    enriched_rows: list[dict[str, Any]] = []
    for row in baseline.to_dict(orient="records"):
        cfbd_team = row["team_cfbd"]
        context = team_context.get(cfbd_team, {})
        per_game = _per_game_usage(pd.Series(row))

        entry: dict[str, Any] = dict(row)
        entry.update(context)
        entry["player_norm"] = _normalise_player_name(row.get("player", ""))
        entry["kickoff_iso"] = row.get("kickoff_iso", row.get("start_dt"))
        entry["book"] = row.get("book", "")
        entry["notes"] = row.get("notes", "")
        entry["wind_mph"] = row.get("wind_mph", 0.0)
        entry["precip_flag"] = row.get("precip_flag", 0.0)
        try:
            entry["wind_mph"] = float(entry["wind_mph"])
        except (TypeError, ValueError):
            entry["wind_mph"] = 0.0
        try:
            entry["precip_flag"] = float(entry["precip_flag"])
        except (TypeError, ValueError):
            entry["precip_flag"] = 0.0
        entry["win_prob_q4"] = _spread_to_win_prob(row.get("spread"))

        key = (row.get("game_id"), row.get("team"), row.get("player"))
        metrics_for_player = metric_lookup.get(key, {})

        metric = row.get("metric")
        if metric == "receiving_yards" or metric == "receptions":
            team_pass_att = context.get("team_pass_attempts_mean") or float("nan")
            if not math.isnan(per_game) and team_pass_att and team_pass_att > 0:
                entry["target_share_mean"] = min(per_game / team_pass_att, 0.95)
            else:
                entry["target_share_mean"] = float("nan")
            entry["tgt_mean"] = per_game
            entry["tgt_var"] = _neg_bin_variance(per_game)
            entry["target_share"] = entry.get("target_share_mean")
            share_for_zi = entry.get("target_share_mean")
            if share_for_zi is not None and not pd.isna(share_for_zi):
                entry["receiving_zero_inflation"] = max(0.0, 0.5 - 1.2 * float(share_for_zi))
            season_targets = row.get("season_targets")
            season_receptions = row.get("season_receptions")
            recep_row = metrics_for_player.get("receptions")
            if pd.isna(season_targets) and recep_row is not None:
                season_targets = recep_row.get("season_targets")
            if pd.isna(season_receptions) and recep_row is not None:
                season_receptions = recep_row.get("season_receptions")
            if season_targets and not pd.isna(season_targets) and season_receptions and not pd.isna(season_receptions):
                try:
                    entry["catch_rate"] = float(season_receptions) / float(season_targets) if season_targets else float("nan")
                except ZeroDivisionError:
                    entry["catch_rate"] = float("nan")
            yards_source = metrics_for_player.get("receiving_yards", row)
            yards_total = yards_source.get("season_total")
            if yards_total and season_receptions and not pd.isna(season_receptions) and season_receptions > 0:
                val = float(yards_total) / float(season_receptions)
                val = max(val, 0.0)
                entry["yds_per_rec_mu"] = val
                entry["yds_per_rec_sd"] = max(max(val, 1.0) * 0.45, 5.0)
            if entry.get("catch_rate") in (None, "") or (isinstance(entry.get("catch_rate"), float) and pd.isna(entry.get("catch_rate"))):
                entry["catch_rate"] = 0.6
        elif metric == "rushing_yards":
            team_rush_att = context.get("team_rush_attempts_mean") or float("nan")
            if not math.isnan(per_game) and team_rush_att and team_rush_att > 0:
                entry["rush_share_mean"] = min(per_game / team_rush_att, 0.95)
            else:
                entry["rush_share_mean"] = float("nan")
            entry["rush_mean"] = per_game
            entry["rush_var"] = _neg_bin_variance(per_game)
            entry["rush_attempt_share"] = entry.get("rush_share_mean")
            season_attempts = row.get("season_attempts")
            season_total = row.get("season_total")
            if season_attempts and not pd.isna(season_attempts) and season_attempts > 0:
                entry["yds_per_rush_mu"] = float(season_total) / float(season_attempts)
                entry["yds_per_rush_sd"] = max(entry["yds_per_rush_mu"] * 0.5, 5.0)
        elif metric == "passing_yards":
            team_pass_att = context.get("team_pass_attempts_mean") or float("nan")
            if not math.isnan(per_game) and team_pass_att and team_pass_att > 0:
                entry["pass_attempt_share"] = min(per_game / team_pass_att, 1.0)
            else:
                entry["pass_attempt_share"] = float("nan")
            entry["att_mean"] = per_game
            entry["att_var"] = _neg_bin_variance(per_game)
            season_attempts = row.get("season_attempts")
            season_completions = row.get("season_completions")
            season_yards = row.get("season_total")
            if season_attempts and not pd.isna(season_attempts) and season_attempts > 0:
                try:
                    entry["comp_rate"] = float(season_completions) / float(season_attempts)
                except Exception:
                    entry["comp_rate"] = float("nan")
            if season_completions and not pd.isna(season_completions) and season_completions > 0:
                entry["yds_per_comp_mu"] = float(season_yards) / float(season_completions)
                entry["yds_per_comp_sd"] = max(entry["yds_per_comp_mu"] * 0.45, 7.0)

        for key_name, default in METRIC_DEFAULTS.items():
            val = entry.get(key_name, default)
            if val in ("", None) or (isinstance(val, float) and pd.isna(val)):
                entry[key_name] = default

        metric_name = str(row.get("metric", "")).lower()
        baseline_val = row.get("baseline")

        def _maybe_set_line(key: str, value: Any) -> None:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return
            if key not in entry or entry[key] in ("", None) or (isinstance(entry[key], float) and pd.isna(entry[key])):
                entry[key] = value

        if metric_name == "receiving_yards":
            _maybe_set_line("line_rec_yds", baseline_val)
            rec_row = metrics_for_player.get("receptions")
            rec_baseline = rec_row.get("baseline") if rec_row else None
            if (rec_baseline is None or (isinstance(rec_baseline, float) and pd.isna(rec_baseline))) and entry.get("tgt_mean") and entry.get("catch_rate"):
                try:
                    rec_baseline = float(entry["tgt_mean"]) * float(entry["catch_rate"])
                except Exception:
                    rec_baseline = None
            _maybe_set_line("line_receptions", rec_baseline)
        elif metric_name == "receptions":
            _maybe_set_line("line_receptions", baseline_val)
        elif metric_name == "rushing_yards":
            _maybe_set_line("line_rush_yds", baseline_val)
        elif metric_name == "passing_yards":
            _maybe_set_line("line_pass_yds", baseline_val)
            comp_line = None
            if entry.get("att_mean") and entry.get("comp_rate"):
                try:
                    comp_line = float(entry["att_mean"]) * float(entry["comp_rate"])
                except Exception:
                    comp_line = None
            _maybe_set_line("line_pass_comp", comp_line)
            _maybe_set_line("line_pass_att", entry.get("att_mean"))
        elif metric_name == "passing_attempts":
            entry["att_mean"] = per_game
            entry["att_var"] = _neg_bin_variance(per_game)
            _maybe_set_line("line_pass_att", baseline_val)
        elif metric_name == "passing_completions":
            _maybe_set_line("line_pass_comp", baseline_val)
        elif metric_name in ("rush_attempts", "rushing_attempts"):
            _maybe_set_line("line_rush_att", baseline_val)
        enriched_rows.append(entry)

    enriched = pd.DataFrame(enriched_rows)
    for col in LINE_COLUMNS + PRICE_COLUMNS:
        if col not in enriched.columns:
            enriched[col] = np.nan

    enriched["wind_mph"] = enriched.get("wind_mph", 0.0).fillna(0.0).astype(float)
    enriched["precip_flag"] = enriched.get("precip_flag", 0.0).fillna(0.0).astype(float)

    pass_adj = 1.0 - 0.02 * np.maximum(0.0, (enriched["wind_mph"] - 10.0) / 10.0)
    rush_adj = 1.0 + 0.01 * enriched["precip_flag"]
    for col in ("yds_per_comp_mu", "yds_per_rec_mu"):
        if col in enriched.columns:
            enriched[col] = enriched[col].astype(float) * pass_adj
    if "yds_per_rush_mu" in enriched.columns:
        enriched["yds_per_rush_mu"] = enriched["yds_per_rush_mu"].astype(float) * rush_adj

    enriched["depth"] = pd.to_numeric(enriched.get("depth", 1), errors="coerce").fillna(1.0)
    enriched["injury_q"] = enriched.get("injury_q", False).fillna(False)
    enriched["alpha_target"] = enriched.get("alpha_target", False).fillna(False)
    enriched["target_share_kappa"] = enriched.apply(
        lambda r: kappa_for_role(r.get("depth"), r.get("injury_q"), r.get("alpha_target")), axis=1
    )
    enriched["rush_share_kappa"] = enriched["target_share_kappa"]

    enriched["kickoff_iso"] = enriched.get("kickoff_iso").fillna(enriched.get("start_dt"))
    enriched["book"] = enriched.get("book", "").fillna("")
    enriched["notes"] = enriched.get("notes", "").fillna("")

    enriched.sort_values(["start_dt", "team", "player", "metric"], inplace=True)

    required_columns = [
        "player",
        "team",
        "opponent",
        "team_plays_mean",
        "team_plays_var",
        "team_pass_rate",
        "team_pass_rate_var",
        "target_share_mean",
        "target_share_kappa",
        "tgt_mean",
        "rush_share_mean",
        "rush_share_kappa",
        "rush_mean",
        "catch_rate",
        "yds_per_rec_mu",
        "yds_per_rec_sd",
        "yds_per_rush_mu",
        "yds_per_rush_sd",
        "att_mean",
        "comp_rate",
        "yds_per_comp_mu",
        "yds_per_comp_sd",
        *LINE_COLUMNS,
        *PRICE_COLUMNS,
        "kickoff_iso",
        "book",
        "notes",
    ]

    ordered_cols = required_columns + [col for col in enriched.columns if col not in required_columns]
    out_df = enriched.reindex(columns=ordered_cols)

    string_cols = LINE_COLUMNS + PRICE_COLUMNS + ["book", "notes"]
    for col in string_cols:
        if col in out_df.columns:
            out_df[col] = out_df[col].fillna("")

    required = {"player", "team", "team_plays_mean", "team_pass_rate"}
    missing = required - set(out_df.columns)
    assert not missing, f"Enhancer missing columns: {missing}"
    if "team_pass_rate" in out_df.columns:
        mask = out_df["team_pass_rate"].notna()
        assert out_df.loc[mask, "team_pass_rate"].between(0, 1).all(), "team_pass_rate outside [0,1]"
    if "yds_per_rec_mu" in out_df.columns:
        mask = out_df["yds_per_rec_mu"].notna()
        assert (out_df.loc[mask, "yds_per_rec_mu"] >= 0).all(), "yds_per_rec_mu negative"

    enriched.sort_values(["start_dt", "team", "player", "metric"], inplace=True)
    if args.dry_run:
        print("[dry-run] Skipping write; preview of columns:")
        print(out_df.head())
        return
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_path, index=False)
    print(f"[info] Enhanced player props → {args.out_path} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
