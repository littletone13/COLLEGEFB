#!/usr/bin/env python3
"""
Generate baseline player prop projections for upcoming FBS games.

The projections are intentionally simple: they use season-to-date PFF
summary totals and convert them into per-game averages for the primary
contributors on each team (starting QB, top rushers, and top receivers).

Outputs:
    out/player_prop_baselines_week10.csv  (configurable via CLI)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from difflib import get_close_matches

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cfb.names import normalize_ascii, normalize_player, normalize_team
from cfb.io import cached_cfbd
from cfb.config import load_config


CONFIG = load_config()
FBS_CONFIG = CONFIG.get("fbs", {}) if isinstance(CONFIG.get("fbs"), dict) else {}
DATA_CONFIG = FBS_CONFIG.get("data", {}) if isinstance(FBS_CONFIG.get("data"), dict) else {}


def _resolve_path(value: Optional[str], default: Path) -> Path:
    if not value:
        return default
    text = str(value).strip()
    if not text or text.lower() == "null":
        return default
    return Path(text).expanduser()


DEFAULT_PFF_DIR = _resolve_path(DATA_CONFIG.get("pff_dir"), Path("data/pff/fbs"))
PASSING_PATH_DEFAULT = (DEFAULT_PFF_DIR / "passing_summary_FBS_FCS_PostWk10.csv")
RECEIVING_PATH_DEFAULT = (DEFAULT_PFF_DIR / "receiving_summary_FBS_FCS_PostWk10.csv")
RUSHING_PATH_DEFAULT = (DEFAULT_PFF_DIR / "rushing_summary_FBS_FCS_PostWk10.csv")
DEFAULT_GAMES_PATH = Path("sims_week10_fbs_2025.csv")
DEFAULT_OUTPUT_PATH = Path("out/player_prop_baselines_week10.csv")

TEAM_OVERRIDES: dict[str, str] = {
    "GEORGIA TECH": "GA TECH",
    "WAKE FOREST": "WAKE",
    "SAN JOSE STATE": "S JOSE ST",
    "SAN JOSE ST": "S JOSE ST",
    "SAN JOSÉ STATE": "S JOSE ST",
    "SAN JOSE": "S JOSE ST",
    "MASSACHUSETTS": "UMASS",
    "NORTHERN ILLINOIS": "N ILLINOIS",
    "SOUTH FLORIDA": "USF",
    "MISSISSIPPI STATE": "MISS STATE",
    "FLORIDA ATLANTIC": "FAU",
    "FLORIDA INTERNATIONAL": "FIU",
    "JACKSONVILLE STATE": "JVILLE ST",
    "MIDDLE TENNESSEE": "MIDDLE TN",
    "LOUISIANA TECH": "LA TECH",
    "LOUISIANA": "LA LAFAYET",
    "CALIFORNIA": "CAL",
}

RUSH_POSITIONS = {"HB", "RB", "TB", "FB"}
RECEIVING_POSITIONS = {"WR", "TE", "HB", "RB"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create baseline FBS player prop projections.")
    parser.add_argument("--games", type=Path, default=DEFAULT_GAMES_PATH, help="Path to FBS sims CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Destination CSV path.")
    parser.add_argument("--top-rushers", type=int, default=2, help="Number of rushers per team to include.")
    parser.add_argument("--top-receivers", type=int, default=3, help="Number of receivers per team to include.")
    parser.add_argument("--year", type=int, help="Season year (optional, informational).")
    parser.add_argument("--week", type=int, help="Target week (optional, informational).")
    parser.add_argument("--passing-path", type=Path, default=PASSING_PATH_DEFAULT, help="Path to passing summary CSV.")
    parser.add_argument("--receiving-path", type=Path, default=RECEIVING_PATH_DEFAULT, help="Path to receiving summary CSV.")
    parser.add_argument("--rushing-path", type=Path, default=RUSHING_PATH_DEFAULT, help="Path to rushing summary CSV.")
    return parser.parse_args()


def resolve_source(path: Path, patterns: Iterable[str]) -> Path:
    path = path.expanduser()
    if path.exists():
        return path
    directory = path.parent
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"Could not locate data file matching {patterns} in {directory}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _normalise_team(value: str | float) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = normalize_ascii(str(value))
    replacements = {
        "&": "AND",
        "ST.": "ST",
        "MTN": "MOUNTAIN",
        "NO.": "NORTH",
        "SOUTHERN MISS": "S MISS",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.strip()


def build_team_lookup(dataframes: Iterable[pd.DataFrame]) -> set[str]:
    teams: set[str] = set()
    for df in dataframes:
        if "team_name" in df.columns:
            teams.update(df["team_name"].dropna().astype(str).str.upper().unique())
    return teams


def match_team(name: str, known: set[str]) -> Optional[str]:
    candidate = _normalise_team(name)
    if not candidate:
        return None
    if candidate in TEAM_OVERRIDES:
        candidate = TEAM_OVERRIDES[candidate]
    if candidate in known:
        return candidate
    candidate_alt = candidate.replace(" STATE", " ST")
    if candidate_alt in known:
        return candidate_alt
    candidate_alt = candidate.replace("SAINT", "ST")
    if candidate_alt in known:
        return candidate_alt
    matches = get_close_matches(candidate, list(known), n=1, cutoff=0.75)
    return matches[0] if matches else None


@dataclass
class GameContext:
    game_id: str
    start_dt: datetime
    team: str
    opponent: str
    team_key: str
    spread: float
    favored: bool


def load_games(path: Path, known: set[str]) -> list[GameContext]:
    if not path.exists():
        raise FileNotFoundError(f"Games file not found: {path}")
    games = pd.read_csv(path)
    if games.empty:
        return []
    games["start_dt"] = pd.to_datetime(games["start_date"], utc=True, errors="coerce")
    now = datetime.now(timezone.utc)
    upcoming = games[games["start_dt"] >= now].copy()
    if upcoming.empty:
        return []
    contexts: list[GameContext] = []
    for row in upcoming.itertuples(index=False):
        spread_value = getattr(row, "spread_home_minus_away", 0.0)
        try:
            spread_value = float(spread_value)
        except (TypeError, ValueError):
            spread_value = 0.0
        if pd.isna(spread_value):
            spread_value = 0.0
        for team, opponent in ((row.home_team, row.away_team), (row.away_team, row.home_team)):
            team_key = match_team(team, known)
            if not team_key:
                print(f"[warn] Unable to match team '{team}' to player datasets.", file=sys.stderr)
                continue
            team_is_home = str(team) == str(row.home_team)
            team_spread = float(spread_value if team_is_home else -spread_value)
            contexts.append(
                GameContext(
                    game_id=str(row.game_id),
                    start_dt=row.start_dt,
                    team=str(team),
                    opponent=str(opponent),
                    team_key=team_key,
                    spread=team_spread,
                    favored=team_spread < 0.0,
                )
            )
    return contexts


def per_game(value: float, games_played: float) -> float:
    if games_played is None or games_played == 0 or pd.isna(games_played):
        return float("nan")
    return float(value) / float(games_played)


def select_qb(df: pd.DataFrame) -> Optional[pd.Series]:
    if df.empty:
        return None
    df = df.sort_values(["attempts", "yards"], ascending=[False, False])
    return df.iloc[0]


def select_top(df: pd.DataFrame, metric: str, top_n: int) -> pd.DataFrame:
    if df.empty or top_n <= 0:
        return pd.DataFrame(columns=df.columns)
    return df.sort_values([metric, "yards"], ascending=[False, False]).head(top_n)


def shrink_by_games(value: float, games: float, prior: float, strength: float = 3.0) -> float:
    if value is None or pd.isna(value):
        return float(prior) if not pd.isna(prior) else float("nan")
    if games is None or pd.isna(games) or games <= 0:
        return float(prior) if not pd.isna(prior) else float(value)
    if prior is None or pd.isna(prior):
        return float(value)
    weight = float(games) / float(games + strength)
    return float(weight * value + (1.0 - weight) * prior)


def _flatten_advanced_team(df: pd.DataFrame) -> pd.DataFrame:
    records = df.to_dict("records")
    flat = pd.json_normalize(records, sep="_")
    renamed = {
        col: col.replace("offense_", "adv_off_").replace("defense_", "adv_def_")
        for col in flat.columns
    }
    flat = flat.rename(columns=renamed)
    flat["team_key"] = flat["team"].apply(_normalise_team)
    return flat


def build_team_modifiers(year: int) -> tuple[pd.DataFrame, dict[str, float]]:
    try:
        advanced_df = cached_cfbd.load_advanced_team(year, season_type="regular")
    except Exception:
        return pd.DataFrame(), {}
    if advanced_df.empty:
        return pd.DataFrame(), {}
    flat = _flatten_advanced_team(advanced_df)
    flat = flat.set_index("team_key")
    stats = {
        "adv_off_success_mean": flat.get("adv_off_success_rate", pd.Series(dtype=float)).mean(),
        "adv_def_success_mean": flat.get("adv_def_success_rate", pd.Series(dtype=float)).mean(),
    }
    return flat, stats


def matchup_factor(
    team_key: str,
    opp_key: str,
    team_adv: pd.DataFrame,
    stats: dict[str, float],
) -> float:
    if team_adv.empty or team_key not in team_adv.index:
        return 1.0
    team_row = team_adv.loc[team_key]
    off_success = float(team_row.get("adv_off_success_rate", 0.0) or 0.0)
    league_off = stats.get("adv_off_success_mean") or 0.0
    if opp_key and opp_key in team_adv.index:
        opp_row = team_adv.loc[opp_key]
        opp_def = float(opp_row.get("adv_def_success_rate", 0.0) or 0.0)
    else:
        opp_def = stats.get("adv_def_success_mean") or 0.0
    league_def = stats.get("adv_def_success_mean") or 0.0
    if league_off <= 0 or league_def <= 0 or opp_def <= 0:
        return 1.0
    raw = (off_success / league_off) * (league_def / opp_def)
    return float(max(0.6, min(1.6, raw)))


def main() -> None:
    args = parse_args()

    passing_path = resolve_source(args.passing_path, ["passing_summary*.csv"])
    receiving_path = resolve_source(args.receiving_path, ["receiving_summary*.csv"])
    rushing_path = resolve_source(args.rushing_path, ["rushing_summary*.csv"])

    passing = pd.read_csv(passing_path)
    receiving = pd.read_csv(receiving_path)
    rushing = pd.read_csv(rushing_path)

    for frame in (passing, receiving, rushing):
        frame["team_key"] = frame["team_name"].astype(str).str.upper()

    known_teams = build_team_lookup((passing, receiving, rushing))
    games = load_games(args.games, known_teams)
    if not games:
        print("[info] No upcoming games detected; nothing to do.")
        return

    season_year = args.year
    if season_year is None:
        season_year = games[0].start_dt.year if games else datetime.now(timezone.utc).year
    team_adv_lookup, adv_stats = build_team_modifiers(season_year)

    rows: list[dict] = []
    for ctx in games:
        opponent_key = _normalise_team(ctx.opponent)
        ctx_factor = matchup_factor(ctx.team_key, opponent_key, team_adv_lookup, adv_stats)

        # Passing projections (QB)
        team_pass = passing[passing["team_key"] == ctx.team_key]
        team_pass_prior = per_game(
            team_pass.get("yards", 0.0).sum(),
            team_pass.get("player_game_count", 0.0).sum(),
        )
        team_pass_prior_att = per_game(
            team_pass.get("attempts", 0.0).sum(),
            team_pass.get("player_game_count", 0.0).sum(),
        )
        team_pass_prior_comp = per_game(
            team_pass.get("completions", 0.0).sum(),
            team_pass.get("player_game_count", 0.0).sum(),
        )

        team_rush = rushing[rushing["team_key"] == ctx.team_key]
        rush_subset = team_rush[team_rush["position"].isin(RUSH_POSITIONS)]
        team_rush_prior = per_game(
            team_rush.get("yards", 0.0).sum(),
            team_rush.get("player_game_count", 0.0).sum(),
        )
        team_rush_prior_att = per_game(
            team_rush.get("attempts", 0.0).sum(),
            team_rush.get("player_game_count", 0.0).sum(),
        )

        qb = select_qb(team_pass)
        if qb is not None:
            qb_games = qb.get("player_game_count", 0)

            def _qb_payload(metric: str, baseline: float, season_total: float, extra: dict[str, float] | None = None) -> dict:
                values = {
                    "game_id": ctx.game_id,
                    "start_dt": ctx.start_dt.isoformat(),
                    "team": ctx.team,
                    "opponent": ctx.opponent,
                    "player": qb["player"],
                    "position": qb["position"],
                    "metric": metric,
                    "baseline": round(baseline or 0.0, 1),
                    "season_total": season_total,
                    "usage": qb.get("attempts", float("nan")),
                    "games_played": qb_games,
                    "season_attempts": qb.get("attempts", float("nan")),
                    "season_completions": qb.get("completions", float("nan")),
                    "season_tds": qb.get("touchdowns", float("nan")),
                    "depth": 1,
                    "alpha_target": True,
                    "injury_q": False,
                    "spread": ctx.spread,
                    "favored": ctx.favored,
                }
                if extra:
                    values.update(extra)
                return values

            pass_yards_raw = per_game(qb.get("yards", 0.0), qb_games)
            pass_yards_adj = shrink_by_games(pass_yards_raw, qb_games, team_pass_prior, strength=2.0)
            pass_yards_adj = (pass_yards_adj or 0.0) * ctx_factor
            rows.append(_qb_payload("passing_yards", pass_yards_adj, qb.get("yards", float("nan"))))

            attempts_raw = per_game(qb.get("attempts", 0.0), qb_games)
            attempts_adj = shrink_by_games(attempts_raw, qb_games, team_pass_prior_att, strength=2.0)
            attempts_adj = (attempts_adj or 0.0) * ctx_factor
            rows.append(_qb_payload("passing_attempts", attempts_adj, qb.get("attempts", float("nan"))))

            completions_raw = per_game(qb.get("completions", 0.0), qb_games)
            completions_adj = shrink_by_games(completions_raw, qb_games, team_pass_prior_comp, strength=2.0)
            completions_adj = (completions_adj or 0.0) * ctx_factor
            rows.append(_qb_payload("passing_completions", completions_adj, qb.get("completions", float("nan"))))

            qb_rush = team_rush[team_rush["player"] == qb["player"]]
            if not qb_rush.empty:
                qb_r = qb_rush.iloc[0]
                qb_r_games = qb_r.get("player_game_count", 0)
                qb_rush_yards = per_game(qb_r.get("yards", 0.0), qb_r_games)
                qb_rush_yards = shrink_by_games(qb_rush_yards, qb_r_games, team_rush_prior)
                qb_rush_yards = (qb_rush_yards or 0.0) * ctx_factor
                qb_rush_att = per_game(qb_r.get("attempts", 0.0), qb_r_games)
                qb_rush_att = shrink_by_games(qb_rush_att, qb_r_games, team_rush_prior_att)
                qb_rush_att = (qb_rush_att or 0.0) * ctx_factor
                rows.append(
                    _qb_payload(
                        "rushing_yards",
                        qb_rush_yards,
                        qb_r.get("yards", float("nan")),
                        {"season_attempts": qb_r.get("attempts", float("nan"))},
                    )
                )
                rows.append(
                    _qb_payload(
                        "rushing_attempts",
                        qb_rush_att,
                        qb_r.get("attempts", float("nan")),
                        {"season_attempts": qb_r.get("attempts", float("nan"))},
                    )
                )

        rushers = select_top(rush_subset, "attempts", args.top_rushers)
        if rushers.empty:
            rushers = select_top(team_rush, "attempts", min(args.top_rushers, 1))
        for depth_idx, rusher in enumerate(rushers.itertuples(index=False), start=1):
            baseline_raw = per_game(rusher.yards, rusher.player_game_count)
            baseline_adj = shrink_by_games(baseline_raw, rusher.player_game_count, team_rush_prior)
            baseline_adj = (baseline_adj or 0.0) * ctx_factor
            attempts_baseline_raw = per_game(rusher.attempts, rusher.player_game_count)
            attempts_baseline = shrink_by_games(attempts_baseline_raw, rusher.player_game_count, team_rush_prior_att)
            attempts_baseline = (attempts_baseline or 0.0) * ctx_factor
            rows.append(
                {
                    "game_id": ctx.game_id,
                    "start_dt": ctx.start_dt.isoformat(),
                    "team": ctx.team,
                    "opponent": ctx.opponent,
                    "player": rusher.player,
                    "position": rusher.position,
                    "metric": "rushing_yards",
                    "baseline": round(baseline_adj or 0.0, 1),
                    "season_total": rusher.yards,
                    "usage": rusher.attempts,
                    "games_played": rusher.player_game_count,
                    "season_attempts": rusher.attempts,
                    "season_tds": getattr(rusher, "touchdowns", float("nan")),
                    "depth": depth_idx,
                    "alpha_target": depth_idx == 1,
                    "injury_q": False,
                    "spread": ctx.spread,
                    "favored": ctx.favored,
                }
            )
            rows.append(
                {
                    "game_id": ctx.game_id,
                    "start_dt": ctx.start_dt.isoformat(),
                    "team": ctx.team,
                    "opponent": ctx.opponent,
                    "player": rusher.player,
                    "position": rusher.position,
                    "metric": "rushing_attempts",
                    "baseline": round(attempts_baseline or 0.0, 1),
                    "season_total": rusher.attempts,
                    "usage": rusher.attempts,
                    "games_played": rusher.player_game_count,
                    "season_attempts": rusher.attempts,
                    "season_tds": getattr(rusher, "touchdowns", float("nan")),
                    "depth": depth_idx,
                    "alpha_target": depth_idx == 1,
                    "injury_q": False,
                    "spread": ctx.spread,
                    "favored": ctx.favored,
                }
            )

        # Receiving projections (yards + receptions)
        team_recv = receiving[receiving["team_key"] == ctx.team_key]
        recv_subset = team_recv[team_recv["position"].isin(RECEIVING_POSITIONS)]
        team_recv_prior_yards = per_game(
            team_recv.get("yards", 0.0).sum(),
            team_recv.get("player_game_count", 0.0).sum(),
        )
        team_recv_prior_recs = per_game(
            team_recv.get("receptions", 0.0).sum(),
            team_recv.get("player_game_count", 0.0).sum(),
        )
        receivers = select_top(recv_subset, "targets", args.top_receivers)
        if receivers.empty:
            receivers = select_top(team_recv, "targets", min(args.top_receivers, 1))
        receiver_players = set(receivers["player"].tolist())
        for depth_idx, receiver in enumerate(receivers.itertuples(index=False), start=1):
            yards_raw = per_game(receiver.yards, receiver.player_game_count)
            yards_baseline_val = shrink_by_games(yards_raw, receiver.player_game_count, team_recv_prior_yards) or 0.0
            yards_baseline_val = max(yards_baseline_val, 0.0)
            yards_baseline_val *= ctx_factor
            yards_baseline = round(yards_baseline_val, 1)
            recs_raw = per_game(receiver.receptions, receiver.player_game_count)
            recs_val = shrink_by_games(recs_raw, receiver.player_game_count, team_recv_prior_recs, strength=2.5) or 0.0
            recs_val = max(recs_val, 0.0)
            recs_val *= 0.5 * (1.0 + ctx_factor)
            recs_baseline = round(recs_val, 1)
            rows.append(
                {
                    "game_id": ctx.game_id,
                    "start_dt": ctx.start_dt.isoformat(),
                    "team": ctx.team,
                    "opponent": ctx.opponent,
                    "player": receiver.player,
                    "position": receiver.position,
                    "metric": "receiving_yards",
                    "baseline": yards_baseline,
                    "season_total": receiver.yards,
                    "usage": receiver.targets,
                    "games_played": receiver.player_game_count,
                    "season_targets": receiver.targets,
                    "season_receptions": receiver.receptions,
                    "season_tds": getattr(receiver, "touchdowns", float("nan")),
                    "depth": depth_idx,
                    "alpha_target": depth_idx == 1,
                    "injury_q": False,
                    "spread": ctx.spread,
                    "favored": ctx.favored,
                }
            )
            rows.append(
                {
                    "game_id": ctx.game_id,
                    "start_dt": ctx.start_dt.isoformat(),
                    "team": ctx.team,
                    "opponent": ctx.opponent,
                    "player": receiver.player,
                    "position": receiver.position,
                    "metric": "receptions",
                    "baseline": recs_baseline,
                    "season_total": receiver.receptions,
                    "usage": receiver.targets,
                    "games_played": receiver.player_game_count,
                    "season_targets": receiver.targets,
                    "season_receptions": receiver.receptions,
                    "season_tds": getattr(receiver, "touchdowns", float("nan")),
                    "depth": depth_idx,
                    "alpha_target": depth_idx == 1,
                    "injury_q": False,
                    "spread": ctx.spread,
                    "favored": ctx.favored,
                }
            )

        # Add receiving projections for rushers not captured above
        for rusher in rushers.itertuples(index=False):
            if rusher.player in receiver_players:
                continue
            recv_match = team_recv[team_recv["player"] == rusher.player]
            if recv_match.empty:
                continue
            recv_entry = recv_match.iloc[0]
            yards_raw = per_game(recv_entry.yards, recv_entry.player_game_count)
            yards_val = shrink_by_games(yards_raw, recv_entry.player_game_count, team_recv_prior_yards) or 0.0
            yards_val = max(yards_val, 0.0)
            yards_val *= ctx_factor
            recs_raw = per_game(recv_entry.receptions, recv_entry.player_game_count)
            recs_val = shrink_by_games(recs_raw, recv_entry.player_game_count, team_recv_prior_recs, strength=2.5) or 0.0
            recs_val = max(recs_val, 0.0)
            recs_val *= 0.5 * (1.0 + ctx_factor)
            rows.append(
                {
                    "game_id": ctx.game_id,
                    "start_dt": ctx.start_dt.isoformat(),
                    "team": ctx.team,
                    "opponent": ctx.opponent,
                    "player": recv_entry.player,
                    "position": recv_entry.position,
                    "metric": "receiving_yards",
                    "baseline": round(yards_val, 1),
                    "season_total": recv_entry.yards,
                    "usage": recv_entry.targets,
                    "games_played": recv_entry.player_game_count,
                    "season_targets": recv_entry.targets,
                    "season_receptions": recv_entry.receptions,
                    "season_tds": getattr(recv_entry, "touchdowns", float("nan")),
                    "depth": getattr(rusher, "depth", 1),
                    "alpha_target": False,
                    "injury_q": False,
                    "spread": ctx.spread,
                    "favored": ctx.favored,
                }
            )
            rows.append(
                {
                    "game_id": ctx.game_id,
                    "start_dt": ctx.start_dt.isoformat(),
                    "team": ctx.team,
                    "opponent": ctx.opponent,
                    "player": recv_entry.player,
                    "position": recv_entry.position,
                    "metric": "receptions",
                    "baseline": round(recs_val, 1),
                    "season_total": recv_entry.receptions,
                    "usage": recv_entry.targets,
                    "games_played": recv_entry.player_game_count,
                    "season_targets": recv_entry.targets,
                    "season_receptions": recv_entry.receptions,
                    "season_tds": getattr(recv_entry, "touchdowns", float("nan")),
                    "depth": getattr(rusher, "depth", 1),
                    "alpha_target": False,
                    "injury_q": False,
                    "spread": ctx.spread,
                    "favored": ctx.favored,
                }
            )

    if not rows:
        print("[warn] No projections generated (team mapping likely failed).")
        return

    output = pd.DataFrame(rows)
    output.sort_values(["start_dt", "team", "player", "metric"], inplace=True)
    output["player"] = output["player"].map(normalize_player)
    output["team"] = output["team"].map(normalize_team)
    output["opponent"] = output["opponent"].map(normalize_team)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"[info] Wrote {len(output)} player prop baselines → {args.output}")


if __name__ == "__main__":
    main()
