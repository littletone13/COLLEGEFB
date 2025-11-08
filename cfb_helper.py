
"""
cfb_helper.py — CSV-driven wrapper for cfb_player_sim.py with optional HTML report output.

Adds:
  --html OUT.html         -> write interactive report (no live odds required)
  --title, --model-tag    -> header text
  --assumed               -> default decimal prices per line_type when odds missing
  --template              -> path to template_report.html (from this repo)
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

# Local imports
import cfb_player_sim as sim
from pricing import edge_vs_assumed, ev_over, breakeven_prob, fair_price_decimal, decimal_to_american
from write_report import parse_assumed_map, ASSUMED_DEFAULT


MARKET_LABELS: Dict[str, str] = {
    "pass_yds": "Pass Yds",
    "pass_comp": "Pass Comp",
    "pass_att": "Pass Att",
    "rec_yds": "Rec Yds",
    "receptions": "Receptions",
    "rush_yds": "Rush Yds",
    "rush_att": "Rush Att",
}


def _float_or_none(x):
    try:
        if x == "" or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _str_or_empty(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _build_lines(row: pd.Series) -> sim.Lines:
    return sim.Lines(
        pass_yds=_float_or_none(row.get("line_pass_yds")),
        pass_comp=_float_or_none(row.get("line_pass_comp")),
        pass_att=_float_or_none(row.get("line_pass_att")),
        rec_yds=_float_or_none(row.get("line_rec_yds")),
        receptions=_float_or_none(row.get("line_receptions")),
        rush_yds=_float_or_none(row.get("line_rush_yds")),
        rush_att=_float_or_none(row.get("line_rush_att")),
    )


def _over_prices(row: pd.Series) -> Dict[str,float]:
    out: Dict[str,float] = {}
    for key in ["pass_yds","pass_comp","pass_att","rec_yds","receptions","rush_yds","rush_att"]:
        val = _float_or_none(row.get(f"price_over_{key}"))
        if val is not None:
            out[key] = val
    return out


def _build_team_params(team_row: pd.Series) -> sim.TeamParams:
    return sim.TeamParams(
        plays_mean=float(team_row.get("plays_mean", team_row.get("team_plays_mean", 0.0))),
        pass_rate=float(team_row.get("pass_rate", 0.5)),
        plays_var=_float_or_none(team_row.get("plays_var", team_row.get("team_plays_var"))),
        pass_rate_var=_float_or_none(team_row.get("pass_rate_var", team_row.get("team_pass_rate_var"))),
        rush_rate=_float_or_none(team_row.get("rush_rate")),
        sack_rate=_float_or_none(team_row.get("team_sack_rate")),
        throwaway_rate=_float_or_none(team_row.get("team_throwaway_rate")),
        scramble_rate=_float_or_none(team_row.get("team_scramble_rate")),
        pass_k=_float_or_none(team_row.get("team_pass_k")),
        rush_k=_float_or_none(team_row.get("team_rush_k")),
        win_prob_q4=_float_or_none(team_row.get("win_prob_q4")),
        favored=bool(team_row.get("favored")) if "favored" in team_row else None,
    )


def _build_role_params(row: pd.Series) -> sim.RoleParams:
    return sim.RoleParams(
        target_share_mean=_float_or_none(row.get("target_share_mean")),
        target_share_kappa=float(row.get("target_share_kappa", 60) or 60),
        rush_share_mean=_float_or_none(row.get("rush_share_mean")),
        rush_share_kappa=float(row.get("rush_share_kappa", 60) or 60),
        position=_str_or_empty(row.get("position")) or None,
        role=_str_or_empty(row.get("role")) or None,
    )


def _simulate_row(row: pd.Series, teams: pd.DataFrame, sims_n: int, seed: int, overdisp: float, corr: float) -> Dict[str, Any]:
    team_name = row["team"]
    if team_name not in teams.index:
        return {"error": f"team '{team_name}' not found in teams.csv"}

    team_params = _build_team_params(teams.loc[team_name])
    role_params = _build_role_params(row)

    def _maybe_bool(val):
        if val is None:
            return None
        if isinstance(val, float) and pd.isna(val):
            return None
        if isinstance(val, str):
            return val.strip().lower() in {"true", "1", "yes", "y"}
        return bool(val)
    stat_value = row.get("stat", row.get("metric", ""))
    stat = str(stat_value).strip().lower()
    if stat in {"receiving_yards", "receptions"}:
        stat = "receiving"
    elif stat in {"rushing_yards", "rush"}:
        stat = "rushing"
    elif stat in {"passing_yards", "pass"}:
        stat = "passing"

    passing = receiving = rushing = None

    if stat == "passing":
        passing = sim.PassingParams(
            att_mean=float(row["att_mean"]),
            comp_rate=float(row["comp_rate"]),
            yds_per_comp_mu=float(row["yds_per_comp_mu"]),
            yds_per_comp_sd=float(row["yds_per_comp_sd"]),
            att_var=_float_or_none(row.get("att_var")),
            wind_mph=float(row.get("wind_mph", 0.0) or 0.0),
            precip=float(row.get("precip_flag", 0.0) or 0.0),
            sack_rate=_float_or_none(row.get("team_sack_rate")),
            throwaway_rate=_float_or_none(row.get("team_throwaway_rate")),
            scramble_rate=_float_or_none(row.get("team_scramble_rate")),
            win_prob_q4=_float_or_none(row.get("win_prob_q4")),
        )
    elif stat == "receiving":
        tgt_mean = _float_or_none(row.get("tgt_mean")) or 0.0
        receiving = sim.ReceivingParams(
            tgt_mean=float(tgt_mean),
            catch_rate=float(row["catch_rate"]),
            yds_per_rec_mu=float(row["yds_per_rec_mu"]),
            yds_per_rec_sd=float(row["yds_per_rec_sd"]),
            tgt_var=_float_or_none(row.get("tgt_var")),
            wind_mph=float(row.get("wind_mph", 0.0) or 0.0),
            precip=float(row.get("precip_flag", 0.0) or 0.0),
            zero_inflation=_float_or_none(row.get("receiving_zero_inflation")),
        )
    elif stat == "rushing":
        rush_mean = _float_or_none(row.get("rush_mean")) or 0.0
        rushing = sim.RushingParams(
            rush_mean=float(rush_mean),
            yds_per_rush_mu=float(row["yds_per_rush_mu"]),
            yds_per_rush_sd=float(row["yds_per_rush_sd"]),
            rush_var=_float_or_none(row.get("rush_var")),
            wind_mph=float(row.get("wind_mph", 0.0) or 0.0),
            precip=float(row.get("precip_flag", 0.0) or 0.0),
            win_prob_q4=_float_or_none(row.get("win_prob_q4")),
            favored=_maybe_bool(row.get("favored")) if "favored" in row else None,
            is_qb=str(row.get("position", "")).upper() == "QB",
        )
    else:
        return {"error": f"unknown stat '{stat}'"}

    lines = _build_lines(row)
    prices = _over_prices(row)

    res = sim.simulate_player(
        passing=passing, receiving=receiving, rushing=rushing,
        lines=lines, sims=sims_n, seed=seed,
        over_prices=prices, overdispersion=overdisp,
        corr_strength=corr, team=team_params, role=role_params
    )

    flat: Dict[str, Any] = {
        "player": row["player"],
        "position": row.get("position", ""),
        "team": team_name,
        "opponent": row.get("opponent",""),
        "stat": stat,
        "book": row.get("book",""),
        "kickoff_iso": row.get("kickoff_iso",""),
        "notes": row.get("notes",""),
        "game_id": row.get("game_id", ""),
    }
    # Map summaries to a unified schema for HTML
    mapping = {
        "pass_att": ["line_pass_att", "price_over_pass_att"],
        "pass_comp": ["line_pass_comp", "price_over_pass_comp"],
        "pass_yds": ["line_pass_yds", "price_over_pass_yds"],
        "rec_yds": ["line_rec_yds", "price_over_rec_yds"],
        "receptions": ["line_receptions", "price_over_receptions"],
        "rush_att": ["line_rush_att", "price_over_rush_att"],
        "rush_yds": ["line_rush_yds", "price_over_rush_yds"],
    }
    rows = []
    for key, (line_col, price_col) in mapping.items():
        if key in res:
            s = res[key]
            row_out = flat.copy()
            row_out["line_type"] = key
            row_out["line"] = _float_or_none(row.get(line_col))
            row_out["price_over"] = _float_or_none(row.get(price_col))
            row_out["prob_over"] = s.get("Pr(> line)")
            row_out["mean"] = s.get("mean")
            row_out["stdev"] = s.get("stdev")
            row_out["p25"] = s.get("p25")
            row_out["p50"] = s.get("p50")
            row_out["p75"] = s.get("p75")
            row_out["p95"] = s.get("p95")
            row_out["p05"] = s.get("p5")
            row_out["p10"] = s.get("p10")
            row_out["p90"] = s.get("p90")
            if "samples" in s:
                row_out["samples"] = s.get("samples")
            rows.append(row_out)
    return rows  # list of prop rows for this player


def run(teams_path: str, players_path: str, out_path: str,
        sims_n: int, seed: int, overdisp: float, corr: float,
        html_path: str | None, html_title: str, html_model_tag: str, html_assumed: str, html_template: str):
    teams = pd.read_csv(teams_path)
    if "team" not in teams.columns:
        raise ValueError("teams.csv must include 'team' column")
    teams = teams.set_index("team")

    players = pd.read_csv(players_path)

    # Fill missing team context from players file when teams.csv lacks an entry
    team_cols_map = {
        "plays_mean": "team_plays_mean",
        "plays_var": "team_plays_var",
        "pass_rate": "team_pass_rate",
        "pass_rate_var": "team_pass_rate_var",
        "pass_attempts_mean": "team_pass_attempts_mean",
        "rush_attempts_mean": "team_rush_attempts_mean",
        "sack_rate": "team_sack_rate",
        "throwaway_rate": "team_throwaway_rate",
        "scramble_rate": "team_scramble_rate",
        "pass_k": "team_pass_k",
        "rush_k": "team_rush_k",
        "win_prob_q4": "win_prob_q4",
        "favored": "favored",
    }
    available_cols = {k: v for k, v in team_cols_map.items() if v in players.columns}
    if available_cols:
        team_context = (
            players[["team", *available_cols.values()]]
            .groupby("team")
            .mean(numeric_only=True)
            .rename(columns={v: k for k, v in available_cols.items()})
        )
        for col in ["plays_var", "pass_rate_var"]:
            if col not in team_context.columns and col in teams.columns:
                team_context[col] = teams[col].groupby(teams.index).mean()
        missing = [t for t in team_context.index if t not in teams.index]
        if missing:
            teams = pd.concat([teams, team_context.loc[missing]], axis=0)

    rows = []
    context = {
        "model_tag": html_model_tag,
        "title": html_title,
        "assumed": html_assumed,
    }
    for _, r in players.iterrows():
        rows.extend(_simulate_row(r, teams, sims_n, seed, overdisp, corr))
    clean_rows: list[Dict[str, Any]] = []
    for item in rows:
        if isinstance(item, dict):
            if "error" in item:
                continue
            clean_rows.append(item)
    df = pd.DataFrame(clean_rows)

    if context:
        for key, value in context.items():
            if key not in df.columns:
                df[key] = value

    if not df.empty:
        if "kickoff_iso" in df.columns:
            df["kickoff_iso"] = pd.to_datetime(df["kickoff_iso"], errors="coerce")
            now = pd.Timestamp.utcnow()
            df = df[df["kickoff_iso"].isna() | (df["kickoff_iso"] > now)]
            df["kickoff_iso"] = df["kickoff_iso"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        subset_cols = [c for c in ["player", "team", "line_type", "line"] if c in df.columns]
        if subset_cols:
            df = df.drop_duplicates(subset=subset_cols).reset_index(drop=True)
        if "line" in df.columns:
            mask = df["line"].isna()
        else:
            mask = pd.Series(False, index=df.index)
        null_cols = [
            "assumed",
            "assumed_price_american",
            "assumed_price_decimal",
            "breakeven_prob",
            "edge_prob_pts",
            "ev_over",
            "fair_price_american",
            "fair_price_decimal",
            "price_over",
            "price_source",
            "prob_over",
        ]
        for col in null_cols:
            if col in df.columns:
                df.loc[mask, col] = None
        if "prob_over" in df.columns:
            prob_mask = df["prob_over"].isna()
            for col in ["fair_price_american", "fair_price_decimal", "ev_over", "edge_prob_pts", "breakeven_prob"]:
                if col in df.columns:
                    df.loc[prob_mask, col] = None

    # If a line and prob exist, compute EV using posted or assumed pricing
    assumed_map = parse_assumed_map(html_assumed)
    def enrich(row):
        p = row.get("prob_over")
        lt = row.get("line_type")
        if p is None or lt is None:
            return row
        price = row.get("price_over")
        if pd.notna(price):
            row["ev_over"] = ev_over(p, price)
            row["breakeven_prob"] = breakeven_prob(price)
            row["fair_price_decimal"] = fair_price_decimal(p)
            row["price_source"] = "posted"
        else:
            assumed = assumed_map.get(lt)
            if assumed is not None:
                e = edge_vs_assumed(p, assumed)
                for k,v in e.items(): row[k] = v
                row["ev_over"] = e["ev_over"]
                row["price_over"] = e["assumed_price_decimal"]
                row["price_source"] = "assumed"
            else:
                row["fair_price_decimal"] = fair_price_decimal(p)
                from pricing import decimal_to_american
                row["fair_price_american"] = decimal_to_american(row["fair_price_decimal"])
                row["price_source"] = None
        return row

    if not df.empty:
        df = df.apply(enrich, axis=1)

    if "team" in df.columns:
        team_series = df["team"].astype(str)
    else:
        team_series = pd.Series(["" for _ in range(len(df))])
    if "opponent" in df.columns:
        opp_series = df["opponent"].astype(str)
    else:
        opp_series = pd.Series(["" for _ in range(len(df))])
    df["matchup"] = team_series + " vs " + opp_series
    if "p50" in df.columns and "fair_value" not in df.columns:
        df["fair_value"] = df["p50"]

    summary_cols = [
        "player",
        "position",
        "team",
        "opponent",
        "matchup",
        "line_type",
        "stat",
        "line",
        "mean",
        "stdev",
        "p50",
        "p10",
        "p90",
        "p05",
        "p25",
        "p75",
        "p95",
        "fair_value",
        "kickoff_iso",
        "game_id",
    ]
    summary_df = df[[col for col in summary_cols if col in df.columns]].copy()
    if not summary_df.empty:
        subset = [col for col in ["player", "line_type"] if col in summary_df.columns]
        if subset:
            summary_df = summary_df.drop_duplicates(subset=subset)

    if out_path:
        summary_df.to_csv(out_path, index=False)

    if html_path:
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        dashboard_rows: list[Dict[str, Any]] = []
        if not df.empty:
            for rec in df.to_dict(orient="records"):
                prob_over = rec.get("prob_over")
                line_val = rec.get("line")
                if prob_over is None or line_val is None:
                    continue
                try:
                    if pd.isna(prob_over) or pd.isna(line_val):
                        continue
                except TypeError:
                    pass

                market_key = rec.get("line_type")
                market_label = MARKET_LABELS.get(market_key, _str_or_empty(market_key)).strip()
                if not market_label:
                    continue

                fair_dec = rec.get("fair_price_decimal")
                if fair_dec is None or (isinstance(fair_dec, float) and pd.isna(fair_dec)):
                    fair_dec = fair_price_decimal(prob_over)
                fair_amer = rec.get("fair_price_american")
                if fair_amer is None or (isinstance(fair_amer, float) and pd.isna(fair_amer)):
                    fair_amer = decimal_to_american(fair_dec)

                offered_dec = rec.get("price_over")
                if offered_dec is None or (isinstance(offered_dec, float) and pd.isna(offered_dec)):
                    offered_dec = fair_dec
                offered_amer = decimal_to_american(offered_dec) if offered_dec else None

                samples_val = rec.get("samples")
                if isinstance(samples_val, list):
                    samples = samples_val
                else:
                    samples = []

                entry = {
                    "player": _str_or_empty(rec.get("player")),
                    "pos": _str_or_empty(rec.get("position")).upper(),
                    "team": _str_or_empty(rec.get("team")),
                    "opp": _str_or_empty(rec.get("opponent")),
                    "market": market_label,
                    "line": float(line_val),
                    "mean": rec.get("mean"),
                    "median": rec.get("p50"),
                    "p05": rec.get("p05"),
                    "p10": rec.get("p10"),
                    "p90": rec.get("p90"),
                    "p95": rec.get("p95"),
                    "p_over": float(prob_over),
                    "side": "Over",
                    "offeredDec": offered_dec,
                    "offeredAmer": offered_amer,
                    "fairDec": fair_dec,
                    "fairAmer": fair_amer,
                    "notes": _str_or_empty(rec.get("notes")),
                    "tags": rec.get("tags") if isinstance(rec.get("tags"), list) else [],
                    "samples": samples,
                    "stdev": rec.get("stdev"),
                    "kickoff": _str_or_empty(rec.get("kickoff_iso")),
                }
                dashboard_rows.append(entry)

        meta = {
            "page_title": html_title,
            "model_tag": html_model_tag,
            "generated_at": generated_at,
            "assumed_note": html_assumed,
            "total_rows": len(dashboard_rows),
        }

        template_text = Path(html_template).read_text(encoding="utf-8")
        html_output = (
            template_text
            .replace("__DATA__", json.dumps(dashboard_rows, separators=(",", ":")))
            .replace("__META__", json.dumps(meta, separators=(",", ":")))
        )
        Path(html_path).write_text(html_output, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams", required=True, help="Path to teams.csv")
    ap.add_argument("--players", required=True, help="Path to players.csv")
    ap.add_argument("--out", default="", help="Optional output CSV for results/edges")

    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overdisp", type=float, default=0.25)
    ap.add_argument("--corr", type=float, default=0.1)

    # HTML report options
    ap.add_argument("--html", default="", help="Optional output HTML report path")
    ap.add_argument("--title", default="Week 10 (Late)")
    ap.add_argument("--model-tag", default="CFB v0.7 - wk10L")
    ap.add_argument("--assumed", default=ASSUMED_DEFAULT)
    ap.add_argument("--template", default="template_report.html")

    args = ap.parse_args()

    run(args.teams, args.players, args.out, args.sims, args.seed, args.overdisp, args.corr,
        args.html or None, args.title, args.model_tag, args.assumed, args.template)


if __name__ == "__main__":
    main()
