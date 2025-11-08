"""Compatibility wrapper around cfb.player_prop_sim for helper/report scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any

import numpy as np

from cfb.player_prop_sim import (
    TeamParams,
    RoleParams,
    YardageModel,
    simulate_passing_yards,
    simulate_receiving,
    simulate_rushing_yards,
    CalibrationMode,
)
from cfb.props_calibration import PropCalibrations, get_default_calibrations


@dataclass
class PassingParams:
    att_mean: float
    comp_rate: float
    yds_per_comp_mu: float
    yds_per_comp_sd: float
    att_var: Optional[float] = None
    wind_mph: float = 0.0
    precip: float = 0.0
    sack_rate: Optional[float] = None
    throwaway_rate: Optional[float] = None
    scramble_rate: Optional[float] = None
    win_prob_q4: Optional[float] = None


@dataclass
class ReceivingParams:
    tgt_mean: float
    catch_rate: float
    yds_per_rec_mu: float
    yds_per_rec_sd: float
    tgt_var: Optional[float] = None
    wind_mph: float = 0.0
    precip: float = 0.0
    zero_inflation: Optional[float] = None


@dataclass
class RushingParams:
    rush_mean: float
    yds_per_rush_mu: float
    yds_per_rush_sd: float
    rush_var: Optional[float] = None
    wind_mph: float = 0.0
    precip: float = 0.0
    win_prob_q4: Optional[float] = None
    favored: Optional[bool] = None
    is_qb: bool = False


@dataclass
class Lines:
    pass_yds: Optional[float] = None
    pass_comp: Optional[float] = None
    pass_att: Optional[float] = None
    rec_yds: Optional[float] = None
    receptions: Optional[float] = None
    rush_yds: Optional[float] = None
    rush_att: Optional[float] = None


def _add_prob_alias(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to expose the naming expected by legacy scripts."""
    if summary is None:
        return summary
    if "prob_over" in summary:
        summary["Pr(> line)"] = summary["prob_over"]
    return summary


def simulate_player(
    *,
    passing: PassingParams | None = None,
    receiving: ReceivingParams | None = None,
    rushing: RushingParams | None = None,
    lines: Lines | None = None,
    sims: int = 25_000,
    seed: int = 42,
    over_prices: Optional[Dict[str, float]] = None,
    overdispersion: float = 0.25,  # kept for API parity; handled in upstream draws
    corr_strength: float = 0.1,
    team: TeamParams | None = None,
    role: RoleParams | None = None,
    calibrations: PropCalibrations | None = None,
    calibration_mode: CalibrationMode = "auto",
) -> Dict[str, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    results: Dict[str, Dict[str, Any]] = {}
    shared_shock = corr_strength > 0

    price_lookup = over_prices or {}
    line_lookup = lines or Lines()
    calibration_source: Optional[PropCalibrations] = None
    if calibration_mode != "heuristic":
        calibration_source = calibrations or get_default_calibrations()

    if passing is not None:
        pass_sim = simulate_passing_yards(
            att_mean=passing.att_mean,
            att_var=passing.att_var,
            comp_rate=passing.comp_rate,
            yards_per_comp=YardageModel(passing.yds_per_comp_mu, passing.yds_per_comp_sd),
            n_sims=sims,
            shared_shock=shared_shock,
            line_yards=line_lookup.pass_yds,
            price_yards=price_lookup.get("pass_yds"),
            line_completions=line_lookup.pass_comp,
            price_completions=price_lookup.get("pass_comp"),
            line_attempts=line_lookup.pass_att,
            price_attempts=price_lookup.get("pass_att"),
            seed=rng,
            wind_mph=passing.wind_mph,
            precip=passing.precip,
            sack_rate=passing.sack_rate if passing.sack_rate is not None else (team.sack_rate if team and team.sack_rate is not None else None),
            throwaway_rate=passing.throwaway_rate if passing.throwaway_rate is not None else (team.throwaway_rate if team and team.throwaway_rate is not None else None),
            scramble_rate=passing.scramble_rate if passing.scramble_rate is not None else (team.scramble_rate if team and team.scramble_rate is not None else None),
            win_prob_q4=passing.win_prob_q4 if passing.win_prob_q4 is not None else (team.win_prob_q4 if team and team.win_prob_q4 is not None else None),
        )
        attempts_summary = pass_sim.pop("attempts_summary", None)
        completions_summary = pass_sim.pop("completions_summary", None)

        pass_yds_summary = _add_prob_alias(pass_sim)
        results["pass_yds"] = pass_yds_summary

        if completions_summary is not None:
            results["pass_comp"] = _add_prob_alias(completions_summary)
        if attempts_summary is not None:
            results["pass_att"] = _add_prob_alias(attempts_summary)

    if receiving is not None:
        recv = simulate_receiving(
            tgt_mean=receiving.tgt_mean,
            tgt_var=receiving.tgt_var,
            catch_rate=receiving.catch_rate,
            yards_per_rec=YardageModel(receiving.yds_per_rec_mu, receiving.yds_per_rec_sd),
            n_sims=sims,
            team=team,
            role=role,
            volume_var=receiving.tgt_var,
            shared_shock=shared_shock,
            line_yards=line_lookup.rec_yds,
            line_receptions=line_lookup.receptions,
            price=None,
            seed=rng,
            wind_mph=receiving.wind_mph,
            precip=receiving.precip,
            zero_inflation=receiving.zero_inflation,
            calibrations=calibration_source,
            calibration_mode=calibration_mode,
        )
        yards_summary = _add_prob_alias(recv.get("yards"))
        rec_summary = _add_prob_alias(recv.get("receptions"))
        if yards_summary:
            results["rec_yds"] = yards_summary
        if rec_summary:
            results["receptions"] = rec_summary

    if rushing is not None:
        rush_metrics = simulate_rushing_yards(
            rush_mean=rushing.rush_mean,
            rush_var=rushing.rush_var,
            yards_per_rush=YardageModel(rushing.yds_per_rush_mu, rushing.yds_per_rush_sd),
            n_sims=sims,
            team=team,
            role=role,
            volume_var=rushing.rush_var,
            shared_shock=shared_shock,
            line_yards=line_lookup.rush_yds,
            line_attempts=line_lookup.rush_att,
            price_yards=price_lookup.get("rush_yds"),
            price_attempts=price_lookup.get("rush_att"),
            seed=rng,
            wind_mph=rushing.wind_mph,
            precip=rushing.precip,
            win_prob_q4=rushing.win_prob_q4 if rushing.win_prob_q4 is not None else (team.win_prob_q4 if team and team.win_prob_q4 is not None else None),
            favored=rushing.favored if rushing.favored is not None else (team.favored if team else None),
            is_qb=rushing.is_qb,
            calibrations=calibration_source,
            calibration_mode=calibration_mode,
        )
        yards_summary = _add_prob_alias(rush_metrics.get("yards"))
        attempts_summary = _add_prob_alias(rush_metrics.get("attempts"))
        if yards_summary:
            results["rush_yds"] = yards_summary
        if attempts_summary:
            results["rush_att"] = attempts_summary

    return results


__all__ = [
    "TeamParams",
    "RoleParams",
    "PassingParams",
    "ReceivingParams",
    "RushingParams",
    "Lines",
    "simulate_player",
]
