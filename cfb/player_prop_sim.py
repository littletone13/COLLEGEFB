"""Monte Carlo simulator for college football player props.

This module is standalone (no external I/O). Callers provide season- or game-
adjusted inputs and (optionally) team context, and the functions return
distribution summaries plus edge diagnostics for a given betting line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from cfb.props_enhancements import (
    YardageModel,
    apply_weather_to_rates,
    garbage_time_multiplier,
    sample_dropbacks_to_attempts,
    sample_kneels,
    zero_inflated_targets,
)

StatKind = Literal["passing_yards", "rushing_yards", "receiving_yards", "receptions"]
YardDistribution = Literal["lognormal", "normal", "gamma"]


@dataclass(frozen=True)
class TeamParams:
    plays_mean: float
    pass_rate: float
    plays_var: Optional[float] = None
    pass_rate_var: Optional[float] = None
    rush_rate: Optional[float] = None
    sack_rate: Optional[float] = None
    throwaway_rate: Optional[float] = None
    scramble_rate: Optional[float] = None
    pass_k: Optional[float] = None
    rush_k: Optional[float] = None
    win_prob_q4: Optional[float] = None
    favored: Optional[bool] = None


@dataclass(frozen=True)
class RoleParams:
    target_share_mean: Optional[float] = None
    target_share_kappa: float = 60.0
    rush_share_mean: Optional[float] = None
    rush_share_kappa: float = 60.0


def _ensure_rng(seed: Optional[int | np.random.Generator]) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _safe_nonnegative(value: float) -> float:
    return float(value) if value and value > 0 else 0.0


def _neg_binomial_draw(
    rng: np.random.Generator,
    mean: float,
    phi: float,
    size: int,
) -> np.ndarray:
    if size <= 0:
        return np.zeros(0, dtype=int)
    mean = _safe_nonnegative(mean)
    if mean == 0:
        return np.zeros(size, dtype=int)
    if phi <= 0 or not np.isfinite(phi):
        return rng.poisson(lam=mean, size=size)
    shape = 1.0 / phi
    p = shape / (shape + mean)
    return rng.negative_binomial(shape, p, size=size)


def _draw_volume(
    rng: np.random.Generator,
    mean: float,
    variance: Optional[float],
    size: int,
) -> np.ndarray:
    mean = _safe_nonnegative(mean)
    if size <= 0:
        return np.zeros(0, dtype=int)
    if mean == 0:
        return np.zeros(size, dtype=int)
    phi = 0.0
    if variance is not None and np.isfinite(variance) and mean > 0:
        phi = max((variance - mean) / (mean * mean), 0.0)
    return _neg_binomial_draw(rng, mean, phi, size=size)


def _sample_gamma_from_mean_sd(
    rng: np.random.Generator,
    size: int,
    mean: float,
    sd: float,
) -> np.ndarray:
    mean = max(mean, 1e-9)
    sd = max(sd, 1e-9)
    shape = (mean / sd) ** 2
    scale = (sd**2) / mean
    return rng.gamma(shape, scale, size=size)


def _sample_yards(
    rng: np.random.Generator,
    size: int,
    model: YardageModel,
) -> np.ndarray:
    mean = _safe_nonnegative(model.mean)
    sd = _safe_nonnegative(model.sd)
    if size <= 0:
        return np.zeros(0, dtype=float)
    if sd == 0 or mean == 0:
        return np.full(size, mean, dtype=float)

    if model.distribution == "normal":
        draws = rng.normal(loc=mean, scale=sd, size=size)
        return np.clip(draws, 0.0, None)
    if model.distribution == "gamma":
        return _sample_gamma_from_mean_sd(rng, size, mean, sd)

    variance = sd * sd
    if mean <= 0:
        return np.zeros(size, dtype=float)
    sigma_sq = np.log(1.0 + variance / (mean * mean))
    sigma = np.sqrt(sigma_sq)
    mu = np.log(mean) - 0.5 * sigma_sq
    return rng.lognormal(mean=mu, sigma=sigma, size=size)


def _american_payout(price: float) -> float:
    if price >= 100:
        return price / 100.0
    if price <= -100:
        return 100.0 / abs(price)
    raise ValueError("American odds must be >= +100 or <= -100.")


def _beta_share_draw(
    rng: np.random.Generator,
    mean: Optional[float],
    kappa: float,
    size: int,
) -> np.ndarray:
    if mean is None or not np.isfinite(mean):
        return np.zeros(size)
    mean = np.clip(mean, 1e-6, 1 - 1e-6)
    kappa = max(kappa, 2.0)
    return rng.beta(mean * kappa, (1 - mean) * kappa, size=size)


def _sample_team_volumes(
    rng: np.random.Generator,
    n_sims: int,
    team: TeamParams,
    *,
    team_shock: Optional[np.ndarray] = None,
    plays_overdispersion: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if team.plays_var and np.isfinite(team.plays_var):
        plays_var = team.plays_var
    else:
        plays_var = team.plays_mean + plays_overdispersion * (team.plays_mean**2)
    phi = 0.0
    if team.plays_mean > 0 and plays_var and np.isfinite(plays_var):
        phi = max((plays_var - team.plays_mean) / (team.plays_mean**2), 0.0)
    plays_draws = _neg_binomial_draw(rng, team.plays_mean, phi, n_sims).astype(int)

    base_pass_rate_val = team.pass_rate if (team.pass_rate is not None and np.isfinite(team.pass_rate)) else 0.5
    base_pass_rate = float(np.clip(base_pass_rate_val, 0.0, 1.0))
    pass_rates = np.full(n_sims, base_pass_rate, dtype=float)

    if team.pass_rate_var and team.pass_rate_var > 0:
        var = min(team.pass_rate_var, base_pass_rate * (1 - base_pass_rate) - 1e-6)
        var = max(var, 1e-6)
        kappa = base_pass_rate * (1 - base_pass_rate) / var - 1.0
        kappa = max(kappa, 2.0)
        pass_rates = np.clip(
            rng.beta(base_pass_rate * kappa, (1 - base_pass_rate) * kappa, size=n_sims),
            0.01,
            0.99,
        )

    if team_shock is not None:
        pass_rates = np.clip(pass_rates + 0.05 * team_shock, 0.01, 0.99)

    rush_rates = (
        np.full(n_sims, team.rush_rate, dtype=float)
        if team.rush_rate is not None
        else 1.0 - pass_rates
    )
    rush_rates = np.clip(rush_rates, 0.0, 1.0)

    pass_attempts = rng.binomial(np.maximum(plays_draws, 0), pass_rates, size=None).astype(int)
    rush_attempts = np.maximum(plays_draws - pass_attempts, 0)
    return plays_draws, pass_attempts, rush_attempts


def _summarise(
    samples: np.ndarray,
    *,
    line: Optional[float],
    price: Optional[float],
    include_preview: bool = False,
) -> dict:
    percentiles = np.percentile(samples, [5, 10, 25, 50, 75, 90, 95])
    summary = {
        "mean": float(np.mean(samples)),
        "stdev": float(np.std(samples, ddof=1)),
        "std": float(np.std(samples, ddof=1)),
        "p5": float(percentiles[0]),
        "p10": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "p50": float(percentiles[3]),
        "p75": float(percentiles[4]),
        "p90": float(percentiles[5]),
        "p95": float(percentiles[6]),
        "n": int(samples.size),
    }
    if include_preview:
        preview = samples[: min(64, samples.size)]
        summary["samples"] = preview.astype(float).tolist()
    if line is None:
        summary.update(
            {"prob_over": None, "prob_under": None, "ev_over": None, "ev_under": None}
        )
        return summary

    prob_over = float(np.mean(samples > line))
    prob_under = float(np.mean(samples < line))
    summary["prob_over"] = prob_over
    summary["prob_under"] = prob_under

    if price is None:
        summary["ev_over"] = None
        summary["ev_under"] = None
    else:
        payout = _american_payout(price)
        summary["ev_over"] = prob_over * payout - (1.0 - prob_over)
        summary["ev_under"] = prob_under * payout - (1.0 - prob_under)
    return summary


def simulate_passing_yards(
    *,
    att_mean: float,
    att_var: Optional[float] = None,
    comp_rate: float,
    yards_per_comp: YardageModel,
    n_sims: int = 25_000,
    shared_shock: bool = False,
    line_yards: Optional[float] = None,
    price_yards: Optional[float] = None,
    line_completions: Optional[float] = None,
    price_completions: Optional[float] = None,
    line_attempts: Optional[float] = None,
    price_attempts: Optional[float] = None,
    seed: Optional[int | np.random.Generator] = None,
    wind_mph: float = 0.0,
    precip: float = 0.0,
    sack_rate: Optional[float] = None,
    throwaway_rate: Optional[float] = None,
    scramble_rate: Optional[float] = None,
    win_prob_q4: Optional[float] = None,
) -> dict:
    rng = _ensure_rng(seed)
    team_shock = rng.normal(size=n_sims) if shared_shock else None

    sack_rate = float(np.clip(sack_rate, 0.0, 0.5)) if sack_rate is not None and np.isfinite(sack_rate) else 0.0
    throwaway_rate = float(np.clip(throwaway_rate, 0.0, 0.5)) if throwaway_rate is not None and np.isfinite(throwaway_rate) else 0.0
    scramble_rate = float(np.clip(scramble_rate, 0.0, 0.5)) if scramble_rate is not None and np.isfinite(scramble_rate) else 0.0

    base_residual = sack_rate + throwaway_rate + scramble_rate
    if base_residual >= 0.95:
        scale = 0.95 / max(base_residual, 1e-6)
        sack_rate *= scale
        throwaway_rate *= scale
        scramble_rate *= scale
        base_residual = sack_rate + throwaway_rate + scramble_rate
    base_attempt_share = max(0.05, 1.0 - base_residual)

    yard_model = yards_per_comp
    attempt_share_weather, comp_rate_weather, yard_model = apply_weather_to_rates(
        base_attempt_share,
        float(np.clip(comp_rate, 0.0, 1.0)),
        yard_model,
        wind_mph=float(wind_mph or 0.0),
        precip=float(precip or 0.0),
    )
    attempt_share = float(np.clip(attempt_share_weather, 0.05, 0.98))
    comp_rate_used = float(np.clip(comp_rate_weather if comp_rate_weather is not None else comp_rate, 0.0, 1.0))

    target_residual = max(0.0, 1.0 - attempt_share)
    if base_residual > 0 and target_residual >= 0:
        scale = target_residual / base_residual if base_residual > 0 else 0.0
        sack_rate *= scale
        throwaway_rate *= scale
        scramble_rate *= scale
    elif base_residual <= 0 and target_residual > 0:
        equal_share = target_residual / 3.0
        sack_rate = throwaway_rate = scramble_rate = equal_share
    else:
        sack_rate = throwaway_rate = scramble_rate = 0.0

    base_attempt_share = max(0.05, base_attempt_share)
    att_scale = attempt_share / base_attempt_share
    att_mean_adj = float(max(0.0, att_mean * att_scale))
    att_var_adj = None if att_var is None or not np.isfinite(att_var) else float(max(0.0, att_var * (att_scale**2)))

    dropback_mean = att_mean_adj / max(attempt_share, 1e-6)
    dropback_var = None if att_var_adj is None else att_var_adj / max(attempt_share**2, 1e-6)

    dropbacks = _draw_volume(rng, dropback_mean, dropback_var, n_sims).astype(int)
    attempts = np.zeros(n_sims, dtype=int)
    sacks = np.zeros(n_sims, dtype=int)
    scrambles = np.zeros(n_sims, dtype=int)
    throwaways = np.zeros(n_sims, dtype=int)

    for i, db in enumerate(dropbacks):
        att_i, sack_i, scramble_i, throw_i = sample_dropbacks_to_attempts(
            rng,
            int(max(db, 0)),
            sack_rate=sack_rate,
            throwaway_rate=throwaway_rate,
            scramble_rate=scramble_rate,
        )
        attempts[i] = att_i
        sacks[i] = sack_i
        scrambles[i] = scramble_i
        throwaways[i] = throw_i

    comp_rates = np.full(n_sims, comp_rate_used, dtype=float)
    if team_shock is not None:
        comp_rates = np.clip(comp_rates + 0.02 * team_shock, 0.0, 1.0)

    completions = np.array(
        [
            rng.binomial(int(max(0, n_att)), comp_rates[i])
            for i, n_att in enumerate(attempts)
        ],
        dtype=int,
    )

    yards_per = _sample_yards(rng, n_sims, yard_model)
    if team_shock is not None:
        yards_per = np.clip(yards_per * (1.0 + 0.03 * team_shock), 0.0, None)

    yards = completions * yards_per
    attempts_summary = _summarise(
        attempts.astype(float),
        line=line_attempts,
        price=price_attempts,
    )
    completions_summary = _summarise(
        completions.astype(float),
        line=line_completions,
        price=price_completions,
    )
    summary = {
        "stat": "passing_yards",
        "attempts_mean": _safe_nonnegative(att_mean_adj),
        "completion_rate": comp_rate_used,
        "dropbacks_mean": float(np.mean(dropbacks)) if dropbacks.size else 0.0,
        "sack_rate": sack_rate,
        "throwaway_rate": throwaway_rate,
        "scramble_rate": scramble_rate,
        "win_prob_q4": float(win_prob_q4) if win_prob_q4 is not None else None,
        "attempts_summary": attempts_summary,
        "completions_summary": completions_summary,
        "sacks_mean": float(np.mean(sacks)) if sacks.size else 0.0,
        "scrambles_mean": float(np.mean(scrambles)) if scrambles.size else 0.0,
        "throwaways_mean": float(np.mean(throwaways)) if throwaways.size else 0.0,
    }
    summary.update(_summarise(yards, line=line_yards, price=price_yards, include_preview=True))
    return summary


def simulate_receiving(
    *,
    tgt_mean: float,
    tgt_var: Optional[float] = None,
    catch_rate: float,
    yards_per_rec: YardageModel,
    n_sims: int = 25_000,
    team: Optional[TeamParams] = None,
    role: Optional[RoleParams] = None,
    volume_var: Optional[float] = None,
    shared_shock: bool = False,
    line_yards: Optional[float] = None,
    line_receptions: Optional[float] = None,
    price: Optional[float] = None,
    seed: Optional[int | np.random.Generator] = None,
    wind_mph: float = 0.0,
    precip: float = 0.0,
    zero_inflation: Optional[float] = None,
) -> dict:
    rng = _ensure_rng(seed)
    team_shock = rng.normal(size=n_sims) if shared_shock else None

    variance = volume_var if volume_var is not None else tgt_var
    tgt_mean_safe = _safe_nonnegative(tgt_mean)
    tgt_draws: np.ndarray

    yard_model = yards_per_rec
    pass_share_weather, catch_rate_weather, yard_model = apply_weather_to_rates(
        1.0,
        float(np.clip(catch_rate, 0.0, 1.0)),
        yard_model,
        wind_mph=float(wind_mph or 0.0),
        precip=float(precip or 0.0),
    )
    pass_rate_scale = float(np.clip(pass_share_weather, 0.05, 1.0))
    catch_rate = float(np.clip(catch_rate_weather if catch_rate_weather is not None else catch_rate, 0.0, 1.0))

    if team is not None:
        _, team_pass_att, _ = _sample_team_volumes(
            rng,
            n_sims,
            team,
            team_shock=team_shock,
        )
        if pass_rate_scale < 1.0:
            team_pass_att = np.maximum(
                np.round(team_pass_att.astype(float) * pass_rate_scale).astype(int),
                0,
            )

        share_mean: Optional[float] = None
        share_kappa: float = 60.0
        if role is not None and role.target_share_mean is not None:
            share_mean = float(role.target_share_mean)
            share_kappa = float(max(role.target_share_kappa, 2.0))
        else:
            expected_pass_att = float(team.pass_rate * team.plays_mean)
            if expected_pass_att > 0:
                share_mean = np.clip(tgt_mean_safe / expected_pass_att, 1e-3, 0.95)

        if share_mean is not None:
            share_draw = _beta_share_draw(rng, share_mean, share_kappa, n_sims)
            share_draw = np.clip(share_draw, 0.0, 1.0)
            tgt_draws = np.array(
                [
                    rng.binomial(int(max(0, n_pass)), share)
                    for n_pass, share in zip(team_pass_att, share_draw)
                ],
                dtype=int,
            )
        else:
            tgt_draws = _draw_volume(rng, tgt_mean_safe, variance, n_sims)
    else:
        tgt_draws = _draw_volume(rng, tgt_mean_safe, variance, n_sims)
        if pass_rate_scale < 1.0:
            tgt_draws = np.floor(tgt_draws.astype(float) * pass_rate_scale).astype(int)

    catch_rate = np.clip(float(catch_rate), 0.0, 1.0)
    if team_shock is not None:
        catch_adj = np.clip(catch_rate + 0.02 * team_shock, 0.0, 1.0)
    else:
        catch_adj = catch_rate

    receptions = np.array(
        [
            rng.binomial(int(max(0, n_targets)), catch_adj[i])
            for i, n_targets in enumerate(tgt_draws)
        ],
        dtype=int,
    )

    if zero_inflation is not None:
        zi_prob = float(np.clip(zero_inflation, 0.0, 0.9))
    elif role is not None and role.target_share_mean is not None:
        route_share = float(np.clip(role.target_share_mean, 0.0, 1.0))
        zi_prob = max(0.0, 0.5 - 1.2 * route_share)
    else:
        zi_prob = 0.0
    if zi_prob > 0.0:
        tgt_draws = zero_inflated_targets(rng, tgt_draws, zi_prob)
        receptions[tgt_draws == 0] = 0

    yards_per = _sample_yards(rng, n_sims, yard_model)
    if team_shock is not None:
        yards_per = np.clip(yards_per * (1.0 + 0.03 * team_shock), 0.0, None)
    yards = receptions * yards_per

    yards_summary = {
        "stat": "receiving_yards",
        "targets_mean": _safe_nonnegative(tgt_mean),
        "catch_rate": catch_rate,
    }
    yards_summary.update(_summarise(yards, line=line_yards, price=price, include_preview=True))

    rec_summary = {
        "stat": "receptions",
        "targets_mean": _safe_nonnegative(tgt_mean),
        "catch_rate": catch_rate,
    }
    rec_summary.update(
        _summarise(receptions.astype(float), line=line_receptions, price=price)
    )
    return {"yards": yards_summary, "receptions": rec_summary}


def simulate_rushing_yards(
    *,
    rush_mean: float,
    rush_var: Optional[float] = None,
    yards_per_rush: YardageModel,
    n_sims: int = 25_000,
    team: Optional[TeamParams] = None,
    role: Optional[RoleParams] = None,
    volume_var: Optional[float] = None,
    shared_shock: bool = False,
    line_yards: Optional[float] = None,
    price_yards: Optional[float] = None,
    line_attempts: Optional[float] = None,
    price_attempts: Optional[float] = None,
    seed: Optional[int | np.random.Generator] = None,
    wind_mph: float = 0.0,
    precip: float = 0.0,
    win_prob_q4: Optional[float] = None,
    favored: Optional[bool] = None,
    is_qb: bool = False,
) -> dict:
    rng = _ensure_rng(seed)
    team_shock = rng.normal(size=n_sims) if shared_shock else None

    usage_mult = 1.0
    if win_prob_q4 is not None and np.isfinite(win_prob_q4):
        usage_mult = garbage_time_multiplier(float(win_prob_q4))

    if team is not None:
        _, _, team_rush_att = _sample_team_volumes(
            rng,
            n_sims,
            team,
            team_shock=team_shock,
        )
        if role and role.rush_share_mean is not None:
            share_mean = role.rush_share_mean
            share_kappa = role.rush_share_kappa
        else:
            denom = max((1 - team.pass_rate) * team.plays_mean, 1e-3)
            share_mean = np.clip(rush_mean / denom, 1e-3, 0.95)
            share_kappa = 60.0
        share_draw = _beta_share_draw(rng, share_mean, share_kappa, n_sims)
        rush_draws = np.array(
            [
                rng.binomial(int(max(0, n_rush)), np.clip(share, 0.0, 1.0))
                for n_rush, share in zip(team_rush_att, share_draw)
            ],
            dtype=int,
        )
    else:
        variance = volume_var if volume_var is not None else rush_var
        rush_draws = _draw_volume(rng, rush_mean, variance, n_sims)

    if usage_mult < 1.0:
        rush_draws = np.floor(rush_draws.astype(float) * usage_mult).astype(int)

    kneels = np.zeros(n_sims, dtype=int)
    if is_qb and win_prob_q4 is not None and np.isfinite(win_prob_q4):
        fav_flag = bool(favored) if favored is not None else False
        for i in range(n_sims):
            kneels[i] = sample_kneels(
                rng,
                favored=fav_flag,
                endgame_wp=float(win_prob_q4),
            )
        if kneels.any():
            rush_draws += kneels

    yard_model = yards_per_rush
    _, _, yard_model = apply_weather_to_rates(
        1.0,
        None,
        yard_model,
        wind_mph=float(wind_mph or 0.0),
        precip=float(precip or 0.0),
    )

    yards_per = _sample_yards(rng, n_sims, yard_model)
    if team_shock is not None:
        yards_per = np.clip(yards_per * (1.0 - 0.03 * team_shock), 0.0, None)
    yards = rush_draws * yards_per
    if kneels.any():
        yards -= kneels.astype(float)

    yards_summary = {
        "stat": "rushing_yards",
        "rush_attempts_mean": _safe_nonnegative(rush_mean),
        "win_prob_q4": float(win_prob_q4) if win_prob_q4 is not None else None,
        "kneels_mean": float(np.mean(kneels)) if kneels.size else 0.0,
    }
    yards_summary.update(_summarise(yards, line=line_yards, price=price_yards, include_preview=True))

    attempts_summary = {
        "stat": "rushing_attempts",
        "rush_attempts_mean": _safe_nonnegative(rush_mean),
    }
    attempts_summary.update(
        _summarise(rush_draws.astype(float), line=line_attempts, price=price_attempts)
    )

    return {"yards": yards_summary, "attempts": attempts_summary}


__all__ = [
    "TeamParams",
    "RoleParams",
    "YardageModel",
    "simulate_passing_yards",
    "simulate_receiving",
    "simulate_rushing_yards",
]
