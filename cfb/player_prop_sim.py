"""Monte Carlo simulator for college football player props.

This module is standalone (no external I/O). Callers provide season- or game-
adjusted inputs and (optionally) team context, and the functions return
distribution summaries plus edge diagnostics for a given betting line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np

from cfb.props_calibration import PropCalibrations, get_default_calibrations
from cfb.props_enhancements import (
    YardageModel,
    apply_weather_to_rates,
    default_receiving_phi,
    default_receiving_zero_inflation,
    default_rushing_phi,
    garbage_time_multiplier,
    sample_dropbacks_to_attempts,
    sample_kneels,
    zero_inflated_targets,
)

StatKind = Literal["passing_yards", "rushing_yards", "receiving_yards", "receptions"]
YardDistribution = Literal["lognormal", "normal", "gamma"]
CalibrationMode = Literal["auto", "heuristic", "calibrated"]


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
    position: Optional[str] = None
    role: Optional[str] = None


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
    *,
    overdispersion: Optional[float] = None,
    variance_scale: float = 1.0,
) -> np.ndarray:
    mean = _safe_nonnegative(mean)
    if size <= 0:
        return np.zeros(0, dtype=int)
    if mean == 0:
        return np.zeros(size, dtype=int)
    var: Optional[float] = None
    if variance is not None and np.isfinite(variance):
        var = max(float(variance), mean)
    elif overdispersion is not None and np.isfinite(overdispersion):
        var = mean + max(overdispersion, 0.0) * (mean**2)
    if var is not None:
        var = max(var * max(variance_scale, 1e-6), mean)
        phi = max((var - mean) / (mean * mean), 0.0) if mean > 0 else 0.0
    else:
        phi = max(float(overdispersion or 0.0), 0.0)
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


def american_to_probability(price: float) -> float:
    """Convert American odds to the implied break-even probability."""

    price = float(price)
    if price >= 100:
        return 100.0 / (price + 100.0)
    if price <= -100:
        return abs(price) / (abs(price) + 100.0)
    raise ValueError("American odds must be >= +100 or <= -100.")


def _poisson_sf(k: int, mean: float) -> float:
    if mean <= 0:
        return 0.0 if k >= 0 else 1.0
    cumulative = 0.0
    term = math.exp(-mean)
    cumulative += term
    for i in range(1, k + 1):
        term *= mean / i
        cumulative += term
    return max(0.0, 1.0 - cumulative)


def _neg_binomial_sf(k: int, mean: float, phi: float) -> float:
    if k < 0:
        return 1.0
    if phi <= 0 or not np.isfinite(phi):
        return _poisson_sf(k, mean)
    r = 1.0 / max(phi, 1e-12)
    p = r / (r + mean)
    log_p = math.log(p)
    log_q = math.log(1.0 - p)
    cdf = 0.0
    for i in range(0, k + 1):
        log_coeff = math.lgamma(i + r) - math.lgamma(r) - math.lgamma(i + 1)
        log_prob = log_coeff + r * log_p + i * log_q
        cdf += math.exp(log_prob)
    return max(0.0, 1.0 - min(cdf, 1.0))


def implied_negbin_moments_from_line(
    line: float,
    price: float,
    phi: float,
    *,
    max_mean: float = 500.0,
) -> tuple[float, float]:
    """Solve for mean/variance consistent with line probability under NegBin."""

    if not np.isfinite(line) or not np.isfinite(price):
        raise ValueError("line and price must be finite for implied moments")
    prob_over = float(np.clip(american_to_probability(price), 1e-4, 1 - 1e-4))
    threshold = int(math.floor(line))
    if threshold < 0:
        threshold = -1

    def survival(mean: float) -> float:
        return _neg_binomial_sf(threshold, mean, phi)

    low = 1e-6
    high = max(line + 5.0, 5.0)
    while survival(high) < prob_over and high < max_mean:
        high *= 1.5
    if survival(high) < prob_over:
        high = max_mean
    for _ in range(80):
        mid = 0.5 * (low + high)
        tail = survival(mid)
        if abs(tail - prob_over) < 1e-6:
            low = high = mid
            break
        if tail < prob_over:
            low = mid
        else:
            high = mid
    mean = 0.5 * (low + high)
    phi_eff = max(phi, 0.0)
    variance = mean + phi_eff * (mean**2)
    return mean, variance


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def implied_lognormal_moments_from_line(
    line: float,
    price: float,
    *,
    mean_hint: float,
    return_sigma: bool = False,
) -> tuple[float, float] | tuple[float, float, float]:
    """Infer lognormal variance consistent with a line probability."""

    if line <= 0 or mean_hint <= 0 or not np.isfinite(price):
        raise ValueError("positive line, mean_hint, and finite price required")
    target_cdf = 1.0 - float(np.clip(american_to_probability(price), 1e-4, 1 - 1e-4))
    log_mean = math.log(mean_hint)

    def cdf_for_sigma(sigma: float) -> float:
        mu = log_mean - 0.5 * sigma * sigma
        z = (math.log(line) - mu) / sigma
        return _norm_cdf(z)

    low = 1e-3
    high = 3.5
    cdf_low = cdf_for_sigma(low)
    cdf_high = cdf_for_sigma(high)
    attempts = 0
    while cdf_low > target_cdf and attempts < 20:
        low *= 0.5
        cdf_low = cdf_for_sigma(low)
        attempts += 1
    attempts = 0
    while cdf_high < target_cdf and attempts < 20:
        high *= 1.5
        cdf_high = cdf_for_sigma(high)
        attempts += 1
    if not (cdf_low <= target_cdf <= cdf_high):
        raise ValueError("unable to bracket target CDF for lognormal calibration")

    sigma = 0.5 * (low + high)
    for _ in range(80):
        cdf_mid = cdf_for_sigma(sigma)
        if abs(cdf_mid - target_cdf) < 1e-6:
            break
        if cdf_mid < target_cdf:
            low = sigma
        else:
            high = sigma
        sigma = 0.5 * (low + high)

    variance = (math.exp(sigma * sigma) - 1.0) * (mean_hint**2)
    if return_sigma:
        return mean_hint, variance, sigma
    return mean_hint, variance


def _resolve_role_keys(
    role: Optional[RoleParams],
    extra: Sequence[str] | None = None,
) -> list[str]:
    keys: list[str] = []
    if role is not None:
        if role.role:
            keys.append(str(role.role).strip().upper())
        if role.position:
            keys.append(str(role.position).strip().upper())
    if extra:
        for val in extra:
            if val:
                keys.append(str(val).strip().upper())
    keys.append("DEFAULT")
    seen: set[str] = set()
    ordered: list[str] = []
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


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
    calibrations: Optional[PropCalibrations] = None,
    calibration_mode: CalibrationMode = "auto",
) -> dict:
    rng = _ensure_rng(seed)
    team_shock = rng.normal(size=n_sims) if shared_shock else None

    variance = volume_var if volume_var is not None else tgt_var
    tgt_mean_input = _safe_nonnegative(tgt_mean)
    tgt_draws: np.ndarray

    calibrations_source = None
    if calibration_mode != "heuristic":
        calibrations_source = calibrations or get_default_calibrations()

    role_keys = _resolve_role_keys(role)
    position = role.position if role else None
    route_share = role.target_share_mean if role else None
    cal_entry = (
        calibrations_source.get_volume("receiving", role_keys, mode=calibration_mode)
        if calibrations_source
        else None
    )
    mean_scale = cal_entry.mean_scale if cal_entry else 1.0
    variance_scale = cal_entry.variance_scale if cal_entry else 1.0
    phi_override = cal_entry.phi if cal_entry else None
    phi_used = (
        float(phi_override)
        if phi_override is not None
        else float(default_receiving_phi(route_share, position))
    )

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

    tgt_mean_effective = tgt_mean_input * mean_scale if mean_scale != 1.0 else tgt_mean_input

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
            if mean_scale != 1.0:
                share_mean = float(np.clip(share_mean * mean_scale, 1e-3, 0.98))
            share_kappa = float(max(role.target_share_kappa, 2.0))
        else:
            expected_pass_att = float(team.pass_rate * team.plays_mean)
            if expected_pass_att > 0:
                share_mean = np.clip(tgt_mean_effective / expected_pass_att, 1e-3, 0.95)

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
            tgt_draws = _draw_volume(
                rng,
                tgt_mean_effective,
                variance,
                n_sims,
                overdispersion=phi_used,
                variance_scale=variance_scale,
            )
    else:
        tgt_draws = _draw_volume(
            rng,
            tgt_mean_effective,
            variance,
            n_sims,
            overdispersion=phi_used,
            variance_scale=variance_scale,
        )
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
    elif cal_entry and cal_entry.zero_inflation is not None:
        zi_prob = float(np.clip(cal_entry.zero_inflation, 0.0, 0.9))
    else:
        zi_prob = float(default_receiving_zero_inflation(route_share, position))
    if zi_prob > 0.0:
        tgt_draws = zero_inflated_targets(rng, tgt_draws, zi_prob)
        receptions[tgt_draws == 0] = 0

    yards_per = _sample_yards(rng, n_sims, yard_model)
    if team_shock is not None:
        yards_per = np.clip(yards_per * (1.0 + 0.03 * team_shock), 0.0, None)
    yards = receptions * yards_per

    yards_summary = {
        "stat": "receiving_yards",
        "targets_mean": tgt_mean_effective,
        "catch_rate": catch_rate,
        "zero_inflation": zi_prob,
        "overdispersion_phi": float(phi_used),
        "variance_scale": float(variance_scale),
        "mean_scale": float(mean_scale),
    }
    yards_summary.update(_summarise(yards, line=line_yards, price=price, include_preview=True))

    rec_summary = {
        "stat": "receptions",
        "targets_mean": tgt_mean_effective,
        "catch_rate": catch_rate,
        "zero_inflation": zi_prob,
        "overdispersion_phi": float(phi_used),
        "variance_scale": float(variance_scale),
        "mean_scale": float(mean_scale),
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
    calibrations: Optional[PropCalibrations] = None,
    calibration_mode: CalibrationMode = "auto",
) -> dict:
    rng = _ensure_rng(seed)
    team_shock = rng.normal(size=n_sims) if shared_shock else None

    usage_mult = 1.0
    if win_prob_q4 is not None and np.isfinite(win_prob_q4):
        usage_mult = garbage_time_multiplier(float(win_prob_q4))

    rush_mean_input = _safe_nonnegative(rush_mean)

    calibrations_source = None
    if calibration_mode != "heuristic":
        calibrations_source = calibrations or get_default_calibrations()

    extra_keys: list[str] = ["QB"] if is_qb else []
    role_keys = _resolve_role_keys(role, extra=extra_keys)
    position = role.position if role else ("QB" if is_qb else None)
    cal_entry = (
        calibrations_source.get_volume("rushing", role_keys, mode=calibration_mode)
        if calibrations_source
        else None
    )
    mean_scale = cal_entry.mean_scale if cal_entry else 1.0
    variance_scale = cal_entry.variance_scale if cal_entry else 1.0
    phi_override = cal_entry.phi if cal_entry else None
    phi_used = (
        float(phi_override)
        if phi_override is not None
        else float(default_rushing_phi(position, is_qb=is_qb))
    )

    rush_mean_effective = rush_mean_input * mean_scale if mean_scale != 1.0 else rush_mean_input

    if team is not None:
        _, _, team_rush_att = _sample_team_volumes(
            rng,
            n_sims,
            team,
            team_shock=team_shock,
        )
        if role and role.rush_share_mean is not None:
            share_mean = float(role.rush_share_mean)
            if mean_scale != 1.0:
                share_mean = float(np.clip(share_mean * mean_scale, 1e-3, 0.98))
            share_kappa = role.rush_share_kappa
        else:
            denom = max((1 - team.pass_rate) * team.plays_mean, 1e-3)
            share_mean = np.clip(rush_mean_effective / denom, 1e-3, 0.95)
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
        rush_draws = _draw_volume(
            rng,
            rush_mean_effective,
            variance,
            n_sims,
            overdispersion=phi_used,
            variance_scale=variance_scale,
        )

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
        "rush_attempts_mean": rush_mean_effective,
        "win_prob_q4": float(win_prob_q4) if win_prob_q4 is not None else None,
        "kneels_mean": float(np.mean(kneels)) if kneels.size else 0.0,
        "overdispersion_phi": float(phi_used),
        "variance_scale": float(variance_scale),
        "mean_scale": float(mean_scale),
    }
    yards_summary.update(_summarise(yards, line=line_yards, price=price_yards, include_preview=True))

    attempts_summary = {
        "stat": "rushing_attempts",
        "rush_attempts_mean": rush_mean_effective,
        "overdispersion_phi": float(phi_used),
        "variance_scale": float(variance_scale),
        "mean_scale": float(mean_scale),
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
    "american_to_probability",
    "implied_negbin_moments_from_line",
    "implied_lognormal_moments_from_line",
]
