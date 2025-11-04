# props_enhancements.py
# Helper utilities for advanced player prop modelling.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class YardageModel:
    mean: float
    sd: float
    distribution: str = "lognormal"


def apply_weather_to_rates(
    pass_rate: float,
    catch_rate: Optional[float],
    yards_model: YardageModel,
    *,
    wind_mph: float = 0.0,
    precip: float = 0.0,
) -> tuple[float, Optional[float], YardageModel]:
    """Adjust pass/catch rates and yards distribution for weather."""

    pr = float(np.clip(pass_rate, 0.0, 1.0))
    cr = None if catch_rate is None else float(np.clip(catch_rate, 0.0, 1.0))
    ym = yards_model

    if wind_mph >= 12.0:
        pr -= 0.02
        if cr is not None:
            cr -= 0.02
        ym = YardageModel(max(0.0, ym.mean * 0.97), max(1e-6, ym.sd * 0.90), ym.distribution)

    if precip >= 1.0:
        pr -= 0.01
        ym = YardageModel(max(0.0, ym.mean * 0.98), max(1e-6, ym.sd * 0.95), ym.distribution)

    pr = float(np.clip(pr, 0.0, 1.0))
    if cr is not None:
        cr = float(np.clip(cr, 0.0, 1.0))
    return pr, cr, ym


def sample_dropbacks_to_attempts(
    rng: np.random.Generator,
    dropbacks: int,
    *,
    sack_rate: float,
    throwaway_rate: float,
    scramble_rate: float,
) -> tuple[int, int, int, int]:
    """Split dropbacks into attempts, sacks, scrambles, throwaways."""

    db = int(max(0, dropbacks))
    srate = float(np.clip(sack_rate, 0.0, 0.5))
    trate = float(np.clip(throwaway_rate, 0.0, 0.5))
    rrate = float(np.clip(scramble_rate, 0.0, 0.5))

    sacks = int(rng.binomial(db, srate))
    remain = db - sacks
    scrambles = int(rng.binomial(max(remain, 0), rrate))
    remain -= scrambles
    throwaways = int(rng.binomial(max(remain, 0), trate))
    attempts = max(remain - throwaways, 0)
    return attempts, sacks, scrambles, throwaways


def garbage_time_multiplier(win_prob_q4: float, coach_tendency: float = 0.8) -> float:
    """Scale late-game usage based on win probability."""

    wp = float(np.clip(win_prob_q4, 0.0, 1.0))
    return float(max(0.6, 1.0 - coach_tendency * max(0.0, wp - 0.5)))


def sample_kneels(
    rng: np.random.Generator,
    *,
    favored: bool,
    endgame_wp: float,
) -> int:
    """Sample kneel attempts (1-3) when favored and likely to win."""

    if not favored or endgame_wp < 0.9:
        return 0
    return int(rng.integers(1, 4))


def dirichlet_room_shares(rng: np.random.Generator, alphas: np.ndarray) -> np.ndarray:
    """Sample room share allocations that sum to 1."""

    alphas = np.asarray(alphas, dtype=float)
    draws = rng.gamma(alphas, 1.0)
    s = draws.sum()
    if s <= 0:
        return np.full_like(draws, 1.0 / len(draws))
    return draws / s


def zero_inflated_targets(
    rng: np.random.Generator,
    base_targets: np.ndarray,
    p0: float | np.ndarray,
) -> np.ndarray:
    """Apply zero-inflation to target counts for fringe players."""

    base_targets = np.asarray(base_targets, dtype=int)
    p0 = np.clip(p0, 0.0, 0.95)
    if np.isscalar(p0):
        mask = rng.random(size=base_targets.shape) < p0
    else:
        p0 = np.asarray(p0, dtype=float)
        if p0.shape != base_targets.shape:
            p0 = np.broadcast_to(p0, base_targets.shape)
        mask = rng.random(size=base_targets.shape) < p0
    out = base_targets.copy()
    out[mask] = 0
    return out


def estimate_overdispersion_k(counts: np.ndarray) -> float:
    """Method-of-moments NegBin k estimator from historical counts."""

    counts = np.asarray(counts, dtype=float)
    if counts.size == 0:
        return float("inf")
    mean = counts.mean()
    var = counts.var(ddof=1) if counts.size > 1 else 0.0
    if var <= mean or mean <= 0:
        return float("inf")
    phi = (var - mean) / (mean * mean)
    if phi <= 0:
        return float("inf")
    return float(1.0 / phi)


__all__ = [
    "YardageModel",
    "apply_weather_to_rates",
    "sample_dropbacks_to_attempts",
    "garbage_time_multiplier",
    "sample_kneels",
    "dirichlet_room_shares",
    "zero_inflated_targets",
    "estimate_overdispersion_k",
]
