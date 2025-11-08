"""Calibrate player-prop parameters from historical odds and results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

STAT_ALIASES = {
    "rec_yds": "receiving_yards",
    "rush_yds": "rushing_yards",
    "rush": "rush_attempts",
}

VOLUME_GROUP = {
    "receptions": "receiving",
    "rush_attempts": "rushing",
}

YARDAGE_STATS = {"receiving_yards", "rushing_yards"}


def american_to_probability(price: float) -> float:
    price = float(price)
    if price >= 100:
        return 100.0 / (price + 100.0)
    if price <= -100:
        return abs(price) / (abs(price) + 100.0)
    raise ValueError("American odds must be >= +100 or <= -100.")


def _poisson_sf(k: int, mean: float) -> float:
    if mean <= 0:
        return 0.0
    total = 0.0
    term = math.exp(-mean)
    total += term
    for i in range(1, k + 1):
        term *= mean / i
        total += term
    return max(0.0, 1.0 - total)


def _negbin_sf(k: int, mean: float, phi: float) -> float:
    if k < 0:
        return 1.0
    if phi <= 0:
        return _poisson_sf(k, mean)
    r = 1.0 / phi
    p = r / (r + mean)
    log_p = math.log(p)
    log_q = math.log(1.0 - p)
    cdf = 0.0
    for i in range(0, k + 1):
        log_coeff = math.lgamma(i + r) - math.lgamma(r) - math.lgamma(i + 1)
        log_prob = log_coeff + r * log_p + i * log_q
        cdf += math.exp(log_prob)
    return max(0.0, 1.0 - min(cdf, 1.0))


def implied_negbin_mean(line: float, price: float, phi: float) -> float:
    threshold = int(math.floor(line))
    prob_over = max(min(american_to_probability(price), 1 - 1e-4), 1e-4)
    low, high = 1e-6, max(line + 5.0, 5.0)
    while _negbin_sf(threshold, high, phi) < prob_over and high < 500:
        high *= 1.5
    for _ in range(60):
        mid = 0.5 * (low + high)
        tail = _negbin_sf(threshold, mid, phi)
        if abs(tail - prob_over) < 1e-6:
            return mid
        if tail < prob_over:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def implied_lognormal_sigma(line: float, price: float, mean_hint: float) -> float:
    target_cdf = 1.0 - max(min(american_to_probability(price), 1 - 1e-4), 1e-4)
    log_mean = math.log(mean_hint)

    def cdf_for_sigma(sigma: float) -> float:
        mu = log_mean - 0.5 * sigma * sigma
        z = (math.log(line) - mu) / sigma
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    low, high = 1e-3, 3.5
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
        return 0.8
    sigma = 0.5 * (low + high)
    for _ in range(60):
        cdf_mid = cdf_for_sigma(sigma)
        if abs(cdf_mid - target_cdf) < 1e-6:
            break
        if cdf_mid < target_cdf:
            low = sigma
        else:
            high = sigma
        sigma = 0.5 * (low + high)
    return sigma


def _group_key(stat: str, row: Dict[str, str]) -> str:
    for field in ("role", "position"):
        val = row.get(field)
        if val:
            clean = val.strip().upper()
            if clean:
                return clean
    return "DEFAULT"


def load_history(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader]


def _volume_entry(records: Iterable[Dict[str, str]]) -> Dict[str, float | int | None]:
    actuals = [float(r["actual"]) for r in records]
    mean_obs = statistics.mean(actuals) if actuals else 0.0
    var_obs = statistics.variance(actuals) if len(actuals) > 1 else 0.0
    zero_rate = sum(1 for value in actuals if value <= 0) / len(actuals) if actuals else 0.0
    phi = 0.0
    if mean_obs > 0 and var_obs > mean_obs:
        phi = max((var_obs - mean_obs) / (mean_obs ** 2), 0.0)

    implied_means: List[float] = []
    for rec in records:
        line = float(rec["closing_line"])
        price = float(rec["over_price"])
        try:
            implied_means.append(implied_negbin_mean(line, price, phi))
        except Exception:
            continue
    mean_scale = 1.0
    if implied_means:
        avg_imp = sum(implied_means) / len(implied_means)
        if avg_imp > 0:
            mean_scale = mean_obs / avg_imp

    variance_base = mean_obs + phi * (mean_obs ** 2) if phi > 0 else mean_obs
    variance_scale = 1.0
    if variance_base > 0 and var_obs > 0:
        variance_scale = var_obs / variance_base

    return {
        "phi": phi,
        "variance_scale": variance_scale,
        "zero_inflation": zero_rate if zero_rate > 0 else None,
        "mean_scale": mean_scale,
        "samples": len(actuals),
    }


def _yardage_entry(records: Iterable[Dict[str, str]]) -> Dict[str, float | int]:
    actuals = [float(r["actual"]) for r in records]
    mean_obs = statistics.mean(actuals) if actuals else 0.0
    var_obs = statistics.variance(actuals) if len(actuals) > 1 else 0.0
    sigmas: List[float] = []
    implied_vars: List[float] = []
    for rec in records:
        line = float(rec["closing_line"])
        if line <= 0 or mean_obs <= 0:
            continue
        price = float(rec["over_price"])
        try:
            sigma = implied_lognormal_sigma(line, price, mean_obs)
        except Exception:
            continue
        sigmas.append(sigma)
        implied_vars.append((math.exp(sigma * sigma) - 1.0) * (mean_obs ** 2))
    if sigmas:
        sigma_avg = sum(sigmas) / len(sigmas)
    elif mean_obs > 0 and var_obs > 0:
        sigma_avg = math.sqrt(max(math.log(1.0 + var_obs / (mean_obs ** 2)), 1e-6))
    else:
        sigma_avg = 0.8

    variance_scale = 1.0
    if implied_vars:
        avg_imp = sum(implied_vars) / len(implied_vars)
        if avg_imp > 0 and var_obs > 0:
            variance_scale = var_obs / avg_imp
    elif mean_obs > 0:
        base_var = (math.exp(sigma_avg * sigma_avg) - 1.0) * (mean_obs ** 2)
        if base_var > 0 and var_obs > 0:
            variance_scale = var_obs / base_var

    return {
        "sigma": sigma_avg,
        "variance_scale": variance_scale,
        "mean_scale": 1.0,
        "samples": len(actuals),
    }


def calibrate_from_history(rows: Iterable[Dict[str, str]], *, min_samples: int = 20) -> Dict[str, object]:
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        stat_raw = row.get("stat", "").strip().lower()
        stat_norm = STAT_ALIASES.get(stat_raw, stat_raw)
        bucket = _group_key(stat_norm, row)
        grouped[(stat_norm, bucket)].append(row)
        grouped[(stat_norm, "DEFAULT")].append(row)

    volume: Dict[str, Dict[str, Dict[str, object]]] = {}
    yardage: Dict[str, Dict[str, Dict[str, object]]] = {}

    for (stat, bucket), records in grouped.items():
        if stat in VOLUME_GROUP:
            group_name = VOLUME_GROUP[stat]
            volume.setdefault(group_name, {})[bucket.lower()] = _volume_entry(records)
        if stat in YARDAGE_STATS:
            yardage.setdefault(stat, {})[bucket.lower()] = _yardage_entry(records)

    meta = {
        "history_rows": sum(len(records) for records in grouped.values()) // 2,
        "min_samples": min_samples,
        "source": "calibrate_player_props.py",
    }

    return {"meta": meta, "volume": volume, "yardage": yardage}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate prop parameters from historical odds and outcomes.")
    parser.add_argument("--history", type=Path, required=True, help="CSV file containing prop history.")
    parser.add_argument("--out", type=Path, required=True, help="Destination JSON path.")
    parser.add_argument("--min-samples", type=int, default=20, help="Minimum samples required for automatic usage.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_history(args.history)
    payload = calibrate_from_history(rows, min_samples=args.min_samples)
    payload["meta"]["min_samples"] = args.min_samples
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[info] wrote calibrations → {args.out} ({payload['meta']['history_rows']} rows)")


if __name__ == "__main__":
    main()
