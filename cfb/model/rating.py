"""Shared rating book utilities for FBS/FCS simulations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from cfb.market import moneyline_from_prob


@dataclass
class RatingConstants:
    avg_total: float
    home_field_advantage: float
    offense_factor: float
    defense_factor: float
    power_factor: float
    spread_sigma: float

    @property
    def avg_team_points(self) -> float:
        return self.avg_total / 2.0


@dataclass
class BayesianConfig:
    enabled: bool = False
    spread_prior_strength: float = 0.0
    total_prior_strength: float = 0.0
    spread_prior_mean: float = 0.0
    total_prior_mean: Optional[float] = None
    min_games: float = 0.0
    max_games: float = 20.0
    default_games: float = 6.0
    prob_sigma_scale: float = 0.0


class RatingBook:
    def __init__(
        self,
        teams: pd.DataFrame,
        constants: RatingConstants,
        power_adjustments: Optional[Dict[str, float]] = None,
        team_games: Optional[Dict[str, float]] = None,
        bayesian_config: Optional[BayesianConfig] = None,
    ) -> None:
        self.teams = teams
        self.constants = constants
        self.power_adjustments = power_adjustments or {}
        self.spread_calibration: Tuple[float, float] = (0.0, 1.0)
        self.total_calibration: Tuple[float, float] = (0.0, 1.0)
        self.prob_sigma: float = constants.spread_sigma
        self.bayesian_config = bayesian_config if bayesian_config and bayesian_config.enabled else None
        if self.bayesian_config:
            normalized_games: Dict[str, float] = {}
            for team, games in (team_games or {}).items():
                key = str(team).lower()
                try:
                    value = float(games)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(value):
                    continue
                normalized_games[key] = value
            self.team_games = normalized_games
        else:
            self.team_games = {}

    def _lookup(self, team: str) -> pd.Series:
        df = self.teams
        mask = df["team"].str.lower() == team.lower()
        if mask.sum() == 1:
            return df.loc[mask].iloc[0]
        mask = df["team"].str.contains(team, case=False, regex=False)
        if mask.sum() == 1:
            return df.loc[mask].iloc[0]
        available = sorted(df["team"].unique())[:10]
        raise KeyError(f"Team '{team}' not found; sample available teams: {available}")

    def get_rating(self, team: str) -> Dict[str, float]:
        row = self._lookup(team)
        return {
            "team": row["team"],
            "offense_rating": row["offense_rating"],
            "defense_rating": row["defense_rating"],
            "power_rating": row["power_rating"] + self.power_adjustments.get(row["team"], 0.0),
        }

    def _predict_raw(self, home: str, away: str, neutral_site: bool = False) -> Dict[str, float]:
        home_row = self._lookup(home)
        away_row = self._lookup(away)
        const = self.constants
        home_power = home_row["power_rating"] + self.power_adjustments.get(home_row["team"], 0.0)
        away_power = away_row["power_rating"] + self.power_adjustments.get(away_row["team"], 0.0)

        def expected_points(off_row: pd.Series, def_row: pd.Series, power_diff: float) -> float:
            return (
                const.avg_team_points
                + const.offense_factor * off_row["offense_rating"]
                - const.defense_factor * def_row["defense_rating"]
                + const.power_factor * power_diff / 2.0
            )

        home_points = expected_points(home_row, away_row, home_power - away_power)
        away_points = expected_points(away_row, home_row, away_power - home_power)

        spread = home_points - away_points
        if not neutral_site:
            spread += const.home_field_advantage
            home_points += const.home_field_advantage / 2.0
            away_points -= const.home_field_advantage / 2.0

        total = home_points + away_points

        home_win_prob = 0.5 * (1.0 + math.erf(spread / (self.prob_sigma * math.sqrt(2))))
        away_win_prob = 1.0 - home_win_prob

        return {
            "home_team": home_row["team"],
            "away_team": away_row["team"],
            "home_points": home_points,
            "away_points": away_points,
            "spread_home_minus_away": spread,
            "total_points": total,
            "home_win_prob": home_win_prob,
            "away_win_prob": away_win_prob,
        }

    def predict(
        self,
        home: str,
        away: str,
        neutral_site: bool = False,
        *,
        calibrated: bool = True,
    ) -> Dict[str, float]:
        raw = self._predict_raw(home, away, neutral_site=neutral_site)
        if not calibrated:
            raw = raw.copy()
            raw["home_moneyline"] = moneyline_from_prob(raw["home_win_prob"])
            raw["away_moneyline"] = moneyline_from_prob(raw["away_win_prob"])
            return raw

        spread = raw["spread_home_minus_away"]
        total = raw["total_points"]
        a_s, b_s = self.spread_calibration
        a_t, b_t = self.total_calibration
        spread = a_s + b_s * spread
        total = a_t + b_t * total
        sigma_local = self.prob_sigma
        spread_weight = total_weight = 1.0
        if self.bayesian_config:
            spread, total, sigma_local, spread_weight, total_weight = self._apply_bayesian_adjustments(
                spread,
                total,
                raw["home_team"],
                raw["away_team"],
            )
        home_points = (total + spread) / 2.0
        away_points = total - home_points
        home_win_prob = 0.5 * (1.0 + math.erf(spread / (sigma_local * math.sqrt(2))))
        away_win_prob = 1.0 - home_win_prob

        result = raw.copy()
        result.update(
            {
                "home_points": home_points,
                "away_points": away_points,
                "spread_home_minus_away": spread,
                "total_points": total,
                "home_win_prob": home_win_prob,
                "away_win_prob": away_win_prob,
                "home_moneyline": moneyline_from_prob(home_win_prob),
                "away_moneyline": moneyline_from_prob(away_win_prob),
            }
        )
        if self.bayesian_config:
            result["prob_sigma_used"] = sigma_local
            result["bayesian_spread_weight"] = spread_weight
            result["bayesian_total_weight"] = total_weight
        return result

    def _team_games(self, team: str) -> float:
        if not self.bayesian_config:
            return 0.0
        key = str(team).lower()
        return self.team_games.get(key, self.bayesian_config.default_games)

    def _clamp_games(self, games: float) -> float:
        cfg = self.bayesian_config
        if cfg is None:
            return games
        try:
            value = float(games)
        except (TypeError, ValueError):
            value = cfg.default_games
        if not math.isfinite(value):
            value = cfg.default_games
        value = max(cfg.min_games, value)
        value = min(cfg.max_games, value)
        return value

    def _apply_bayesian_adjustments(
        self,
        spread: float,
        total: float,
        home_team: str,
        away_team: str,
    ) -> Tuple[float, float, float, float, float]:
        cfg = self.bayesian_config
        if cfg is None:
            return spread, total, self.prob_sigma, 1.0, 1.0

        home_games = self._clamp_games(self._team_games(home_team))
        away_games = self._clamp_games(self._team_games(away_team))
        combined_games = home_games + away_games
        if combined_games <= 0:
            combined_games = max(cfg.default_games * 2.0, 1.0)

        spread_strength = max(cfg.spread_prior_strength, 0.0)
        if spread_strength > 0:
            spread_weight = combined_games / (combined_games + spread_strength)
        else:
            spread_weight = 1.0
        spread_weight = float(np.clip(spread_weight, 0.0, 1.0))
        spread_prior = cfg.spread_prior_mean if cfg.spread_prior_mean is not None else 0.0
        adjusted_spread = spread_weight * spread + (1.0 - spread_weight) * spread_prior

        total_strength = max(cfg.total_prior_strength, 0.0)
        if total_strength > 0:
            total_weight = combined_games / (combined_games + total_strength)
        else:
            total_weight = 1.0
        total_weight = float(np.clip(total_weight, 0.0, 1.0))
        total_prior = cfg.total_prior_mean if cfg.total_prior_mean is not None else self.constants.avg_total
        adjusted_total = total_weight * total + (1.0 - total_weight) * total_prior

        sigma_scale = max(cfg.prob_sigma_scale, 0.0)
        sigma_adjusted = self.prob_sigma * (1.0 + sigma_scale / max(combined_games, 1.0))

        return adjusted_spread, adjusted_total, sigma_adjusted, spread_weight, total_weight


def fit_linear_calibrations(book: RatingBook, games: Iterable[dict]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    margins_pred: list[float] = []
    margins_actual: list[float] = []
    margins_weight: list[float] = []
    totals_pred: list[float] = []
    totals_actual: list[float] = []
    totals_weight: list[float] = []

    for game in games:
        if game.get("completed") is not True:
            continue
        if game.get("homeClassification") != "fbs" or game.get("awayClassification") != "fbs":
            continue
        home_points = game.get("homePoints")
        away_points = game.get("awayPoints")
        if home_points is None or away_points is None:
            continue
        try:
            raw = book.predict(
                game["homeTeam"],
                game["awayTeam"],
                neutral_site=game.get("neutralSite", False),
                calibrated=False,
            )
        except KeyError:
            continue
        weight = float(game.get("__cal_weight", 1.0))
        if not math.isfinite(weight) or weight <= 0.0:
            continue
        margins_pred.append(raw["spread_home_minus_away"])
        margins_actual.append(home_points - away_points)
        margins_weight.append(weight)
        totals_pred.append(raw["total_points"])
        totals_actual.append(home_points + away_points)
        totals_weight.append(weight)

    def _fit(pred: list[float], actual: list[float], weights: list[float], *, stretch: bool = False) -> Tuple[float, float]:
        if len(pred) < 30:
            return (0.0, 1.0)
        x = np.array(pred)
        y = np.array(actual)
        w = np.clip(np.array(weights, dtype=float), 1e-4, None)
        avg_pred = float(np.average(x, weights=w))
        avg_actual = float(np.average(y, weights=w))
        if stretch:
            var_pred = float(np.average((x - avg_pred) ** 2, weights=w))
            var_actual = float(np.average((y - avg_actual) ** 2, weights=w))
            if var_pred <= 1e-6:
                slope = 1.0
            else:
                slope = var_actual ** 0.5 / (var_pred ** 0.5 if var_pred > 0 else 1.0)
            if not np.isfinite(slope) or slope <= 0.0:
                slope = 1.0
        else:
            slope, intercept_poly = np.polyfit(x, y, 1, w=np.sqrt(w))
            if not np.isfinite(slope) or abs(slope) < 1e-6:
                slope = 1.0
            if not np.isfinite(intercept_poly):
                intercept_poly = 0.0
            avg_actual = float(np.average(y, weights=w))
            avg_pred = float(np.average(x, weights=w))
            intercept = float(intercept_poly)
            slope = float(slope)
            intercept = float(np.clip(intercept, -15.0, 15.0))
            slope = float(np.clip(slope, 0.3, 1.5))
            return (intercept, slope)

        slope = float(np.clip(slope, 0.6, 2.5))
        intercept = float(avg_actual - slope * avg_pred)
        intercept = float(np.clip(intercept, -15.0, 15.0))
        return (float(intercept), float(slope))

    return _fit(margins_pred, margins_actual, margins_weight, stretch=True), _fit(totals_pred, totals_actual, totals_weight, stretch=False)


def fit_probability_sigma(book: RatingBook, games: Iterable[dict]) -> Optional[float]:
    spreads = []
    outcomes = []
    weights = []
    for game in games:
        if game.get("completed") is not True:
            continue
        if game.get("homeClassification") != "fbs" or game.get("awayClassification") != "fbs":
            continue
        home_points = game.get("homePoints")
        away_points = game.get("awayPoints")
        if home_points is None or away_points is None:
            continue
        try:
            pred = book.predict(game["homeTeam"], game["awayTeam"], neutral_site=game.get("neutralSite", False))
        except KeyError:
            continue
        weight = float(game.get("__cal_weight", 1.0))
        if not math.isfinite(weight) or weight <= 0.0:
            continue
        spreads.append(pred["spread_home_minus_away"])
        outcomes.append(1.0 if home_points > away_points else 0.0)
        weights.append(weight)
    if len(spreads) < 25:
        return None
    spreads = np.array(spreads)
    outcomes = np.array(outcomes)
    weights_arr = np.clip(np.array(weights, dtype=float), 1e-4, None)
    candidates = np.linspace(6.0, 25.0, 200)
    best_sigma = None
    best_loss = float("inf")
    for sigma in candidates:
        scaled = spreads / (sigma * math.sqrt(2))
        erf_vals = np.vectorize(math.erf)(scaled)
        probs = 0.5 * (1.0 + erf_vals)
        loss = np.average((probs - outcomes) ** 2, weights=weights_arr)
        if loss < best_loss:
            best_loss = loss
            best_sigma = sigma
    if best_sigma is None:
        return None
    return float(best_sigma)
