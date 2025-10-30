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


class RatingBook:
    def __init__(
        self,
        teams: pd.DataFrame,
        constants: RatingConstants,
        power_adjustments: Optional[Dict[str, float]] = None,
    ) -> None:
        self.teams = teams
        self.constants = constants
        self.power_adjustments = power_adjustments or {}
        self.spread_calibration: Tuple[float, float] = (0.0, 1.0)
        self.total_calibration: Tuple[float, float] = (0.0, 1.0)
        self.prob_sigma: float = constants.spread_sigma

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
        home_points = (total + spread) / 2.0
        away_points = total - home_points
        home_win_prob = 0.5 * (1.0 + math.erf(spread / (self.prob_sigma * math.sqrt(2))))
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
        return result


def fit_linear_calibrations(book: RatingBook, games: Iterable[dict]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    margins_pred: list[float] = []
    margins_actual: list[float] = []
    totals_pred: list[float] = []
    totals_actual: list[float] = []

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
        margins_pred.append(raw["spread_home_minus_away"])
        margins_actual.append(home_points - away_points)
        totals_pred.append(raw["total_points"])
        totals_actual.append(home_points + away_points)

    def _fit(pred: list[float], actual: list[float]) -> Tuple[float, float]:
        if len(pred) < 30:
            return (0.0, 1.0)
        x = np.array(pred)
        y = np.array(actual)
        slope, intercept = np.polyfit(x, y, 1)
        if not np.isfinite(slope) or abs(slope) < 1e-6:
            slope = 1.0
        if not np.isfinite(intercept):
            intercept = 0.0
        slope = float(np.clip(slope, 0.3, 1.5))
        intercept = float(np.clip(intercept, -15.0, 15.0))
        return (float(intercept), float(slope))

    return _fit(margins_pred, margins_actual), _fit(totals_pred, totals_actual)


def fit_probability_sigma(book: RatingBook, games: Iterable[dict]) -> Optional[float]:
    spreads = []
    outcomes = []
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
        spreads.append(pred["spread_home_minus_away"])
        outcomes.append(1.0 if home_points > away_points else 0.0)
    if len(spreads) < 25:
        return None
    spreads = np.array(spreads)
    outcomes = np.array(outcomes)
    candidates = np.linspace(6.0, 25.0, 200)
    best_sigma = None
    best_loss = float("inf")
    for sigma in candidates:
        scaled = spreads / (sigma * math.sqrt(2))
        erf_vals = np.vectorize(math.erf)(scaled)
        probs = 0.5 * (1.0 + erf_vals)
        loss = np.mean((probs - outcomes) ** 2)
        if loss < best_loss:
            best_loss = loss
            best_sigma = sigma
    if best_sigma is None:
        return None
    return float(best_sigma)
