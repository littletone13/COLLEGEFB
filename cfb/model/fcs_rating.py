"""FCS-specific rating book implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd


@dataclass
class RatingConstants:
    avg_total: float
    home_field_advantage: float
    offense_factor: float
    defense_factor: float
    special_teams_factor: float
    spread_sigma: float

    @property
    def avg_team_points(self) -> float:
        return self.avg_total / 2.0


@dataclass
class TeamRatings:
    team_name: str
    offense_rating: float
    defense_rating: float
    special_teams_rating: float
    power_rating: float
    spread_rating: float


class RatingBook:
    """Rating book tailored to the FCS heuristic model."""

    def __init__(
        self,
        teams: pd.DataFrame,
        constants: RatingConstants,
        *,
        spread_calibration: Tuple[float, float],
        total_calibration: Tuple[float, float],
        prob_sigma: float,
        spread_model: Optional[Dict[str, float]] = None,
    ) -> None:
        self.teams = teams
        self.constants = constants
        self.spread_calibration = spread_calibration
        self.total_calibration = total_calibration
        self.prob_sigma = prob_sigma
        self.spread_model = spread_model or {}

    def _find_row(self, team: str) -> pd.Series:
        df = self.teams
        mask = df["team_name"].str.lower() == team.lower()
        if mask.sum() == 1:
            return df.loc[mask].iloc[0]
        mask = df["team_name"].str.contains(team, case=False, regex=False)
        if mask.sum() == 1:
            return df.loc[mask].iloc[0]
        available = ", ".join(sorted(df["team_name"].unique()))
        raise KeyError(f"Team '{team}' not found. Available teams: {available}")

    def get(self, team: str) -> TeamRatings:
        row = self._find_row(team)
        return TeamRatings(
            team_name=row["team_name"],
            offense_rating=float(row["offense_rating"]),
            defense_rating=float(row["defense_rating"]),
            special_teams_rating=float(row["special_rating"]),
            power_rating=float(row["power_rating"]),
            spread_rating=float(row.get("spread_rating", row["power_rating"])),
        )

    def predict(
        self,
        team_one: str,
        team_two: str,
        *,
        neutral_site: bool = False,
    ) -> Dict[str, float]:
        a = self.get(team_one)
        b = self.get(team_two)
        const = self.constants

        a_points = self._expected_points(a, b)
        b_points = self._expected_points(b, a)

        spread_raw = self._spread_from_model(a, b, a_points, b_points)
        if not neutral_site:
            spread_raw += const.home_field_advantage
            a_points += const.home_field_advantage / 2.0
            b_points -= const.home_field_advantage / 2.0

        total_raw = a_points + b_points

        spread = self.spread_calibration[0] + self.spread_calibration[1] * spread_raw
        total = self.total_calibration[0] + self.total_calibration[1] * total_raw
        total = max(20.0, total)

        a_points = (total + spread) / 2.0
        b_points = total - a_points

        win_prob_a = self._spread_to_win_prob(spread)
        win_prob_b = 1.0 - win_prob_a

        return {
            "team_one": a.team_name,
            "team_two": b.team_name,
            "team_one_points": a_points,
            "team_two_points": b_points,
            "spread_team_one_minus_team_two": spread,
            "total_points": total,
            "team_one_win_prob": win_prob_a,
            "team_two_win_prob": win_prob_b,
            "team_one_moneyline": self._prob_to_moneyline(win_prob_a),
            "team_two_moneyline": self._prob_to_moneyline(win_prob_b),
        }

    def _spread_from_model(self, a: TeamRatings, b: TeamRatings, a_points: float, b_points: float) -> float:
        if not self.spread_model:
            return a_points - b_points
        coeffs = self.spread_model
        spread_diff = a.spread_rating - b.spread_rating
        power_diff = a.power_rating - b.power_rating
        offense_diff = a.offense_rating - b.offense_rating
        defense_diff = b.defense_rating - a.defense_rating
        special_diff = a.special_teams_rating - b.special_teams_rating
        sr_power = spread_diff * power_diff
        off_def = offense_diff * defense_diff
        power_special = power_diff * special_diff
        power_sq = power_diff * power_diff
        return (
            coeffs.get("intercept", 0.0)
            + coeffs.get("spread_rating_diff", 1.0) * spread_diff
            + coeffs.get("power_diff", 0.0) * power_diff
            + coeffs.get("offense_diff", 0.0) * offense_diff
            + coeffs.get("defense_diff", 0.0) * defense_diff
            + coeffs.get("special_diff", 0.0) * special_diff
            + coeffs.get("sr_x_power", 0.0) * sr_power
            + coeffs.get("off_x_def", 0.0) * off_def
            + coeffs.get("power_x_special", 0.0) * power_special
            + coeffs.get("power_sq", 0.0) * power_sq
        )

    def _expected_points(self, offense: TeamRatings, defense: TeamRatings) -> float:
        const = self.constants
        return (
            const.avg_team_points
            + const.offense_factor * offense.offense_rating
            - const.defense_factor * defense.defense_rating
            + const.special_teams_factor * (offense.special_teams_rating - defense.special_teams_rating) / 2.0
        )

    def _spread_to_win_prob(self, spread: float) -> float:
        sigma = self.prob_sigma
        z = spread / (sigma * math.sqrt(2))
        return 0.5 * (1.0 + math.erf(z))

    @staticmethod
    def _prob_to_moneyline(prob: float) -> Optional[float]:
        if prob <= 0.0 or prob >= 1.0:
            return None
        if prob >= 0.5:
            return -100.0 * prob / (1.0 - prob)
        return 100.0 * (1.0 - prob) / prob
