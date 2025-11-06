"""FCS-specific rating book implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np

from cfb.model.rating import BayesianConfig


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
        team_games: Optional[Dict[str, float]] = None,
        bayesian_config: Optional[BayesianConfig] = None,
    ) -> None:
        self.teams = teams
        self.constants = constants
        self.spread_calibration = spread_calibration
        self.total_calibration = total_calibration
        self.prob_sigma = prob_sigma
        self.spread_model = spread_model or {}
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
        calibrated: bool = True,
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

        spread = spread_raw
        total = total_raw
        sigma_local = self.prob_sigma
        spread_weight = total_weight = 1.0
        if calibrated:
            spread = self.spread_calibration[0] + self.spread_calibration[1] * spread
            total = self.total_calibration[0] + self.total_calibration[1] * total
            if self.bayesian_config:
                spread, total, sigma_local, spread_weight, total_weight = self._apply_bayesian_adjustments(
                    spread,
                    total,
                    a.team_name,
                    b.team_name,
                )

        total = max(20.0, total)
        a_points = (total + spread) / 2.0
        b_points = total - a_points

        win_prob_a = self._spread_to_win_prob(spread, sigma_local)
        win_prob_b = 1.0 - win_prob_a

        result = {
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
        if calibrated and self.bayesian_config:
            result["prob_sigma_used"] = sigma_local
            result["bayesian_spread_weight"] = spread_weight
            result["bayesian_total_weight"] = total_weight
        return result

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

    def _spread_to_win_prob(self, spread: float, sigma: float) -> float:
        z = spread / (sigma * math.sqrt(2))
        return 0.5 * (1.0 + math.erf(z))

    @staticmethod
    def _prob_to_moneyline(prob: float) -> Optional[float]:
        if prob <= 0.0 or prob >= 1.0:
            return None
        if prob >= 0.5:
            return -100.0 * prob / (1.0 - prob)
        return 100.0 * (1.0 - prob) / prob

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
        team_one: str,
        team_two: str,
    ) -> Tuple[float, float, float, float, float]:
        cfg = self.bayesian_config
        if cfg is None:
            return spread, total, self.prob_sigma, 1.0, 1.0

        games_one = self._clamp_games(self._team_games(team_one))
        games_two = self._clamp_games(self._team_games(team_two))
        combined_games = games_one + games_two
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
