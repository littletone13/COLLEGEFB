from __future__ import annotations

from datetime import datetime, timezone

import pytest

from cfb.io import the_odds_api
from cfb.fcs_aliases import map_team as map_fcs_team
from fbs import _log_missing_odds as log_missing_fbs


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def test_american_decimal_conversion_round_trip() -> None:
    assert the_odds_api.american_to_decimal(-110) == pytest.approx(1.90909, rel=1e-5)
    assert the_odds_api.american_to_decimal(150) == pytest.approx(2.5, rel=1e-9)
    assert the_odds_api.decimal_to_american(1.90909) == pytest.approx(-110, abs=1e-3)
    assert the_odds_api.decimal_to_american(2.5) == pytest.approx(150, rel=1e-9)


def test_normalise_prices_maps_home_away() -> None:
    event = {
        "id": "event123",
        "commence_time": "2025-10-01T18:00:00Z",
        "home_team": "Home Team",
        "away_team": "Away Team",
        "bookmakers": [
            {
                "key": "fanduel",
                "last_update": "2025-10-01T12:00:00Z",
                "markets": [
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": "Home Team", "price": -110, "point": -3.5},
                            {"name": "Away Team", "price": -102, "point": 3.5},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": -115, "point": 51.5},
                            {"name": "Under", "price": -105, "point": 51.5},
                        ],
                    },
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Home Team", "price": -150},
                            {"name": "Away Team", "price": 130},
                        ],
                    },
                ],
            }
        ],
    }
    rows = the_odds_api.normalise_prices(event)
    assert len(rows) == 3
    spread = next(row for row in rows if row["market"] == "spread")
    assert spread["point"] == -3.5
    assert spread["price_home"] == -110
    assert spread["price_away"] == -102
    total = next(row for row in rows if row["market"] == "total")
    assert total["point"] == 51.5
    assert total["price_over"] == -115
    assert total["price_under"] == -105
    moneyline = next(row for row in rows if row["market"] == "moneyline")
    assert moneyline["price_home"] == -150
    assert moneyline["price_away"] == 130


def test_iter_history_snapshots_retries(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_fetch(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] < 2:
            raise the_odds_api.TheOddsAPIError("temporary")
        return {
            "timestamp": "2025-10-01T12:00:00Z",
            "next_timestamp": None,
            "data": [],
        }

    monkeypatch.setattr(the_odds_api, "fetch_history_snapshot", fake_fetch)
    snapshots = list(
        the_odds_api.iter_history_snapshots(
            "americanfootball_ncaaf",
            start=_utc(2025, 10, 1, 12, 0),
            end=_utc(2025, 10, 1, 12, 30),
            retry_attempts=2,
            retry_backoff=0,
        )
    )
    assert len(snapshots) == 1
    assert calls["count"] == 2


def test_iter_history_snapshots_raises_after_retries(monkeypatch) -> None:
    def always_fail(*args, **kwargs):
        raise the_odds_api.TheOddsAPIError("hard failure")

    monkeypatch.setattr(the_odds_api, "fetch_history_snapshot", always_fail)
    with pytest.raises(the_odds_api.TheOddsAPIError):
        list(
            the_odds_api.iter_history_snapshots(
                "americanfootball_ncaaf",
                start=_utc(2025, 10, 1, 12, 0),
                end=_utc(2025, 10, 1, 12, 30),
                retry_attempts=2,
                retry_backoff=0,
            )
        )


def test_log_missing_odds_warns(caplog) -> None:
    lookup = {
        1: {"home_team": "Alpha", "away_team": "Beta"},
        2: {"home_team": "Gamma", "away_team": "Delta"},
    }
    caplog.set_level("WARNING", logger="fbs")
    log_missing_fbs(lookup, {1}, label="FBS")
    assert "Alpha vs Beta" not in caplog.text  # matched game suppressed
    assert "Gamma vs Delta" in caplog.text
    assert "The Odds API returned no FBS odds" in caplog.text

    caplog.clear()
    log_missing_fbs(lookup, {1, 2}, label="FBS")
    assert not caplog.records

    caplog.clear()
    log_missing_fbs(lookup, None, label="FBS")
    assert not caplog.records


def test_fcs_alias_variants_collapse() -> None:
    assert map_fcs_team("St. Thomas") == "STTHOM"
    assert map_fcs_team("St. Thomas (MN)") == "STTHOM"
    assert map_fcs_team("Stephen F. Austin") == "STF AUSTIN"
    assert map_fcs_team("LIU Sharks") == "LIUSHAR"
