import pytest

pd = pytest.importorskip("pandas")

from cfb.market import EdgeFilterConfig, annotate_edges, allow_spread_bet, edge_filter_mask


def test_annotate_edges_creates_expected_columns():
    df = pd.DataFrame(
        {
            "spread": [3.5, -1.0],
            "market_spread": [2.0, -1.5],
            "total": [52.5, 45.0],
            "market_total": [50.0, 46.0],
            "home_win_prob": [0.62, 0.48],
            "market_provider_count": [3, 1],
        }
    )

    result = annotate_edges(
        df,
        model_spread_col="spread",
        market_spread_col="market_spread",
        model_total_col="total",
        market_total_col="market_total",
        win_prob_col="home_win_prob",
        provider_count_col="market_provider_count",
    )

    assert "spread_edge" in result.columns
    assert result.loc[0, "spread_edge"] == pytest.approx(1.5)
    assert result.loc[1, "spread_edge"] == pytest.approx(0.5)
    assert result.loc[0, "total_edge"] == pytest.approx(2.5)
    assert result.loc[1, "total_edge"] == pytest.approx(-1.0)
    assert result.loc[0, "provider_count"] == 3


def test_edge_filter_mask_respects_thresholds():
    df = pd.DataFrame(
        {
            "spread_edge": [0.8, 1.2, 2.0],
            "total_edge": [0.1, 0.5, 0.2],
            "provider_count": [1, 2, 2],
        }
    )

    config = EdgeFilterConfig(spread_edge_min=1.0, min_provider_count=2)
    mask = edge_filter_mask(df, config)
    assert list(mask) == [False, True, True]


def test_allow_spread_bet_handles_missing_values():
    config = EdgeFilterConfig(spread_edge_min=1.0, min_provider_count=1)
    assert not allow_spread_bet(None, 2, config)
    assert not allow_spread_bet(float("nan"), 2, config)
    assert not allow_spread_bet(0.5, 2, config)
    assert not allow_spread_bet(1.5, 0, config)
    assert allow_spread_bet(1.5, 2, config)
