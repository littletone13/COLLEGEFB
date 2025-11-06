
from __future__ import annotations

def decimal_to_american(d: float) -> int:
    if d <= 1.0:
        return 0
    return int(round(100*(d-1))) if d >= 2.0 else int(round(-100/(d-1)))

def american_to_decimal(a: float) -> float:
    return 1 + (a/100.0) if a > 0 else 1 + (100.0/abs(a))

def ev_over(prob_over: float, price_decimal: float) -> float:
    return prob_over*price_decimal - 1.0

def breakeven_prob(price_decimal: float) -> float:
    return 1.0 / price_decimal

def fair_price_decimal(prob_over: float) -> float:
    return 1.0 / max(1e-9, prob_over)

def edge_vs_assumed(prob_over: float, assumed_price: float) -> dict:
    be = breakeven_prob(assumed_price)
    ev = ev_over(prob_over, assumed_price)
    return {
        "assumed_price_decimal": assumed_price,
        "assumed_price_american": decimal_to_american(assumed_price),
        "breakeven_prob": be,
        "edge_prob_pts": prob_over - be,
        "ev_over": ev,
        "fair_price_decimal": fair_price_decimal(prob_over),
        "fair_price_american": decimal_to_american(fair_price_decimal(prob_over)),
    }
