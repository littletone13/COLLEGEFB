from __future__ import annotations

import os
from datetime import date
from typing import Any, Dict

import pandas as pd
import requests
import streamlit as st

DEFAULT_API_BASE = os.environ.get("SIMS_API_URL", "http://localhost:8000")
DEFAULT_API_KEY = os.environ.get("SIMS_API_KEY")


def _call_api(base_url: str, path: str, params: Dict[str, Any], api_key: str | None) -> Dict[str, Any]:
    payload = dict(params)
    if api_key:
        payload["auth"] = api_key
    resp = requests.get(f"{base_url}{path}", params=payload, timeout=45)
    resp.raise_for_status()
    return resp.json()


def _render_dataframe(df: pd.DataFrame, *, title: str) -> None:
    if df.empty:
        st.info(f"No rows returned for {title} with the current filters.")
        return
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    highlight_cols = [col for col in ["spread_edge", "total_edge", "win_prob_edge"] if col in df.columns]
    styled = df.style
    for col in highlight_cols:
        styled = styled.background_gradient(subset=[col], cmap="RdYlGn")
    st.dataframe(styled.format(precision=3), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="College Football Model Dashboard", layout="wide")
    st.title("College Football Model Dashboard")

    with st.sidebar:
        st.header("API Settings")
        api_base = st.text_input("API Base URL", value=DEFAULT_API_BASE)
        api_key = st.text_input("API Key", value=DEFAULT_API_KEY or "", type="password") or None
        st.markdown("Edge filters apply on the server via the FastAPI layer.")

    tabs = st.tabs(["FBS Week", "FCS Window", "OddsLogic Coverage"])

    with tabs[0]:
        st.subheader("FBS Weekly Sims")
        today = date.today()
        default_year = today.year + 1 if today.month >= 7 else today.year
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            year = st.number_input("Year", min_value=2000, max_value=2100, value=default_year, step=1)
        with col_b:
            week = st.number_input("Week", min_value=1, max_value=20, value=10, step=1)
        with col_c:
            min_provider_count = st.number_input("Min providers", min_value=0, max_value=10, value=1, step=1)
        include_completed = st.checkbox("Include completed games", value=False)
        neutral_default = st.checkbox("Treat missing neutral flag as neutral site", value=False)
        providers = st.text_input("Provider filter (comma separated)")
        min_spread_edge = st.slider("Min spread edge (pts)", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
        min_total_edge = st.slider("Min total edge (pts)", min_value=0.0, max_value=10.0, value=0.0, step=0.5)

        if st.button("Run FBS Sims", type="primary"):
            params = {
                "year": int(year),
                "week": int(week),
                "include_completed": include_completed,
                "neutral_default": neutral_default,
                "min_provider_count": int(min_provider_count),
                "min_spread_edge": float(min_spread_edge),
                "min_total_edge": float(min_total_edge),
            }
            if providers:
                params["providers"] = providers
            try:
                payload = _call_api(api_base, "/fbs/week", params, api_key)
            except requests.HTTPError as exc:
                st.error(f"API error: {exc}")
            except requests.RequestException as exc:
                st.error(f"Network error: {exc}")
            else:
                games = pd.DataFrame(payload.get("games", []))
                meta = payload.get("meta", {})
                st.caption(
                    f"Returned {meta.get('returned_games', len(games))} of {meta.get('total_games', len(games))} games"
                )
                _render_dataframe(games, title="FBS Sims")

    with tabs[1]:
        st.subheader("FCS Slate Sims")
        default_start = date(today.year, today.month, today.day)
        start_date = st.date_input("Start date", value=default_start, format="YYYY-MM-DD")
        days = st.slider("Window (days)", min_value=1, max_value=7, value=3)
        week_hint = st.text_input("Optional CFBD week (blank for auto)")
        providers_fcs = st.text_input("Provider filter (comma separated)", key="providers_fcs")
        min_spread_edge_fcs = st.slider("Min spread edge", min_value=0.0, max_value=10.0, value=1.0, step=0.5, key="spread_edge_fcs")
        min_total_edge_fcs = st.slider("Min total edge", min_value=0.0, max_value=10.0, value=0.0, step=0.5, key="total_edge_fcs")
        min_provider_count_fcs = st.number_input("Min providers", min_value=0, max_value=10, value=1, step=1, key="providers_count_fcs")

        if st.button("Run FCS Sims", type="primary"):
            params = {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "days": int(days),
                "min_spread_edge": float(min_spread_edge_fcs),
                "min_total_edge": float(min_total_edge_fcs),
                "min_provider_count": int(min_provider_count_fcs),
            }
            if week_hint.strip():
                params["week"] = int(week_hint)
            if providers_fcs:
                params["providers"] = providers_fcs
            try:
                payload = _call_api(api_base, "/fcs/window", params, api_key)
            except requests.HTTPError as exc:
                st.error(f"API error: {exc}")
            except requests.RequestException as exc:
                st.error(f"Network error: {exc}")
            else:
                games = pd.DataFrame(payload.get("games", []))
                meta = payload.get("meta", {})
                st.caption(
                    f"Returned {meta.get('returned_games', len(games))} of {meta.get('total_games', len(games))} games"
                )
                _render_dataframe(games, title="FCS Sims")

    with tabs[2]:
        st.subheader("OddsLogic Coverage Snapshot")
        classification = st.selectbox("Classification", options=["all", "fbs", "fcs"], index=0)
        if st.button("Fetch Coverage"):
            params: Dict[str, Any] = {}
            if classification != "all":
                params["classification"] = classification
            try:
                payload = _call_api(api_base, "/oddslogic/coverage", params, api_key)
            except requests.HTTPError as exc:
                st.error(f"API error: {exc}")
            except requests.RequestException as exc:
                st.error(f"Network error: {exc}")
            else:
                coverage = pd.DataFrame(payload.get("coverage", []))
                providers = pd.DataFrame(payload.get("provider_coverage", []))
                st.markdown("### Coverage by Classification")
                _render_dataframe(coverage, title="Coverage summary")
                if not providers.empty:
                    st.markdown("### Provider Counts")
                    _render_dataframe(providers, title="Provider coverage")


if __name__ == "__main__":
    main()
