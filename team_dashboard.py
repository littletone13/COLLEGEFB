"""Interactive Streamlit dashboard for auditing FCS team-level inputs."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd
import streamlit as st

import fcs


CATEGORY_GROUPS = {
    "Rating Snapshot": [
        "spread_rating",
        "offense_rating",
        "defense_rating",
        "special_rating",
        "power_rating",
    ],
    "PFF Skill / Receiving": [
        "receiving_grade_offense",
        "receiving_grade_route",
        "receiving_grade_pass_block",
        "receiving_target_qb_rating",
        "receiving_yprr",
        "receiving_catch_rate",
    ],
    "PFF Line / Blocking": [
        "blocking_grade_offense",
        "blocking_grade_pass",
        "blocking_grade_run",
        "blocking_pbe",
        "pressures_allowed_total",
        "sacks_allowed_total",
    ],
    "PFF Defense": [
        "defense_grade_overall",
        "defense_grade_coverage",
        "defense_grade_run",
        "defense_grade_pass_rush",
        "defense_missed_tackle_rate",
        "defense_qb_rating_against",
        "defense_turnovers",
        "defense_sacks",
        "defense_pressures",
    ],
    "Special Teams": [
        "special_grade_misc",
        "special_grade_return",
        "special_grade_punt_return",
        "special_grade_kickoff",
        "special_grade_fg_offense",
        "special_tackles",
    ],
    "Offense Efficiency": [
        "plays_per_game",
        "offense_ypp",
        "offense_ypg",
        "third_down_pct",
        "team_pass_eff",
        "points_per_game",
        "avg_time_of_possession",
        "red_zone_pct",
    ],
    "Defense Efficiency": [
        "defense_ypp_allowed",
        "defense_ypg_allowed",
        "third_down_def_pct",
        "opp_pass_eff",
        "points_allowed_per_game",
        "red_zone_def_pct",
    ],
    "Opponent Adjustments": [
        "opp_avg_offense_ypp",
        "opp_avg_defense_ypp_allowed",
        "offense_ypp_adj",
        "points_per_game_adj",
        "team_pass_eff_adj",
        "defense_ypp_allowed_adj",
        "points_allowed_per_game_adj",
    ],
    "Ball Control / Discipline": [
        "turnover_gain",
        "turnover_loss",
        "turnover_margin_total",
        "turnover_margin_avg",
        "penalties_per_game",
        "penalty_yards_per_game",
    ],
    "Yardage Splits": [
        "rush_yards_per_game",
        "rush_yards_allowed_per_game",
        "pass_yards_per_game",
        "pass_yards_allowed_per_game",
        "sacks_per_game",
        "sacks_allowed_per_game",
        "tfl_per_game",
        "tfl_allowed_per_game",
    ],
}


def friendly_metric_name(name: str) -> str:
    return name.replace("_", " ").title()


@st.cache_data(show_spinner=False)
def load_ratings(season_year: int, data_dir: str) -> pd.DataFrame:
    return fcs.load_team_ratings(season_year=season_year, data_dir=Path(data_dir))


def format_metric_table(team_row: pd.Series, columns: Iterable[str]) -> pd.DataFrame:
    records: List[dict] = []
    for col in columns:
        if col not in team_row.index:
            continue
        value = team_row[col]
        if pd.isna(value):
            continue
        record = {
            "Metric": friendly_metric_name(col),
            "Value": value,
        }
        z_col = f"{col}_z"
        if z_col in team_row.index and pd.notna(team_row[z_col]):
            record["Z-Score"] = team_row[z_col]
        records.append(record)
    return pd.DataFrame(records)


def render_team_details(team_row: pd.Series) -> None:
    st.subheader(f"{team_row['team_name']} Inputs", divider="gray")
    cols = st.columns(5)
    cols[0].metric("Spread Rating", f"{team_row['spread_rating']:.2f}")
    cols[1].metric("Offense", f"{team_row['offense_rating']:.2f}")
    cols[2].metric("Defense", f"{team_row['defense_rating']:.2f}")
    cols[3].metric("Special Teams", f"{team_row['special_rating']:.2f}")
    cols[4].metric("Power", f"{team_row['power_rating']:.2f}")

    for title, columns in CATEGORY_GROUPS.items():
        table = format_metric_table(team_row, columns)
        if table.empty:
            continue
        st.markdown(f"#### {title}")
        st.dataframe(
            table.set_index("Metric").sort_index(),
            use_container_width=True,
        )


def render_compare_table(teams: pd.DataFrame, selected: list[str]) -> None:
    if not selected:
        return
    subset = (
        teams[teams["team_name"].isin(selected)]
        .set_index("team_name")
        [["spread_rating", "offense_rating", "defense_rating", "special_rating", "power_rating"]]
        .sort_values("power_rating", ascending=False)
    )
    st.markdown("#### Comparison Table")
    st.dataframe(subset, use_container_width=True)


def render_scatter(teams: pd.DataFrame) -> None:
    chart_df = teams[["team_name", "offense_rating", "defense_rating", "power_rating"]]
    chart_df = chart_df.rename(columns={
        "offense_rating": "Offense Rating",
        "defense_rating": "Defense Rating",
        "power_rating": "Power Rating",
    })
    st.markdown("#### Offense vs Defense")
    st.scatter_chart(
        chart_df,
        x="Offense Rating",
        y="Defense Rating",
        color="Power Rating",
        size="Power Rating",
    )


def main() -> None:
    st.set_page_config(page_title="FCS Team Dashboard", layout="wide")
    st.title("FCS Team Input Dashboard")
    st.caption("Explore aggregated PFF + NCAA features that feed the model.")

    sidebar = st.sidebar
    sidebar.header("Controls")
    default_dir = str(fcs.DATA_DIR_DEFAULT)
    season_year = sidebar.number_input("Season Year", min_value=2018, max_value=2030, value=2025, step=1)
    data_dir = sidebar.text_input("PFF Data Directory", value=default_dir)

    try:
        teams = load_ratings(season_year=season_year, data_dir=data_dir)
    except FileNotFoundError as exc:
        st.error(f"Failed to load ratings: {exc}")
        return

    teams = teams.sort_values("team_name").reset_index(drop=True)
    team_names = teams["team_name"].tolist()

    selected_team = sidebar.selectbox("Primary Team", options=team_names, index=0)
    compare_choices = sidebar.multiselect("Compare Teams", options=team_names, default=[selected_team])

    render_scatter(teams)
    render_compare_table(teams, compare_choices)

    team_row = teams.loc[teams["team_name"] == selected_team].iloc[0]
    render_team_details(team_row)

    st.markdown("---")
    st.markdown("#### Full Dataset")
    st.dataframe(
        teams[["team_name", "spread_rating", "offense_rating", "defense_rating", "special_rating", "power_rating"]],
        use_container_width=True,
    )
    st.download_button(
        "Download Ratings CSV",
        data=teams.to_csv(index=False).encode("utf-8"),
        file_name=f"fcs_team_ratings_{season_year}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
