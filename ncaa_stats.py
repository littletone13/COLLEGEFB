"""Utilities for ingesting NCAA.com FCS statistics.

The site exposes per-category tables for both team and individual stats.
This helper downloads the relevant tables (handling pagination), extracts
team slugs, and normalises them to match the PFF naming convention used
in ``fcs.py``.

Network requests are lightweight (<10 pages per stat) but, to be a good
citizen, callers should reuse the cached data between runs when
possible.
"""
from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.ncaa.com/stats/football/fcs/current"
SCOREBOARD_BASE_URL = "https://www.ncaa.com/scoreboard/football/fcs"
USER_AGENT = "Mozilla/5.0 (compatible; CodexBot/0.1; +https://openai.com)"
HEADERS = {"User-Agent": USER_AGENT}

# Manual crosswalk between NCAA school slugs and the PFF naming used in the
# local dataset. The slug set comes from the NCAA stats tables.
SLUG_TO_PFF: Dict[str, str] = {
    'abilene-christian': 'ABILENE CH',
    'alabama-am': 'ALAB A&M',
    'alabama-st': 'ALABAMA ST',
    'albany-ny': 'ALBANY',
    'alcorn': 'ALCORN ST',
    'ark-pine-bluff': 'ARKAPB',
    'austin-peay': 'AUSTINPEAY',
    'bethune-cookman': 'BETH COOK',
    'brown': 'BROWN',
    'bryant': 'BRYANT',
    'bucknell': 'BUCKNELL',
    'butler': 'BUTLER',
    'cal-poly': 'CAL POLY',
    'campbell': 'CAMPBELL',
    'central-ark': 'CENT ARK',
    'central-conn-st': 'CENT CT ST',
    'charleston-so': 'CHARLES SO',
    'chattanooga': 'CHATTNOOGA',
    'citadel': 'CITADEL',
    'colgate': 'COLGATE',
    'columbia': 'COLUMBIA',
    'cornell': 'CORNELL',
    'dartmouth': 'DARTMOUTH',
    'davidson': 'DAVIDSON',
    'dayton': 'DAYTON',
    'delaware-st': 'DELWARE ST',
    'drake': 'DRAKE',
    'duquesne': 'DUQUESNE',
    'east-tenn-st': 'E TENN ST',
    'eastern-ill': 'E ILLINOIS',
    'eastern-ky': 'E KENTUCKY',
    'eastern-wash': 'E WASHGTON',
    'elon': 'ELON',
    'florida-am': 'FL A&M',
    'fordham': 'FORDHAM',
    'furman': 'FURMAN',
    'gardner-webb': 'GARD-WEBB',
    'georgetown': 'GEORGETOWN',
    'grambling': 'GRAMBLING',
    'hampton': 'HAMPTON',
    'harvard': 'HARVARD',
    'holy-cross': 'HOLY CROSS',
    'houston-christian': 'HOUCHR',
    'howard': 'HOWARD',
    'idaho': 'IDAHO',
    'idaho-st': 'IDAHO ST',
    'illinois-st': 'ILL STATE',
    'indiana-st': 'INDIANA ST',
    'jackson-st': 'JACKSON ST',
    'lafayette': 'LAFAYETTE',
    'lamar': 'LAMAR',
    'lehigh': 'LEHIGH',
    'lindenwood-mo': 'LNDNWD',
    'long-island': 'LIUSHAR',
    'mercyhurst': 'MERCYHURST',
    'new-haven': 'NEWHVN',
    'maine': 'MAINE',
    'marist': 'MARIST',
    'mcneese': 'MCNEESE',
    'mercer': 'MERCER',
    'merrimack': 'MERRIMACK',
    'mississippi-val': 'MS VLY ST',
    'monmouth': 'MONMOUTH',
    'montana': 'MONTANA',
    'montana-st': 'MONTANA ST',
    'morehead-st': 'MOREHEAD',
    'morgan-st': 'MORGAN ST',
    'murray-st': 'MURRAY ST',
    'nc-at': 'NC A&T',
    'nc-central': 'NC CENT',
    'new-hampshire': 'NEW HAMP',
    'nicholls-st': 'NICHOLLS',
    'norfolk-st': 'NORFOLK',
    'north-ala': 'N ALABAMA',
    'north-dakota': 'N DAKOTA',
    'north-dakota-st': 'N DAK ST',
    'northern-ariz': 'N ARIZONA',
    'northern-colo': 'N COLORADO',
    'northwestern-st': 'NWSTATE',
    'penn': 'PENN',
    'portland-st': 'PORTLAND',
    'prairie-view': 'PRVIEW A&M',
    'presbyterian': 'PRESBYTERN',
    'princeton': 'PRINCETON',
    'rhode-island': 'RHODE ISLD',
    'richmond': 'RICHMOND',
    'robert-morris': 'ROB MORRIS',
    'sacramento-st': 'SACRAMENTO',
    'sacred-heart': 'SACR HEART',
    'samford': 'SAMFORD',
    'san-diego': 'SAN DIEGO',
    'south-carolina-st': 'SCAR STATE',
    'south-dakota': 'S DAKOTA',
    'south-dakota-st': 'S DAK ST',
    'southeast-mo-st': 'SE MO ST',
    'southeastern-la': 'SE LA',
    'southern-ill': 'S ILLINOIS',
    'southern-u': 'SOUTHERN U',
    'southern-utah': 'SO UTAH',
    'st-francis-pa': 'ST FRANCIS',
    'st-thomas-mn': 'STTHOM',
    'stephen-f-austin': 'STF AUSTIN',
    'stetson': 'STETSON',
    'stonehill': 'STHLSK',
    'stony-brook': 'STNY BROOK',
    'tarleton-st': 'TARLETON',
    'tennessee-st': 'TENN STATE',
    'tennessee-tech': 'TENN TECH',
    'tex-am-commerce': 'TXAMCO',
    'texas-southern': 'TEXAS STHN',
    'towson': 'TOWSON',
    'uc-davis': 'UC DAVIS',
    'uiw': 'INCAR WORD',
    'uni': 'N IOWA',
    'ut-martin': 'TENN MARTN',
    'utah-tech': 'UTAHTC',
    'utrgv': 'TXGV',
    'west-ga': 'W GEORGIA',
    'valparaiso': 'VALPO',
    'villanova': 'VILLANOVA',
    'vmi': 'VA MILT IN',
    'wagner': 'WAGNER',
    'weber-st': 'WEBER ST',
    'western-caro': 'W CAROLINA',
    'western-ill': 'W ILLINOIS',
    'william-mary': 'WM & MARY',
    'wofford': 'WOFFORD',
    'yale': 'YALE',
    'youngstown-st': 'YNGTOWN ST',
}


@dataclass
class NcaaStatTable:
    stat_type: str
    stat_id: int
    label: str


TEAM_TABLES: Dict[str, NcaaStatTable] = {
    "total_offense": NcaaStatTable("team", 21, "Total Offense"),
    "total_defense": NcaaStatTable("team", 22, "Total Defense"),
    "third_down_offense": NcaaStatTable("team", 699, "Third Down Conversion Pct"),
    "third_down_defense": NcaaStatTable("team", 701, "Third Down Conversion Pct Defense"),
    "team_passing_efficiency": NcaaStatTable("team", 465, "Team Passing Efficiency"),
    "team_passing_efficiency_def": NcaaStatTable("team", 40, "Team Passing Efficiency Defense"),
    "time_of_possession": NcaaStatTable("team", 705, "Time of Possession"),
    "scoring_offense": NcaaStatTable("team", 27, "Scoring Offense"),
    "scoring_defense": NcaaStatTable("team", 28, "Scoring Defense"),
    "red_zone_offense": NcaaStatTable("team", 703, "Red Zone Offense"),
    "red_zone_defense": NcaaStatTable("team", 704, "Red Zone Defense"),
    "turnover_margin": NcaaStatTable("team", 29, "Turnover Margin"),
    "penalties_pg": NcaaStatTable("team", 697, "Fewest Penalties Per Game"),
    "penalty_yards_pg": NcaaStatTable("team", 698, "Fewest Penalty Yards Per Game"),
    "rushing_offense": NcaaStatTable("team", 23, "Rushing Offense"),
    "rushing_defense": NcaaStatTable("team", 24, "Rushing Defense"),
    "passing_offense": NcaaStatTable("team", 25, "Passing Offense"),
    "passing_defense": NcaaStatTable("team", 695, "Passing Yards Allowed"),
    "team_sacks": NcaaStatTable("team", 466, "Team Sacks"),
    "team_tfl": NcaaStatTable("team", 467, "Team Tackles for Loss"),
    "sacks_allowed": NcaaStatTable("team", 468, "Sacks Allowed"),
    "tfl_allowed": NcaaStatTable("team", 696, "Tackles for Loss Allowed"),
    "net_punting": NcaaStatTable("team", 98, "Net Punting"),
    "kickoff_returns": NcaaStatTable("team", 96, "Kickoff Returns"),
    "kickoff_return_defense": NcaaStatTable("team", 463, "Kickoff Return Defense"),
    "punt_returns": NcaaStatTable("team", 97, "Punt Returns"),
    "punt_return_defense": NcaaStatTable("team", 462, "Punt Return Defense"),
}


@dataclass
class NcaaIndividualStat:
    stat_id: int
    label: str


INDIVIDUAL_TABLES: Dict[str, NcaaIndividualStat] = {
    "passing_efficiency": NcaaIndividualStat(8, "Passing Efficiency"),
    "rushing_yards_per_game": NcaaIndividualStat(7, "Rushing Yards Per Game"),
    "receiving_yards": NcaaIndividualStat(455, "Receiving Yards"),
}


def fetch_table(stat: NcaaStatTable, session: Optional[requests.Session] = None) -> pd.DataFrame:
    """Download the full NCAA stat table (all pages) and return a DataFrame."""

    sess = session or requests.Session()
    rows: List[List[str]] = []
    header: Optional[List[str]] = None
    page = 1

    while True:
        url = f"{BASE_URL}/{stat.stat_type}/{stat.stat_id}"
        if page > 1:
            url += f"/p{page}"
        resp = sess.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            logging.warning("NCAA stats request failed (%s): %s", resp.status_code, url)
            break
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table")
        if not table:
            break
        header = [th.get_text(strip=True) for th in table.find("tr").find_all("th")]
        page_rows: List[List[str]] = []
        for tr in table.find_all("tr")[1:]:
            cells = tr.find_all("td")
            if not cells:
                continue
            slug = None
            values: List[str] = []
            for idx, cell in enumerate(cells):
                text = cell.get_text(strip=True)
                values.append(text)
                if idx == 1:
                    anchor = cell.find("a")
                    if anchor and anchor.get("href"):
                        slug = anchor["href"].rstrip("/").split("/")[-1]
            if slug is None:
                continue
            values.append(slug)
            page_rows.append(values)
        if not page_rows:
            break
        rows.extend(page_rows)
        pager_next = soup.find("li", class_="stats-pager__li--next")
        if not pager_next or not pager_next.find("a") or not pager_next.find("a").get("href"):
            break
        page += 1

    if not rows or not header:
        return pd.DataFrame()

    columns = header + ["slug"]
    df = pd.DataFrame(rows, columns=columns)
    return df


def _coerce_numeric(df: pd.DataFrame, exclude: Iterable[str]) -> pd.DataFrame:
    for col in df.columns:
        if col in exclude:
            continue
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    return df


def fetch_individual_table(stat: NcaaIndividualStat, session: Optional[requests.Session] = None) -> pd.DataFrame:
    """Download an individual leaderboard and annotate with school slug."""

    sess = session or requests.Session()
    rows: List[List[str]] = []
    header: Optional[List[str]] = None
    page = 1

    while True:
        url = f"{BASE_URL}/individual/{stat.stat_id}"
        if page > 1:
            url += f"/p{page}"
        resp = sess.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            break
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table")
        if not table:
            break
        header = [th.get_text(strip=True) for th in table.find("tr").find_all("th")]
        page_rows: List[List[str]] = []
        for tr in table.find_all("tr")[1:]:
            cells = tr.find_all("td")
            if not cells:
                continue
            slug = None
            values: List[str] = []
            for idx, cell in enumerate(cells):
                text = cell.get_text(strip=True)
                values.append(text)
                if idx == 2:
                    anchor = cell.find("a")
                    if anchor and anchor.get("href"):
                        slug = anchor["href"].rstrip("/").split("/")[-1]
            if slug is None:
                continue
            values.append(slug)
            page_rows.append(values)
        if not page_rows:
            break
        rows.extend(page_rows)
        pager_next = soup.find("li", class_="stats-pager__li--next")
        if not pager_next or not pager_next.find("a") or not pager_next.find("a").get("href"):
            break
        page += 1

    if not rows or not header:
        return pd.DataFrame()

    columns = header + ["slug"]
    return pd.DataFrame(rows, columns=columns)


def build_individual_features(session: Optional[requests.Session] = None) -> pd.DataFrame:
    """Aggregate key individual leaderboards into team-level signals."""

    session = session or requests.Session()
    aggregations: List[pd.DataFrame] = []

    stat_specs = [
        ("passing_efficiency", "Pass Eff", "qb_pass_eff"),
        ("rushing_yards_per_game", "YPG", "rb_rush_ypg"),
        ("receiving_yards", "Rec Yds", "wr_rec_yards"),
    ]

    for key, value_col, out_col in stat_specs:
        spec = INDIVIDUAL_TABLES.get(key)
        if spec is None:
            continue
        table = fetch_individual_table(spec, session=session)
        if table.empty or value_col not in table.columns:
            continue
        exclude = {"Rank", "Name", "Team", "Cl", "Position", "slug"}
        table = _coerce_numeric(table, exclude)
        table["team_name"] = table["slug"].map(SLUG_TO_PFF)
        table = table.dropna(subset=["team_name", value_col])
        table = table.sort_values(value_col, ascending=False)
        top = table.groupby("team_name", as_index=False).first()[["team_name", value_col]]
        top = top.rename(columns={value_col: out_col})
        aggregations.append(top)

    if not aggregations:
        return pd.DataFrame(columns=["team_name"])

    features = aggregations[0]
    for frame in aggregations[1:]:
        features = features.merge(frame, on="team_name", how="outer")
    return features


def _date_range_for_season(season_year: int) -> List[date]:
    start = date(season_year, 8, 1)
    today = date.today()
    end = min(today, date(season_year, 12, 31))
    days = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def fetch_scoreboard_games(season_year: int, session: Optional[requests.Session] = None) -> pd.DataFrame:
    session = session or requests.Session()
    records: List[Dict[str, object]] = []

    for day in _date_range_for_season(season_year):
        url = f"{SCOREBOARD_BASE_URL}/{day:%Y/%m/%d}"
        try:
            resp = session.get(url, headers=HEADERS, timeout=15)
        except requests.RequestException:
            continue
        if resp.status_code != 200:
            continue
        soup = BeautifulSoup(resp.text, "html.parser")
        script = soup.find("script", {"data-drupal-selector": "drupal-settings-json"})
        if not script:
            continue
        try:
            settings = json.loads(script.string)
        except json.JSONDecodeError:
            continue
        scoreboard = settings.get("scoreboard")
        if not scoreboard:
            continue
        for game in scoreboard.get("initialGames", []):
            teams = game.get("teams", [])
            if len(teams) != 2:
                continue
            home = next((team for team in teams if team.get("isHome")), None)
            away = next((team for team in teams if not team.get("isHome")), None)
            if not home or not away:
                continue
            records.append({
                "date": day,
                "start_date": game.get("startDate"),
                "game_state": game.get("gameState"),
                "status": game.get("statusCodeDisplay"),
                "home_slug": home.get("seoname"),
                "away_slug": away.get("seoname"),
                "home_score": home.get("score"),
                "away_score": away.get("score"),
            })

    return pd.DataFrame(records)


def apply_opponent_adjustments(
    teams: pd.DataFrame,
    season_year: int,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    schedule = fetch_scoreboard_games(season_year, session=session)
    if schedule.empty:
        adjustment_columns = [
            "opp_avg_offense_ypp",
            "opp_avg_offense_ypg",
            "opp_avg_third_down_pct",
            "opp_avg_team_pass_eff",
            "opp_avg_points_per_game",
            "opp_avg_defense_ypp_allowed",
            "opp_avg_defense_ypg_allowed",
            "opp_avg_third_down_def_pct",
            "opp_avg_opp_pass_eff",
            "opp_avg_points_allowed_per_game",
            "offense_ypp_adj",
            "points_per_game_adj",
            "team_pass_eff_adj",
            "defense_ypp_allowed_adj",
            "points_allowed_per_game_adj",
        ]
        for col in adjustment_columns:
            if col not in teams:
                teams[col] = float("nan")
        return teams

    schedule["home_score"] = pd.to_numeric(schedule["home_score"], errors="coerce")
    schedule["away_score"] = pd.to_numeric(schedule["away_score"], errors="coerce")
    completed = schedule.dropna(subset=["home_score", "away_score"])
    if completed.empty:
        return teams

    records: List[Dict[str, object]] = []
    for row in completed.itertuples(index=False):
        home = SLUG_TO_PFF.get(row.home_slug)
        away = SLUG_TO_PFF.get(row.away_slug)
        if not home or not away:
            continue
        records.append({
            "team": home,
            "opponent": away,
            "points_for": row.home_score,
            "points_against": row.away_score,
        })
        records.append({
            "team": away,
            "opponent": home,
            "points_for": row.away_score,
            "points_against": row.home_score,
        })

    if not records:
        return teams

    games_df = pd.DataFrame(records)
    metrics = [
        "offense_ypp",
        "offense_ypg",
        "third_down_pct",
        "team_pass_eff",
        "points_per_game",
        "defense_ypp_allowed",
        "defense_ypg_allowed",
        "third_down_def_pct",
        "opp_pass_eff",
        "points_allowed_per_game",
    ]
    opp_metrics = teams[["team_name"] + [m for m in metrics if m in teams.columns]].rename(columns={"team_name": "opponent"})
    merged = games_df.merge(opp_metrics, on="opponent", how="left")
    opp_avgs = merged.groupby("team", as_index=True).mean(numeric_only=True)
    opp_avgs = opp_avgs.add_prefix("opp_avg_")
    teams = teams.merge(opp_avgs, left_on="team_name", right_index=True, how="left")

    if "opp_avg_defense_ypp_allowed" in teams.columns:
        teams["offense_ypp_adj"] = teams["offense_ypp"] - teams["opp_avg_defense_ypp_allowed"]
    if "opp_avg_points_allowed_per_game" in teams.columns:
        teams["points_per_game_adj"] = teams["points_per_game"] - teams["opp_avg_points_allowed_per_game"]
    if "opp_avg_opp_pass_eff" in teams.columns:
        teams["team_pass_eff_adj"] = teams["team_pass_eff"] - teams["opp_avg_opp_pass_eff"]
    if "opp_avg_offense_ypp" in teams.columns:
        teams["defense_ypp_allowed_adj"] = teams["defense_ypp_allowed"] - teams["opp_avg_offense_ypp"]
    if "opp_avg_points_per_game" in teams.columns:
        teams["points_allowed_per_game_adj"] = teams["points_allowed_per_game"] - teams["opp_avg_points_per_game"]

    return teams


def build_team_feature_frame(season_year: Optional[int] = None) -> pd.DataFrame:
    """Collect selected NCAA tables and engineer consolidated team metrics."""

    session = requests.Session()
    tables = {key: fetch_table(tbl, session=session) for key, tbl in TEAM_TABLES.items()}

    def _merge_feature(base: pd.DataFrame, key: str, columns: List[str], rename: Dict[str, str]) -> pd.DataFrame:
        table = tables.get(key)
        if table is None or table.empty:
            return base
        frame = _coerce_numeric(table.copy(), {"Rank", "Team", "slug"})
        cleaned_cols = []
        for col in frame.columns:
            cleaned = col.replace('<br/>', ' ')
            cleaned = ' '.join(cleaned.split())
            cleaned_cols.append(cleaned)
        frame.columns = cleaned_cols
        subset = frame[columns + ["slug"]].rename(columns=rename)
        return base.merge(subset, on="slug", how="left")

    if tables["total_offense"].empty:
        raise RuntimeError("Failed to download NCAA team stats; aborting feature build.")

    offense = _coerce_numeric(tables["total_offense"], {"Rank", "Team", "slug"}).copy()
    offense["plays_per_game"] = offense["Plays"] / offense["G"]
    offense_features = offense[["slug", "plays_per_game", "Yds/Play", "YPG"]].rename(
        columns={
            "Yds/Play": "offense_ypp",
            "YPG": "offense_ypg",
        }
    )

    defense = _coerce_numeric(tables["total_defense"], {"Rank", "Team", "slug"}).copy()
    defense_features = defense[["slug", "Yds/Play", "YPG"]].rename(
        columns={
            "Yds/Play": "defense_ypp_allowed",
            "YPG": "defense_ypg_allowed",
        }
    )

    third_down_off = _coerce_numeric(tables["third_down_offense"], {"Rank", "Team", "slug"}).copy()
    third_down_off["third_down_pct"] = third_down_off["Pct"]
    third_down_off_features = third_down_off[["slug", "third_down_pct"]]

    third_down_def = _coerce_numeric(tables["third_down_defense"], {"Rank", "Team", "slug"}).copy()
    third_down_def["third_down_def_pct"] = third_down_def["Pct"]
    third_down_def_features = third_down_def[["slug", "third_down_def_pct"]]

    pass_eff = _coerce_numeric(tables["team_passing_efficiency"], {"Rank", "Team", "slug"}).copy()
    pass_eff_features = pass_eff[["slug", "Pass Eff"]].rename(columns={"Pass Eff": "team_pass_eff"})

    pass_eff_def = _coerce_numeric(tables["team_passing_efficiency_def"], {"Rank", "Team", "slug"}).copy()
    pass_eff_def_features = pass_eff_def[["slug", "Pass Eff"]].rename(columns={"Pass Eff": "opp_pass_eff"})

    scoring_off = _coerce_numeric(tables["scoring_offense"], {"Rank", "Team", "slug"}).copy()
    off_col = "PPG" if "PPG" in scoring_off.columns else "Avg"
    scoring_off_features = scoring_off[["slug", off_col]].rename(columns={off_col: "points_per_game"})

    scoring_def = _coerce_numeric(tables["scoring_defense"], {"Rank", "Team", "slug"}).copy()
    def_col = "PPG" if "PPG" in scoring_def.columns else "Avg"
    scoring_def_features = scoring_def[["slug", def_col]].rename(columns={def_col: "points_allowed_per_game"})

    top = _coerce_numeric(tables["time_of_possession"], {"Rank", "Team", "slug", "TOP"}).copy()
    top_features = top[["slug", "AvgTOP"]].rename(columns={"AvgTOP": "avg_time_of_possession"})

    features = offense_features
    for frame in [
        defense_features,
        third_down_off_features,
        third_down_def_features,
        pass_eff_features,
        pass_eff_def_features,
        scoring_off_features,
        scoring_def_features,
        top_features,
    ]:
        features = features.merge(frame, on="slug", how="outer")

    features = _merge_feature(
        features,
        "red_zone_offense",
        ["RZAtt", "RZ Rush TD", "RZ Pass TD", "RZ FG Made", "RZScores", "Pct"],
        {
            "RZAtt": "red_zone_attempts",
            "RZ Rush TD": "red_zone_rush_td",
            "RZ Pass TD": "red_zone_pass_td",
            "RZ FG Made": "red_zone_fg_made",
            "RZScores": "red_zone_scores",
            "Pct": "red_zone_pct",
        },
    )
    features = _merge_feature(
        features,
        "red_zone_defense",
        ["Opp RZAtt", "Opp RZ Rush TD", "Opp RZ Pass TD", "Opp RZ FG Made", "Opp RZScores", "Pct"],
        {
            "Opp RZAtt": "red_zone_def_attempts",
            "Opp RZ Rush TD": "red_zone_def_rush_td",
            "Opp RZ Pass TD": "red_zone_def_pass_td",
            "Opp RZ FG Made": "red_zone_def_fg_made",
            "Opp RZScores": "red_zone_def_scores",
            "Pct": "red_zone_def_pct",
        },
    )
    features = _merge_feature(
        features,
        "turnover_margin",
        ["Turn Gain", "Turn Lost", "Margin", "Avg"],
        {
            "Turn Gain": "turnover_gain",
            "Turn Lost": "turnover_loss",
            "Margin": "turnover_margin_total",
            "Avg": "turnover_margin_avg",
        },
    )
    features = _merge_feature(
        features,
        "penalties_pg",
        ["PenPerGame"],
        {
            "PenPerGame": "penalties_per_game",
        },
    )
    features = _merge_feature(
        features,
        "penalty_yards_pg",
        ["YPG"],
        {
            "YPG": "penalty_yards_per_game",
        },
    )
    features = _merge_feature(
        features,
        "rushing_offense",
        ["Rush Yds", "Yds/Rush", "YPG"],
        {
            "Rush Yds": "rush_yards_total",
            "Yds/Rush": "rush_yards_per_carry",
            "YPG": "rush_yards_per_game",
        },
    )
    features = _merge_feature(
        features,
        "rushing_defense",
        ["Opp Rush Yds", "Yds/Rush", "YPG"],
        {
            "Opp Rush Yds": "rush_yards_allowed_total",
            "Yds/Rush": "rush_yards_allowed_per_carry",
            "YPG": "rush_yards_allowed_per_game",
        },
    )
    features = _merge_feature(
        features,
        "passing_offense",
        ["Pass Yds", "Yds/Att", "YPG"],
        {
            "Pass Yds": "pass_yards_total",
            "Yds/Att": "pass_yards_per_attempt",
            "YPG": "pass_yards_per_game",
        },
    )
    features = _merge_feature(
        features,
        "passing_defense",
        ["Opp Pass Yds", "Yds/Att", "YPG"],
        {
            "Opp Pass Yds": "pass_yards_allowed_total",
            "Yds/Att": "pass_yards_allowed_per_attempt",
            "YPG": "pass_yards_allowed_per_game",
        },
    )
    features = _merge_feature(
        features,
        "team_sacks",
        ["Sacks", "Sack Yds", "Avg"],
        {
            "Sacks": "sacks_total",
            "Sack Yds": "sack_yards",
            "Avg": "sacks_per_game",
        },
    )
    features = _merge_feature(
        features,
        "team_tfl",
        ["TTFL", "TFLPG"],
        {
            "TTFL": "tfl_total",
            "TFLPG": "tfl_per_game",
        },
    )
    features = _merge_feature(
        features,
        "sacks_allowed",
        ["Opp Sacks", "Opp Sack Yds", "Avg"],
        {
            "Opp Sacks": "sacks_allowed",
            "Opp Sack Yds": "sacks_allowed_yards",
            "Avg": "sacks_allowed_per_game",
        },
    )
    features = _merge_feature(
        features,
        "tfl_allowed",
        ["Opp STFL", "Opp ATFL", "Opp TFL", "Opp Tackle Yds", "Avg"],
        {
            "Opp STFL": "tfl_allowed_solo",
            "Opp ATFL": "tfl_allowed_assisted",
            "Opp TFL": "tfl_allowed_total",
            "Opp Tackle Yds": "tfl_allowed_yards",
            "Avg": "tfl_allowed_per_game",
        },
    )
    features = _merge_feature(
        features,
        "net_punting",
        ["Net Yds"],
        {
            "Net Yds": "net_punting_avg",
        },
    )
    features = _merge_feature(
        features,
        "kickoff_returns",
        ["Avg"],
        {
            "Avg": "kick_return_avg",
        },
    )
    features = _merge_feature(
        features,
        "kickoff_return_defense",
        ["Avg"],
        {
            "Avg": "kick_return_defense_avg",
        },
    )
    features = _merge_feature(
        features,
        "punt_returns",
        ["Avg"],
        {
            "Avg": "punt_return_avg",
        },
    )
    features = _merge_feature(
        features,
        "punt_return_defense",
        ["Avg"],
        {
            "Avg": "punt_return_defense_avg",
        },
    )

    features["team_name"] = features["slug"].map(SLUG_TO_PFF)
    features = features.dropna(subset=["team_name"]).drop(columns=["slug"])

    individual = build_individual_features(session=session)
    if not individual.empty:
        features = features.merge(individual, on="team_name", how="left")

    target_season = season_year or datetime.now().year
    features = apply_opponent_adjustments(features, target_season, session=session)

    return features
