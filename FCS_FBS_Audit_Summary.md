# FBS & FCS Modeling Audit (Week 10, 2025)

## Executive Summary
- The current stack ingests CFBD advanced metrics, weather, and market lines, and blends them with opponent-adjusted ridge metrics for FCS. PFF positional grades are available but still rely on manual CSV drops.
- Core scripts (`fbs.py`, `fcs.py`, weekly sim and backtest drivers) produce usable matchup edges, yet processes remain manual and calibration to market is partial. Injury modeling is effectively absent.
- To reach a Rufus Peabody / Andrew Mack standard, the program needs stronger automation, sturdier calibrations, and a decision layer that emphasizes closing-line value (CLV), disciplined staking, and high-quality intel feeds.

## Architecture Overview
### Data Inputs
- CollegeFootballData REST + GraphQL for ratings, play-by-play, weather, and lines.
- Locally maintained PFF exports (`~/Desktop/POST WEEK 9 FBS & FCS DATA`, etc.) for snap counts and unit grades.
- Ridge-regressed opponent adjustments for FCS (scripts in `fcs_opponent_adjust.py` and `fcs_adjusted/` assets).

### Processing & Simulation
- `fbs.py` / `fcs.py` construct base team ratings, apply weather adjustments, and blend market priors (currently 80% weight).
- Weekly scripts (`simulate_fbs_week.py`, `simulate_fcs_week.py`) generate CSV/HTML artifacts with projected spreads, totals, and market deltas.
- Backtesting utilities (`backtest_fbs.py`, `backtest_fcs.py`) evaluate MAE/Brier and ROI (assumes flat staking on any positive edge).

### Outputs & Distribution
- Latest HTML/CSV reports (Week 10, 2025) land in repo root and Desktop; no automated archival or versioning by date.
- Betting deltas are single-source (one market snapshot). No distribution tooling beyond static files.

### DevOps & Versioning
- GitHub repo (`littletone13/COLLEGEFB`) mirrors local state; however, large CSV artifacts are committed and secrets risk exposure.
- No CI/CD, linting, or test harness beyond ad hoc manual runs.

## Strengths Worth Preserving
- Solid baseline modeling ethos: opponent adjustments, weather-aware totals, and market blending already in place.
- Comprehensive CFBD integration with modular helpers allows future expansion (e.g., player props) once data supports it.
- Ridge-adjusted FCS feed closes the gap between FCS and FBS workflows.
- Backtests provide quick signal on calibration drift, giving a foothold for future validation discipline.
- Fresh ridge-fit rating constants (FBS & FCS) captured in `calibration/` and now wired into the core model.

## Gaps Blocking Market-Grade Edge
- **Injury/availability**: No authoritative feed; scripts log warnings and proceed with stale assumptions.
- **Manual data ops**: PFF uploads, ridge runs, and weather refreshes depend on human intervention; no air-gapped history.
- **Calibration**: Rating constants, spread/total mappings, and betting thresholds are heuristics; ROI backtests bet every edge.
- **Market coverage**: Single-book lines; no multi-book consensus, steam detection, or live updates.
- **Governance**: Secrets appear in command history, large binaries in repo, no automated QA to catch data mismatches.

## Recommendations
### 1. Data Infrastructure (Weeks 0–2)
- Orchestrate nightly jobs (Cron/Airflow/Prefect) to pull CFBD data, recompute ridge adjustments, and ingest PFF exports via scripted API/scraper.
- Store snapshots in dated folders or cloud object storage; checksum files and validate schema before promotion.
- Stand up a lightweight injury pipeline: start with publicly scraped depth charts/beat reports; plan migration to a paid feed.

### 2. Modeling & Calibration (Weeks 2–5)
- Refit `RatingConstants` and weather coefficients using ridge/elastic models on the completed-game set; maintain yearly calibration files.
- Split historical weeks into train/validation; tune market weights, weather caps, and home-field advantage to minimize MAE on validation.
- Introduce scenario simulations (weather/tempo variance) and propagate uncertainty into edge calculations.

### 3. Market Integration & Decision Layer (Weeks 3–6)
- Aggregate odds from multiple books using CFBD odds endpoints and partner APIs; calculate consensus and best-line deltas.
- Add edge gating (e.g., bet only when modeled advantage > threshold and expected CLV positive) and fractional Kelly staking.
- Track CLV, bet outcomes, and bankroll drawdown; generate daily reports that mirror Rufus-style risk management dashboards.

### 4. Workflow, Tooling, & Security (Weeks 4–8)
- Refactor shared logic into a Python package to avoid duplication between FBS/FCS scripts; drive configs via YAML/JSON.
- Build a private Streamlit/FastAPI interface for triggering sims, reviewing edges, and exporting artifacts. Gate behind authentication.
- Lock down secrets via `.env` + secret manager, purge credentials from repo history, and expand `.gitignore` to cover exports.
- Implement pytest-based smoke tests for data ingestion and regression checks for calibration routines.

### 5. Research Backlog for Extra Edge
- Develop opponent-adjusted unit strength models for FBS mirroring the FCS ridge approach; explore unit clustering to capture matchup leverage.
- Leverage PFF participation data to model substitution impact (QB/OL rotations, defensive fatigue).
- Evaluate third-party injury intel (Action Labs, CFB Nerds, team beat scrapes) to algorithmically downgrade units when key starters sit.
- Build historical prop baselines once injury data stabilizes; align with market makers' approach before offering player props.

## Deployment Roadmap
1. **Foundation (Weeks 0–2)**: Automate data pulls, enforce QA, surface alerting when feeds fail; create data catalog.
2. **Calibration Sprint (Weeks 2–4)**: Refit constants, hold-out validation, implement CLV-aware staking.
3. **Productization (Weeks 4–6)**: Ship internal dashboard, multi-book odds ingestion, automated HTML/CSV archiving to Desktop + cloud.
4. **Advanced Intelligence (Weeks 6–8)**: Injury feed integration, scenario modeling, prop research kickoff.

## KPI & Monitoring Framework
- **Accuracy**: Weekly MAE/Brier vs. market closes; track drift by unit (spread/total/moneyline).
- **Market Fit**: CLV distribution, percentage of bets beating close, average edge at time of bet.
- **Risk**: Bankroll volatility, peak drawdown, Kelly utilization, exposure by conference.
- **Operational Health**: Data freshness dashboard, ingestion failure alerts, completeness reports for PFF/injury feeds.

## Closing Notes
- Aligning with Rufus Peabody and Andrew Mack’s standards means emphasizing robustness, verifiable CLV, and disciplined bankroll management alongside world-class data hygiene.
- Prioritize automation and calibration first; once those pillars are set, layer on richer intel (injury/snap detail) and productize the workflow for repeatable, audit-ready execution.
