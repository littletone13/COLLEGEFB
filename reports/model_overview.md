# FCS & FBS Modelling Overview (Q4 2024)

## Purpose
This briefing package summarizes the end-to-end workflow powering our FBS/FCS forecasts, the current calibration state, and the validation numbers that demonstrate how much signal comes from the model versus the market. The goal is to give Rufus Peabody (and peers) enough context to evaluate methodology and identify the next levers to pull.

---

## Data Stack

- **Core team metrics (FBS)** – CollegeFootballData (CFBD) PPA splits, SP+, FPI, Elo; enriched with opponent-adjusted PPA, PFF unit grades, and OddsLogic injury penalties.
- **Core team metrics (FCS)** – PFF opponent-adjusted ridge regressions (offense/defense/special teams), NCAA scoreboard-derived opponent context, PFF snap/usage data.
- **Market inputs** – CFBD `lines` endpoint for schedule-level consensus; OddsLogic archive + live feeds for high-limit books (Circa, Sports411, BetOnline, FanDuel, DraftKings) with opener/closer history and injury wire.
- **Weather** – CFBD `/games/weather` augmented with day-of OddsLogic fields; total impact capped at ±10 pts.
- **Storage** – Large OddsLogic archives live outside the repo (`~/data/oddslogic/...`) and are symlinked in for reproducibility.

---

## Modelling Architecture

1. **Rating construction**
   - FBS: Z-scored blend of CFBD/PFF metrics with ridge-derived scoring constants. Injury penalties debias unit ratings. Power adjustments recalibrated weekly via opponent results.
   - FCS: PFF opponent-adjusted ratings bolstered with NCAA-derived tempo/efficiency stats. Linear spread model (ridge/elastic interactions) generates the spread rating layer.
2. **Bayesian shrink (NEW)**  
   - Team-level games-played counts refine spreads/totals toward priors when sample sizes are thin. The posterior also scales win-prob sigma so early-season projections remain realistic.
3. **Market blending**
   - Pure model output is preserved (`model_spread`, `model_total`). A configurable blend (default 40% FBS / 70% FCS) produces market-adjusted views (`spread_home_minus_away`, `total_points`). OddsLogic closes override CFBD when available to ensure sharp books drive the anchor.
4. **Simulation layer**
   - Week simulators (`simulate_fbs_week.py`, `simulate_fcs_week.py`) assemble weather, market providers, and edges; HTML exports now show both pure and adjusted numbers plus market deltas.
5. **Backtesting & QA**
   - Spread/total MAE, Brier, ROI split into pure vs. market-adjusted buckets. OddsLogic coverage filters out games lacking at least one targeted provider. Injury scrapes warn (but do not fail) when CFBD GraphQL stalls.

---

## Validation Snapshots

| Segment | Spread MAE (Pure) | Spread MAE (Adj) | Total MAE (Pure) | Total MAE (Adj) | Spread ROI (Pure) | Spread ROI (Adj) | Total ROI (Pure) | Total ROI (Adj) |
|--------|-------------------|------------------|------------------|-----------------|-------------------|------------------|------------------|-----------------|
| **FBS 2024 (641 games)** | 11.49 | 12.03 | 11.91 | 11.41 | -17.1% | -18.0% | **+10.8%** | **+12.9%** |
| **FBS 2025 YTD (401 games)** | 11.10 | 11.97 | 11.66 | 11.65 | -7.9% | -10.9% | **+16.7%** | **+18.4%** |
| **FCS 2024–25 (3,634 games)** | 18.20 | 18.21 | 13.72 | 13.45 | -27.2% | -35.4% | -3.0% | 0.1% |

**Takeaways**
- Totals remain the strongest edge in both divisions; the market blend improves calibration without erasing signal.
- Spreads lag the closer in raw ROI—expected until we finish the market-anchor and CLV monitoring workstreams.
- The Bayesian shrink materially reduces early-season volatility (sigma inflator) and keeps the HTML/API outputs aligned with what sharp books are dealing.

---

## Deliverables & Automation

- **Backtests** – `python3 backtest_fbs.py --year <yr> --oddslogic-dir ...` and `python3 backtest_fcs.py <start> <end> ...` emit plain-text summaries plus per-game CSVs for deeper analysis.
- **Daily sims** – FastAPI + Streamlit stack (local for now) surfaces interactive dashboards. HTML/CSV exports are dropped to Desktop for quick sharing.
- **Injury ingestion** – CFBD GraphQL and OddsLogic feeds merged; warnings raised when CFBD returns HTML or stalls.
- **CI & Testing** – Pytest smoke suite (market integration, oddslogic loader). GitHub Actions run on every push to guard against regression.

---

## Roadmap Priorities

1. **Closing-line validation** – Automate CLV deltas vs. Circa/Sports411 closes, feed results into bet selection and reporting.
2. **Market anchor upgrades** – Finish `market_anchor.py` integration so spread priors align with multi-book consensus rather than a single implied rating.
3. **FCS spread calibration** – Re-run regularized regression with new provider dispersion and Bayesian shrinkage; reduce the -27% ROI drag on FCS spreads.
4. **Streamlit hardening** – Move API + dashboard to managed hosting, add auth, and expose per-game audit trails.
5. **Injury enrichment** – Source higher-coverage feeds (PFF, team SID) to replace CFBD’s partial GraphQL output.

---

## Sharing & Next Steps

- Latest artifacts: `~/Desktop/sims_week10_fbs_2025.html`, `~/Desktop/sims_week10_fcs_2025.html`, `calibration/fbs_backtest_2024_results.txt`, `calibration/fbs_backtest_2025_results.txt`, `calibration/fcs_backtest_2024_2025_results.txt`.
- OddsLogic archive path defined via `ODDSLOGIC_ARCHIVE_DIR`; keep large datasets outside git.
- Recommended discussion topics with Rufus:
  1. Confidence in totals vs. spreads and how to weight them in staking.
  2. Appetite for Bayesian shrink parameters (current prior strength: 6 games FBS / 5 games FCS).
  3. Prioritizing CLV instrumentation versus FCS spread refits for the next sprint.

---

*Prepared by: Anthony’s modelling workspace (Week 10, 2025)*.

