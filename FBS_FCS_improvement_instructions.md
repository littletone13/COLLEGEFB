# FBS and FCS Backtest Improvement Checklist

## FBS Pipeline
1. **Isolate training data to past games only.**
   * Rebuild the rating book week by week so calibration games only include contests that kicked before the matchup you are grading.
   * Persist the spread/total intercept + slope and win-probability sigma learned from the off-season training export (e.g., store them in `config.yaml`) instead of re-estimating on the evaluation season.

2. **Remove post-game weather leakage.**
   * When replaying historical slates, turn off weather adjustments or replace them with the archived pre-game forecast feed so totals are not corrected by realized weather.

3. **Prevent season-long metric leakage.**
   * Replace the current full-season CFBD/SP+/FPI/Elo pulls with weekly snapshots, or regress toward priors based on how many snaps have been played.
   * Introduce Bayesian/ridge shrinkage so early-season ratings do not benefit from end-of-year data.

4. **Tighten market data handling.**
   * Promote The Odds API as the canonical live line feed for both FBS and FCS (details in the "The Odds API integration" section below) and only fall back to CFBD closing lines when coverage gaps exist. 
   * When falling back to CFBD closing lines, sort entries by the vendor timestamp before selecting the final price and keep that timestamp on disk.
   * Store the provider counts and individual book entries so the edge filter can require confirmation from multiple books.
   * Use the actual moneyline/spread/total juice in ROI math rather than assuming -110.

5. **Retain refreshed calibration coefficients.**
   * After `calibrate_fbs_from_training.py` produces new intercept/slope, write them to disk alongside a version/timestamp and reload them in `build_rating_book` during simulations.

6. **Log residual stability checks.**
   * Persist weekly model-vs-market residuals and run rolling MAE/CUSUM diagnostics to flag drift during the season.

## FCS Pipeline
1. **Finish spread ROI bookkeeping.**
   * Mirror the totals evaluation to create actual spread bets, using the OddsLogic `spread_price` column when available.

2. **Respect neutral-site flags.**
   * Set `neutralSite` based on CFBD or OddsLogic metadata so home-field advantage is not automatically granted.

3. **Leverage actual market prices.**
   * Feed the observed juice from OddsLogic (or CFBD) into the ROI math instead of assuming -110 on every ticket.

4. **Refresh regression weights and calibrations.**
   * Schedule a backtest-driven re-fit of the static coefficient dictionaries and calibrations so they can react to new seasons.

5. **Harden opponent-adjusted data loading.**
   * When adjusted CSVs are missing, fall back to the unadjusted aggregation path with a warning rather than aborting the run.

6. **Fill gaps when OddsLogic or The Odds API miss games.**
   * Drop back to your single-book CSV/API imports so the graded sample is not biased toward games that the main vendors carried.

## The Odds API integration

1. **Configure credentials securely.**
   * Store the purchased key `b8454a2cf3e5a607b33c2c5f85871a06` in a secrets manager or environment variable (`THE_ODDS_API_KEY`) instead of hard-coding it in source. Update `.env.example`/`config.yaml` to document the variable and load it wherever line ingestion runs.

2. **Adopt The Odds API for current lines.**
   * Replace existing live line pulls (OddsLogic, CFBD `/lines`) in both `simulate_fbs_week.py` and `simulate_fcs_week.py` with calls to The Odds API `/v4/sports/{sport_key}/odds/` endpoint, requesting `regions=us,us2` (or preferred books) and `markets=spreads,totals,moneyline`. Normalize the response schema into your existing line tables, preserving `bookmaker_key`, `market`, `price`, `point`, and `last_update` fields.

3. **Backfill historical lines from The Odds API.**
   * Use `/v4/sports/{sport_key}/odds-history/` (or the paid bulk export) to download full-season archives for both `americanfootball_ncaaf` (FBS) and `americanfootball_ncaaf_fcs`. Schedule a backfill job that iterates season-by-season and stores parquet snapshots in `data/lines/the_odds_api/season={year}/week={week}` so backtests can query by kickoff date.
   * Start with the most recent and strategically important years (e.g., current season, prior season, playoff years) before expanding deeper into history so you can validate the integration quickly while conserving tokens.

4. **Blend The Odds API with CFBD stats.**
   * Keep CFBD as the source of truth for statistical features, training exports, and roster data. When constructing model inputs, join The Odds API line snapshots on the CFBD `game_id`/`season` combination to align market numbers with the corresponding team stats.

5. **Update ROI and calibration tooling.**
   * Modify `backtest_fbs.py`, `backtest_fcs.py`, and the calibration scripts to read spread/total/moneyline prices from the new normalized The Odds API tables. Ensure the tooling respects the per-book juice, timestamps, and provider counts so edges are computed against the precise market state that existed before kickoff.

6. **Retain fallback + monitoring.**
   * When The Odds API fails to deliver a specific matchup (maintenance, book exclusion), continue to fall back to CFBD or OddsLogic feeds while logging the event. Add automated alerts when the primary feed misses more than a configurable threshold of games so you can investigate API quota or coverage shifts quickly.

7. **Plan around The Odds API token quotas.**
   * A standard request to `/odds/` costs 1 token per bookmaker-market bundle and `/odds-history/` starts at 5 tokens per event snapshot; a full FBS + FCS season (≈1,300 + 1,000 games) with spreads/totals/moneylines from four books therefore consumes roughly 30–40k tokens even before retries. Twenty thousand tokens comfortably covers weekly live pulls but will not stretch across multi-season historical backfills—budget for the higher tier or stage the backfill over multiple billing periods.
   * Cache every successful response to disk/object storage before transforming it so you only spend tokens once per game/week. Wire weekly cron jobs to skip previously archived windows and alert when token consumption deviates from expectations (e.g., spike >10% week over week), preventing runaway costs during bulk imports.

## Historical line data sourcing
1. **Prioritize multi-year, timestamped feeds.**
   * For FBS, pull the archived `Lines` collections from [The Power Rank](https://thepowerrank.com) or CFB-Graphs' Kaggle datasets when you need decade-deep history, and augment with the free CFBD `/lines` endpoint for seasons 2019-present. Both sources carry open/close numbers with book identifiers and UTC timestamps, letting you verify that closes precede kickoff.
   * For FCS, subscribe to OddsJam or SpankOdds' low-tier historical export—each includes Pinnacle/Circa closing spreads, totals, and prices dating back to 2018 with minute-level timestamps. If budget is a concern, SportsDataIO's college-football package offers FCS closes from 2016 onward at a lower cadence (hourly snapshots).

2. **Normalize vendor schemas once on ingest.**
   * Write a translation layer that maps each provider's book codes (e.g., `PN`, `CIR`, `DK`) to a unified enum and converts timestamps to UTC. Persist raw JSON/CSV alongside the normalized parquet so you can reprocess when new fields appear.

3. **Archive lines contemporaneously.**
   * Mirror the API feed into object storage (S3/GCS) during the season with a daily cron so you own the historical record even if vendors tighten access. Store game identifiers plus `line_type`, `price`, and `updated_at` fields so you can reconstruct close, open, and mid-week snapshots for backtesting.

4. **Cross-check across tiers.**
   * When both OddsLogic and the new provider return prices for the same game, compare close timestamps and prices; flag discrepancies above 0.5 points or 5 cents of juice so you can audit which source drifted after kickoff.

5. **Document availability and gaps.**
   * Maintain a README in `data/lines/` that lists coverage by season, subdivision, and book. Note any missing weeks or vendors so future analysts know how reliable the archive is before running ROI studies.
