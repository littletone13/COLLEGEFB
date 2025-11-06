# COLLEGEFB

## Environment
- Copy `.env.example` to `.env` and populate the API keys.
- `THE_ODDS_API_KEY` must be set to your purchased The Odds API credential; the client reads this for both FBS and FCS simulations.
- Optional overrides: `THE_ODDS_API_REGIONS` and `THE_ODDS_API_MARKETS` default to `us,us2` and `spreads,totals,h2h` if unset.

## Running simulations
- Ensure the CFBD API key (`CFBD_API_KEY`) is configured before running the weekly simulators.
- The market ingestion prioritizes The Odds API and falls back to CFBD or archived OddsLogic data when a matchup is missing from the primary feed.
