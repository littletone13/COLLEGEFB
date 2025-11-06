# CFBD Cache Workflow

This repository now includes a lightweight caching layer so that modelling code can read CollegeFootballData payloads from disk rather than relying on the live API.

## Populating the cache

Run the helper script with your CFBD key (either pass `--api-key` or set `CFBD_API_KEY`):

```bash
PYTHONWARNINGS=ignore \
CFBD_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx python scripts/cache_cfbd.py \
  --start-year 2025 \
  --seasons 5 \
  --season-type regular
```

Key details:

- Data is written under `data/cache/cfbd/season_<YEAR>/`.
- The script retrieves team-level season advanced stats, game-level advanced stats (per week), and play-by-play (per week). Use `--skip-*` flags to disable individual pieces.
- Requests are paced with exponential backoff and a custom `User-Agent`; if Cloudflare challenges persist, rerun later or add `--max-week` to limit the scope.

## Loading cached data

Use the new helper module `cfb/io/cached_cfbd.py` inside your notebooks or pipelines:

```python
from cfb.io import cached_cfbd

team_df = cached_cfbd.load_advanced_team(2025)        # season-level stats
week_df = cached_cfbd.load_game_advanced(2025, 11)    # week-level advanced metrics
pbp_df  = cached_cfbd.load_plays(2025, 11)            # play-by-play
```

- If the requested file is missing and `CFBD_API_KEY` is set, the loader will fall back to a live fetch (respecting the same backoff rules) and store the result under the season directory.
- To point the cache to another location (for example, on a shared volume), call `cached_cfbd.configure_cache_root(Path("/path/to/cache"))`.

With these utilities in place the FBS/FCS models, player prop pipeline, and backtests can rely on cached data while still supporting live updates when needed.
