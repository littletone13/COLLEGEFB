# Predictive Model Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│  1. Data Layer (CFBD cache + live fallback)                         │
│  ├─ scripts/cache_cfbd.py → data/cache/cfbd/season_<year>/…        │
│  └─ cfb/io/cached_cfbd.py  → load_* helpers (plays, team stats)     │
└────────────────────────────────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────────────────┐
│  2. Feature Engineering Layer                                      │
│  ├─ Team features: tempo, run/pass rates, EPA splits, opponent adj.│
│  ├─ Matchup context: offense vs defense success/explosive, line yrs│
│  └─ Player usage: snaps, carries, targets, red-zone share          │
│         (derived from cached play-by-play + depth/roster info)     │
└────────────────────────────────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────────────────┐
│  3. Model Layer                                                     │
│  ├─ FBS rating book (fbs.py)                                       │
│  │    • consumes team features, refits offense/defense weights      │
│  │    • outputs spread/total probabilities used by sims             │
│  └─ Player prop models                                              │
│       • supervised regressors for yards/receptions/TDs              │
│       • mean + variance per player per stat                         │
└────────────────────────────────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────────────────┐
│  4. Simulation & Output Layer                                      │
│  ├─ simulate_fbs_week.py → matchup projections                      │
│  └─ Prop pipeline (generate_player_prop_baselines.py → outputs)     │
│       • CSV/HTML with projected stats, edges, Kelly sizing          │
└────────────────────────────────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────────────────┐
│  5. Validation & Monitoring                                        │
│  ├─ Backtests stored in calibration/                               │
│  ├─ Drift metrics: projected vs actual usage                       │
│  └─ Scheduled retrains as new weeks finish                         │
└────────────────────────────────────────────────────────────────────┘
```

With this layout:

- **Data Layer** keeps the last five seasons cached locally while still permitting a live fetch when new weeks are missing.
- **Feature Layer** is the single source of truth for both the rating book and player-prop models so tempo and matchup style stay aligned.
- **Model Layer** updates spreads/totals and prop expectations using the richer features.
- **Outputs** stay consistent across simulations, dashboards, and prop reports.
- **Validation** ensures we only promote changes that improve predictive accuracy.
