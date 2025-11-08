#!/bin/bash
set -euo pipefail

cd /Users/anthonyeding/FCS
LOG_DIR="logs/odds_capture"
mkdir -p "$LOG_DIR"
timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"

PYTHONPATH='.' CFBD_API_KEY='8Td8MlATFPwulj8kAGttr4gxfpjJtlNxoqGyvtLdGc9GEJh5kxd3yKpJGNCAUSQF' \
THE_ODDS_API_KEY='b8454a2cf3e5a607b33c2c5f85871a06' \
python3 scripts/capture_live_odds.py \
  --sport fbs \
  --year 2025 \
  --week 11 \
  --season-type regular \
  --output data/lines/live >> "$LOG_DIR/capture_${timestamp}.log" 2>&1
