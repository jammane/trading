#!/usr/bin/env bash
# prune_runs.sh — cap how much disk finished training runs hold on the droplet.
#
# A run directory is ~3-4 GB, and ~95% of that is StockNN weights: the 5-day elite history
# ({ind}_hist.bin, 186 MB x 12), the 20 elites per industry (74 MB x 12), and the same elites again
# under champion/. MT1/MT2 files are tens of MB, so they are left alone.
#
# Weights are only needed to seed a future run (--load-dir) or to convert to .pt for production.
# ANALYSIS only needs the logs. So: keep weights for the N most recent runs, keep every log forever.
#
# Deletion is a WHITELIST of StockNN weight patterns, never "every .bin but X": the old version
# deleted every top-level *.bin except mt_training_log.bin, which silently took mt1_dataset.bin
# (a log, ~70 MB) with it. Logs that must survive: train.log, *.csv, mt_training_log.bin,
# mt1_dataset.bin.
#
# Scans every /root/ht_* run directory (ht_train*, ht_race*, ht_collect*, ...), not just ht_train*.
#
# Usage: prune_runs.sh [KEEP] [--dry-run]     (KEEP default 2)
set -euo pipefail
KEEP=2; DRY=""
for a in "$@"; do
    case "$a" in
        --dry-run) DRY=1 ;;
        ''|*[!0-9]*) echo "usage: $0 [KEEP] [--dry-run]" >&2; exit 2 ;;
        *) KEEP="$a" ;;
    esac
done

# StockNN weights only. mt2_hist.bin is MT2 and small; excluded explicitly.
weight_files() {
    find "$1" -maxdepth 2 -type f \( -name '*_elite_*.bin' -o -name '*_elite_*.pt' \
        -o \( -name '*_hist.bin' ! -name 'mt2_hist.bin' \) \) 2>/dev/null
}

# Never touch a run that is currently being written.
LIVE=""
if pgrep -x training_v4_cpp >/dev/null 2>&1; then
    LIVE=$(pgrep -a training_v4_cpp | grep -oP '(?<=--output )\S+' | head -1 || true)
    LIVE="${LIVE%/}"
    [ -n "$LIVE" ] && echo "run in flight, will skip: $LIVE"
fi

mapfile -t DIRS < <(ls -dt /root/ht_*/ 2>/dev/null | sed 's:/$::' || true)
kept=0
for d in "${DIRS[@]}"; do
    [ -d "$d" ] || continue
    [ "$d" = "$LIVE" ] && continue
    n=$(weight_files "$d" | wc -l)
    [ "$n" -eq 0 ] && continue                    # already logs-only
    kept=$((kept+1))
    if [ "$kept" -le "$KEEP" ]; then
        echo "keep weights : $d ($(du -sh "$d" | cut -f1))"
        continue
    fi
    sz=$(du -sh "$d" | cut -f1)
    if [ -n "$DRY" ]; then
        echo "WOULD strip  : $d ($sz, $n weight files; logs preserved)"
    else
        weight_files "$d" | xargs -r rm -f
        [ -d "$d/champion" ] && find "$d/champion" -maxdepth 0 -empty -delete
        echo "stripped     : $d ($sz -> $(du -sh "$d" | cut -f1); logs preserved)"
    fi
done
echo "--- disk ---"; df -h /root | tail -1
