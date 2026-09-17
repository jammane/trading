#!/usr/bin/env python3
"""
download_daily.py — Incremental daily update of stock_data/*.json.

Run 30 minutes after market close (4:30 PM ET, weekdays). For each symbol
in the current universe:
  - File exists → fetch the past 7 calendar days from yfinance, merge by
    date (dedup, new data wins), trim to MAX_HISTORY_DAYS most recent days.
  - File missing → download full 5-year history (new-symbol case).

All symbols end up with at most MAX_HISTORY_DAYS (1255) days of history,
so older data is pruned automatically on each run.

Dead-ticker detection
---------------------
A delisted symbol does not raise: yfinance returns an empty frame, the merge is
a no-op, and the symbol reports `+ 0 day(s)` exactly like a healthy symbol on a
weekend. The file keeps its full MAX_HISTORY_DAYS, so nothing downstream looks
wrong either — but the days are the WRONG days, frozen at the delisting date.

That matters more than it looks: the C++ trainer indexes over the UNION of all
symbols' dates, so one frozen file with an older start stretches the union and
pushes `--stop-day` backwards, silently discarding the most recent sessions for
every healthy symbol.

Detection is cohort-relative rather than calendar-based: all symbols trade the
same sessions, so the newest date across the universe is the reference, and any
symbol lagging it by more than MAX_STALE_DAYS is dead or halted. This needs no
market-holiday table and is correct on weekends, when every symbol lags "today"
but none lags the cohort.

Exits non-zero when anything is stale or errored, so cron surfaces it.

Usage:
  python download_daily.py
"""
import datetime as dt
import json
import math
import os
import time

import yfinance as yf

from universe import ALL_SYMBOLS

MAX_HISTORY_DAYS = 1255
STOCK_DATA_DIR   = 'stock_data'
FETCH_LOOKBACK   = '7d'   # covers weekends and the current day robustly
MAX_STALE_DAYS   = 7      # calendar days a symbol may lag the cohort before it is called dead


def _load_existing(sym: str) -> list:
    path = os.path.join(STOCK_DATA_DIR, f'{sym}.json')
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            days = json.load(f).get('days', [])
        # Purge bars already written as non-finite. _merge keeps an existing bar when the fetch
        # no longer returns it, so without this a bad bar survives every future run -- dropping
        # it at the source only stops NEW ones. See _fetch for why they are poison.
        clean = [d for d in days
                 if all(isinstance(d.get(k), (int, float))
                        and math.isfinite(d[k]) and d[k] > 0
                        for k in ('open', 'high', 'low', 'close'))]
        if len(clean) != len(days):
            print(f'  {sym}: purged {len(days) - len(clean)} unusable bar(s) already on disk')
        return clean
    except (OSError, json.JSONDecodeError) as e:
        # Returning [] triggers a full 5-year refetch for this symbol — say so, because silently
        # refetching every run would look like normal behaviour.
        print(f"WARNING: {path} unreadable ({e}) — treating as empty, will refetch in full")
        return []


def _fetch(sym: str, period: str) -> list:
    """Fetch daily bars, dropping any that are not usable.

    yfinance returns NaN for a session it has no data for, and `json.dump` writes that as a bare
    `NaN` literal. Downstream, the C++ trainer's `atof()` parses it happily, so the bar looked
    valid and NaN entered the portfolio valuation -- `(int)NaN` is undefined behaviour and landed
    on INT32_MIN, surfacing as `prod=$-2147483648` for SCCO 2026-08-11. It silently destroyed
    every statistic computed over that industry-pass. Cheaper to never write it.
    """
    ticker = yf.Ticker(sym)
    hist   = ticker.history(period=period, interval='1d')
    days   = []
    dropped = 0
    for date, row in hist.iterrows():
        o, h, low, c = (float(row['Open']), float(row['High']),
                        float(row['Low']), float(row['Close']))
        if not all(math.isfinite(v) and v > 0 for v in (o, h, low, c)) or h < low:
            dropped += 1
            continue
        v = float(row['Volume'])
        days.append({
            'date':   date.strftime('%Y-%m-%d'),
            'open':   o,
            'high':   h,
            'low':    low,
            'close':  c,
            'volume': int(v) if math.isfinite(v) and v >= 0 else 0,
        })
    if dropped:
        print(f'    NOTE: dropped {dropped} unusable bar(s) for {sym} (non-finite or non-positive)')
    return days


def _merge(existing: list, fetched: list) -> list:
    """Union by date; fetched data wins on any conflict."""
    by_date = {d['date']: d for d in existing}
    for d in fetched:
        by_date[d['date']] = d
    return sorted(by_date.values(), key=lambda d: d['date'])


def _save(sym: str, days: list) -> None:
    days = days[-MAX_HISTORY_DAYS:]
    path = os.path.join(STOCK_DATA_DIR, f'{sym}.json')
    with open(path, 'w') as f:
        json.dump({'days': days}, f)


def find_stale_symbols(last_dates: dict, max_stale_days: int = MAX_STALE_DAYS) -> tuple:
    """Symbols whose newest bar lags the universe's newest bar by too much.

    `last_dates` maps symbol -> 'YYYY-MM-DD' of its most recent bar. Returns
    (cohort_last, {sym: (last_date, days_behind)}) sorted most-stale first.
    The cohort maximum is the reference, so this is correct on weekends and
    holidays, when every symbol lags today's date but none lags the cohort.
    Returns (None, {}) when there is nothing to compare.
    """
    if not last_dates:
        return None, {}
    parsed = {s: dt.date.fromisoformat(d) for s, d in last_dates.items()}
    cohort_last = max(parsed.values())
    stale = {}
    for sym, d in parsed.items():
        behind = (cohort_last - d).days
        if behind > max_stale_days:
            stale[sym] = (d.isoformat(), behind)
    ordered = dict(sorted(stale.items(), key=lambda kv: -kv[1][1]))
    return cohort_last.isoformat(), ordered


def main() -> int:
    os.makedirs(STOCK_DATA_DIR, exist_ok=True)
    symbols = ALL_SYMBOLS
    print(f'Updating {len(symbols)} symbols → trimmed to {MAX_HISTORY_DAYS} days each.')

    updated = new_sym = errors = 0
    last_dates: dict = {}
    no_rows: list = []

    for i, sym in enumerate(symbols, 1):
        existing = _load_existing(sym)
        try:
            if existing:
                fetched = _fetch(sym, FETCH_LOOKBACK)
                merged  = _merge(existing, fetched)
                added   = len(merged) - len(existing)
                _save(sym, merged)
                total   = min(len(merged), MAX_HISTORY_DAYS)
                last    = merged[-1]['date']
                last_dates[sym] = last
                # An empty fetch is the direct delisting signal: yfinance returns no rows rather
                # than raising. Distinguish it here from a genuine "no new sessions" result.
                flag = ''
                if not fetched:
                    no_rows.append(sym)
                    flag = '  ** no rows returned (delisted/halted?)'
                print(f'  [{i:3d}/{len(symbols)}] {sym:<6s}  +{added:2d} day(s)  '
                      f'→ {total} days  last={last}{flag}')
                updated += 1
            else:
                print(f'  [{i:3d}/{len(symbols)}] {sym:<6s}  new — full 5-year download...')
                fetched = _fetch(sym, '5y')
                if fetched:
                    _save(sym, fetched)
                    last_dates[sym] = fetched[-1]['date']
                    print(f'    saved {min(len(fetched), MAX_HISTORY_DAYS)} days')
                    new_sym += 1
                else:
                    print(f'    WARNING: no data returned for {sym}')
                    errors += 1
        except Exception as e:
            print(f'  [{i:3d}/{len(symbols)}] {sym:<6s}  ERROR: {e}')
            errors += 1
        time.sleep(1.2)

    cohort_last, stale = find_stale_symbols(last_dates)
    print(f'\nDone: {updated} updated, {new_sym} new, {errors} errors.')
    if cohort_last:
        print(f'Newest session across the universe: {cohort_last}')

    if stale:
        print(f'\n{"=" * 72}')
        print(f'DEAD / STALE TICKERS — {len(stale)} symbol(s) lag the universe by '
              f'more than {MAX_STALE_DAYS} days')
        print('=' * 72)
        for sym, (last, behind) in stale.items():
            print(f'  {sym:<6s} last bar {last}  ({behind} days behind {cohort_last})')
        print('\nA stale file keeps its full history, so nothing downstream errors — but the')
        print('trainer indexes over the UNION of all symbols\' dates, so a frozen file pushes')
        print('--stop-day backwards and silently drops recent sessions for every other symbol.')
        print('\nPer the swap rules a defunct ticker is an immediate swap, not a watch:')
        for sym in stale:
            print(f"  ./swap_symbols.sh '{{\"{sym}\": \"REPLACEMENT\"}}'")
    elif no_rows:
        # Fetched nothing but not yet behind the cohort — e.g. a halt that started today.
        print(f'\nNOTE: {len(no_rows)} symbol(s) returned no rows this run but are not yet stale: '
              f'{", ".join(no_rows)}')

    return 1 if (stale or errors) else 0


if __name__ == '__main__':
    raise SystemExit(main())
