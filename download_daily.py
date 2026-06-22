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

Usage:
  python download_daily.py
"""
import json
import os
import time

import yfinance as yf

from universe import ALL_SYMBOLS

MAX_HISTORY_DAYS = 1255
STOCK_DATA_DIR   = 'stock_data'
FETCH_LOOKBACK   = '7d'   # covers weekends and the current day robustly


def _load_existing(sym: str) -> list:
    path = os.path.join(STOCK_DATA_DIR, f'{sym}.json')
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return json.load(f).get('days', [])
    except Exception:
        return []


def _fetch(sym: str, period: str) -> list:
    ticker = yf.Ticker(sym)
    hist   = ticker.history(period=period, interval='1d')
    days   = []
    for date, row in hist.iterrows():
        days.append({
            'date':   date.strftime('%Y-%m-%d'),
            'open':   float(row['Open']),
            'high':   float(row['High']),
            'low':    float(row['Low']),
            'close':  float(row['Close']),
            'volume': int(row['Volume']),
        })
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


def main() -> None:
    os.makedirs(STOCK_DATA_DIR, exist_ok=True)
    symbols = ALL_SYMBOLS
    print(f'Updating {len(symbols)} symbols → trimmed to {MAX_HISTORY_DAYS} days each.')

    updated = new_sym = errors = 0
    for i, sym in enumerate(symbols, 1):
        existing = _load_existing(sym)
        try:
            if existing:
                fetched = _fetch(sym, FETCH_LOOKBACK)
                merged  = _merge(existing, fetched)
                added   = len(merged) - len(existing)
                _save(sym, merged)
                total   = min(len(merged), MAX_HISTORY_DAYS)
                print(f'  [{i:3d}/{len(symbols)}] {sym:<6s}  +{added:2d} day(s)  → {total} days')
                updated += 1
            else:
                print(f'  [{i:3d}/{len(symbols)}] {sym:<6s}  new — full 5-year download...')
                fetched = _fetch(sym, '5y')
                if fetched:
                    _save(sym, fetched)
                    print(f'    saved {min(len(fetched), MAX_HISTORY_DAYS)} days')
                    new_sym += 1
                else:
                    print(f'    WARNING: no data returned for {sym}')
                    errors += 1
        except Exception as e:
            print(f'  [{i:3d}/{len(symbols)}] {sym:<6s}  ERROR: {e}')
            errors += 1
        time.sleep(1.2)

    print(f'\nDone: {updated} updated, {new_sym} new, {errors} errors.')


if __name__ == '__main__':
    main()
