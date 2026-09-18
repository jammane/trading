#!/usr/bin/env python3
"""
read_mt1_dataset.py — reader for mt1_dataset.bin, the artefact the horizon decision rests on.

Each record is one training day and carries, per industry:

    feat[74]      the exact slice handed to mt1_step_day  (37 market ‖ 37 portfolio)
    book_prev     the book at TODAY's close, before the day's trades
    book_now      the book at the NEXT close, after them
    mkt_move      baseline − book_prev    prices moved, holdings fixed
    trade_delta   slot0_score − baseline  holdings moved, prices fixed

book P&L = book_now − book_prev = mkt_move + trade_delta, exactly. `verify_identity` checks that
on every row; a failure means the three marks in step_industry have drifted apart.

Forward-h book P&L is a sum over consecutive records, so h is an offline sweep rather than one
training run per candidate. `forward_pnl` builds it.

Usage:
  python read_mt1_dataset.py logs/acct0/training/mt1_dataset.bin
"""
import argparse
import os
import struct
import sys

import numpy as np

HEADER_SIZE  = 16
DS_MAGIC     = 0x4D543144        # "MT1D" — must match DS_LOG_MAGIC in training_v4.cpp
DS_VERSION   = 1
N_IND        = 12
DS_FEAT      = 74

RECORD_FMT  = '<II' + 'f' * (DS_FEAT * N_IND) + 'f' * (4 * N_IND)
RECORD_SIZE = struct.calcsize(RECORD_FMT)
assert RECORD_SIZE == 3752, f'dataset record must be 3752 bytes, got {RECORD_SIZE}'

COMPONENTS = ('book_prev', 'book_now', 'mkt_move', 'trade_delta')


def parse(path):
    """Return a dict of arrays. feat is (T, N_IND, 74); the rest are (T, N_IND)."""
    with open(path, 'rb') as f:
        blob = f.read()
    if len(blob) < HEADER_SIZE:
        sys.exit(f'{path}: too short to hold a header')
    magic, version, n_ind, n_feat = struct.unpack('<4I', blob[:HEADER_SIZE])
    if magic != DS_MAGIC:
        sys.exit(f'{path}: bad magic 0x{magic:08X} — not an mt1_dataset.bin')
    if version != DS_VERSION:
        sys.exit(f'{path}: dataset version {version}, this reader handles {DS_VERSION}')
    if (n_ind, n_feat) != (N_IND, DS_FEAT):
        sys.exit(f'{path}: {n_ind} industries x {n_feat} features, expected {N_IND} x {DS_FEAT}')

    body = blob[HEADER_SIZE:]
    n = len(body) // RECORD_SIZE
    if len(body) % RECORD_SIZE:
        print(f'Warning: {len(body) % RECORD_SIZE} trailing bytes ignored (run in flight?)',
              file=sys.stderr)
    if n == 0:
        return {'pass': np.zeros(0, np.int64), 'day': np.zeros(0, np.int64),
                'feat': np.zeros((0, N_IND, DS_FEAT), np.float32),
                **{k: np.zeros((0, N_IND), np.float32) for k in COMPONENTS}}

    raw = np.frombuffer(body[:n * RECORD_SIZE], dtype=np.uint8).reshape(n, RECORD_SIZE)
    head = raw[:, :8].copy().view(np.uint32)
    rest = raw[:, 8:].copy().view(np.float32)
    out = {'pass': head[:, 0].astype(np.int64), 'day': head[:, 1].astype(np.int64),
           'feat': rest[:, :DS_FEAT * N_IND].reshape(n, N_IND, DS_FEAT)}
    off = DS_FEAT * N_IND
    for k in COMPONENTS:
        out[k] = rest[:, off:off + N_IND]
        off += N_IND
    return out


def verify_identity(ds, tol=1e-2):
    """book_now - book_prev must equal mkt_move + trade_delta on every row.

    Returns (ok, max_abs_error). The tolerance is in DOLLARS and loose on purpose: these are
    float32 differences of ~$25,000 quantities, so ~1e-3 of rounding is expected. A failure means
    the decomposition in step_industry no longer holds, not that the arithmetic is imprecise.
    """
    if len(ds['day']) == 0:
        return True, 0.0
    lhs = ds['book_now'] - ds['book_prev']
    rhs = ds['mkt_move'] + ds['trade_delta']
    err = float(np.max(np.abs(lhs - rhs)))
    return err <= tol, err


def forward_pnl(ds, h, component='book'):
    """Forward h-day P&L per industry, in dollars, aligned to the day the prediction is made.

    Row t holds the sum of the component over days t+1 .. t+h, so it is the outcome of a
    prediction made with feat[t]. The last h rows are NaN — their window has not closed.

    component: 'book' (market + trade), 'mkt' (market move only), or 'trade' (trade delta only).
    """
    per_day = {'book': ds['book_now'] - ds['book_prev'],
               'mkt': ds['mkt_move'],
               'trade': ds['trade_delta']}[component].astype(np.float64)
    T = per_day.shape[0]
    out = np.full_like(per_day, np.nan)
    if T <= h:
        return out
    cum = np.vstack([np.zeros((1, per_day.shape[1])), np.cumsum(per_day, axis=0)])
    # sum over (t, t+h] = cum[t+h+1] - cum[t+1]
    out[:T - h] = cum[1 + h:T + 1] - cum[1:T - h + 1]
    return out


def main():
    ap = argparse.ArgumentParser(description='Summarize mt1_dataset.bin')
    ap.add_argument('dataset')
    ap.add_argument('--pass', dest='passnum', type=int, default=None)
    args = ap.parse_args()
    if not os.path.exists(args.dataset):
        sys.exit(f'{args.dataset}: not found')
    ds = parse(args.dataset)
    n = len(ds['day'])
    if n == 0:
        sys.exit(f'{args.dataset}: header valid, no records yet')
    if args.passnum is not None:
        m = ds['pass'] == args.passnum - 1
        ds = {k: v[m] for k, v in ds.items()}
        n = len(ds['day'])

    ok, err = verify_identity(ds)
    print(f'{args.dataset}: {n} days, passes {sorted(set(ds["pass"].tolist()))}, '
          f'days {ds["day"][0]}-{ds["day"][-1]}')
    print(f'decomposition identity: {"OK" if ok else "FAILED"}  (max error ${err:.4f})')
    if not ok:
        sys.exit('book_now - book_prev != mkt_move + trade_delta — the three marks have drifted')

    book = ds['book_now'] - ds['book_prev']
    print(f'\n{"component":<14} {"mean $/day":>11} {"sd $":>9} {"share of |book|":>16}')
    denom = np.mean(np.abs(book))
    for label, v in (('book P&L', book), ('market move', ds['mkt_move']),
                     ('trade delta', ds['trade_delta'])):
        print(f'{label:<14} {v.mean():>+11.2f} {v.std():>9.0f} '
              f'{np.mean(np.abs(v)) / denom:>15.0%}')
    print(f'\nfeatures: {ds["feat"].shape}  '
          f'finite {np.isfinite(ds["feat"]).mean():.2%}  '
          f'|x| p99 {np.percentile(np.abs(ds["feat"]), 99):.3g}')


if __name__ == '__main__':
    main()
