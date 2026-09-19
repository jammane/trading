#!/usr/bin/env python3
"""
read_mt1_dataset.py — reader for mt1_dataset.bin, the artefact the horizon decision rests on.

Each record is one training day and carries, per industry:

    feat[74]      the exact slice handed to mt1_step_day  (37 market ‖ 37 portfolio)
    book_prev     the book at TODAY's close, before the day's trades
    book_now      the book at the NEXT close, after them
    mkt_move      baseline − book_prev    prices moved, holdings fixed
    trade_delta   slot0_score − baseline  holdings moved, prices fixed

  and, from v2, the deployed model's ORDER INTENT — what it decided to do before anything filled:

    oi_n_buy      symbols carrying a buy order (0..12);  oi_n_sell likewise
    oi_buy_aggr   mean (buy limit − close_t)/span_t; > 0 = limit above today's close
    oi_sell_aggr  same for the sell_all limit
    oi_buy_disp   sd of buy_price_frac across ordering symbols
    oi_buy_val    intended buy in dollars at today's close;  oi_sell_val likewise

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
DS_VERSION   = 3
N_IND        = 12
DS_FEAT      = 74
CFEAT        = 146       # MT1CNet's input width; mirrors MT1C_IN in mt1_pool.h

COMPONENTS = ('book_prev', 'book_now', 'mkt_move', 'trade_delta',
              # v2: slot-0 order INTENT. Causal — limit prices are anchored to TODAY's bar and
              # the quantities come off StockNN's forward pass on today's data. Contrast
              # buy_exec/sell_exec, which count what FILLED and are decided by the NEXT day's
              # bar; those are look-ahead and do not exist yet at production decision time.
              'oi_n_buy', 'oi_n_sell', 'oi_buy_aggr', 'oi_sell_aggr',
              'oi_buy_disp', 'oi_buy_val', 'oi_sell_val')

# Both pools' deployed prediction and its score, paired on the same row.
PAIRED = ('m_pred', 'm_score', 'c_pred', 'c_score')

RECORD_FMT  = ('<II' + 'f' * (DS_FEAT * N_IND) + 'f' * (len(COMPONENTS) * N_IND)
               + 'f' * (CFEAT * N_IND) + 'f' * (len(PAIRED) * N_IND))
RECORD_SIZE = struct.calcsize(RECORD_FMT)
assert RECORD_SIZE == 11288, f'dataset record must be 11288 bytes, got {RECORD_SIZE}'

# Field offsets inside each symbol's 12-wide block of cfeat — mirrors the CN_* constants.
# All scale-free. See build_mt1c_input in training_v4.cpp for why.
CN_FIELDS = ('open_over_c', 'high_over_c', 'low_over_c', 'close_rel', 'close_pos',
             'close_vs_wap', 'range_over_a', 'pos_weight',
             'buy_frac_avail', 'buy_price_frac', 'sell_all_price_frac', 'sell_frac_held')
CN_SYMS, CN_PER_SYM = 12, 12
CN_CASH, CN_BOOK = CN_SYMS * CN_PER_SYM, CN_SYMS * CN_PER_SYM + 1


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
                'cfeat': np.zeros((0, N_IND, CFEAT), np.float32),
                **{k: np.zeros((0, N_IND), np.float32) for k in PAIRED},
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
    out['cfeat'] = rest[:, off:off + CFEAT * N_IND].reshape(n, N_IND, CFEAT)
    off += CFEAT * N_IND
    for k in PAIRED:
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
    print(f'\n{"order intent":<14} {"mean":>11} {"sd":>9}')
    for k in COMPONENTS[4:]:
        print(f'{k:<14} {ds[k].mean():>+11.3f} {ds[k].std():>9.3f}')
    sc = [ds[k] for k in ('m_score', 'c_score')]
    if np.any(sc[0] > 0):
        print(f'\n{"evolved pools":<18} {"mean score":>11} {"mean |pred|":>12}')
        for lbl, k in (('MT1Net  (74)', 'm'), ('MT1CNet (146)', 'c')):
            m = ds[f'{k}_score'] > 0
            print(f'{lbl:<18} {ds[f"{k}_score"][m].mean():>11.4f} '
                  f'{np.abs(ds[f"{k}_pred"][m]).mean():>12.1f}')
    print(f'\ncompetitor input: {ds["cfeat"].shape}  '
          f'finite {np.isfinite(ds["cfeat"]).mean():.2%}')
    print(f'features: {ds["feat"].shape}  '
          f'finite {np.isfinite(ds["feat"]).mean():.2%}  '
          f'|x| p99 {np.percentile(np.abs(ds["feat"]), 99):.3g}')


if __name__ == '__main__':
    main()
