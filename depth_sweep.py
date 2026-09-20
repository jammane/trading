#!/usr/bin/env python3
"""Does conditioning on MORE past days help? 5 memory kernels x 1-5 days of input.

STATE = the k days ORDERED BY PROFITABILITY, plus how many of them were positive.

That is the natural generalisation of the six-state scheme, and it reproduces the earlier methods
exactly rather than approximately:

    k     states = k! * (k+1)
    1      2      sign(today)                        == "method 1"
    2      6      the six accel/decel/flip states    == "method 2", the current model
    3     24
    4    120
    5    720

At k = 2 the mapping is exact: knowing the profitability order of the two days AND how many were
positive determines which of up_more / up_less / up_down / dn_more / dn_less / dn_up it is, because
the positive days are always the top `m` in the profitability order. `verify_k2` asserts it.

Every state of every k is scored; nothing is grouped or dropped.

WHAT THIS REALLY MEASURES is a bias-variance trade. Each extra day multiplies the state space by
more than k while the data stays fixed. Post burn-in there are ~9,000 industry-days: ~4,500 per
state at k = 1, and ~12 at k = 5. More conditioning can only win if the extra days carry enough
signal to outrun that collapse. The walk-forward predictive log-likelihood answers it directly
because it is comparable across state-space sizes -- an in-sample fit is not, since a larger state
space always fits better.

Strictly causal: the table used to predict day t has only ever observed days before t.

    python depth_sweep.py mt1_dataset.bin --burn-in 500
"""
import argparse
from itertools import permutations
from math import factorial

import numpy as np

import read_mt1_dataset as D

_PERM_IDX = {k: {p: i for i, p in enumerate(permutations(range(k)))} for k in range(1, 6)}


def n_states(k):
    """k = 0 is the no-conditioning control: one global P(tomorrow keeps today's sign)."""
    return 1 if k == 0 else factorial(k) * (k + 1)


def state_k(vals):
    """State from k consecutive days, oldest first. None if any day is flat or non-finite.

    k = 0 returns state 0 for every day: the control that conditions on nothing, so any column
    that fails to beat it is not using its conditioning at all.

    The rank vector says which day was most profitable, second, and so on; `m` says how many were
    profitable at all. Together they are the ordered set the state is built from.
    """
    for v in vals:
        if v == 0 or not np.isfinite(v):
            return None
    k = len(vals)
    if k == 0:
        return 0
    ranks = tuple(int(r) for r in np.argsort(np.argsort(np.asarray(vals, float))))
    m = int(sum(1 for v in vals if v > 0))
    return _PERM_IDX[k][ranks] * (k + 1) + m


def verify_k2():
    """At k = 2 the ordered-set state must be a relabelling of the original six, one-to-one."""
    def original(prev, cur):
        if prev > 0:
            return 'up_more' if cur > 0 and cur >= prev else ('up_less' if cur > 0 else 'up_down')
        return 'dn_more' if cur < 0 and cur <= prev else ('dn_less' if cur < 0 else 'dn_up')

    seen = {}
    rng = np.random.default_rng(0)
    for _ in range(20000):
        a, b = rng.normal(0, 100), rng.normal(0, 100)
        if a == 0 or b == 0:
            continue
        s = state_k([a, b])
        o = original(a, b)
        if s in seen and seen[s] != o:
            raise AssertionError(f'state {s} maps to both {seen[s]} and {o}')
        seen[s] = o
    if len(set(seen.values())) != 6 or len(seen) != 6:
        raise AssertionError(f'expected a 6<->6 bijection, got {len(seen)} states '
                             f'-> {len(set(seen.values()))} names')
    return seen


def run(pnl, k, half_life, window, warmup=250, prior=2.0):
    """Walk-forward predictive log-likelihood of the sign outcome, per observation.

    Pooled across industries: the between-industry variance of this effect measured SMALLER than
    sampling noise alone (tau^2 = 0), so per-industry tables would be fitting noise -- and at
    k = 5 there would be about one observation per industry per state.
    """
    T, N = pnl.shape
    S = n_states(k)
    w_n, w_same = np.zeros(S), np.zeros(S)
    ring = []
    gamma = 0.5 ** (1.0 / half_life) if window <= 0 else 1.0
    ll, n_obs = 0.0, 0
    seen = np.zeros(S, bool)

    for t in range(max(k, 1), T):
        states, sames = [], []
        for i in range(N):
            st = state_k(pnl[t - k:t, i]) if k else 0
            if k == 0 and (pnl[t - 1, i] == 0 or not np.isfinite(pnl[t - 1, i])):
                st = None
            if st is None or pnl[t, i] == 0 or not np.isfinite(pnl[t, i]):
                continue
            same = float(np.sign(pnl[t, i]) == np.sign(pnl[t - 1, i]))
            states.append(st)
            sames.append(same)
            if t > warmup:
                p = (w_same[st] + prior * 0.5) / (w_n[st] + prior)
                p = min(max(p, 1e-6), 1 - 1e-6)
                ll += np.log(p if same else 1 - p)
                n_obs += 1
                seen[st] = True
        if window <= 0:
            w_n *= gamma
            w_same *= gamma
        for st, same in zip(states, sames, strict=True):
            w_n[st] += 1.0
            w_same[st] += same
        if window > 0:
            ring.append(list(zip(states, sames, strict=True)))
            while len(ring) > window:
                for st, same in ring.pop(0):
                    w_n[st] -= 1.0
                    w_same[st] -= same
    occ = int(seen.sum())
    return (ll / n_obs if n_obs else float('nan')), n_obs, occ, float(w_n.sum() / max(occ, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('--burn-in', type=int, default=500)
    a = ap.parse_args()

    mapping = verify_k2()
    print(f'k=2 reproduces the original six states exactly '
          f'({len(mapping)} <-> {len(set(mapping.values()))} bijection)\n')

    ds = D.read(a.dataset) if hasattr(D, 'read') else D.parse(a.dataset)
    day = np.asarray(ds['day'])
    raw = ds['book_now'].astype(float) - ds['book_prev'].astype(float)
    keep = day >= a.burn_in
    cols = [raw[keep, i][np.abs(raw[keep, i]) > 1e-9] for i in range(raw.shape[1])]
    m = min(len(c) for c in cols)
    P = np.column_stack([c[-m:] for c in cols])
    coin = -np.log(2.0)
    print(f'StockNN book P&L, day >= {a.burn_in}: {P.shape[0]} days x {P.shape[1]} industries')
    print(f'coin flip = {coin:.5f}; table is MILLINATS above it (higher is better)\n')

    kernels = [('1y decay', 252.0, 0), ('2y decay', 504.0, 0),
               ('1y window', 1e9, 252), ('2y window', 1e9, 504),
               ('never forget', 1e9, 0)]
    print(f'  {"kernel":<13} ' + ' '.join(f'{"k=" + str(k):>9}' for k in range(0, 6)))
    best, grid = (None, None, -1e9), {}
    for name, hl, win in kernels:
        cells = []
        for k in range(0, 6):
            ll, n, occ, per = run(P, k, hl, win)
            grid[(name, k)] = (ll, n, occ, per)
            cells.append(f'{1000 * (ll - coin):>+9.2f}')
            if ll > best[2]:
                best = (name, k, ll)
        print(f'  {name:<13} ' + ' '.join(cells))
    print(f'\n  best: {best[0]}, k = {best[1]}  ({1000 * (best[2] - coin):+.2f} millinats)')

    print('\n  the cost of more input days (k=0 conditions on nothing -- the control):')
    print(f'  {"k":>3} {"states":>8} {"occupied":>9} {"obs/state":>10}')
    for k in range(0, 6):
        _ll, _n, occ, per = grid[('never forget', k)]
        print(f'  {k:>3} {n_states(k):>8} {occ:>9} {per:>10.1f}')


if __name__ == '__main__':
    main()
