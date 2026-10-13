#!/usr/bin/env python3
"""How long does an industry's P&L behaviour persist?

A different question from `mt1_analysis.py`. That asks whether today's FEATURES predict forward
P&L, and answered no at h=1-2 with a working control. This asks whether the P&L series has
exploitable structure AT ALL, independent of any model or feature set: if an industry is earning
now, how long does that last, and how long does an estimate of its rate stay valid?

That matters because it bounds what any allocator could achieve. If runs are no longer than
coin-flipping and the rate estimate decays immediately, then no MT1/MT2 architecture can rank
industries usefully and the target is the problem, not the network. If there IS persistence, it
names the holding period an allocator should use.

FOUR MEASURES, each against a shuffled control (same values, time order destroyed):

  1. SIGN RUNS     -- consecutive days of the same sign. Longer than shuffled = momentum.
  2. VARIANCE RATIO -- Lo-MacKinlay VR(q) = Var(q-day sum) / (q * Var(1-day)).
                       VR > 1 trending, = 1 random walk, < 1 mean-reverting. This is the direct
                       statistical answer to "over what horizon does P&L accumulate".
  3. SLOPE LIFE    -- fit the mean daily rate over an opening window, then walk forward until the
                       running mean leaves a tolerance band around it. How long the rate holds.
  4. RANK LIFE     -- rank the 12 industries by trailing P&L, then measure how long that ranking
                       survives. The allocation question stated directly.

Reset days are dropped. `step_industry` recapitalises a book below $22,500, and the reset path
writes book_prev == book_now so the day contributes exactly $0. Those are not real flat days and
would both shorten runs and inflate the zero bin.

    python pnl_persistence.py mt1_dataset.bin
    python pnl_persistence.py mt1_dataset.bin --burn-in 500
"""
import argparse
import sys

import numpy as np

import read_mt1_dataset as D

IND = ['hardware', 'software', 'financial', 'discret', 'services', 'health',
       'industrl', 'staples', 'energy', 'utilitie', 'land', 'materials']


# ── helpers ─────────────────────────────────────────────────────────────────────

def block_boot(fn, T, n_boot, block, seed):
    """Moving-block bootstrap over TIME. Returns (se, lo, hi) of fn(idx).

    Every measure here is computed on overlapping windows of a serially dependent series, so iid
    resampling would understate the SE badly -- the same error that once produced a t of -7.20 in
    this project on a signal indistinguishable from zero.
    """
    rng = np.random.default_rng(seed)
    n_blocks = max(1, int(np.ceil(T / block)))
    out = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, T - block), size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block, T)) for s in starts])[:T]
        v = fn(idx)
        if v is not None and np.isfinite(v):
            out.append(v)
    if len(out) < 20:
        return np.nan, np.nan, np.nan
    a = np.array(out)
    return a.std(ddof=1), np.percentile(a, 2.5), np.percentile(a, 97.5)


def sign_runs(x):
    """Lengths of maximal same-sign runs in a 1-D series, zeros dropped."""
    s = np.sign(x)
    s = s[s != 0]
    if s.size == 0:
        return np.array([])
    brk = np.flatnonzero(np.diff(s) != 0)
    ends = np.concatenate([brk, [s.size - 1]])
    starts = np.concatenate([[0], brk + 1])
    return (ends - starts + 1).astype(float)


def variance_ratio(x, q):
    """Lo-MacKinlay VR(q) with the heteroskedasticity-robust z statistic.

    x is a series of per-period increments (daily P&L). Under a random walk in the cumulative
    series, VR(q) = 1 for every q.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    T = x.size
    if 4 * q > T:
        return np.nan, np.nan
    mu = x.mean()
    d = x - mu
    var1 = (d @ d) / (T - 1)
    if var1 <= 0:
        return np.nan, np.nan
    csum = np.concatenate([[0.0], np.cumsum(x)])
    qsum = csum[q:] - csum[:-q]                       # overlapping q-day sums
    # Lo-MacKinlay unbiased scaling: m = q(T-q+1)(1 - q/T).
    # This read (1 - 1/q) in the first draft, which inflates varq by 1/(1-1/q) -- a factor of 2.0
    # at q=2, 1.5 at q=3, 1.25 at q=5. It produced VR(2)=1.97 with z +89.9 on REAL data, which
    # looks like overwhelming evidence of trending. The shuffled control reported VR(2)=1.966 on
    # the same data with time order destroyed, where the true value is 1.0 by construction, and
    # that is the only reason the bug was caught rather than published.
    m = q * (T - q + 1) * (1.0 - float(q) / T)
    varq = ((qsum - q * mu) ** 2).sum() / m
    vr = varq / var1
    # robust variance: sum of weighted squared-increment autocovariance ratios
    theta = 0.0
    for j in range(1, q):
        num = ((d[j:] ** 2) * (d[:-j] ** 2)).sum()
        den = (d @ d) ** 2 / T
        dj = num / den if den > 0 else 0.0
        theta += ((2.0 * (q - j) / q) ** 2) * dj
    z = (vr - 1.0) / np.sqrt(theta / T) if theta > 0 else np.nan
    return vr, z


def slope_life(x, w0=20, tol=1.0, max_h=250):
    """How many days an opening estimate of the mean daily rate stays valid.

    From each start t: s0 = mean(x[t : t+w0]), se0 = sd(x[t : t+w0]) / sqrt(w0). Walk m forward
    and stop at the first m where |mean(x[t : t+m]) - s0| > tol * se0. Returns those lifetimes.

    Read this only against the shuffled control. Under an iid series the running mean also drifts
    away from a noisy opening estimate, so a finite lifetime is NOT by itself evidence of regime
    change -- only a lifetime that differs from the shuffled one is.
    """
    x = np.asarray(x, float)
    T = x.size
    lives = []
    for t in range(0, T - w0 - 1):
        w = x[t:t + w0]
        if not np.all(np.isfinite(w)):
            continue
        s0 = w.mean()
        se0 = w.std(ddof=1) / np.sqrt(w0)
        if not np.isfinite(se0) or se0 <= 0:
            continue
        hi = min(T - t, max_h)
        run = np.cumsum(x[t:t + hi])
        m = np.arange(1, hi + 1)
        dev = np.abs(run / m - s0)
        bad = np.flatnonzero((m > w0) & (dev > tol * se0))
        lives.append(float(bad[0] + 1) if bad.size else float(hi))
    return np.array(lives)


def rank_life(pnl, k, lags):
    """Does the trailing-k ranking of industries survive `lag` days?

    Returns, per lag: the Spearman correlation between the trailing-k rank at t and the trailing-k
    rank at t+lag, and P(an industry in the top tercile at t is still there at t+lag).
    """
    T, N = pnl.shape
    csum = np.vstack([np.zeros((1, N)), np.cumsum(pnl, axis=0)])
    trail = csum[k:] - csum[:-k]                      # (T-k+1, N) trailing-k sums
    rk = np.argsort(np.argsort(trail, axis=1), axis=1).astype(float)
    top = rk >= (N - N // 3)
    out = {}
    for lag in lags:
        if trail.shape[0] <= lag:
            continue
        a, b = rk[:-lag], rk[lag:]
        cs = [np.corrcoef(a[i], b[i])[0, 1] for i in range(a.shape[0])]
        out[lag] = (float(np.nanmean(cs)),
                    float((top[lag:][top[:-lag]]).mean()))
    return out


# ── report ──────────────────────────────────────────────────────────────────────

def analyse(pnl, label, n_boot, seed):
    T, N = pnl.shape
    rng = np.random.default_rng(seed)
    shuf = np.column_stack([rng.permutation(pnl[:, i]) for i in range(N)])

    print(f'\n{"=" * 78}\n{label}   ({T} days x {N} industries)\n{"=" * 78}')
    pos = float((pnl > 0).mean())
    print(f'daily P&L: mean ${pnl.mean():+.2f}  sd ${pnl.std():.0f}  positive {100*pos:.1f}%')

    # 1 ── sign runs
    obs = np.concatenate([sign_runs(pnl[:, i]) for i in range(N)])
    nul = np.concatenate([sign_runs(shuf[:, i]) for i in range(N)])
    def run_stat(idx):
        return np.concatenate([sign_runs(pnl[idx, i]) for i in range(N)]).mean()
    se, lo, hi = block_boot(run_stat, T, n_boot, 20, seed)
    diff = obs.mean() - nul.mean()
    t = diff / se if se and np.isfinite(se) and se > 0 else np.nan
    print(f'\n1. SIGN RUNS       observed mean {obs.mean():.2f} d   shuffled {nul.mean():.2f} d   '
          f'diff {diff:+.3f} (t {t:+.1f})')
    print(f'   longest observed {obs.max():.0f} d, shuffled {nul.max():.0f} d; '
          f'median {np.median(obs):.0f} vs {np.median(nul):.0f}')

    # 2 ── variance ratio
    print('\n2. VARIANCE RATIO  (VR>1 trending, =1 random walk, <1 mean-reverting)')
    print('      q     VR     z        shuffled VR')
    for q in (2, 3, 5, 10, 20):
        vrs = [variance_ratio(pnl[:, i], q) for i in range(N)]
        zs = [z for _, z in vrs if np.isfinite(z)]
        vv = [v for v, _ in vrs if np.isfinite(v)]
        sv = [variance_ratio(shuf[:, i], q)[0] for i in range(N)]
        sv = [v for v in sv if np.isfinite(v)]
        if not vv:
            continue
        zbar = np.mean(zs) * np.sqrt(len(zs)) if zs else np.nan   # Stouffer across industries
        star = '*' if np.isfinite(zbar) and abs(zbar) > 2 else ' '
        print(f'    {q:3d}  {np.mean(vv):5.3f}  {zbar:+6.1f}{star}      {np.mean(sv):5.3f}')

    # 3 ── slope life
    ob = np.concatenate([slope_life(pnl[:, i]) for i in range(N)])
    nu = np.concatenate([slope_life(shuf[:, i]) for i in range(N)])
    def sl_stat(idx):
        return np.concatenate([slope_life(pnl[idx, i]) for i in range(N)]).mean()
    se2, _, _ = block_boot(sl_stat, T, max(40, n_boot // 4), 40, seed)
    d2 = ob.mean() - nu.mean()
    t2 = d2 / se2 if se2 and np.isfinite(se2) and se2 > 0 else np.nan
    print(f'\n3. SLOPE LIFE      observed median {np.median(ob):.0f} d  mean {ob.mean():.1f} d   '
          f'shuffled median {np.median(nu):.0f} d  mean {nu.mean():.1f} d')
    print(f'   difference {d2:+.2f} d (t {t2:+.1f});  quartiles obs '
          f'{np.percentile(ob,25):.0f}/{np.percentile(ob,75):.0f}  '
          f'shuf {np.percentile(nu,25):.0f}/{np.percentile(nu,75):.0f}')

    # 4 ── rank life
    print('\n4. RANK LIFE       trailing-k ranking vs the ranking `lag` days later')
    lags = [1, 2, 3, 5, 10, 20, 40]
    for k in (5, 20, 60):
        r = rank_life(pnl, k, lags)
        s = rank_life(shuf, k, lags)
        cells = '  '.join(f'{lag}d {r[lag][0]:+.2f}' for lag in lags if lag in r)
        print(f'   k={k:3d} rank corr : {cells}')
        cells = '  '.join(f'{lag}d {s[lag][0]:+.2f}' for lag in lags if lag in s)
        print(f'         shuffled  : {cells}')
        cells = '  '.join(f'{lag}d {100*r[lag][1]:.0f}%' for lag in lags if lag in r)
        print(f'         top-tercile stays top: {cells}   (chance 33%)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('--burn-in', type=int, default=500,
                    help='days of StockNN learning to exclude in the second pass (default 500)')
    ap.add_argument('--boot', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    ds = D.read(a.dataset) if hasattr(D, 'read') else D.parse(a.dataset)
    day = np.asarray(ds['day'])
    pnl = (ds['book_now'].astype(float) - ds['book_prev'].astype(float))

    reset = (np.abs(pnl) < 1e-9)
    print(f'{a.dataset}: {pnl.shape[0]} days x {pnl.shape[1]} industries, '
          f'day {day[0]}-{day[-1]}')
    print(f'hard-floor reset / exactly-flat industry-days dropped: '
          f'{int(reset.sum())} ({100*reset.mean():.2f}%)')
    pnl = np.where(reset, np.nan, pnl)
    # column-wise compaction: each industry keeps its own non-reset days, so runs are not broken
    # by another industry's reset.
    cols = [pnl[:, i][np.isfinite(pnl[:, i])] for i in range(pnl.shape[1])]
    m = min(len(c) for c in cols)
    clean = np.column_stack([c[-m:] for c in cols])

    analyse(clean, 'FULL SAMPLE', a.boot, a.seed)
    if a.burn_in > 0:
        keep = day >= a.burn_in
        if keep.sum() > 200:
            cols = [pnl[keep, i][np.isfinite(pnl[keep, i])] for i in range(pnl.shape[1])]
            m = min(len(c) for c in cols)
            late = np.column_stack([c[-m:] for c in cols])
            analyse(late, f'AFTER BURN-IN (day >= {a.burn_in})', a.boot, a.seed)
        else:
            print(f'\n(burn-in {a.burn_in} leaves too few days)', file=sys.stderr)


if __name__ == '__main__':
    main()
