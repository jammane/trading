#!/usr/bin/env python3
"""
mt1_analysis.py — decide A, B and C from mt1_dataset.bin, without another training run.

  A  which formulation predicts at all:  per-industry | pooled | cross-sectional
  B  the horizon h
  C  which of MT1/MT2 has to be a network, and which can be arithmetic

Three things this harness takes seriously, because each has already produced a wrong answer in
this project:

  NO LOOK-AHEAD.  A model predicting the window [t+1, t+h] may only be fitted on windows that
  CLOSED before t. Not "started before t" — closed. An off-by-one makes every horizon look
  predictable, and it is invisible in the output.

  OVERLAP-CORRECT ERROR BARS.  Consecutive daily h-day predictions share h-1 days, and the 12
  industries move together. Treating T x 12 rows as independent overstates t by roughly
  sqrt(h * 12). That produced a t of -7.20 here on a signal indistinguishable from zero. All
  standard errors come from a moving-block bootstrap over TIME, resampling whole blocks with all
  industries attached, so both dependencies are preserved.

  CONTROLS, ALWAYS.  Every run reports a shuffled-target control (must give IC ~ 0 — if it does
  not, the protocol leaks) and a planted-signal control (must be recovered — if it is not, a null
  result means nothing). An IC of 0.02 is meaningless without both.
"""
import numpy as np

import read_mt1_dataset as D

# ── statistics ───────────────────────────────────────────────────────────────────


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4 or a[m].std() < 1e-12 or b[m].std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4:
        return np.nan
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    return pearson(ra, rb)


def block_bootstrap(stat_fn, n_rows, block, n_boot=400, seed=0):
    """SE and percentile CI for a statistic over rows that are serially dependent.

    stat_fn(row_index_array) -> float. Rows are resampled in contiguous blocks so that
    within-block dependence (overlapping forward windows) survives resampling. Caller passes
    row indices that share a time axis, so resampling a block carries all its industries.

    block must exceed the dependence length, which here is the horizon h. Too short and the SE
    collapses back toward the independent-sample answer, which is the error being corrected.
    """
    rng = np.random.default_rng(seed)
    if n_rows <= block:
        return np.nan, (np.nan, np.nan)
    n_blocks = int(np.ceil(n_rows / block))
    starts_max = n_rows - block
    out = []
    for _ in range(n_boot):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n_rows]
        v = stat_fn(idx)
        if np.isfinite(v):
            out.append(v)
    if len(out) < 20:
        return np.nan, (np.nan, np.nan)
    out = np.array(out)
    return float(out.std(ddof=1)), (float(np.percentile(out, 2.5)),
                                    float(np.percentile(out, 97.5)))


# ── fitting ──────────────────────────────────────────────────────────────────────


def ridge_fit(X, y, lam):
    Xm, ym = X.mean(0), y.mean()
    Xc = X - Xm
    A = Xc.T @ Xc + lam * np.eye(X.shape[1])
    w = np.linalg.solve(A, Xc.T @ (y - ym))
    return w, Xm, ym


def walk_forward(X, y, h, min_train=250, refit_every=20, lam=10.0):
    """Predict y[t] from X[t], fitting only on rows whose forward window closed before t.

    Row t's target covers days t+1..t+h, so it is known only at t+h. A fit made for test day t may
    therefore use rows up to t-h-1 inclusive. `refit_every` reuses a fit for the following K days,
    which is both far cheaper and what production would actually do.

    Returns (pred, actual, rows) for the test rows only.
    """
    T = len(y)
    pred = np.full(T, np.nan)
    w = Xm = ym = None
    last_fit = -10**9
    for t in range(min_train + h, T):
        avail = t - h                                   # rows 0..avail-1 have closed windows
        if avail < min_train:
            continue
        if t - last_fit >= refit_every or w is None:
            m = np.isfinite(y[:avail])
            if m.sum() < min_train:
                continue
            w, Xm, ym = ridge_fit(X[:avail][m], y[:avail][m], lam)
            last_fit = t
        pred[t] = (X[t] - Xm) @ w + ym
    rows = np.where(np.isfinite(pred) & np.isfinite(y))[0]
    return pred[rows], y[rows], rows


# ── formulations ─────────────────────────────────────────────────────────────────


def _target(ds, h, component, standardize):
    """Forward-h P&L per industry, optionally scaled by each industry's TRAILING sd.

    Standardizing matters for the pooled fit: industry P&L sds differ several-fold, so a raw pool
    is dominated by the loudest industries. The scale is trailing-only, so it stays causal.
    """
    y = D.forward_pnl(ds, h, component)
    if not standardize:
        return y
    out = np.full_like(y, np.nan)
    per_day = (ds['book_now'] - ds['book_prev']).astype(float)
    for i in range(y.shape[1]):
        s = np.full(len(y), np.nan)
        for t in range(60, len(y)):
            w = per_day[max(0, t - 250):t, i]
            sd = w.std()
            s[t] = sd if sd > 1e-9 else np.nan
        out[:, i] = y[:, i] / s
    return out


def run_formulation(ds, h, formulation, component='book', standardize=True, **kw):
    """Fit and evaluate one formulation. Returns per-industry predictions aligned on a time axis."""
    feat = ds['feat'].astype(float)                     # (T, 12, 74)
    T, N, F = feat.shape
    y = _target(ds, h, component, standardize)
    P = np.full((T, N), np.nan)
    A = np.full((T, N), np.nan)

    if formulation == 'per-industry':
        for i in range(N):
            p, a, rows = walk_forward(feat[:, i, :], y[:, i], h, **kw)
            P[rows, i] = p
            A[rows, i] = a

    elif formulation == 'pooled':
        # One model for all 12. The fit sees 12x the rows; the walk-forward cut is still by TIME,
        # so no industry's future leaks into another industry's past.
        # Stack industries along a pseudo-row axis; row r belongs to time r // N, so the
        # walk-forward cut by TIME still excludes every industry's future.
        big_X = feat.reshape(T * N, F)
        big_y = y.reshape(T * N)
        # a row's time index is row // N
        w = Xm = ym = None
        last_fit = -10**9
        for t in range(kw.get('min_train', 250) + h, T):
            avail = t - h
            if avail < kw.get('min_train', 250):
                continue
            if t - last_fit >= kw.get('refit_every', 20) or w is None:
                sel = slice(0, avail * N)
                m = np.isfinite(big_y[sel])
                if m.sum() < kw.get('min_train', 250):
                    continue
                w, Xm, ym = ridge_fit(big_X[sel][m], big_y[sel][m], kw.get('lam', 10.0))
                last_fit = t
            for i in range(N):
                if np.isfinite(y[t, i]):
                    P[t, i] = (feat[t, i] - Xm) @ w + ym
                    A[t, i] = y[t, i]

    elif formulation == 'cross-sectional':
        # Every industry sees ALL 888 features — MT2's job done directly, no MT1 in between.
        X = feat.reshape(T, N * F)
        for i in range(N):
            p, a, rows = walk_forward(X, y[:, i], h, **kw)
            P[rows, i] = p
            A[rows, i] = a

    else:
        raise ValueError(f'unknown formulation {formulation!r}')

    return P, A


# ── evaluation ───────────────────────────────────────────────────────────────────


def evaluate(P, A, h, n_boot=400, seed=0):
    """IC, rank IC and a top-3 spread, each with a block-bootstrap SE over time blocks.

    Rank IC and the top-3 spread are per-DAY quantities: a day's cross-sectional ranking does not
    change depending on which other days were resampled. So they are computed once here and the
    bootstrap averages a subset, rather than being recomputed inside every one of 400 iterations.
    That is ~400x less work and is why this is fast enough to run in the test suite.
    """
    rows = np.where(np.isfinite(P).any(1) & np.isfinite(A).any(1))[0]
    if len(rows) < 100:
        return None
    Pr, Ar = P[rows], A[rows]
    n = len(rows)
    block = max(2 * h, 20)

    # per-day series, computed once
    daily_rank = np.full(n, np.nan)
    daily_top = np.full(n, np.nan)
    for t in range(n):
        p, a = Pr[t], Ar[t]
        m = np.isfinite(p) & np.isfinite(a)
        if m.sum() >= 4:
            daily_rank[t] = spearman(p[m], a[m])
        if m.sum() >= 6:
            pi, ai = p[m], a[m]
            daily_top[t] = ai[np.argsort(pi)[-3:]].mean() - ai.mean()

    flatP, flatA = Pr.ravel(), Ar.ravel()
    width = Pr.shape[1]

    def flat_ic(idx):
        sel = (idx[:, None] * width + np.arange(width)).ravel()
        return pearson(flatP[sel], flatA[sel])

    def rank_ic(idx):
        v = daily_rank[idx]
        return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan

    def top_spread(idx):
        v = daily_top[idx]
        return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan

    res = {'n_days': n, 'block': block}
    all_idx = np.arange(n)
    for name, fn in (('ic', flat_ic), ('rank_ic', rank_ic), ('top3_spread', top_spread)):
        v = fn(all_idx)
        se, ci = block_bootstrap(fn, n, block, n_boot=n_boot, seed=seed)
        res[name] = v
        res[name + '_se'] = se
        res[name + '_t'] = v / se if (se and np.isfinite(se) and se > 0) else np.nan
        res[name + '_ci'] = ci
    return res


def shuffle_control(ds, h, formulation, seed=0, **kw):
    """Same pipeline, targets shuffled in time. Must give IC ~ 0 or the protocol leaks."""
    rng = np.random.default_rng(seed)
    ds2 = dict(ds)
    perm = rng.permutation(len(ds['day']))
    for k in ('book_prev', 'book_now', 'mkt_move', 'trade_delta'):
        ds2[k] = ds[k][perm]
    return evaluate(*run_formulation(ds2, h, formulation, **kw), h, n_boot=200, seed=seed)


def _plant(ds, h, strength, component='book'):
    """Return a copy of `ds` with a known signal added to the component being measured.

    The signal must land in the field `forward_pnl()` actually READS for this component. It used
    to always go into `book_now`, which meant that under `--component mkt` or `--component trade`
    the plant never reached the target at all: the control silently measured nothing, reported no
    power, and so the nulls those two runs produced were uninterpretable rather than evidence of
    absence. A control that cannot fail is not a control.

    Two invariants, both load-bearing:

    * The plant is scaled to the sd of the component BEING MEASURED, not always the book's. The
      trade delta's sd is several times smaller than the book's (measured: $419 vs $691), so a
      book-scaled plant injected into the trade leg would be relatively enormous and the control
      would pass trivially, which tells us nothing about detecting a trade-sized effect.
    * The decomposition identity `book_now - book_prev == mkt_move + trade_delta` survives.
      Adding to `book_now` AND to exactly one of the two legs keeps both sides equal, so the
      planted copy stays a physically consistent dataset rather than one that only happens not to
      be checked here.
    """
    f0 = ds['feat'][:, :, 0].astype(float)
    f0 = (f0 - np.nanmean(f0)) / (np.nanstd(f0) + 1e-9)
    base = {'book': ds['book_now'] - ds['book_prev'],
            'mkt': ds['mkt_move'],
            'trade': ds['trade_delta']}[component].astype(float)
    per_day = strength * np.nanstd(base) * f0 / h
    add = np.zeros_like(per_day)
    for k in range(1, h + 1):
        add[k:] += per_day[:-k]
    ds2 = dict(ds)
    ds2['book_now'] = (ds['book_now'] + add).astype(np.float32)
    # 'book' is carried on the market leg by convention; either leg would do, but it must be
    # exactly one of them or the identity breaks.
    leg = 'trade_delta' if component == 'trade' else 'mkt_move'
    ds2[leg] = (ds[leg] + add).astype(np.float32)
    return ds2


def planted_control(ds, h, formulation, strength=0.30, seed=0, **kw):
    """Same pipeline, with a known signal added to the target from feature 0.

    The plant must fill the WHOLE forward window, not one day of it. A one-day kick tested at h=5
    is diluted by sqrt(5) and by the four days it does not touch, so the control fails and the
    null it is supposed to validate becomes uninterpretable. That happened: at h=5 a one-day plant
    read IC +0.020 (t +1.0) while the same plant at h=1 read +0.164 (t +10.4).

    So day t's feature drives days t+1..t+h, each by strength/h of the daily sd. The forward-h sum
    from t then contains the full `strength * sd * f0[t]` exactly once, at every h.

    If this control is not clearly positive, the harness has no power at that horizon and any null
    it reports there is uninformative rather than evidence of absence.

    The plant is placed by `_plant()`, which routes it to the field the chosen `component` is
    actually read from. See that function for why.
    """
    ds2 = _plant(ds, h, strength, kw.get('component', 'book'))
    return evaluate(*run_formulation(ds2, h, formulation, **kw), h, n_boot=200, seed=seed)


# ── CLI ──────────────────────────────────────────────────────────────────────────


def _fmt(r, label):
    if r is None:
        return f'  {label:<18}  (too few test days)'
    def cell(k):
        v, t = r[k], r[k + '_t']
        star = '*' if np.isfinite(t) and abs(t) > 2 else ' '
        return f'{v:>+8.4f} ({t:>+5.1f}){star}'
    return (f'  {label:<18} {r["n_days"]:>5} '
            f'{cell("ic")} {cell("rank_ic")} '
            f'{r["top3_spread"]:>+9.4f} ({r["top3_spread_t"]:>+5.1f})')


def main():
    import argparse
    ap = argparse.ArgumentParser(description='A/B/C from mt1_dataset.bin')
    ap.add_argument('dataset')
    ap.add_argument('--horizons', default='1,2,3,5,10,20')
    ap.add_argument('--component', default='book', choices=['book', 'mkt', 'trade'])
    ap.add_argument('--formulations', default='per-industry,pooled,cross-sectional')
    ap.add_argument('--start-day', type=int, default=0,
                    help='drop days below this (skip the StockNN learning curve)')
    ap.add_argument('--raw-target', action='store_true',
                    help='do not scale the target by each industry trailing sd')
    ap.add_argument('--boot', type=int, default=400)
    ap.add_argument('--no-controls', action='store_true')
    args = ap.parse_args()

    ds = D.parse(args.dataset)
    ok, err = D.verify_identity(ds)
    if not ok:
        raise SystemExit(f'decomposition identity failed (max error ${err:.4f}) — do not trust this file')
    if args.start_day:
        m = ds['day'] >= args.start_day
        ds = {k: v[m] for k, v in ds.items()}
    print(f'{args.dataset}: {len(ds["day"])} days '
          f'(day {ds["day"][0]}-{ds["day"][-1]}), component={args.component}, '
          f'target={"raw $" if args.raw_target else "trailing-sd scaled"}')
    print('identity OK.  SEs are moving-block bootstrap over time blocks '
          '(block = max(2h, 20)); * = |t| > 2\n')

    kw = dict(standardize=not args.raw_target)
    for h in [int(x) for x in args.horizons.split(',')]:
        print(f'h = {h}')
        print(f'  {"formulation":<18} {"days":>5} {"IC (t)":>18} {"rank IC (t)":>18} '
              f'{"top3 spread (t)":>19}')
        for f in args.formulations.split(','):
            P, A = run_formulation(ds, h, f, component=args.component, **kw)
            print(_fmt(evaluate(P, A, h, n_boot=args.boot), f))
        if not args.no_controls:
            print(_fmt(shuffle_control(ds, h, 'per-industry', component=args.component, **kw),
                       'CONTROL shuffled'))
            print(_fmt(planted_control(ds, h, 'per-industry', component=args.component, **kw),
                       'CONTROL planted'))
        print()
    print('  shuffled must be ~0 (else the protocol leaks); planted must be clearly positive')
    print('  (else the harness has no power and a null result means nothing).')


if __name__ == '__main__':
    main()
