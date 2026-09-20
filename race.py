#!/usr/bin/env python3
"""The race: MT1Net vs MT1CNet vs MT2INet. Pre-registered, written before the data landed.

All three ran in the same pool, through the same step, lifecycle, score and target. Only the input
differs:

    MT1Net    74 = own industry, trailing curves
    MT1CNet  146 = own industry, TODAY only + StockNN's order intent
    MT2INet  888 = ALL industries, trailing  (the independent allocator, no MT1 in the loop)

WIN CONDITION — reliably detecting, BY ORDER AND BY WEIGHT, which industries' StockNN will make
the most money, so investment can be focused there and still diversified responsibly.

That is CROSS-SECTIONAL. Per-industry accuracy is a means, not the test: a model can predict each
industry passably and still order them wrongly, and an ordering that is right but mis-weighted
either concentrates recklessly or dilutes the edge away. So both halves are measured, and both
must hold.

  PRIMARY — the win condition itself, swept over the HOLDING PERIOD h
      ORDER    cross-sectional rank IC (Spearman across the 12 industries) against the forward
               h-day return > 0, block-bootstrap t > 3. (3, not 2: three candidates at once.)
      WEIGHT   allocating on the predictions BEATS EQUAL WEIGHT over the same h, block-bootstrap
               t > 2. Weights follow the production path — positive predictions split into
               terciles at 1 : 1.5 : 2.25 — so "diversify responsibly" is built in, not assumed.

      h is NOT yet decided, so it is not assumed here. All three were TRAINED at h = 1 (the
      trainer's target is the single-day book P&L), but the question the win condition asks is
      over what holding period acting on that signal pays — which is a different question and the
      one the horizon sweep exists to answer. A candidate WORKS if it clears both bars at SOME h,
      and that h is the answer to the horizon question. Overlapping holding periods are handled by
      a block bootstrap with block >= 2h.

  SUPPORTING — diagnostic only, never sufficient on its own
      per-industry corr(prediction, realised P&L), and the deployed score against the constant-$0
      benchmark. A candidate can pass these and still lose the race, which is the point.

  WORKING       both PRIMARY criteria hold, and ORDER survives in the second half of the pass.
  FAILED        needs POSITIVE evidence: its planted control was recovered (the harness could have
                seen an effect this size), its score has flattened (converged, not out of runway),
                and it is not within noise of a candidate that passed.
  INCONCLUSIVE  everything else. Gets more data, NOT deletion. MT2INet carries 28,929 parameters
                against MT1Net's 3,501, so a null at day 1,238 can mean "not finished" rather than
                "never". Abandoning a candidate that was merely unconvincing is the expensive
                mistake here, because the other two get deleted on the strength of it.

If NO candidate reaches WORKING, that is a statement about the TARGET, not a ranking of the three,
and nothing should be abandoned on differences between three nulls.

ADDED AFTER THE FIRST RUN (v0.8.1.11): a family of CONDITIONAL candidates, scored against the
identical win condition and bars. They cost nothing at training time -- unlike the three networks
they need no pool and no evolution, only the logged per-industry book P&L -- so they are evaluated
in the same batch pass after training completes.

    cond-1y-decay / cond-2y-decay      exponential memory, 252 / 504 day half-life
    cond-1y-win   / cond-2y-win        rectangular memory, last 252 / 504 days
    cond-forever                       no forgetting

Each predicts TOMORROW from (yesterday, today) via the six accelerate/decelerate/flip states, the
expectation estimated walk-forward on days strictly before the one being predicted. The streak
length N is deliberately NOT part of the prediction: measured across every state it sits within
0.25 days of the 2.0 a coin flip gives, and the observed and derived estimators agree to 0.065
days, so there is no streak to extrapolate.

TWO HONESTY NOTES ON ADDING THEM LATE:
  * The bars are unchanged. Raising or lowering a pre-registered threshold after seeing data is
    the failure this file exists to prevent.
  * Eight candidates is more multiple testing than three. But the five conditional variants differ
    only in how far back they remember, so they are near-duplicates rather than five independent
    shots -- their pairwise prediction correlation is printed below so that is visible and not
    merely asserted.
"""
import sys

import numpy as np

sys.path.insert(0, '.')
import mt1_analysis as M
import read_mt1_dataset as D

NET_CANDIDATES = [('MT1Net',  'm', '74  own industry, trailing'),
                  ('MT1CNet', 'c', '146 own industry, today + intent'),
                  ('MT2INet', 'i', '888 all industries, trailing')]

# (label, half_life, window). window = 0 selects exponential decay.
COND_KERNELS = [('1y-decay', 252.0, 0), ('2y-decay', 504.0, 0),
                ('1y-win', 1e9, 252), ('2y-win', 1e9, 504),
                ('forever', 1e9, 0)]
# Depth: how many past days the state is built from. State = those days ORDERED BY
# PROFITABILITY plus the profitability inversion point, so k! * (k+1) states -- k=1 gives 2
# (sign of today) and k=2 gives the original six. Only 1 and 2 are raced: depth_sweep.py measured
# k=3 at roughly break-even and k=4-5 at 8-50 millinats WORSE than a coin flip, because the state
# space outruns the data (720 states over ~9,000 industry-days is 14 observations each).
COND_DEPTHS = [1, 2]
COND_CANDIDATES = [(f'cond-k{k}-{lab}', hl, win, k,
                    f'ordered-set k={k}, {lab}')
                   for k in COND_DEPTHS for lab, hl, win in COND_KERNELS]
TIER_W = (1.0, 1.5, 2.25)

ds = D.parse(sys.argv[1] if len(sys.argv) > 1 else 'mt1_dataset.bin')
ok, err = D.verify_identity(ds)
if not ok:
    raise SystemExit(f'decomposition identity failed (${err:.4f}) — do not trust this file')

T = len(ds['day'])
book = (ds['book_now'] - ds['book_prev']).astype(float)
prev = np.maximum(ds['book_prev'].astype(float), 1.0)
ret = book / prev                      # per-industry realised return, comparable across industries
print(f'{T} days (day {ds["day"][0]}-{ds["day"][-1]}), identity OK\n')


def boot_t(v, block=20, n=400):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 50:
        return np.nan, np.nan
    m = v.mean()
    se, _ = M.block_bootstrap(lambda idx: float(v[idx].mean()), len(v), block, n)
    return m, (m / se if se else np.nan)


def tier_weights(p):
    """Production weighting: positive predictions into terciles at 1 : 1.5 : 2.25, normalised.
    Everything non-positive gets zero. Diversification is a property of the scheme, not a hope."""
    w = np.zeros_like(p)
    pos = np.where(p > 0)[0]
    if len(pos) == 0:
        return None
    order = pos[np.argsort(p[pos])]
    n = len(order)
    if n <= 3:
        for r, k in enumerate(order):
            w[k] = TIER_W[min(r, 2)]
    else:
        base, rem = n // 3, n % 3
        n1 = base + (1 if rem >= 1 else 0)
        n2 = base + (1 if rem >= 2 else 0)
        for r, k in enumerate(order):
            w[k] = TIER_W[0] if r < n1 else (TIER_W[1] if r < n1 + n2 else TIER_W[2])
    return w / w.sum()


def forward_return(h):
    """Per-industry return over the NEXT h sessions — what a position opened today actually earns
    if held for h days. h = 1 is `ret`."""
    fwd = D.forward_pnl(ds, h - 1, 'book') if h > 1 else np.zeros_like(book)
    tot = book + np.nan_to_num(fwd, nan=0.0) if h > 1 else book
    out = tot / prev
    if h > 1:
        out[-(h - 1):] = np.nan          # windows that never closed
    return out


def allocation_edge(pred, mask, fwd, h, lo=0, hi=None):
    """(weighted forward-h return − equal-weight forward-h return), in basis points per period.
    This IS the win condition's second half: focus the money and still diversify."""
    hi = hi if hi is not None else T
    out = []
    for t in range(lo, hi):
        m = mask[t] & np.isfinite(pred[t]) & np.isfinite(fwd[t])
        if m.sum() < 6:
            continue
        p, r = pred[t][m], fwd[t][m]
        w = tier_weights(p)
        if w is None:
            continue
        out.append((w @ r - r.mean()) * 1e4)
    return np.array(out)


def rank_ic(pred, mask, fwd, h, lo=0, hi=None):
    hi = hi if hi is not None else T
    out = []
    for t in range(lo, hi):
        m = mask[t] & np.isfinite(pred[t]) & np.isfinite(fwd[t])
        if m.sum() >= 6:
            out.append(M.spearman(pred[t][m], fwd[t][m]))
    return np.array(out)


def conditional_pred(book_arr, half_life, window, k, warmup=250, prior=50.0):
    """Walk-forward E[tomorrow's P&L | state], one column per industry, for any depth k.

    Alignment matches the logged candidates: pred[t] is the prediction FOR day t's outcome, so it
    may only use information through t-1. The state is read from days t-k..t-1 and the estimator
    sees day t only AFTER the prediction for day t has been made.

    Pooled across industries, shrunk toward the global mean. The between-industry variance of this
    effect measured SMALLER than sampling noise alone (tau^2 = 0), so per-industry tables would be
    fitting noise.
    """
    from depth_sweep import n_states, state_k
    Tn, Nn = book_arr.shape
    S = n_states(k)
    w_n, w_sum = np.zeros(S), np.zeros(S)
    g_n = g_sum = 0.0
    ring = []
    gamma = 0.5 ** (1.0 / half_life) if window <= 0 else 1.0
    out = np.full((Tn, Nn), np.nan)
    for t in range(k + 1, Tn):
        obs = []
        for i in range(Nn):
            st = state_k(book_arr[t - 1 - k:t - 1, i])
            if st is None:
                continue
            if t > warmup and (w_n[st] + g_n) > 0:
                gm = g_sum / g_n if g_n > 0 else 0.0
                out[t, i] = (w_sum[st] + prior * gm) / (w_n[st] + prior)
            y = book_arr[t - 1, i]
            if np.isfinite(y) and y != 0:
                obs.append((st, float(y)))
        if window <= 0:
            w_n *= gamma
            w_sum *= gamma
            g_n *= gamma
            g_sum *= gamma
        for st, y in obs:
            w_n[st] += 1.0
            w_sum[st] += y
            g_n += 1.0
            g_sum += y
        if window > 0:
            ring.append(obs)
            while len(ring) > window:
                for st, y in ring.pop(0):
                    w_n[st] -= 1.0
                    w_sum[st] -= y
                    g_n -= 1.0
                    g_sum -= y
    return out


# Build every candidate's (prediction, mask, score-or-None) up front.
PRED = {}
for _n, _k, _d in NET_CANDIDATES:
    PRED[_n] = (ds[f'{_k}_pred'].astype(float), ds[f'{_k}_score'].astype(float) > 0,
                ds[f'{_k}_score'].astype(float), _d)
_book_clean = np.where(np.abs(book) < 1e-9, np.nan, book)
for _n, _hl, _w, _k, _d in COND_CANDIDATES:
    _p = conditional_pred(_book_clean, _hl, _w, _k)
    PRED[_n] = (_p, np.isfinite(_p), None, _d)

CANDIDATES = [(n, None, d) for n, _, d in NET_CANDIDATES] + \
             [(n, None, d) for n, _, _, _, d in COND_CANDIDATES]

_cn = [n for n, _, _, _, _ in COND_CANDIDATES]
_flat = {n: PRED[n][0].ravel() for n in _cn}
_msk = np.ones_like(next(iter(_flat.values())), bool)
for v in _flat.values():
    _msk &= np.isfinite(v)
if _msk.sum() > 100:
    _C = np.corrcoef(np.vstack([_flat[n][_msk] for n in _cn]))
    _cors = [abs(_C[i, j]) for i in range(len(_cn)) for j in range(i + 1, len(_cn))]
    # Effective number of independent tests (Cheverud-Nyholt): correlated candidates are not
    # separate shots on goal. Computed rather than asserted -- the first version of this line
    # claimed "~1 effective test" by eye, which the median correlation of 0.87 does not support.
    _lam = np.linalg.eigvalsh(_C)
    _meff = 1 + (len(_cn) - 1) * (1 - _lam.var(ddof=1) / len(_cn))
    print(f'conditional family: {len(_cn)} variants, pairwise |corr| '
          f'{min(_cors):.3f}-{max(_cors):.3f} (median {np.median(_cors):.3f}); '
          f'effective independent tests ~{_meff:.1f}, not {len(_cn)}\n')

half = T // 2
third = 2 * T // 3
HORIZONS = [1, 2, 3, 5, 10]

print('PRIMARY — the win condition, swept over the holding period')
print('(all three were TRAINED at h = 1; this asks over what h acting on the signal pays)\n')
res = {}
for name, _k, _d in CANDIDATES:
    pr, m, sc, _d = PRED[name]
    if m.sum() < 200:
        print(f'{name:<10}   (not scored)')
        res[name] = dict(scored=False)
        continue
    if sc is None:
        # A closed-form estimator has no training runway, so "may not have converged" is not
        # available to it as an excuse. drift = 0 makes it eligible for a FAILED verdict.
        r = dict(scored=True, drift=0.0, score=None, pred=pr, mask=m, h={})
    else:
        early = sc[:third][m[:third]].mean()
        late = sc[third:][m[third:]].mean()
        r = dict(scored=True, drift=late - early, score=sc[m].mean(), pred=pr, mask=m, h={})
    print(f'{name}   ({_d})')
    print(f'  {"h":>3} {"rank IC":>9} {"t":>6} {"IC 2nd half":>12} {"t":>6} '
          f'{"edge bp/period":>15} {"t":>6}')
    for h in HORIZONS:
        fwd = forward_return(h)
        blk = max(2 * h, 20)
        ic, t_ic = boot_t(rank_ic(pr, m, fwd, h), blk)
        ic2, t_ic2 = boot_t(rank_ic(pr, m, fwd, h, lo=half), blk)
        ed, t_ed = boot_t(allocation_edge(pr, m, fwd, h), blk)
        r['h'][h] = dict(ic=ic, t_ic=t_ic, ic2=ic2, t_ic2=t_ic2, edge=ed, t_edge=t_ed)
        star = '*' if (t_ic > 3 and t_ed > 2 and t_ic2 > 0) else ' '
        print(f'  {h:>3} {ic:>+9.4f} {t_ic:>+6.1f} {ic2:>+12.4f} {t_ic2:>+6.1f} '
              f'{ed:>+15.2f} {t_ed:>+6.1f} {star}')
    res[name] = r
    print()

# supporting diagnostics
zero = np.full_like(book, np.nan)
for i in range(D.N_IND):
    a = book[:, i]
    for t in range(20, T):
        w_ = a[max(0, t - 20):t]
        d = max(abs(a[t] - w_.mean()), max(np.abs(w_[-10:]).mean() * 0.5, 1.0), 1e-6)
        zero[t, i] = d / (abs(a[t]) + d)
any_sc = ds['m_score'] > 0
bar0 = np.nanmean(zero[any_sc])
print(f'\nSUPPORTING (diagnostic only; constant-$0 benchmark = {bar0:.4f})')
print(f'{"candidate":<10} {"score":>8} {"vs $0":>7} {"per-ind corr":>13} {"t":>6} {"Δscore last⅓":>13}')
for name, _k, _d in CANDIDATES:
    r = res[name]
    if not r.get('scored') or r.get('score') is None:
        continue
    vals = [M.pearson(r['pred'][:, i][r['mask'][:, i]], book[:, i][r['mask'][:, i]])
            for i in range(D.N_IND) if r['mask'][:, i].sum() > 80]
    c = np.nanmean(vals)
    tc = c / (np.nanstd(vals) / np.sqrt(len(vals)))
    r['beats0'] = r['score'] > bar0
    print(f'{name:<10} {r["score"]:>8.4f} {"yes" if r["beats0"] else "no":>7} '
          f'{c:>+13.4f} {tc:>+6.1f} {r["drift"]:>+13.4f}')

def best_h(r):
    """The horizon at which a candidate clears both bars, if any."""
    for h, v in r.get('h', {}).items():
        if v['t_ic'] > 3 and v['t_edge'] > 2 and v['t_ic2'] > 0:
            return h, v
    return None, None


print('VERDICTS')
winners = {n: best_h(r)[0] for n, r in res.items()
           if r.get('scored') and best_h(r)[0] is not None}
for name, _k, desc in CANDIDATES:
    r = res[name]
    if not r.get('scored'):
        print(f'  {name:<10} INCONCLUSIVE   {desc}\n  {"":<10} never scored')
        continue
    if name in winners:
        h = winners[name]
        v = r['h'][h]
        print(f'  {name:<10} WORKING        {desc}')
        print(f'  {"":<10} at h = {h}: orders correctly (t {v["t_ic"]:+.1f}) and the weighting '
              f'beats equal weight by {v["edge"]:+.2f} bp/period (t {v["t_edge"]:+.1f})')
        continue
    bh = max(r['h'].items(), key=lambda kv: kv[1]['t_ic'])
    miss = [f'best horizon h = {bh[0]}: rank IC t {bh[1]["t_ic"]:+.1f} (need > 3), '
            f'allocation edge t {bh[1]["t_edge"]:+.1f} (need > 2)']
    converged = abs(r['drift']) < 0.01
    if winners and converged:
        verdict, why = 'FAILED', '; '.join(miss) + '; score has flattened, so it had its run'
    elif not converged:
        verdict, why = 'INCONCLUSIVE', (f'score still moving ({r["drift"]:+.4f} over the last '
                                        'third) — may not have converged. DO NOT ABANDON')
    else:
        verdict, why = 'INCONCLUSIVE', '; '.join(miss) + '; no candidate passed, so this ranks nothing'
    print(f'  {name:<10} {verdict:<14} {desc}\n  {"":<10} {why}')

if not winners:
    print('\n  NO CANDIDATE REACHED WORKING AT ANY HORIZON — a statement about the TARGET, not a')
    print(f'  ranking of the {len(CANDIDATES)}. Nothing is abandoned on differences between nulls, and')
    print('  the horizon question stays open rather than being settled by a null.')
elif len(winners) > 1:
    print(f'\n  MORE THAN ONE PASSED: {winners}. Prefer the one with the larger allocation edge at')
    print('  its own best horizon; the others are still not FAILED, only second.')
print('\n  Read the planted control at each input width before treating any null as failure.')
