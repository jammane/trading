#!/usr/bin/env python3
"""PRE-REGISTERED out-of-sample confirmation of the VOL allocation signal.

Written and committed BEFORE the confirming run finished. Nothing below may be tuned after
seeing the result -- the whole value of this file is that its parameters and its bar were fixed
in advance, because VOL was found after roughly twenty other tests on the discovery sample and
is therefore exactly the kind of result that needs an untouched sample to survive.

THE PREDICTOR, frozen:
    VOL_i(t) = standard deviation of industry i's daily book return over the trailing 60 days,
               ending at t-1. Weights are VOL normalised to sum to 1, long only.
    Sample:    day >= 400 (the first 400 days are learning-phase and were excluded on the
               discovery sample too).

THE BAR, frozen, identical to race.py's pre-registered win condition:
    ORDER    cross-sectional rank IC vs next-day return, block-bootstrap t > 3
    WEIGHT   allocation edge over equal weight, block-bootstrap t > 2
    CONTROL  permutation test (same weights, industries shuffled) p < 0.05
    All three must hold. Any two is not a pass.

ALSO REPORTED, not part of the bar: beta against equal weight and alpha after removing it.
On the discovery sample VOL gave rank IC +0.0415 (t 4.7), edge +1.72 bp/day (t 3.9),
permutation p 0.003, beta 1.04, annualised alpha +3.8% (t 3.5).

WHAT THIS DOES AND DOES NOT TEST:
    DOES     a different StockNN evolution -- new mutation noise, new portfolios, a genuinely
             new realisation of the P&L series the predictor is scored against.
    DOES NOT a different market. The confirming run walks the same calendar days and the same
             prices, so the market regime is shared. A pass here means the signal survives a
             new draw of the strategy, NOT that it survives a market that falls.
"""
import argparse

import numpy as np

import mt1_analysis as M
import read_mt1_dataset as D

W = 60
BURN_IN = 400
BAR_ORDER_T = 3.0
BAR_EDGE_T = 2.0
BAR_PERM_P = 0.05
N_PERM = 500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    ds = D.read(a.dataset) if hasattr(D, 'read') else D.parse(a.dataset)
    day = np.asarray(ds['day'])
    bp = ds['book_prev'].astype(float)
    r = np.where(bp > 1.0, (ds['book_now'].astype(float) - bp) / np.maximum(bp, 1.0), 0.0)
    T, N = r.shape
    rows = list(range(max(W, int(np.argmax(day >= BURN_IN))), T - 1))
    if len(rows) < 200:
        raise SystemExit(f'only {len(rows)} scorable days -- run did not reach day {BURN_IN}+')

    ics, ws, nx = [], [], []
    for t in rows:
        v = r[max(0, t - W):t].std(axis=0)
        if not np.all(np.isfinite(v)) or v.sum() <= 0:
            continue
        ics.append(M.spearman(v, r[t + 1]))
        ws.append(v / v.sum())
        nx.append(r[t + 1])
    ics = np.array([x for x in ics if np.isfinite(x)])
    ws, nx = np.array(ws), np.array(nx)
    flat = nx.mean(axis=1)
    real = (ws * nx).sum(axis=1)
    d = real - flat

    se_ic, _ = M.block_bootstrap(lambda i: float(ics[i].mean()), len(ics), 20, 500)
    se_ed, _ = M.block_bootstrap(lambda i: float(d[i].mean()), len(d), 20, 500)
    t_ic = ics.mean() / se_ic if se_ic else np.nan
    t_ed = d.mean() / se_ed if se_ed else np.nan

    rng = np.random.default_rng(a.seed)
    nulls = np.array([(((ws[:, rng.permutation(N)] * nx).sum(axis=1)) - flat).mean()
                      for _ in range(N_PERM)])
    pv = (np.sum(np.abs(nulls - nulls.mean()) >= abs(d.mean() - nulls.mean())) + 1) / (N_PERM + 1)

    beta = np.polyfit(flat, real, 1)[0]
    resid = real - beta * flat
    se_a, _ = M.block_bootstrap(lambda i: float(resid[i].mean()), len(resid), 20, 500)

    print(f'{a.dataset}: days {day[0]}-{day[-1]}, {len(d)} scored days from day {BURN_IN}\n')
    print(f'  {"criterion":<28} {"value":>12} {"t / p":>10} {"bar":>10}  result')
    o = 'PASS' if t_ic > BAR_ORDER_T else 'FAIL'
    e = 'PASS' if t_ed > BAR_EDGE_T else 'FAIL'
    c = 'PASS' if pv < BAR_PERM_P else 'FAIL'
    print(f'  {"ORDER  rank IC":<28} {ics.mean():>+12.4f} {t_ic:>10.1f} {"t > 3":>10}  {o}')
    print(f'  {"WEIGHT edge (bp/day)":<28} {1e4*d.mean():>+12.2f} {t_ed:>10.1f} {"t > 2":>10}  {e}')
    print(f'  {"CONTROL permutation":<28} {1e4*nulls.mean():>+12.2f} {pv:>10.3f} {"p < .05":>10}  {c}')
    print('\n  diagnostics (not part of the bar):')
    print(f'    cumulative  VOL {100*(np.prod(1+real)-1):+.1f}%   flat {100*(np.prod(1+flat)-1):+.1f}%')
    print(f'    annualised  ret {100*252*real.mean():+.1f}%  vol {100*np.sqrt(252)*real.std():.1f}%  '
          f'Sharpe {252*real.mean()/(np.sqrt(252)*real.std()):.2f}')
    print(f'    beta {beta:.2f}   annualised alpha {100*252*resid.mean():+.1f}%  '
          f'(t {resid.mean()/se_a if se_a else np.nan:.1f})')
    verdict = 'CONFIRMED' if (o == e == c == 'PASS') else 'NOT CONFIRMED'
    print(f'\n  VERDICT: {verdict}')
    print('  (discovery sample gave rank IC +0.0415 t 4.7, edge +1.72 t 3.9, perm p 0.003)')
    print('  A pass means the signal survives a new StockNN evolution. It does NOT mean it')
    print('  survives a different market -- the confirming run shares the same calendar days.')


if __name__ == '__main__':
    main()
