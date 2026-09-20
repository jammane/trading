"""Where StockNN's money comes from, and where it leaks.

Run after a training pass, against mt1_dataset.bin + holdings_log.csv + stock_data.

THE CORRECTED PICTURE (v0.8.1.7 run, day >= 400, 855 days, 12 industries):

    StockNN, flat allocation          +54.2%
    benchmark, eq-wt rebalanced daily +80.6%
    gap                               -26.4pp
      of which cash drag              -14.2pp   17.6% of the book sits in cash
      of which selection + trading    -12.2pp

FOUR CORRECTIONS THIS FILE EXISTS TO PREVENT, all of which were reported wrong first:

  1. Reset injections. step_industry recapitalises a book below $22,500 to $25,000. Those days log
     $0 P&L, so they do not inflate the P&L sum -- but they do raise the capital base. $126,955
     was injected over the run ($106,032 of it before day 400). Ignoring it turned a 2.6-7.9pp
     UNDERperformance into a reported +8.9pp outperformance.

  2. Arithmetic vs log differences. "Industry return minus benchmark return" does not sum across
     periods. Hardware read +108pp on that measure and +2.8% on the log ratio that actually
     decomposes; the difference is hardware's volatility, not its skill.

  3. Two different benchmarks. Point-to-point buy & hold is +61.7%; equal-weight rebalanced daily
     is +80.6%. Only the second is like-for-like against a daily-rebalanced strategy.

  4. Cash drag was never measured at all, and it is the biggest single identifiable term.

THE ORIGINAL TERM this file computes: how much of the book is actually invested?

Holding cash in a rising market underperforms buy-and-hold without any trading skill being
involved. That term has never been measured here, and it is the obvious candidate for the gap
between StockNN and its benchmark.

  book return  ~  invested_fraction x market return  +  selection  +  trade delta  +  cash drag
  cash drag    =  -(1 - invested_fraction) x benchmark return
"""
import json
import sys
import warnings
from collections import defaultdict

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

with open('/root/trading-ht/universe.json') as _uf:
    U = json.load(_uf)
inds = list(U)
SHORT = ['hardware', 'software', 'financial', 'discret', 'services', 'health',
         'industrl', 'staples', 'energy', 'utilitie', 'land', 'materials']

px = {}
for ind in inds:
    for s in U[ind]:
        try:
            with open(f'/root/trading/stock_data/{s}.json') as fh:
                px[s] = np.array([b['close'] for b in json.load(fh)['days']], float)
        except Exception:
            px[s] = None

rows = defaultdict(dict)
with open('/root/abc3/holdings_log.csv') as f:
    hdr = f.readline().strip().split(',')
    for ln in f:
        p = ln.strip().split(',')
        d = int(p[1]); ind = p[2]
        rows[d][ind] = (float(p[3]), [float(x) for x in p[4:16]])

days = sorted(rows)
print(f'holdings log: {len(days)} days, {days[0]}..{days[-1]}\n')

inv_frac = defaultdict(list)
for d in days:
    for k, ind in enumerate(inds):
        if ind not in rows[d]:
            continue
        cash, q = rows[d][ind]
        held = 0.0
        for j, s in enumerate(U[ind]):
            c = px.get(s)
            if c is None:
                continue
            i = len(c) - 1255 + d
            if 0 <= i < len(c):
                held += q[j]*c[i]
        tot = held + cash
        if tot > 1.0:
            inv_frac[k].append((d, held/tot))

print('=== invested fraction of the book (1.00 = fully deployed, 0 = all cash) ===')
print(f'  {"industry":<11} {"all days":>9} {"day<400":>9} {"day>=400":>9} {"last 250d":>10}')
allf = []
for k in range(12):
    a = np.array(inv_frac[k])
    if not len(a):
        continue
    d_, f_ = a[:, 0], a[:, 1]
    allf.append(f_[d_ >= 400].mean())
    print(f'  {SHORT[k]:<11} {f_.mean():>9.3f} {f_[d_<400].mean():>9.3f} '
          f'{f_[d_>=400].mean():>9.3f} {f_[d_>=days[-1]-250].mean():>10.3f}')
print(f'  {"MEAN":<11} {np.mean([np.array(inv_frac[k])[:,1].mean() for k in range(12)]):>9.3f} '
      f'{"":>9} {np.mean(allf):>9.3f}')

f_bar = float(np.mean(allf))
print('\n=== what cash drag alone explains, day >= 400 ===')
print(f'  mean invested fraction        {f_bar:.3f}  -> {100*(1-f_bar):.1f}% sits in cash')
BH = 80.6
print(f'  benchmark (eq-wt, rebalanced) {BH:+.1f}%')
print(f'  a fully passive book holding only {f_bar:.3f} of that benchmark would return '
      f'{f_bar*BH:+.1f}%')
print('  StockNN actually returned     +54.2%  (flat allocation across the 12)')
print(f'  gap explained by cash drag    {BH - f_bar*BH:>+.1f}pp of the {BH-54.2:+.1f}pp total')
print(f'  residual (selection + trades) {54.2 - f_bar*BH:>+.1f}pp')
