"""Tier allocation overlay across look-back intervals. d0 = today; d = look-back.

First 400 days discarded. Two rebalance conventions, because "d-interval determines the
allocation" admits both:
   DAILY   : look back d days, re-weight every day
   EVERY-d : look back d days, hold that allocation for the next d days

Overlay only -- StockNN's training is untouched; this redistributes capital across the realized
per-industry returns. Returns are treated as scale-invariant, which whole-share rounding and the
MAX_SINGLE_STOCK_PCT cap make approximately but not exactly true.
"""
import json, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0,'.')
import read_mt1_dataset as D
import mt1_analysis as M
exec(open('/root/abc3/tiered.py').read().split("print(f'{T} days")[0])

D0 = 400
start_i = int(np.argmax(day >= D0))
NPERM = 300
rng = np.random.default_rng(0)

def series(d, every_d):
    ws, nx = [], []
    hold = None; age = 0
    for t in range(max(d, start_i), T-1):
        if day[t] < D0: continue
        if every_d:
            if hold is None or age >= d:
                w = weights(t, d); hold = np.full(N,1.0/N) if w is None else w; age = 0
            w = hold; age += 1
        else:
            w = weights(t, d); w = np.full(N,1.0/N) if w is None else w
        ws.append(w); nx.append(r[t+1])
    return np.array(ws), np.array(nx)

for every_d in (False, True):
    print(f'=== {"REBALANCE EVERY d DAYS" if every_d else "DAILY REBALANCE"}, look-back d, day >= {D0} ===')
    print(f'  {"d":>4} {"tiered %":>10} {"flat %":>9} {"vs flat":>9} {"bp/day":>8} {"boot t":>7} '
          f'{"perm z":>7} {"perm p":>7} {"maxwt%":>7} {"t2 size":>8}')
    for d in (5,10,20,30,60,90,120):
        ws, nx = series(d, every_d)
        flat = nx.mean(axis=1); real = (ws*nx).sum(axis=1)
        diff = real - flat
        se,_ = M.block_bootstrap(lambda i: float(diff[i].mean()), len(diff), 20, 400)
        nulls = []
        for _ in range(NPERM):
            p = rng.permutation(N)
            nulls.append(((ws[:,p]*nx).sum(axis=1)-flat).mean())
        nulls = np.array(nulls)
        z = (diff.mean()-nulls.mean())/nulls.std(ddof=1)
        pv = (np.sum(np.abs(nulls-nulls.mean())>=abs(diff.mean()-nulls.mean()))+1)/(NPERM+1)
        ct, cf = 100*(np.prod(1+real)-1), 100*(np.prod(1+flat)-1)
        star = '*' if pv < 0.05 else ' '
        print(f'  {d:>4} {ct:>+10.1f} {cf:>+9.1f} {ct-cf:>+9.1f} {1e4*diff.mean():>+8.2f} '
              f'{diff.mean()/se if se else np.nan:>7.1f} {z:>+7.1f} {pv:>7.3f}{star} '
              f'{100*ws.max(axis=1).mean():>7.1f} {(ws<1e-9).sum(axis=1).mean():>8.2f}')
    print()
reb = 100*np.mean([np.prod(1+bh[start_i:T-1,k])-1 for k in range(N)])
print(f'  benchmark, equal-weight rebalanced daily: {reb:+.1f}%')
