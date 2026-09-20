"""Push the look-back below 5 and measure turnover + cost sensitivity.

If the edge keeps RISING as d -> 1, that is the signature of the fill-model artifact (the
simulation executes at intraday extremes and marks at the close, which manufactures short-horizon
structure) rather than a tradeable signal. Turnover tells us what transaction costs would do to it
-- the overlay so far has charged nothing for reallocating capital.
"""
import json, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0,'.')
import read_mt1_dataset as D
import mt1_analysis as M
exec(open('/root/abc3/tiered.py').read().split("print(f'{T} days")[0])

D0 = 400; start_i = int(np.argmax(day >= D0)); NPERM = 300
rng = np.random.default_rng(0)
print(f'  {"d":>4} {"tiered %":>10} {"vs flat":>9} {"bp/day":>8} {"boot t":>7} {"perm p":>7} '
      f'{"turnover":>9} {"bp @5bp":>8} {"bp @10bp":>9} {"net vs flat":>12}')
for d in (1,2,3,5,7,10):
    ws, nx = [], []
    for t in range(max(d, start_i), T-1):
        if day[t] < D0: continue
        w = weights(t, d); ws.append(np.full(N,1.0/N) if w is None else w); nx.append(r[t+1])
    ws, nx = np.array(ws), np.array(nx)
    flat = nx.mean(axis=1); real = (ws*nx).sum(axis=1); diff = real-flat
    # one-way turnover per day: half the L1 change in weights
    to = 0.5*np.abs(np.diff(ws, axis=0)).sum(axis=1)
    to_m = to.mean()
    se,_ = M.block_bootstrap(lambda i: float(diff[i].mean()), len(diff), 20, 400)
    nulls = []
    for _ in range(NPERM):
        p = rng.permutation(N)
        nulls.append((((ws[:,p]*nx).sum(axis=1))-flat).mean())
    nulls = np.array(nulls)
    pv = (np.sum(np.abs(nulls-nulls.mean())>=abs(diff.mean()-nulls.mean()))+1)/(NPERM+1)
    ct = 100*(np.prod(1+real)-1); cf = 100*(np.prod(1+flat)-1)
    c5, c10 = 1e4*diff.mean()-to_m*2*5, 1e4*diff.mean()-to_m*2*10
    # net cumulative at 10bp round trip
    netr = real - np.concatenate([[0], to])*2*10e-4
    star = '*' if pv < 0.05 else ' '
    print(f'  {d:>4} {ct:>+10.1f} {ct-cf:>+9.1f} {1e4*diff.mean():>+8.2f} '
          f'{diff.mean()/se if se else np.nan:>7.1f} {pv:>7.3f}{star} {100*to_m:>8.1f}% '
          f'{c5:>+8.2f} {c10:>+9.2f} {100*(np.prod(1+netr)-1)-cf:>+11.1f}')
print(f'\n  flat {100*(np.prod(1+flat)-1):+.1f}%   benchmark (eq-wt rebalanced daily) '
      f'{100*np.mean([np.prod(1+bh[start_i:T-1,k])-1 for k in range(N)]):+.1f}%')
print('  "bp @Xbp" = edge net of X basis points per side on the turnover shown.')
