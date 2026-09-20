"""Quarterly efficiency: is an industry's edge over its own benchmark persistent, and is there
a trend? Efficiency = industry return - its equal-weight buy&hold return, over a 63-day quarter.

Three questions:
  1. PERSISTENCE  does quarter q's efficiency predict q+1's?  (the allocation question)
  2. TREND        is efficiency improving as StockNN learns?  (the learning question)
  3. TRADEABLE    does allocating quarterly on it beat flat, net of turnover?
"""
import json, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0,'.')
import read_mt1_dataset as D
import mt1_analysis as M
exec(open('/root/abc3/tiered.py').read().split("print(f'{T} days")[0])

IND = ['hardware','software','financial','discret','services','health',
       'industrl','staples','energy','utilitie','land','materials']
D0, Q = 400, 63
s0 = int(np.argmax(day >= D0))
bounds = list(range(s0, T-1, Q))
if len(bounds) and (T-1-bounds[-1]) < Q//2: bounds = bounds[:-1]
nq = len(bounds)-1 if bounds[-1]+Q > T-1 else len(bounds)
qs = [(bounds[k], min(bounds[k]+Q, T-1)) for k in range(nq)]
eff = np.zeros((len(qs), N)); ind_r = np.zeros_like(eff); bh_r = np.zeros_like(eff)
for k,(a,b) in enumerate(qs):
    ind_r[k] = np.prod(1+r[a:b], axis=0)-1
    bh_r[k]  = np.prod(1+bh[a:b], axis=0)-1
    # LOG difference: these sum across quarters to the log of the cumulative ratio, so the
    # quarterly figures reconcile with the whole-period comparison. Arithmetic differences do
    # not -- hardware showed -0.8%/quarter mean while beating its benchmark by ~80pp cumulative,
    # which is compounding, not a contradiction.
    eff[k]   = np.log1p(np.maximum(ind_r[k],-0.99)) - np.log1p(np.maximum(bh_r[k],-0.99))

print(f'day >= {D0}, {Q}-day quarters, {len(qs)} quarters\n')
print('=== efficiency by quarter: log(1+industry) - log(1+buy&hold), in % ===')
print(f'  {"industry":<10} ' + ' '.join(f'{"Q"+str(k+1):>7}' for k in range(len(qs))) + f' {"mean":>8}')
for i in range(N):
    print(f'  {IND[i]:<10} ' + ' '.join(f'{100*eff[k,i]:>+7.1f}' for k in range(len(qs)))
          + f' {100*eff[:,i].mean():>+8.1f}')
print(f'  {"MEAN":<10} ' + ' '.join(f'{100*eff[k].mean():>+7.1f}' for k in range(len(qs)))
      + f' {100*eff.mean():>+8.1f}')

print('\n=== 1. PERSISTENCE: does quarter q predict q+1? ===')
a = eff[:-1].ravel(); b = eff[1:].ravel()
print(f'  pooled corr(eff[q], eff[q+1])      {np.corrcoef(a,b)[0,1]:+.3f}   n={len(a)}')
rk = [M.spearman(eff[k], eff[k+1]) for k in range(len(qs)-1)]
rk = np.array([x for x in rk if np.isfinite(x)])
se = rk.std(ddof=1)/np.sqrt(len(rk))
print(f'  mean cross-sectional rank IC       {rk.mean():+.3f}  (t {rk.mean()/se:+.1f}, '
      f'{len(rk)} quarter pairs)')
print(f'  quarters with positive rank IC     {int((rk>0).sum())}/{len(rk)}')
h = len(qs)//2
sh = np.corrcoef(eff[:h].sum(axis=0), eff[h:].sum(axis=0))[0,1]
print(f'  SPLIT-HALF corr(first-half total eff, second-half total eff): {sh:+.3f}')
print(f'  cumulative log-eff per industry (sums to the whole-period gap):')
tot = eff.sum(axis=0)
for i in np.argsort(-tot):
    print(f'     {IND[i]:<10} {100*tot[i]:>+8.1f}  ->  x{np.exp(tot[i]):.2f} vs its benchmark')

print('\n=== 2. TREND: is efficiency improving over time? ===')
x = np.arange(len(qs)); y = eff.mean(axis=1)
sl = np.polyfit(x, y, 1)[0]
r_ = np.corrcoef(x, y)[0,1]
print(f'  mean efficiency per quarter: first half {100*y[:h].mean():+.2f}%  '
      f'second half {100*y[h:].mean():+.2f}%')
print(f'  slope {100*sl:+.3f}%/quarter   corr(quarter, efficiency) {r_:+.3f}')

print('\n=== 3. TRADEABLE: allocate quarterly on the prior quarter, hold the quarter ===')
print(f'  {"rule":<22} {"cum %":>9} {"vs flat":>9} {"turnover/qtr":>13} {"net @10bp":>10} {"perm p":>8}')
rng = np.random.default_rng(0)
flat_c = 1.0
for k in range(1, len(qs)):
    flat_c *= (1+ind_r[k].mean())
def tier_w(e, ir):
    t0 = np.where(e > 0)[0]; t1 = np.array([i for i in range(N) if i not in set(t0.tolist()) and ir[i] > 0], int)
    w = np.zeros(N)
    if len(t0)==0 and len(t1)==0: return np.full(N,1.0/N)
    if len(t0)==0: w[t1]=ir[t1]/ir[t1].sum(); return w
    if len(t1)==0: w[t0]=e[t0]/e[t0].sum(); return w
    w[t0]=(2/3)*e[t0]/e[t0].sum(); w[t1]=(1/3)*ir[t1]/ir[t1].sum()
    for _ in range(50):
        cap=w[t0].min(); over=w[t1]-cap
        if (over<=1e-12).all(): break
        mv=over[over>0].sum(); w[t1]=np.minimum(w[t1],cap); w[t0]+=mv*(e[t0]/e[t0].sum())
    return w/w.sum()
for lab, fn in (('tier on prior-qtr eff', lambda k: tier_w(eff[k-1], ind_r[k-1])),
                ('top-6 by prior eff', lambda k: (lambda m: m/m.sum())(
                    np.isin(np.arange(N), np.argsort(-eff[k-1])[:6]).astype(float)))):
    ws = np.array([fn(k) for k in range(1, len(qs))])
    rets = np.array([ws[k-1] @ ind_r[k] for k in range(1, len(qs))])
    to = 0.5*np.abs(np.diff(np.vstack([np.full((1,N),1.0/N), ws]), axis=0)).sum(axis=1).mean()
    cum = np.prod(1+rets); net = np.prod(1+rets-to*2*10e-4)
    nulls=[]
    for _ in range(400):
        p = rng.permutation(N)
        nulls.append(np.prod(1+np.array([ws[k-1][p] @ ind_r[k] for k in range(1,len(qs))])))
    nulls=np.array(nulls)
    pv=(np.sum(np.abs(nulls-nulls.mean())>=abs(cum-nulls.mean()))+1)/401
    print(f'  {lab:<22} {100*(cum-1):>+9.1f} {100*(cum-flat_c):>+9.1f} {100*to:>12.1f}% '
          f'{100*(net-1):>+9.1f} {pv:>8.3f}')
print(f'  {"flat":<22} {100*(flat_c-1):>+9.1f}')
