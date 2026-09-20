"""Rolling tiered allocation vs flat.

RULE (as specified):
  tier 0 = industries whose trailing-W return BEATS their own equal-weight buy&hold
  tier 1 = the rest, provided their trailing-W return is net POSITIVE
  tier 2 = everything else -> no allocation
  2/3 of capital to tier 0, 1/3 to tier 1, weights inside a tier proportional to that
  industry's share of the tier's total (excess for tier 0, gain for tier 1),
  subject to: no tier-1 industry may be weighted above any tier-0 industry.

CAUSALITY: tiers are formed from days <= t and applied to day t+1's return. The trailing window
never contains the day being traded.

Returns are book P&L / book_prev -- scale free, and reset days contribute exactly 0, so the
hard-floor injections cannot flatter any strategy.
"""
import json, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0,'.')
import read_mt1_dataset as D
import mt1_analysis as M

IND = ['hardware','software','financial','discret','services','health',
       'industrl','staples','energy','utilitie','land','materials']
U = json.load(open('/root/trading-ht/universe.json')); inds = list(U)
ds = D.read('mt1_dataset.bin') if hasattr(D,'read') else D.parse('mt1_dataset.bin')
day = np.asarray(ds['day'])
bp = ds['book_prev'].astype(float); bn = ds['book_now'].astype(float)
r = np.where(bp > 1.0, (bn-bp)/np.maximum(bp,1.0), 0.0)        # industry daily return
T, N = r.shape
d0, d1 = int(day[0]), int(day[-1])

def closes(s):
    return np.array([b['close'] for b in json.load(
        open(f'/root/trading/stock_data/{s}.json'))['days']], float)
bh = np.zeros((T, N))
for k, ind in enumerate(inds):
    syms = U[ind] if isinstance(U[ind], list) else U[ind].get('symbols', [])
    acc, cnt = np.zeros(T), 0
    for s in syms:
        try: c = closes(s)
        except Exception: continue
        off = len(c)-1255
        idx = off + day
        if idx.min() < 1 or idx.max() >= len(c): continue
        acc += c[idx]/c[idx-1] - 1.0; cnt += 1
    if cnt: bh[:, k] = acc/cnt

def weights(t, W):
    """Tier weights from the trailing W days ending at t (inclusive). None -> use flat."""
    lo = max(0, t-W+1)
    ind_ret = np.prod(1.0+r[lo:t+1], axis=0) - 1.0
    bh_ret  = np.prod(1.0+bh[lo:t+1], axis=0) - 1.0
    exc = ind_ret - bh_ret
    t0 = np.where(exc > 0)[0]
    t1 = np.array([i for i in range(N) if i not in set(t0.tolist()) and ind_ret[i] > 0], int)
    w = np.zeros(N)
    if len(t0) == 0 and len(t1) == 0:
        return None
    if len(t0) == 0:
        w[t1] = ind_ret[t1]/ind_ret[t1].sum(); return w
    if len(t1) == 0:
        w[t0] = exc[t0]/exc[t0].sum(); return w
    w[t0] = (2.0/3.0) * exc[t0]/exc[t0].sum()
    w[t1] = (1.0/3.0) * ind_ret[t1]/ind_ret[t1].sum()
    # enforce: no tier-1 weight above any tier-0 weight; move the excess into tier 0
    for _ in range(50):
        cap = w[t0].min()
        over = w[t1] - cap
        if (over <= 1e-12).all(): break
        move = over[over > 0].sum()
        w[t1] = np.minimum(w[t1], cap)
        w[t0] += (2.0/3.0 and 1.0) * move * (exc[t0]/exc[t0].sum())
    return w/w.sum()

print(f'{T} days, {N} industries. Weights from days <= t, applied to day t+1.\n')
for D0 in (400, 500):
    print(f'=== day >= {D0} ===')
    print(f'  {"window":>7} {"tiered %":>10} {"flat %":>9} {"diff":>9} {"daily bp":>9} '
          f'{"t":>6} {"tier0 size":>11} {"flat days":>10}')
    for W in (30, 60, 90):
        tw, fl, dif, empt = [], [], [], 0
        t0sz = []
        for t in range(max(W, int(np.argmax(day >= D0))), T-1):
            if day[t] < D0: continue
            w = weights(t, W)
            if w is None:
                w = np.full(N, 1.0/N); empt += 1
            else:
                t0sz.append(int((w > 0).sum()))
            nxt = r[t+1]
            tw.append(float(w @ nxt)); fl.append(float(nxt.mean()))
        tw, fl = np.array(tw), np.array(fl)
        d = tw - fl
        se, _ = M.block_bootstrap(lambda i: float(d[i].mean()), len(d), 20, 500)
        cum_t = 100*(np.prod(1+tw)-1); cum_f = 100*(np.prod(1+fl)-1)
        print(f'  {W:>7} {cum_t:>+10.1f} {cum_f:>+9.1f} {cum_t-cum_f:>+9.1f} '
              f'{1e4*d.mean():>+9.2f} {d.mean()/se if se else float("nan"):>6.1f} '
              f'{np.mean(t0sz):>11.1f} {empt:>10}')
    print()
