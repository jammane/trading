"""Quarterly scorecard for OFFLINE trees across the expanded interval set.

Same metric as the live table: hit rate minus always-up, per 63-day quarter. Offline trees are
conditioned on slot 0's series (that is all mt1_dataset carries); the live ones are conditioned on
all 20 elites. Side by side, that is the comparison.
"""
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import read_mt1_dataset as D
from depth_sweep import n_states, state_k

ds = D.read('mt1_dataset.bin') if hasattr(D, 'read') else D.parse('mt1_dataset.bin')
day = np.asarray(ds['day'])
raw = ds['book_now'].astype(float) - ds['book_prev'].astype(float)
keep = day >= 400
cols = [raw[keep, i][np.abs(raw[keep, i]) > 1e-9] for i in range(raw.shape[1])]
m = min(len(c) for c in cols)
P = np.column_stack([c[-m:] for c in cols])
T, N = P.shape
Q, WARM = 63, 150

def scorecard(k, window, half_life, prior=2.0):
    S = n_states(k)
    w_n, w_same = np.zeros(S), np.zeros(S)
    ring = []
    g = 0.5 ** (1.0 / half_life) if window <= 0 else 1.0
    per = {}
    for t in range(k, T):
        obs = []
        for i in range(N):
            st = state_k(P[t-k:t, i])
            y = P[t, i]
            if st is None or y == 0 or not np.isfinite(y):
                continue
            same = float(np.sign(y) == np.sign(P[t-1, i]))
            if t > WARM:
                p = (w_same[st] + prior*0.5) / (w_n[st] + prior)
                call = np.sign(P[t-1, i]) if p > 0.5 else -np.sign(P[t-1, i])
                q = (t - WARM - 1)//Q
                a = per.setdefault(q, [0, 0, 0])
                a[0] += 1; a[1] += int(call == np.sign(y)); a[2] += int(y > 0)
            obs.append((st, same))
        if window <= 0:
            w_n *= g; w_same *= g
        for st, same in obs:
            w_n[st] += 1.0; w_same[st] += same
        if window > 0:
            ring.append(obs)
            while len(ring) > window:
                for st, same in ring.pop(0):
                    w_n[st] -= 1.0; w_same[st] -= same
    return per

CFG = [(w, f'{w}d win', w, 1e9) for w in (5, 10, 20, 30, 60, 90, 120, 252, 504)] + \
      [(0, 'forever', 0, 1e9)]
qmax = (T - WARM - 2)//Q
DASH = '-'
for k in (1, 2):
    print(f'\n=== OFFLINE, k={k}, day >= 400, {qmax+1} quarters of {Q} '
          f'(cell = hit rate minus always-up, pp) ===')
    hdr = f'  {"memory":<10}' + ''.join(f'{"Q"+str(q+1):>7}' for q in range(qmax+1)) + f'{"ALL":>8}'
    print(hdr); print('  ' + '-'*(len(hdr)-2))
    for _, lab, win, hl in CFG:
        per = scorecard(k, win, hl)
        tp = th = tb = 0; cells = []
        for q in range(qmax+1):
            p, h, b = per.get(q, [0, 0, 0])
            tp += p; th += h; tb += b
            cells.append(f'{100*(h/p-b/p):>+7.2f}' if p > 30 else f'{DASH:>7}')
        print(f'  {lab:<10}' + ''.join(cells) + (f'{100*(th/tp-tb/tp):>+8.2f}' if tp else ''))
    per = scorecard(k, 0, 1e9)
    print('  ' + '-'*(len(hdr)-2))
    print(f'  {"always-up%":<10}' + ''.join(
        f'{100*per[q][2]/per[q][0]:>7.1f}' if per.get(q, [0])[0] > 30 else f'{DASH:>7}'
        for q in range(qmax+1)))
