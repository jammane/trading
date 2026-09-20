"""Does feeding the trees the first 400 days of learning-phase data damage them?

Two variants, SCORED ON IDENTICAL DAYS (post-400 quarters only) so the comparison is clean:
  from d17   accumulate counts from the start of the run, like the LIVE models do
  from d400  accumulate only after the learning phase, like the offline table does
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
cols_all = [raw[:, i][np.abs(raw[:, i]) > 1e-9] for i in range(raw.shape[1])]
dcols = [day[np.abs(raw[:, i]) > 1e-9] for i in range(raw.shape[1])]
m = min(len(c) for c in cols_all)
P = np.column_stack([c[-m:] for c in cols_all])
DY = dcols[0][-m:]
T, N = P.shape
Q = 63
score_from = int(np.argmax(DY >= 400)) + 150      # same warmup offset as before

def scorecard(k, learn_from, window=0, half_life=1e9, prior=2.0):
    S = n_states(k)
    w_n, w_same = np.zeros(S), np.zeros(S)
    ring = []
    g = 0.5 ** (1.0/half_life) if window <= 0 else 1.0
    per = {}
    for t in range(k, T):
        obs = []
        for i in range(N):
            st = state_k(P[t-k:t, i]); y = P[t, i]
            if st is None or y == 0 or not np.isfinite(y):
                continue
            same = float(np.sign(y) == np.sign(P[t-1, i]))
            if t >= score_from:
                p = (w_same[st] + prior*0.5)/(w_n[st] + prior)
                call = np.sign(P[t-1, i]) if p > 0.5 else -np.sign(P[t-1, i])
                q = (t - score_from)//Q
                a = per.setdefault(q, [0, 0, 0])
                a[0] += 1; a[1] += int(call == np.sign(y)); a[2] += int(y > 0)
            if t >= learn_from:
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

d400 = int(np.argmax(DY >= 400))
qmax = (T - score_from - 1)//Q
DASH = '-'
print(f'scored on identical days: post-400 quarters, {qmax+1} of {Q}\n')
hdr = f'  {"variant":<22}' + ''.join(f'{"Q"+str(q+1):>7}' for q in range(qmax+1)) + f'{"ALL":>8}'
for k in (1, 2):
    print(f'=== k={k} ===')
    print(hdr); print('  ' + '-'*(len(hdr)-2))
    for lab, lf in ((f'forever from d17', 0), (f'forever from d400', d400)):
        per = scorecard(k, lf)
        tp = th = tb = 0; cells = []
        for q in range(qmax+1):
            p, h, b = per.get(q, [0, 0, 0])
            tp += p; th += h; tb += b
            cells.append(f'{100*(h/p-b/p):>+7.2f}' if p > 30 else f'{DASH:>7}')
        print(f'  {lab:<22}' + ''.join(cells) + (f'{100*(th/tp-tb/tp):>+8.2f}' if tp else ''))
    print()
