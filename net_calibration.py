#!/usr/bin/env python3
"""Is the networks' problem SIGNAL or CALIBRATION?

MT1 is trained per-industry: its score d/(err+d) rewards getting one industry's dollar figure
close. The win condition is CROSS-SECTIONAL: rank the 12 against each other. Those are different
objectives, and optimising the first can actively hurt the second -- twelve independently
calibrated estimators, each shrunk toward its own trailing mean, rank badly because the ranking
then reflects differences in those means rather than differences in expected surprise. CLAUDE.md
flags this as "fixable programmatically by z-scoring each output against its own trailing
distribution"; it has never been tested.

This tests it, offline, on the predictions already logged -- no retraining. Three transforms, each
strictly causal (trailing window only, never including the day being transformed):

    raw          the prediction as logged
    z            (pred - trailing mean) / trailing sd, per industry
    pct          the prediction's percentile within its own trailing distribution
    sign         sign(pred) alone -- keeps direction, discards magnitude entirely

If a transform lifts rank IC materially, the networks HAVE cross-sectional signal that their
calibration is hiding, and the fix is a post-processing step rather than a new architecture. If
none does, the signal is not there and no rescaling will conjure it.

    python net_calibration.py mt1_dataset.bin
"""
import argparse

import numpy as np

import mt1_analysis as M
import read_mt1_dataset as D

NETS = [('MT1Net', 'm'), ('MT1CNet', 'c'), ('MT2INet', 'i')]
WIN = 250


def trailing_transform(pred, mask, kind):
    """Causal per-industry rescaling: day t uses only days < t."""
    T, N = pred.shape
    out = np.full((T, N), np.nan)
    for i in range(N):
        for t in range(WIN, T):
            if not mask[t, i] or not np.isfinite(pred[t, i]):
                continue
            w = pred[max(0, t - WIN):t, i]
            w = w[np.isfinite(w) & mask[max(0, t - WIN):t, i]]
            if len(w) < 60:
                continue
            if kind == 'z':
                sd = w.std(ddof=1)
                out[t, i] = (pred[t, i] - w.mean()) / sd if sd > 1e-12 else 0.0
            elif kind == 'pct':
                out[t, i] = (w < pred[t, i]).mean()
            elif kind == 'sign':
                out[t, i] = np.sign(pred[t, i])
            else:
                out[t, i] = pred[t, i]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('--boot', type=int, default=400)
    a = ap.parse_args()

    ds = D.read(a.dataset) if hasattr(D, 'read') else D.parse(a.dataset)
    T = len(ds['day'])
    book = (ds['book_now'] - ds['book_prev']).astype(float)
    prev = np.maximum(ds['book_prev'].astype(float), 1.0)

    def fwd(h):
        f = D.forward_pnl(ds, h - 1, 'book') if h > 1 else np.zeros_like(book)
        tot = book + np.nan_to_num(f, nan=0.0) if h > 1 else book
        r = tot / prev
        if h > 1:
            r[-(h - 1):] = np.nan
        return r

    def rank_ic_t(pred, mask, f, h):
        vals = []
        for t in range(T):
            m = mask[t] & np.isfinite(pred[t]) & np.isfinite(f[t])
            if m.sum() >= 6:
                vals.append(M.spearman(pred[t][m], f[t][m]))
        v = np.array([x for x in vals if np.isfinite(x)])
        if len(v) < 50:
            return np.nan, np.nan
        se, _ = M.block_bootstrap(lambda idx: float(v[idx].mean()), len(v), max(2 * h, 20), a.boot)
        return v.mean(), (v.mean() / se if se else np.nan)

    print(f'{T} days. Cross-sectional rank IC (block-bootstrap t), by transform.\n')
    print(f'  {"net":<9} {"transform":<10} ' + ' '.join(f'{"h=" + str(h):>16}' for h in (1, 2, 3)))
    for name, k in NETS:
        pred = ds[f'{k}_pred'].astype(float)
        mask = ds[f'{k}_score'].astype(float) > 0
        if mask.sum() < 200:
            continue
        for kind in ('raw', 'z', 'pct', 'sign'):
            p = trailing_transform(pred, mask, kind)
            cells = []
            for h in (1, 2, 3):
                ic, t = rank_ic_t(p, mask & np.isfinite(p), fwd(h), h)
                star = '*' if np.isfinite(t) and abs(t) > 2 else ' '
                cells.append(f'{ic:>+9.4f} ({t:>+4.1f}){star}')
            print(f'  {name:<9} {kind:<10} ' + ' '.join(cells))
        print()


if __name__ == '__main__':
    main()
