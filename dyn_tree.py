#!/usr/bin/env python3
"""An explicit Bayesian network with a DYNAMIC (adaptively-split) context tree.

WHY A TREE. depth_sweep.py measured fixed-depth conditioning and found it collapses past k=2:
k=3 breaks even, k=4-5 run 8-50 millinats WORSE than a coin flip, because a fixed k splits EVERY
state whether or not the data supports that state being split (720 states over ~9,000
industry-days is 14 observations each). A tree splits only where the evidence justifies it, so
depth is paid for per-branch rather than globally.

STRUCTURE. A prefix tree over the recent history, one level per extra day:

    level 1   sign(today)                       2 branches   == k=1
    level 2   today vs yesterday: accel / decel / flip        == k=2, the original six states
    level 3   yesterday vs the day before       3 branches
    ...

At levels 1 and 2 this is exactly the encoding already raced (the k=2 bijection is proven in
depth_sweep.verify_k2). It diverges from the ordered-set encoding only at level 3+, where the
ordered set would give 24 states and this gives 18 -- and in practice the tree rarely reaches
there, which is the finding rather than a detail.

DYNAMIC. A node uses its own estimate only if splitting it is justified, tested every time it is
consulted:
    * each child needs >= MIN_OBS effective observations, and
    * a likelihood-ratio test of split-vs-pooled must clear a chi-square threshold.
Counts decay (or leave a rectangular window), so a split that stops being justified is abandoned:
the tree grows AND shrinks as the data moves. That is the "dynamic" part -- not merely that it was
grown once from data.

Prediction walks down the current context and stops at the deepest node whose split is justified.

    python dyn_tree.py mt1_dataset.bin --burn-in 400 --judge 50
"""
import argparse
import math

import numpy as np

import read_mt1_dataset as D

ACCEL, DECEL, FLIP = 0, 1, 2


def relation(prev, cur):
    if (cur > 0) != (prev > 0):
        return FLIP
    return ACCEL if abs(cur) >= abs(prev) else DECEL


def context(col, t, max_depth):
    """Path symbols for predicting day t, most-recent first. None if the path is unusable."""
    if t < 1 or col[t - 1] == 0 or not np.isfinite(col[t - 1]):
        return None
    path = [0 if col[t - 1] > 0 else 1]
    for d in range(2, max_depth + 1):
        j = t - d
        if j < 0 or col[j] == 0 or not np.isfinite(col[j]):
            break
        path.append(relation(col[j], col[j + 1]))
    return path


class Node:
    __slots__ = ('n', 'k', 'kids')

    def __init__(self):
        self.n = 0.0      # effective observations
        self.k = 0.0      # of which "tomorrow kept today's sign"
        self.kids = {}

    def p(self, prior=2.0):
        return (self.k + prior * 0.5) / (self.n + prior)


def chi2_crit(df, z=2.3263):                      # 99th percentile, Wilson-Hilferty
    k = max(df, 1)
    return k * (1.0 - 2.0 / (9.0 * k) + z * math.sqrt(2.0 / (9.0 * k))) ** 3


class DynTree:
    def __init__(self, max_depth=5, min_obs=40, half_life=1e9, window=0):
        self.root = Node()
        self.max_depth = max_depth
        self.min_obs = min_obs
        self.half_life = half_life
        self.window = window
        self.ring = []

    def _walk(self, path, create=False):
        nodes = [self.root]
        cur = self.root
        for s in path:
            if s not in cur.kids:
                if not create:
                    break
                cur.kids[s] = Node()
            cur = cur.kids[s]
            nodes.append(cur)
        return nodes

    def _split_justified(self, node):
        """Likelihood-ratio test: do the children's rates differ beyond sampling noise?"""
        kids = [c for c in node.kids.values() if c.n >= self.min_obs]
        if len(kids) < 2:
            return False
        n_tot = sum(c.n for c in kids)
        k_tot = sum(c.k for c in kids)
        p0 = k_tot / n_tot if n_tot > 0 else 0.5
        if p0 <= 0 or p0 >= 1:
            return False
        g = 0.0
        for c in kids:
            p1 = min(max(c.k / c.n, 1e-9), 1 - 1e-9)
            g += 2.0 * (c.k * math.log(p1 / p0) + (c.n - c.k) * math.log((1 - p1) / (1 - p0)))
        return g > chi2_crit(len(kids) - 1)

    def _differs(self, node, ref):
        """Does `node`'s rate differ from `ref`'s beyond sampling noise? LR test, 1 df, 99%."""
        if node.n < self.min_obs:
            return False
        p0 = min(max(ref.p(), 1e-9), 1 - 1e-9)
        p1 = min(max(node.k / node.n, 1e-9), 1 - 1e-9)
        g = 2.0 * (node.k * math.log(p1 / p0) + (node.n - node.k) * math.log((1 - p1) / (1 - p0)))
        return g > chi2_crit(1)

    def predict(self, path):
        """Deepest node that differs from the best shallower one; its rate, and the depth used.

        Descend on DATA (a child needs min_obs), then accept depth on EVIDENCE, comparing each
        candidate against the deepest node accepted so far rather than against its immediate
        parent. The distinction is not cosmetic: an earlier version required each level's split to
        be justified before descending, which made structure at level 2 unreachable whenever
        level 1 carried nothing. Its power test failed -- the tree never reached depth 2 on data
        with depth-2 structure planted in it -- and it would have reported "no deeper structure"
        for any input, which is precisely the conclusion being drawn from this file.
        """
        nodes = [self.root]
        cur = self.root
        for s in path:
            if s not in cur.kids or cur.kids[s].n < self.min_obs:
                break
            cur = cur.kids[s]
            nodes.append(cur)
        best, best_d = self.root, 0
        for d in range(1, len(nodes)):
            if self._differs(nodes[d], best):
                best, best_d = nodes[d], d
        return best.p(), best_d

    def observe(self, obs):
        """obs: list of (path, same). Applied after every prediction for the day has been made."""
        if self.window <= 0:
            g = 0.5 ** (1.0 / self.half_life)
            self._decay(self.root, g)
        for path, same in obs:
            for nd in self._walk(path, create=True):
                nd.n += 1.0
                nd.k += same
        if self.window > 0:
            self.ring.append(obs)
            while len(self.ring) > self.window:
                for path, same in self.ring.pop(0):
                    for nd in self._walk(path, create=False):
                        nd.n -= 1.0
                        nd.k -= same

    def _decay(self, node, g):
        node.n *= g
        node.k *= g
        for c in node.kids.values():
            self._decay(c, g)


def run(pnl, half_life, window, judge, max_depth=5, min_obs=40):
    T, N = pnl.shape
    tree = DynTree(max_depth, min_obs, half_life, window)
    hits, lls, depths, base = [], [], [], []
    start = T - judge
    for t in range(1, T):
        obs = []
        for i in range(N):
            path = context(pnl[:, i], t, max_depth)
            y, ytd = pnl[t, i], pnl[t - 1, i]
            if path is None or not np.isfinite(y) or y == 0:
                continue
            same = float(np.sign(y) == np.sign(ytd))
            if t >= start:
                p, d = tree.predict(path)
                call = np.sign(ytd) if p > 0.5 else -np.sign(ytd)
                hits.append(float(call == np.sign(y)))
                lls.append(math.log(max(min(p if same else 1 - p, 1 - 1e-9), 1e-9)))
                depths.append(d)
                base.append(float(y > 0))
            obs.append((path, same))
        tree.observe(obs)
    return (np.array(hits), np.array(lls), np.array(depths), np.array(base))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('--burn-in', type=int, default=400)
    ap.add_argument('--judge', type=int, default=50, help='judge on the last N days only')
    ap.add_argument('--min-obs', type=int, default=40)
    a = ap.parse_args()

    ds = D.read(a.dataset) if hasattr(D, 'read') else D.parse(a.dataset)
    day = np.asarray(ds['day'])
    raw = ds['book_now'].astype(float) - ds['book_prev'].astype(float)
    keep = day >= a.burn_in
    cols = [raw[keep, i][np.abs(raw[keep, i]) > 1e-9] for i in range(raw.shape[1])]
    m = min(len(c) for c in cols)
    P = np.column_stack([c[-m:] for c in cols])
    print(f'StockNN book P&L, day >= {a.burn_in}: {P.shape[0]} days x {P.shape[1]} industries')
    print(f'judging on the LAST {a.judge} days only\n')

    kernels = [('1y decay', 252.0, 0), ('2y decay', 504.0, 0),
               ('1y window', 1e9, 252), ('2y window', 1e9, 504),
               ('never forget', 1e9, 0)]
    coin = -math.log(2)
    print(f'  {"kernel":<13} {"n":>6} {"hit rate":>10} {"vs always-up":>14} '
          f'{"logL/obs":>10} {"vs coin":>10} {"mean depth":>11} {"max":>4}')
    rows = []
    for name, hl, win in kernels:
        h, ll, dep, bs = run(P, hl, win, a.judge, min_obs=a.min_obs)
        if len(h) == 0:
            continue
        rows.append((name, h, ll, dep, bs))
        print(f'  {name:<13} {len(h):>6} {100*h.mean():>9.2f}% '
              f'{100*(h.mean()-bs.mean()):>+13.2f}pp {ll.mean():>10.5f} '
              f'{1000*(ll.mean()-coin):>+9.2f} {dep.mean():>11.2f} {int(dep.max()):>4}')

    bs = rows[0][4]
    se = np.sqrt(0.25 / len(bs))
    print(f'\n  ALWAYS-UP on this window: {100*bs.mean():.2f}%')
    print(f'  POWER: n = {len(bs)} predictions over {a.judge} days. Naive SE on a hit rate is '
          f'{100*se:.2f}pp,')
    print('  and the 12 industries move together (mean pairwise corr +0.26) so the effective')
    print(f'  sample is nearer {a.judge} than {len(bs)} -- SE ~{100*np.sqrt(0.25/a.judge):.1f}pp.')
    print(f'  Differences below roughly {2*100*np.sqrt(0.25/a.judge):.0f}pp are not resolvable here.')


if __name__ == '__main__':
    main()
