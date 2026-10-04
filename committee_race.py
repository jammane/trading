#!/usr/bin/env python3
"""Post-race race: committees of race entries VOTE on the 4 industries to fund.

Reads the trainer's race_daily.csv (last RACE_LOG_DAYS of every pass: each entry's 12
predictions per day, plus a REALIZED row of each industry's actual return in bp). Each pass is a
separately trained version of the same models, so the read is CONSISTENCY across passes.

Entrants (user design, 2026-10-04): the entries with the most passes beating flat -- at least 6,
plus any tied with the 6th -- together with the 6 best by mean gap (overlap expected). Committees
are every combination of 4, 5 and 6 entrants.

Methods. `pick` weights are per position in a member's own top 4 (#1..#4); `member` weights
order the committee by entrant standing (wins, then mean), each member 1.1x the one below it.
  votes        each member's top 4, one vote each                          (user)
  votes_pick   top-4 positions weighted 1.1^3, 1.1^2, 1.1, 1               (user, 1.1 per step)
  votes_member members weighted 1.1^(k-1) .. 1.1, 1 by standing            (user, 1.1 per step)
  votes_both   both of the above, multiplied
  borda        mean rank over all 12 industries, not just the top 4        (suggested)
  zrecord      mean within-day z-score, each member weighted by its own
               recency-weighted gap so far (0.995/day, causal)             (suggested)
  bayes_tree   a Bayesian tree over the vote pattern: root -> number of votes an industry got
               -> exactly which members voted for it. Each node holds a recency-discounted
               (0.995/day) Gaussian posterior of the industry's next-day return minus flat,
               shrunk toward its parent node (kappa = 20 effective days), so a pattern seen
               rarely falls back on "how many voted". Fund the 4 highest leaf posterior means;
               it learns online, decides day t from days < t only.     (user: a Bayesian option)
Ties in any method break by vote count, then the members' mean z-score.

Per configuration and pass: gap = mean daily (top4 return - flat) in bp. Reported:
  w_flat  passes where gap > 0
  w_best  passes where gap > EVERY member's own gap (the committee beat its best member)
  w_mean  passes where gap > the mean of its members' gaps

usage: committee_race.py race_daily.csv [--sizes 4,5,6] [--pool "A,B,..."] [--top 40]
                         [--out committee_race.csv]
"""
import argparse
import csv
import itertools
import math
import sys
from collections import defaultdict

import numpy as np

RECENCY = 0.995        # = PASS_JUDGE_RECENCY / ALLOC_RECENCY in the trainer
STEP = 1.1             # user: each weight 1.1x the weaker one beneath it
N_PICK = 4
MIN_BY_WINS = 6
N_BY_MEAN = 6
TREE_KAPPA = 20.0
METHODS = ("votes", "votes_pick", "votes_member", "votes_both", "borda", "zrecord", "bayes_tree")
PICK_W = np.array([STEP ** 3, STEP ** 2, STEP, 1.0])


def load(path):
    """-> entries, {pass: dict(real[T,12], ok[T,12], flat[T], pred{e: [T,12]}, gap{e: [T]})}

    Only days on which every entry has a row are kept, so every configuration in a pass is
    scored on the same days.
    """
    rows = list(csv.DictReader(open(path)))
    if not rows:
        sys.exit(f"{path}: empty")
    inds = list(rows[0].keys())[12:]
    by = defaultdict(dict)
    for r in rows:
        by[(int(r["pass"]), int(r["day"]))][r["entry"]] = r
    entries = sorted({r["entry"] for r in rows} - {"REALIZED"})

    def vec(r):
        return [float(r[i]) if r[i] != "" else math.nan for i in inds]

    passes = {}
    for p in sorted({k[0] for k in by}):
        days = sorted(d for (q, d), v in by.items()
                      if q == p and "REALIZED" in v and all(e in v for e in entries))
        if not days:
            continue
        real = np.array([vec(by[(p, d)]["REALIZED"]) for d in days])
        ok = ~np.isnan(real)
        passes[p] = dict(
            real=np.nan_to_num(real), ok=ok,
            flat=np.array([real[t][ok[t]].mean() for t in range(len(days))]),
            pred={e: np.array([vec(by[(p, d)][e]) for d in days]) for e in entries},
            gap={e: np.array([float(by[(p, d)][e]["gap_bp"]) for d in days]) for e in entries})
    return entries, passes


def standings(entries, passes):
    """-> {entry: (wins, mean gap, [gap per pass])}, the single-entry baseline."""
    ps = sorted(passes)
    out = {}
    for e in entries:
        g = np.array([passes[p]["gap"][e].mean() for p in ps])
        out[e] = (int((g > 0).sum()), float(g.mean()), g)
    return out


def entrant_pool(stand, min_wins=MIN_BY_WINS, n_mean=N_BY_MEAN):
    """User rule: the most wins (at least min_wins entries, plus ties with the last one) united
    with the n_mean best means. Returned strongest first (wins, then mean)."""
    by_w = sorted(stand, key=lambda e: (-stand[e][0], -stand[e][1]))
    cut = stand[by_w[min(min_wins, len(by_w)) - 1]][0]
    pool = {e for e in by_w if stand[e][0] >= cut}
    pool |= set(sorted(stand, key=lambda e: -stand[e][1])[:n_mean])
    return [e for e in by_w if e in pool]


def per_entry(pred, ok):
    """Within-day features of one entry: rank (0 worst .. n-1 best, ties averaged), z-score,
    and top-4 position (0 = its #1 pick .. 3, -1 = not picked). NaN / -1 on non-ok industries."""
    T, n = pred.shape
    rk = np.zeros((T, n))
    z = np.zeros((T, n))
    pos = np.full((T, n), -1)
    for t in range(T):
        m = np.flatnonzero(ok[t])
        x = pred[t, m]
        order = np.argsort(-x, kind="stable")
        pos[t, m[order[:N_PICK]]] = np.arange(min(N_PICK, len(m)))
        r = np.empty(len(x))
        r[np.argsort(x, kind="stable")] = np.arange(len(x), dtype=float)
        for v in np.unique(x):
            sel = x == v
            if sel.sum() > 1:
                r[sel] = r[sel].mean()
        rk[t, m] = r
        sd = x.std()
        z[t, m] = (x - x.mean()) / sd if sd > 0 else 0.0
    return rk, z, pos


def record(gap, recency=RECENCY):
    """Causal recency-weighted mean of a member's own daily gap: out[t] uses gap[:t] only."""
    out = np.zeros(len(gap))
    num = den = 0.0
    for t in range(len(gap)):
        out[t] = num / den if den > 0 else 0.0
        num = recency * num + gap[t]
        den = recency * den + 1.0
    return out


def fund(score, ok, votes, zmean):
    """[T,12] bool: the 4 best ok industries by score, ties -> votes -> mean z -> lowest index."""
    T, n = score.shape
    # rounded so weights equal by design (1.1 * 1.21 vs 1.1 ** 3) tie exactly
    s = np.where(ok, np.round(score, 9), -np.inf)
    idx = np.broadcast_to(np.arange(n), (T, n))
    order = np.lexsort((idx, -zmean, -votes, -s), axis=-1)       # last key is primary
    pick = np.zeros((T, n), bool)
    np.put_along_axis(pick, order[:, :N_PICK], True, axis=-1)
    return pick


def bayes_tree(pos, real, flat, ok, kappa=TREE_KAPPA, recency=RECENCY):
    """[T,12] scores from the vote-pattern tree; day t uses only days < t.
    pos: [k,T,12] top-4 positions of the k members."""
    k, T, n = pos.shape
    voted = pos >= 0
    pattern = (voted * (1 << np.arange(k))[:, None, None]).sum(0)        # [T,12] leaf id
    count = voted.sum(0)                                                  # [T,12] vote-count node
    root = np.zeros(2)                                                    # (weight, weighted sum)
    mid = np.zeros((k + 1, 2))
    leaf = np.zeros((1 << k, 2))
    score = np.zeros((T, n))

    for t in range(T):
        r0 = root[1] / root[0] if root[0] > 0 else 0.0
        c, lf = count[t], pattern[t]
        m1 = (mid[c, 1] + kappa * r0) / (mid[c, 0] + kappa)
        score[t] = (leaf[lf, 1] + kappa * m1) / (leaf[lf, 0] + kappa)
        root *= recency
        mid *= recency
        leaf *= recency
        o = ok[t]
        y = real[t, o] - flat[t]
        root += (o.sum(), y.sum())
        np.add.at(mid, (c[o], 0), 1.0)
        np.add.at(mid, (c[o], 1), y)
        np.add.at(leaf, (lf[o], 0), 1.0)
        np.add.at(leaf, (lf[o], 1), y)
    return score


def run_committee(members, rank_w, P, F):
    """members: entrant names, strongest first; rank_w: their member weights (1.1 steps).
    P: one pass; F: {entry: (rank, z, pos, rec)} for that pass. -> {method: gap bp/day}."""
    rk = np.array([F[e][0] for e in members])
    z = np.array([F[e][1] for e in members])
    pos = np.array([F[e][2] for e in members])
    rec = np.array([F[e][3] for e in members])
    real, ok, flat = P["real"], P["ok"], P["flat"]
    voted = (pos >= 0).astype(float)
    pw = np.where(pos >= 0, PICK_W[np.clip(pos, 0, 3)], 0.0)
    mw = np.asarray(rank_w)[:, None, None]
    votes = voted.sum(0)
    zmean = z.mean(0)
    rw = np.maximum(rec, 0.0)[:, :, None]
    rw = np.where(rw.sum(0, keepdims=True) <= 0, 1.0, rw)
    scores = {
        "votes": votes,
        "votes_pick": pw.sum(0),
        "votes_member": (mw * voted).sum(0),
        "votes_both": (mw * pw).sum(0),
        "borda": rk.mean(0),
        "zrecord": (rw * z).sum(0) / rw.sum(0),
        "bayes_tree": bayes_tree(pos, real, flat, ok),
    }
    out = {}
    for meth, s in scores.items():
        pick = fund(s, ok, votes, zmean)
        out[meth] = float(((pick * real).sum(1) / N_PICK - flat).mean())
    return out


def run(path, sizes=(4, 5, 6), pool=None):
    """-> (stand, pool, ps, results[list of dict])"""
    entries, passes = load(path)
    stand = standings(entries, passes)
    if pool:
        missing = set(pool) - set(entries)
        if missing:
            sys.exit(f"not in {path}: {', '.join(sorted(missing))}")
        pool = sorted(pool, key=lambda e: (-stand[e][0], -stand[e][1]))
    else:
        pool = entrant_pool(stand)
    ps = sorted(passes)
    feats = {}
    for p in ps:
        P = passes[p]
        feats[p] = {e: (*per_entry(P["pred"][e], P["ok"]), record(P["gap"][e])) for e in pool}
    results = []
    for k in sizes:
        if k > len(pool):
            continue
        rank_w = [STEP ** (k - 1 - j) for j in range(k)]           # strongest first
        for combo in itertools.combinations(pool, k):              # keeps pool order
            per = [run_committee(combo, rank_w, passes[p], feats[p]) for p in ps]
            mem = np.array([stand[e][2] for e in combo])           # [k,P]
            for meth in METHODS:
                g = np.array([x[meth] for x in per])
                results.append(dict(committee=" + ".join(combo), k=k, method=meth, gaps=g,
                                    w_flat=int((g > 0).sum()),
                                    w_best=int((g > mem.max(0)).sum()),
                                    w_mean=int((g > mem.mean(0)).sum())))
    return stand, pool, ps, results


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv")
    ap.add_argument("--sizes", default="4,5,6")
    ap.add_argument("--pool", default="", help="override the entrant list (comma separated)")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    sizes = tuple(int(x) for x in a.sizes.split(","))
    pool = [x.strip() for x in a.pool.split(",") if x.strip()] or None
    stand, pool, ps, res = run(a.csv, sizes, pool)
    n = len(ps)
    print(f"{a.csv}: {len(stand)} entries, {n} passes\n")
    print("singles, gap vs flat bp/day per pass  (* = entrant)")
    print(f"{'entry':24}{'wins':>6}{'mean':>8}  " + " ".join(f"{'p' + str(p):>6}" for p in ps))
    for e in sorted(stand, key=lambda e: (-stand[e][0], -stand[e][1])):
        w, m, g = stand[e]
        print(f"{('* ' if e in pool else '  ') + e:24}{w:>3}/{n:<2}{m:+8.2f}  "
              + " ".join(f"{x:+6.2f}" for x in g))
    print(f"\n{len(pool)} entrants -> {len(res) // len(METHODS)} committees x "
          f"{len(METHODS)} methods = {len(res)} configurations")
    res.sort(key=lambda r: (-r["w_flat"], -r["w_best"], -r["gaps"].mean()))
    print(f"\ntop {a.top} (sorted: beat flat, beat best member, mean)")
    print(f"{'w_flat':>6}{'w_best':>7}{'w_mean':>7}{'mean':>8}  {'method':12} committee")
    for r in res[:a.top]:
        print(f"{r['w_flat']:>3}/{n:<2}{r['w_best']:>4}/{n:<2}{r['w_mean']:>4}/{n:<2}"
              f"{r['gaps'].mean():+8.2f}  {r['method']:12} {r['committee']}")
    print("\nby method: configurations beating flat in >= 80% of passes, and mean gap")
    for meth in METHODS:
        rs = [r for r in res if r["method"] == meth]
        hi = sum(r["w_flat"] >= 0.8 * n for r in rs)
        print(f"  {meth:12} {hi:>5}/{len(rs):<5} {np.mean([r['gaps'].mean() for r in rs]):+7.2f}")
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["committee", "k", "method", "w_flat", "w_best", "w_mean", "mean"]
                       + [f"p{p}" for p in ps])
            for r in res:
                w.writerow([r["committee"], r["k"], r["method"], r["w_flat"], r["w_best"],
                            r["w_mean"], f"{r['gaps'].mean():.4f}"]
                           + [f"{x:.4f}" for x in r["gaps"]])
        print(f"\nall configurations -> {a.out}")


if __name__ == "__main__":
    main()
