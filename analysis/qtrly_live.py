"""Quarterly scorecard for the ONLINE trees training inside the live pass.

Cell = that model's hit rate minus the always-up baseline over the quarter, in pp.
Positive means it beat the trivial rule that quarter.
"""
import collections
import csv
import sys

C = "/root/trading-ht/logs/acct0/training/bayes_online.csv"
Q = 63
rows = list(csv.DictReader(open(C)))
if not rows:
    sys.exit("csv empty")
days = sorted({int(r["day"]) for r in rows})
d0 = days[0]
agg = collections.defaultdict(lambda: [0, 0, 0])      # (model, qtr) -> preds, hits, base_up
for r in rows:
    q = (int(r["day"]) - d0) // Q
    a = agg[(r["model"], q)]
    a[0] += int(r["preds"]); a[1] += int(r["hits"]); a[2] += int(r["base_up"])
models = sorted({r["model"] for r in rows})
qmax = max(q for _, q in agg)
DASH = "-"
print(f"live run: days {d0}..{days[-1]}  ({len(days)} days, {qmax+1} quarters of {Q})")
print("cell = model hit rate MINUS always-up, in pp\n")
hdr = f"  {'model':<14}" + "".join(f"{'Q'+str(q+1):>7}" for q in range(qmax+1)) + f"{'ALL':>8}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for m in models:
    tp = th = tb = 0
    cells = []
    for q in range(qmax + 1):
        p, h, b = agg.get((m, q), [0, 0, 0])
        tp += p; th += h; tb += b
        cells.append(f"{100*(h/p-b/p):>+7.2f}" if p > 30 else f"{DASH:>7}")
    tail = f"{100*(th/tp-tb/tp):>+8.2f}" if tp else ""
    print(f"  {m:<14}" + "".join(cells) + tail)
print("  " + "-" * (len(hdr) - 2))
bl, nd = [], []
for q in range(qmax + 1):
    p, h, b = agg.get((models[0], q), [0, 0, 0])
    bl.append(f"{100*b/p:>7.1f}" if p > 30 else f"{DASH:>7}")
    nd.append(f"{p:>7}")
print(f"  {'always-up %':<14}" + "".join(bl))
print(f"  {'n preds':<14}" + "".join(nd))
