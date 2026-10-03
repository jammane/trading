#!/usr/bin/env python3
"""Per-pass top4-vs-flat and within-day rank for the entries that won 4/5 passes in v3.
usage: race_watch.py [train.log] [--all] [--trend]

v5+ logs also carry a recency-weighted gap (0.995/day) and a 100-day trend; both are
printed when present.

In the v3 log (<= v0.8.1.34) the MT1S-sum seed replicate was named "MT1S-stk evo"; it is
read as "MT1S-rep evo" there so the two runs line up.
"""
import glob
import os
import re
import sys

newest = max(glob.glob("/root/ht_race_v*/train.log"), key=os.path.getmtime, default="")
log = next((a for a in sys.argv[1:] if not a.startswith("-")), newest)
WATCH = ["MT1S-rep evo", "MT1S-sum evo", "MT1 grdC", "ALLOC bayes mean", "ALLOC bayes thompson"]
text = open(log, errors="replace").read()
v3_names = "MT1S-rep" not in text

p = cur = None
res, order, flat = {}, [], {}
recw, trend = {}, {}
for line in text.split("\n"):
    m = re.search(r"MT1 race, pass (\d+)", line)
    if m:
        p = int(m.group(1))
        continue
    if p is None:
        continue
    m = re.search(r"\]\s+(MT1\S*|TANH32)\s+(\w+)\s+loss.*within-day rank\s+(\S+)", line)
    if m:
        cur = m.group(1) + " " + m.group(2)
        if v3_names and cur == "MT1S-stk evo":
            cur = "MT1S-rep evo"
        if cur not in order:
            order.append(cur)
        res.setdefault(cur, {})[p] = [m.group(3), None]
        continue
    m = re.search(r"alloc bp/day: flat (\S+) \| top4 \S+ \((\S+) vs flat\)", line)
    if m and cur:
        flat[p] = float(m.group(1))
        res[cur][p][1] = float(m.group(2))
    m = re.search(r"alloc recency-wtd .*top4 vs flat ([+-][\d.]+)", line)
    if m and cur:
        recw.setdefault(cur, {})[p] = float(m.group(1))
    m = re.search(r"alloc by day \(top4 vs flat, bp/day\):(.*)", line)
    if m and cur:
        trend.setdefault(cur, {})[p] = m.group(1).strip()
    m = re.search(r"ALLOC bayes (mean|thompson).*\(([+-][\d.]+) vs flat\)", line)
    if m:
        k = "ALLOC bayes " + m.group(1)
        cur = k
        if k not in order:
            order.append(k)
        res.setdefault(k, {})[p] = ["", float(m.group(2))]

ps = sorted(flat)
if not ps:
    print(f"{log}: no pass table yet")
    sys.exit()
rows = order if "--all" in sys.argv else [e for e in WATCH if e in res]
print(f"{log}   flat bp/day: " + "  ".join(f"p{q} {flat[q]:+.2f}" for q in ps))
print(f"{'entry':22}" + "".join(f"  p{q} top4-flat/rank  " for q in ps) + "   mean  wins")
for e in rows:
    vals = [res[e].get(q, ["", None]) for q in ps]
    xs = [x[1] for x in vals if x[1] is not None]
    cells = "".join(f"  {x[1]:+7.2f} / {x[0]:>7}  " if x[1] is not None else " " * 21
                    for x in vals)
    mean = sum(xs) / len(xs) if xs else float("nan")
    print(f"{e:22}{cells}  {mean:+6.2f}  {sum(x > 0 for x in xs)}/{len(xs)}")

if recw:
    print("\nrecency-weighted top4 vs flat (0.995/day, half-life ~138 days)")
    print(f"{'entry':22}" + "".join(f"{'p' + str(q):>9}" for q in ps) + "     mean  wins")
    for e in rows:
        xs = [recw.get(e, {}).get(q) for q in ps]
        got = [x for x in xs if x is not None]
        if not got:
            continue
        print(f"{e:22}" + "".join(f"{x:+9.2f}" if x is not None else " " * 9 for x in xs)
              + f"   {sum(got) / len(got):+6.2f}  {sum(x > 0 for x in got)}/{len(got)}")
if trend and "--trend" in sys.argv:
    print("\n100-day trend, top4 vs flat bp/day (segment start day: value)")
    for e in rows:
        for q in ps:
            if q in trend.get(e, {}):
                print(f"{e:22} p{q}  {trend[e][q]}")
