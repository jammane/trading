#!/usr/bin/env python3
"""Gate for race_daily.csv: shape, entry set, and that each row's numbers are self-consistent
with that day's REALIZED returns. usage: race_daily_check.py CSV ROWS BAYES -> prints 'ok ...' or a reason."""
import csv
import sys
from collections import defaultdict

path, rows_want, bayes = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
with open(path) as f:
    r = list(csv.DictReader(f))
hdr = list(r[0].keys()) if r else []
inds = hdr[12:]
if len(hdr) != 24:
    sys.exit(f"header has {len(hdr)} columns, want 24")
real = {(x["pass"], x["day"]): x for x in r if x["entry"] == "REALIZED"}
ent = defaultdict(int)
bad = 0
for x in r:
    e = x["entry"]
    if e == "REALIZED":
        continue
    ent[e] += 1
    rz = real.get((x["pass"], x["day"]))
    picks = [x[f"pick{q}"] for q in range(1, 5)]
    if rz is None or "" in picks:
        bad += 1
        continue
    t4 = sum(float(rz[p]) for p in picks) / 4
    vals = {i: float(x[i]) for i in inds if x[i] != ""}
    top = sorted(vals, key=lambda i: -vals[i])[:4]
    if (abs(t4 - float(x["top4_bp"])) > 0.01
            or abs(float(x["top4_bp"]) - float(x["flat_bp"]) - float(x["gap_bp"])) > 0.01
            or sorted(top) != sorted(picks) and len(set(vals.values())) == len(vals)):
        bad += 1
race = [e for e in ent if not e.startswith("ALLOC bayes")]
nb = sum(1 for e in ent if e.startswith("ALLOC bayes"))
if not real:
    sys.exit("no REALIZED rows")
if len(race) != rows_want:
    sys.exit(f"{len(race)} race entries logged, want {rows_want}")
if not bayes and nb:
    sys.exit("bayes rows present but the allocator was skipped")
if bad:
    sys.exit(f"{bad} rows inconsistent with REALIZED")
print(f"ok {len(real)} days, {len(race)} entries, {nb} bayes lines, {len(r)} rows")
