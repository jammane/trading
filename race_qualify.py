#!/usr/bin/env python3
"""Qualifier: which MT1 race entries go on to the next race.

Rule (set 2026-10-03): an entry must beat flat allocation -- top-4 vs flat, long-only, the
whole-pass `alloc bp/day` line -- in MORE THAN 40% of its counted passes.

  * entries unchanged since v3: counted over v3 + v4 passes  (5/10 qualifies, 4/10 does not)
  * entries whose model changed in v4: v4 passes only        (3/5 qualifies, 2/5 does not)

Changed in v4: TANH32 grad and gdmn (per-row Adam -> one batched step, v0.8.1.36) and the
per-stock MT1S-stk entries (new in v0.8.1.35). v3's "MT1S-stk" rows were the shared encoder and
the seed replicate, renamed MT1S-shr / MT1S-rep; they are read under their v4 names.

"ALLOC bayes" is one allocator printed as two lines (mean, thompson); it qualifies if either
line does. MT1C grad (the deployed entry) and MT1 evo (the logged pool) are read by index
elsewhere in the trainer, so they always run -- reported, but marked infrastructure.

usage: race_qualify.py V3_LOG V4_LOG [--skip-only]
Prints the table, then `RACE_SKIP=<comma list>` for training_v4_cpp --race-skip.
Exits 2 if either log is missing a pass, so a partial race never decides the field.
"""
import re
import sys

THRESHOLD = 0.40
V3_PASSES = V4_PASSES = 5
CHANGED_IN_V4 = {"TANH32 grad", "TANH32 gdmn", "MT1S-stk evo", "MT1S-stk grad", "MT1S-stk grdC"}
V3_RENAME = {"MT1S-stk evo": "MT1S-rep evo", "MT1S-stk grad": "MT1S-shr grad",
             "MT1S-stk grdC": "MT1S-shr grdC"}
INFRA = {"MT1C grad": "deployed entry", "MT1 evo": "logged pool"}


def parse(path, rename=None):
    """{entry: {pass: top4-minus-flat bp/day}} from a race train.log."""
    rename = rename or {}
    out, p, cur = {}, None, None
    for line in open(path, errors="replace"):
        m = re.search(r"MT1 race, pass (\d+)", line)
        if m:
            p = int(m.group(1))
            continue
        if p is None:
            continue
        m = re.search(r"\]\s+(MT1\S*|TANH32)\s+(\w+)\s+loss", line)
        if m:
            cur = m.group(1) + " " + m.group(2)
            cur = rename.get(cur, cur)
            continue
        m = re.search(r"alloc bp/day: flat \S+ \| top4 \S+ \(([+-][\d.]+) vs flat\)", line)
        if m and cur:
            out.setdefault(cur, {})[p] = float(m.group(1))
            continue
        m = re.search(r"ALLOC bayes (mean|thompson).*\(([+-][\d.]+) vs flat\)", line)
        if m:
            out.setdefault("ALLOC bayes " + m.group(1), {})[p] = float(m.group(2))
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if len(args) != 2:
        sys.exit(__doc__)
    v3 = parse(args[0], V3_RENAME)
    v4 = parse(args[1])
    for name, log, want in (("v3", v3, V3_PASSES), ("v4", v4, V4_PASSES)):
        got = max((len(x) for x in log.values()), default=0)
        if got < want:
            print(f"ERROR: {name} log has {got} pass tables, need {want} -- not deciding the field")
            sys.exit(2)

    rows = []
    for e in v4:                                  # the v4 roster is the field
        own = list(v4[e].values())
        if e in CHANGED_IN_V4 or e not in v3:
            xs, basis = own, "v4"
        else:
            xs, basis = list(v3[e].values()) + own, "v3+v4"
        wins = sum(x > 0 for x in xs)
        rows.append((e, basis, wins, len(xs), sum(xs) / len(xs), wins / len(xs) > THRESHOLD))

    bayes_ok = any(q for e, *_, q in rows if e.startswith("ALLOC bayes"))
    skip = []
    if not args or "--skip-only" not in sys.argv:
        print(f"{'entry':22} {'counted':>7} {'wins':>7} {'mean':>7}  verdict   (rule: > {THRESHOLD:.0%} of passes beat flat)")
    for e, basis, w, n, mean, ok in sorted(rows, key=lambda r: (-r[2] / r[3], -r[4])):
        if e in INFRA:
            verdict = "QUALIFY" if ok else f"runs ({INFRA[e]})"
        elif e.startswith("ALLOC bayes"):
            verdict = "QUALIFY" if ok else ("allocator qualifies" if bayes_ok else "OUT")
        else:
            verdict = "QUALIFY" if ok else "OUT"
            if not ok:
                skip.append(e)
        if "--skip-only" not in sys.argv:
            print(f"{e:22} {basis:>7} {w:>3}/{n:<3} {mean:+7.2f}  {verdict}")
    if not bayes_ok:
        skip.append("ALLOC bayes")
    print("RACE_SKIP=" + ",".join(skip))


if __name__ == "__main__":
    main()
