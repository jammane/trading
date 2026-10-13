"""race_daily_check.py: the gate on training_v4_cpp's race_daily.csv.

It must accept a consistent file and reject each way the writer could go wrong: a pick that is
not the entry's top 4 by prediction, a top4 that does not match that day's REALIZED returns,
a missing entry, and bayes rows when the allocator was skipped.
"""
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "race_daily_check.py"
INDS = [f"i{k}" for k in range(12)]
HDR = "pass,day,date,entry,ok_n,flat_bp,top4_bp,gap_bp,pick1,pick2,pick3,pick4," + ",".join(INDS)
RET = [float(k) for k in range(12)]                     # industry k returns k bp; flat = 5.5


def _entry_row(day, name, preds, picks=None):
    top = picks or sorted(range(12), key=lambda k: -preds[k])[:4]
    t4 = sum(RET[k] for k in top) / 4
    return (f"1,{day},2025-01-01,{name},12,5.5,{t4:.4f},{t4 - 5.5:.4f},"
            + ",".join(INDS[k] for k in top) + "," + ",".join(f"{p:g}" for p in preds))


def _csv(tmp_path, entries, bayes=False, bad_pick=False):
    lines = [HDR]
    for day in (400, 401):
        lines.append(f"1,{day},2025-01-01,REALIZED,12,5.5,,,,,,," + ",".join(f"{r:g}" for r in RET))
        for j, name in enumerate(entries):
            preds = [((k * 7 + j) % 12) + 0.5 for k in range(12)]
            picks = [0, 1, 2, 3] if bad_pick and j == 0 else None
            lines.append(_entry_row(day, name, preds, picks))
        if bayes:
            lines.append(_entry_row(day, "ALLOC bayes mean", [float(k) for k in range(12)]))
    p = tmp_path / "race_daily.csv"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def _run(path, rows, bayes):
    r = subprocess.run([sys.executable, str(SCRIPT), path, str(rows), "1" if bayes else "0"],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def test_consistent_file_passes(tmp_path):
    rc, out = _run(_csv(tmp_path, ["MT1C grad", "MT1 evo"], bayes=True), 2, True)
    assert rc == 0 and out.startswith("ok 2 days, 2 entries, 1 bayes")


def test_pick_not_in_top4_by_prediction_fails(tmp_path):
    rc, out = _run(_csv(tmp_path, ["MT1C grad", "MT1 evo"], bad_pick=True), 2, False)
    assert rc != 0 and "inconsistent" in out


def test_top4_not_matching_realized_fails(tmp_path):
    p = _csv(tmp_path, ["MT1C grad"])
    # MT1C grad's top 4 are i5, i10, i3, i8: move i5's realised return out from under its top4
    txt = "\n".join(ln.replace(",4,5,6,", ",4,95,6,") if ",REALIZED," in ln else ln
                    for ln in Path(p).read_text().splitlines())
    Path(p).write_text(txt + "\n")
    rc, out = _run(p, 1, False)
    assert rc != 0 and "inconsistent" in out


def test_missing_entry_fails(tmp_path):
    rc, out = _run(_csv(tmp_path, ["MT1C grad"]), 2, False)
    assert rc != 0 and "1 race entries logged, want 2" in out


def test_bayes_rows_when_skipped_fail(tmp_path):
    rc, out = _run(_csv(tmp_path, ["MT1C grad"], bayes=True), 1, False)
    assert rc != 0 and "skipped" in out
