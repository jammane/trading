#!/usr/bin/env python3
"""
plot_training.py — Download training logs from the droplet and generate SVG progress charts.

Usage:
    python plot_training.py                       # use existing logs_local/ files
    python plot_training.py --download            # pull fresh logs first
    python plot_training.py --download --account acct0 --host root@165.22.6.112

Outputs (git-ignored):
    plots/industry_performance.svg   — StockNN elite portfolio value per industry
    plots/mt1_performance.svg        — MT1 composite score per industry
    plots/mt2_performance.svg        — MT2 allocation score (elite pool stats)

Lines: solid = mean, dashed = max & min  (slot0 excluded per design)
"""

import argparse
import csv
import struct
import subprocess
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import matplotlib.lines as mlines
except ImportError:
    sys.exit("matplotlib is required: pip install matplotlib")

# ── Configuration ─────────────────────────────────────────────────────────────
DROPLET_HOST  = "root@165.22.6.112"
REMOTE_LOG    = "/root/trading/logs/{account}/training"
LOCAL_LOG_DIR = Path("logs_local")
PLOT_DIR      = Path("plots")

DAYS_PER_PASS = 1255
FIG_W, FIG_H  = 16, 9   # 16:9 — matches widescreen laptop aspect ratio

INDUSTRIES = [
    "tech_hardware", "tech_software_ai", "financials", "consumer_discretionary",
    "consumer_services", "health_care", "industrials", "consumer_staples",
    "energy", "utilities", "real_estate", "materials",
]
IND_LABEL = {
    "tech_hardware":          "Hardware",
    "tech_software_ai":       "Software/AI",
    "financials":             "Financials",
    "consumer_discretionary": "Cons. Discr.",
    "consumer_services":      "Cons. Svc",
    "health_care":            "Health Care",
    "industrials":            "Industrials",
    "consumer_staples":       "Cons. Staples",
    "energy":                 "Energy",
    "utilities":              "Utilities",
    "real_estate":            "Real Estate",
    "materials":              "Materials",
}
# 12 perceptually distinct colors (matplotlib tab10 + tab20 supplement)
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#f0a500", "#00897b",
]
IND_COLOR = dict(zip(INDUSTRIES, COLORS))

# ── Binary log constants (v4, 984 bytes/record) ───────────────────────────────
MT_LOG_MAGIC   = 0x4D543132  # 'MT12'
RECORD_SIZE_V4 = 984
HEADER_SIZE    = 16
# 2×uint32 + 240 floats (5 components × 4 stats × 12 inds) + 3 floats (MT2) + uint8 + 3 pad
_RECORD_STRUCT = struct.Struct("<II" + "f" * 243 + "Bxxx")
assert _RECORD_STRUCT.size == RECORD_SIZE_V4, f"struct size mismatch: {_RECORD_STRUCT.size}"

_COMP_NAMES = ["composite", "direction", "range", "accuracy", "confidence"]
_STAT_NAMES = ["best", "slot0", "mean", "min"]

# ── Download ──────────────────────────────────────────────────────────────────
def download_logs(host: str, account: str) -> None:
    remote = f"{host}:{REMOTE_LOG.format(account=account)}"
    LOCAL_LOG_DIR.mkdir(exist_ok=True)
    for fname in ("training_log.csv", "mt_training_log.bin"):
        src = f"{remote}/{fname}"
        dst = LOCAL_LOG_DIR / fname
        print(f"  scp {src} → {dst}")
        result = subprocess.run(["scp", src, str(dst)], capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(f"scp failed: {result.stderr.strip()}")

# ── Parse training_log.csv ────────────────────────────────────────────────────
def load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v.strip()) for k, v in row.items()})
    return rows

def csv_x(row: dict) -> int:
    return int(row["pass"] - 1) * DAYS_PER_PASS + int(row["day"])

# ── Parse mt_training_log.bin ─────────────────────────────────────────────────
def load_binary_log(path: Path) -> list[dict]:
    data = path.read_bytes()
    if len(data) < HEADER_SIZE:
        sys.exit(f"{path}: too small to be a valid log ({len(data)} bytes)")

    magic, version, n_ind, _ = struct.unpack_from("<IIII", data, 0)
    if magic != MT_LOG_MAGIC:
        sys.exit(f"{path}: bad magic {magic:#010x} (expected {MT_LOG_MAGIC:#010x})")
    if version != 3:
        sys.exit(f"{path}: log version {version} — this tool requires v3 (984-byte records)")

    records = []
    offset = HEADER_SIZE
    while offset + RECORD_SIZE_V4 <= len(data):
        raw = _RECORD_STRUCT.unpack_from(data, offset)
        pass_num   = raw[0]
        actual_day = raw[1]

        # raw[2..241] = 240 MT1 floats; raw[242..244] = MT2; raw[245] = injected
        f = raw[2:242]  # slice of 240 floats
        mt1 = {}
        for ci, comp in enumerate(_COMP_NAMES):
            mt1[comp] = {}
            for si, stat in enumerate(_STAT_NAMES):
                base = ci * 48 + si * 12
                mt1[comp][stat] = list(f[base : base + 12])

        records.append({
            "pass":      pass_num,
            "day":       actual_day,
            "x":         pass_num * DAYS_PER_PASS + actual_day,
            "mt1":       mt1,
            "mt2_best":  raw[242],
            "mt2_slot0": raw[243],
            "mt2_ideal": raw[244],
            "mt2_inj":   raw[245],
        })
        offset += RECORD_SIZE_V4

    return records

# ── Shared plot helpers ───────────────────────────────────────────────────────
def _style_ax(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.tick_params(labelsize=8)

def _add_pass_dividers(ax, xs: list) -> None:
    if not xs:
        return
    max_x = max(xs)
    p = 1
    while p * DAYS_PER_PASS < max_x:
        ax.axvline(p * DAYS_PER_PASS, color="#cccccc", linewidth=0.6, linestyle=":")
        p += 1

def _industry_legend(ax) -> None:
    handles = [
        mlines.Line2D([], [], color=IND_COLOR[ind], linewidth=1.8, label=IND_LABEL[ind])
        for ind in INDUSTRIES
    ]
    handles += [
        mlines.Line2D([], [], color="#444444", linewidth=1.8, linestyle="-",  label="─── mean"),
        mlines.Line2D([], [], color="#444444", linewidth=0.9, linestyle="--", label="- - max / min"),
    ]
    ax.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5, -0.14), ncol=7,
        fontsize=8, framealpha=0.95, edgecolor="#cccccc",
    )

def _plot_band(ax, xs, means, maxes, mins, color: str) -> None:
    ax.plot(xs, means, color=color, linewidth=1.5, alpha=0.90, zorder=3)
    ax.plot(xs, maxes, color=color, linewidth=0.75, linestyle="--", alpha=0.50, zorder=2)
    ax.plot(xs, mins,  color=color, linewidth=0.75, linestyle="--", alpha=0.50, zorder=2)

# ── Industry performance SVG ──────────────────────────────────────────────────
def plot_industry(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(bottom=0.23)

    xs = [csv_x(r) for r in rows]
    for ind in INDUSTRIES:
        _plot_band(
            ax, xs,
            means=[r[f"{ind}_elite_mean"] for r in rows],
            maxes=[r[f"{ind}_elite_max"]  for r in rows],
            mins= [r[f"{ind}_elite_min"]  for r in rows],
            color=IND_COLOR[ind],
        )

    ax.axhline(25_000, color="#aaaaaa", linewidth=0.6, linestyle=":", zorder=1)
    _add_pass_dividers(ax, xs)
    _style_ax(ax, "Industry Elite Performance (StockNN)", "Training Day", "Portfolio Value ($)")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    _industry_legend(ax)

    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")

# ── MT1 composite SVG ─────────────────────────────────────────────────────────
def plot_mt1(records: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(bottom=0.23)

    xs = [r["x"] for r in records]
    for i, ind in enumerate(INDUSTRIES):
        _plot_band(
            ax, xs,
            means=[r["mt1"]["composite"]["mean"][i] for r in records],
            maxes=[r["mt1"]["composite"]["best"][i] for r in records],
            mins= [r["mt1"]["composite"]["min"][i]  for r in records],
            color=IND_COLOR[ind],
        )

    ax.set_ylim(0, 1)
    _add_pass_dividers(ax, xs)
    _style_ax(ax, "MT1 Composite Score by Industry", "Training Day", "Score (0 – 1)")
    _industry_legend(ax)

    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")

# ── MT2 SVG ───────────────────────────────────────────────────────────────────
def plot_mt2(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(bottom=0.13)

    xs     = [csv_x(r) for r in rows]
    means  = [r["mt2_elite_mean_pts"] for r in rows]
    maxes  = [r["mt2_elite_max_pts"]  for r in rows]
    mins   = [r["mt2_elite_min_pts"]  for r in rows]
    ideals = [r["mt2_ideal_pts"]      for r in rows]

    ax.plot(xs, means,  color="#1f77b4", linewidth=1.5, alpha=0.90, label="Elite mean")
    ax.plot(xs, maxes,  color="#1f77b4", linewidth=0.75, linestyle="--", alpha=0.55, label="Elite max")
    ax.plot(xs, mins,   color="#1f77b4", linewidth=0.75, linestyle="--", alpha=0.55, label="Elite min")
    ax.plot(xs, ideals, color="#aaaaaa", linewidth=0.8,  linestyle=":",  alpha=0.85, label="Ideal (slot0 basis)")
    ax.axhline(0, color="#dddddd", linewidth=0.5, zorder=1)
    _add_pass_dividers(ax, xs)
    _style_ax(ax, "MT2 Allocation Score (Elite Pool)", "Training Day", "Score (pts)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4,
              fontsize=9, framealpha=0.95, edgecolor="#cccccc")

    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate training progress SVGs")
    ap.add_argument("--download", action="store_true", help="Pull fresh logs from droplet")
    ap.add_argument("--host",    default=DROPLET_HOST, help="Droplet SSH host")
    ap.add_argument("--account", default="acct0",      help="Account (e.g. acct0)")
    args = ap.parse_args()

    PLOT_DIR.mkdir(exist_ok=True)
    LOCAL_LOG_DIR.mkdir(exist_ok=True)

    if args.download:
        print("Downloading logs from droplet...")
        download_logs(args.host, args.account)

    csv_path = LOCAL_LOG_DIR / "training_log.csv"
    bin_path = LOCAL_LOG_DIR / "mt_training_log.bin"

    for p in (csv_path, bin_path):
        if not p.exists():
            sys.exit(f"Missing {p} — run with --download or copy manually")

    print("Parsing logs...")
    rows    = load_csv(csv_path)
    records = load_binary_log(bin_path)
    if not rows:
        sys.exit("training_log.csv is empty")
    if not records:
        sys.exit("mt_training_log.bin has no records (MT1 activates at day 25)")
    max_day = max(r["x"] for r in records)
    print(f"  CSV: {len(rows)} rows  |  Binary: {len(records)} MT1 records  |  latest x={max_day}")

    print("Generating SVGs...")
    plot_industry(rows,    PLOT_DIR / "industry_performance.svg")
    plot_mt1(records,      PLOT_DIR / "mt1_performance.svg")
    plot_mt2(rows,         PLOT_DIR / "mt2_performance.svg")
    print("Done.")

if __name__ == "__main__":
    main()
