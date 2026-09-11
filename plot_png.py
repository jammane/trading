#!/usr/bin/env python3
"""
Local helper (untracked): pull the heads/tails training logs from /root/ht_train on the droplet
and render the plot_training.py charts as PNGs (so they can be viewed inline). Not committed —
just a monitoring convenience for the mt1-heads-tails 1-pass run.

  python plot_png.py            # download + render to plots/pass<N>/*.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_training as pt

# The 1-pass experiment writes logs co-located in /root/ht_train (not logs/acct0/training).
pt.REMOTE_LOG = "/root/ht_train"

# Render PNG instead of SVG.
def _save_png(fig, out_path):
    p = Path(out_path).with_suffix(".png")
    fig.savefig(p, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  {p}")

pt._save = _save_png

if __name__ == "__main__":
    import sys
    sys.argv = ["plot_png.py", "--download"]
    pt.main()
