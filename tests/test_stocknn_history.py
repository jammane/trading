"""StockNN's history input in production must be the trainer's: the 15 valid bars BEFORE the
decision day.

Until 2026-10-13 production fed a history that already ended with the decision day's bar, so every
day-over-day delta in the `today` input read zero. Measured on the v5 champions with the trainer's
slot 0 on the same day and book (fill_check's parity run): 27 of 144 symbols got the trainer's
outputs; with the history cut at the day before, 144 of 144 to float precision. The first paper
session traded on the wrong inputs because no test held the two sides to each other.
"""
import json
from pathlib import Path

import pytest

pv = pytest.importorskip("production_v2")


def _write(tmp_path, days):
    d = tmp_path / "stock_data"
    d.mkdir()
    (d / "AAA.json").write_text(json.dumps({"days": days}))
    return str(d)


def _bar(date, close):
    return dict(date=date, open=close, high=close + 1, low=close - 1, close=close, volume=100)


def test_history_ends_the_session_before_the_decision_day(tmp_path, monkeypatch):
    days = [_bar(f"2026-09-{k:02d}", 10 + k) for k in range(1, 21)] + [_bar("2026-10-13", 99)]
    monkeypatch.setattr(pv, "STOCK_DATA_DIR", _write(tmp_path, days))
    h = pv.load_stock_data(["AAA"], before="2026-10-13")["AAA"]
    assert len(h) == 15 and h[-1][1] == 30 and 99 not in [r[1] for r in h]
    assert pv.load_stock_data(["AAA"])["AAA"][-1][1] == 99       # without a cutoff: everything


def test_invalid_bars_are_dropped_before_the_last_15_are_taken(tmp_path, monkeypatch):
    """The trainer's ring skips an invalid bar, so it still holds 15 valid ones."""
    days = [_bar(f"2026-09-{k:02d}", 10 + k) for k in range(1, 21)]
    days[-2]["close"] = float("nan")
    monkeypatch.setattr(pv, "STOCK_DATA_DIR", _write(tmp_path, days))
    h = pv.load_stock_data(["AAA"], before="2026-10-13")["AAA"]
    assert len(h) == 15 and h[0][1] == 15


def test_both_stocknn_call_sites_use_the_cut_history():
    src = (Path(__file__).resolve().parent.parent / "production_v2.py").read_text()
    assert "model_hist = load_stock_data(symbols, before=session)" in src
    assert "stocknn_outputs(model, symbols, model_hist," in src
    assert "committee_decision(model_dir, industries, model_hist," in src
