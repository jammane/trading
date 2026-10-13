"""upkeep_cpp.py and production's daily-training glue (training_v4_cpp --upkeep).

The book and the state are text formats shared with the C++ trainer, which indexes symbols by
position: a book written in the wrong symbol order would train every industry on another stock's
holdings without any error. The bayes seed must apply the trainer's exact update, or the allocator
starts from a posterior the race never had.
"""
import math

import pytest

import upkeep_cpp as u

UNI = {"ind_a": ["A1", "A2", "A3"], "ind_b": ["B1", "B2", "B3"]}


def _books():
    return {"ind_a": {"cash": 1000.0, "holdings": {"A3": 5, "A1": 2}, "stop_prices": {"A1": 9.5}},
            "ind_b": {"cash": 0.0, "holdings": {}, "stop_prices": {}}}


def test_book_writes_symbols_in_universe_order(tmp_path):
    p = tmp_path / "book.txt"
    u.write_book(str(p), "2026-10-09", _books(), {"ind_a": 2000.0, "ind_b": 1500.0}, universe=UNI)
    lines = p.read_text().splitlines()
    assert lines[:2] == ["upkeep_book 1", "date 2026-10-09"]
    a = lines[2].split()
    assert a[:6] == ["ind", "ind_a", "base", "2000.000000", "cash", "1000.000000"]
    hold = [float(x) for x in a[7:10]]
    stop = [float(x) for x in a[11:14]]
    assert hold == [2.0, 0.0, 5.0] and stop == [9.5, 0.0, 0.0]        # A1, A2, A3
    assert lines[3].split()[1] == "ind_b"


def test_book_refuses_a_missing_industry_or_a_zero_base(tmp_path):
    books = _books()
    del books["ind_b"]
    with pytest.raises(ValueError):
        u.write_book(str(tmp_path / "b"), "2026-10-09", books, {"ind_a": 1.0}, universe=UNI)
    with pytest.raises(ValueError):
        u.write_book(str(tmp_path / "b"), "2026-10-09", _books(), {"ind_a": 1.0, "ind_b": 0.0},
                     universe=UNI)


def test_book_refuses_non_finite_values(tmp_path):
    books = _books()
    books["ind_a"]["cash"] = math.nan
    with pytest.raises(ValueError):
        u.write_book(str(tmp_path / "b"), "2026-10-09", books, {"ind_a": 1.0, "ind_b": 1.0},
                     universe=UNI)


def test_state_round_trip(tmp_path):
    st = dict(date="2026-10-09", hist_count=2,
              mkt={"ind_a": [1.0, 2.0], "ind_b": [3.0, 4.0]},
              pf={"ind_a": [5.0, 6.0], "ind_b": [7.0, 8.0]},
              bayes={"ind_a": (0.001, 0.02, 49.5), "ind_b": (-0.002, 0.01, 30.0)},
              kv={"lr TANH32___grad ind_a": 0.003, "adam_step MT1______grdC": 1234.0})
    u.write_state(str(tmp_path), st, list(UNI))
    back = u.read_state(str(tmp_path))
    for k in ("date", "hist_count", "mkt", "pf", "bayes", "kv"):
        assert back[k] == st[k], k


def test_absent_state_reads_as_none(tmp_path):
    assert u.read_state(str(tmp_path)) is None


def test_bayes_replay_is_the_trainers_update():
    """w <- g w + 1; mu += (x - mu)/w; m2 <- g m2 + (x - mu_old)(x - mu_new), hand-worked."""
    g = u.BALLOC_GAMMA
    xs = [0.01, -0.02, 0.005]
    mu, m2, w = 0.0, 0.0, 0.0
    for x in xs:
        w = g * w + 1.0
        d = x - mu
        mu += d / w
        m2 = g * m2 + d * (x - mu)
    rmu, rm2, rw = u.bayes_replay([[x, 0.0] for x in xs], [[True, False]] * 3)
    assert math.isclose(rmu[0], mu) and math.isclose(rm2[0], m2) and math.isclose(rw[0], w)
    assert (rmu[1], rm2[1], rw[1]) == (0.0, 0.0, 0.0)           # never observed: untouched


def test_bayes_mean_is_the_discounted_mean():
    """With gamma weights the posterior mean is sum g^k x_{t-k} / sum g^k."""
    xs = [0.003 * ((-1) ** k) + 0.001 * k for k in range(60)]
    mu, _, w = u.bayes_replay([[x] for x in xs], [[True]] * 60)
    wts = [u.BALLOC_GAMMA ** (59 - k) for k in range(60)]
    assert math.isclose(mu[0], sum(a * b for a, b in zip(wts, xs, strict=True)) / sum(wts))
    assert math.isclose(w[0], sum(wts))


def test_seed_state_refuses_to_overwrite(tmp_path):
    u.write_state(str(tmp_path), dict(date="2026-10-09"), list(UNI))
    with pytest.raises(FileExistsError):
        u.seed_state(str(tmp_path), "unused.csv", "unused.csv")


def test_run_upkeep_passes_the_committee_and_the_race_settings(tmp_path, monkeypatch):
    seen = {}

    class R:
        returncode = 3

    def fake_run(cmd, stdout, stderr, timeout):
        seen["cmd"] = cmd
        return R()
    monkeypatch.setattr(u.subprocess, "run", fake_run)
    rc = u.run_upkeep("bin", "book", "state", "seed", ["TANH32 grad", "MT1 evo"],
                      str(tmp_path / "log" / "u.log"))
    assert rc == 3
    c = seen["cmd"]
    assert c[:7] == ["bin", "--upkeep", "book", "--output", "state", "--load-dir", "seed"]
    assert c[c.index("--race-only") + 1] == "TANH32 grad,MT1 evo"
    assert c[c.index("--sigma") + 1] == "0.008" and c[c.index("--master-sigma") + 1] == "0.006"


class _Pos:
    def __init__(self, symbol, qty):
        self.symbol, self.qty = symbol, qty


class _Client:
    def __init__(self, pos):
        self.pos = pos

    def get_all_positions(self):
        return [_Pos(s, q) for s, q in self.pos.items()]


def test_production_records_the_real_book_for_tomorrow(tmp_path, monkeypatch):
    pv = pytest.importorskip("production_v2")
    monkeypatch.setattr(u, "universe_order", lambda path="universe.json": UNI)
    orders = [{"symbol": "A1", "action": "stop_loss", "quantity": 1, "price": 9.0},
              {"symbol": "A1", "action": "stop_loss", "quantity": 1, "price": 9.5},
              {"symbol": "B1", "action": "buy", "quantity": 1, "price": 20.0}]
    day = {s: {"close": 10.0} for syms in UNI.values() for s in syms}
    pv.save_upkeep_book(str(tmp_path), _Client({"A1": 3, "B2": 4}), UNI,
                        {"ind_a": 500.0, "ind_b": 700.0}, {"ind_b"}, orders, day)
    st = (tmp_path / "cpp" / "book_next.txt").read_text().splitlines()
    a, b = st[2].split(), st[3].split()
    assert a[5] == "500.000000" and [float(x) for x in a[7:10]] == [3.0, 0.0, 0.0]
    assert float(a[11]) == 9.5                                    # the higher of two stops
    assert b[5] == "0.000000"                                     # sell-only: no cash to buy
    assert [float(x) for x in b[7:10]] == [0.0, 4.0, 0.0]
    # base is fixed on the first write: 500 + 3x10 for ind_a, 0 + 4x10 for ind_b
    import json
    base = json.loads((tmp_path / "cpp" / "book_base.json").read_text())
    assert base == {"ind_a": 530.0, "ind_b": 40.0}
    pv.save_upkeep_book(str(tmp_path), _Client({}), UNI, {"ind_a": 1.0, "ind_b": 1.0}, set(),
                        [], day)
    assert json.loads((tmp_path / "cpp" / "book_base.json").read_text()) == base
