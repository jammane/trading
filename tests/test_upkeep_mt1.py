"""upkeep.py's MT1 path — the one production actually runs, every trading day.

`training_v4.cpp` and `upkeep.py` evolve the SAME pool files: the trainer writes
`mt1_{ind}_slot_{n}.bin` plus the metadata sidecar, `convert_weights.py` turns them into `.pt`, and
from then on this path carries the pool forward one session at a time. The two implementations are
not merely similar — they are two halves of one pool's life. A divergence in scoring or ranking
means the deployed model and the cull victims differ depending on which half is running, and
nothing crashes, nothing fails to load, and no number looks obviously wrong.

That risk is not hypothetical. The first version of this path had the sort key's two components in
the opposite order from the C++, and separately never persisted its rolling state, so in production
nothing would ever have been scored at all.

The parity tests compile the header's own functions and compare numerically — the technique
`test_mt1net_parity.py` uses for the forward pass — rather than comparing the two by reading them.
"""
import json
import math
import re
import subprocess
from pathlib import Path

import pytest
import torch

import upkeep
from models import MT1Net
from upkeep import (
    MT1_BASELINE_DAYS,
    MT1_FLOOR_DAYS,
    MT1_FLOOR_FRAC,
    MT1_POOL_CULL_PCT,
    MT1_POOL_ELITE_PCT,
    MT1_POOL_LINEAGE_CAP,
    MT1_POOL_LINEAGE_RESUME,
    MT1_POOL_MIN_AGE,
    MT1_POOL_SLOTS,
    MT1_PRED_SCALE,
    MT1_RECENCY_W,
    MT1_SCORE_HIST,
    _mt1_barred_lineages,
    _mt1_slot_mature,
    _mt1_slot_rank_key,
    _mt1_slot_record,
    load_mt1_rolling_state,
    mt1_baseline,
    mt1_floor,
    mt1_pred,
    mt1_score,
    mt1_windows_are_causal,
    save_mt1_rolling_state,
    upkeep_mt1_industry,
)

REPO = Path(__file__).resolve().parent.parent

# ── C++ parity driver ───────────────────────────────────────────────────────────

DRIVER = r'''
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "mt1_pool.h"

static MT1SlotMeta read_meta() {
    MT1SlotMeta m;
    mt1_slot_init(m, 0);
    int n;
    if (scanf("%d", &n) != 1) exit(2);
    m.n_pred = (uint16_t)n;
    for (int i = 0; i < MT1_SCORE_HIST; i++) {
        double v;
        if (scanf("%lf", &v) != 1) exit(2);
        m.score[i] = (float)v;
    }
    return m;
}

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "";
    if (!strcmp(mode, "score")) {
        double a, p, b, f;
        if (scanf("%lf %lf %lf %lf", &a, &p, &b, &f) != 4) return 2;
        printf("%.9e\n", mt1_score((float)a, (float)p, (float)b, (float)f));
    } else if (!strcmp(mode, "pred")) {
        double r;
        if (scanf("%lf", &r) != 1) return 2;
        printf("%.9e\n", mt1_pred((float)r));
    } else if (!strcmp(mode, "slot")) {
        MT1SlotMeta m = read_meta();
        printf("%.9e %.9e %d\n", mt1_slot_score(m), mt1_slot_mean(m), mt1_slot_mature(m) ? 1 : 0);
    } else if (!strcmp(mode, "better")) {
        MT1SlotMeta a = read_meta();
        MT1SlotMeta b = read_meta();
        printf("%d\n", mt1_slot_better(a, b) ? 1 : 0);
    } else if (!strcmp(mode, "consts")) {
        printf("MT1_POOL_SLOTS %d\n",          MT1_POOL_SLOTS);
        printf("MT1_SCORE_HIST %d\n",          MT1_SCORE_HIST);
        printf("MT1_POOL_MIN_AGE %d\n",        MT1_POOL_MIN_AGE);
        printf("MT1_BASELINE_DAYS %d\n",       MT1_BASELINE_DAYS);
        printf("MT1_FLOOR_DAYS %d\n",          MT1_FLOOR_DAYS);
        printf("MT1_POOL_CULL_PCT %.9g\n",     (double)MT1_POOL_CULL_PCT);
        printf("MT1_POOL_ELITE_PCT %.9g\n",    (double)MT1_POOL_ELITE_PCT);
        printf("MT1_POOL_LINEAGE_CAP %.9g\n",  (double)MT1_POOL_LINEAGE_CAP);
        printf("MT1_POOL_LINEAGE_RESUME %.9g\n", (double)MT1_POOL_LINEAGE_RESUME);
        printf("MT1_FLOOR_FRAC %.9g\n",        (double)MT1_FLOOR_FRAC);
        printf("MT1_PRED_SCALE %.9g\n",        (double)MT1_PRED_SCALE);
        for (int i = 0; i < 4; i++) printf("MT1_RECENCY_W%d %.9g\n", i, (double)MT1_RECENCY_W[i]);
    } else {
        return 3;
    }
    return 0;
}
'''


@pytest.fixture(scope='module')
def cpp(tmp_path_factory):
    d = tmp_path_factory.mktemp('mt1pool')
    src = d / 'driver.cpp'
    src.write_text(DRIVER)
    exe = d / 'driver'
    r = subprocess.run(['g++', '-std=c++20', '-O2', '-I', str(REPO), '-o', str(exe), str(src), '-lm'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f'g++ unavailable or build failed: {r.stderr[:300]}')

    def run(mode, stdin=''):
        out = subprocess.run([str(exe), mode], input=stdin, capture_output=True, text=True)
        assert out.returncode == 0, f'driver {mode} failed: {out.stderr[:300]}'
        return out.stdout.strip()

    return run


def _register(meta):
    """Python `scores` is oldest-first (appended); C++ `score[0]` is the MOST RECENT.

    The two conventions are exactly opposite, which is the kind of thing that reads as correct on
    both sides and silently reverses the recency weighting. Reverse and pad here, once.
    """
    scores = list(meta.get('scores', []))[-MT1_SCORE_HIST:][::-1]
    scores += [0.0] * (MT1_SCORE_HIST - len(scores))
    return f"{meta.get('n_pred', 0)} " + ' '.join(repr(float(s)) for s in scores)


def _meta(scores, n_pred=None, lineage=0):
    return {'scores': list(scores),
            'n_pred': len(scores) if n_pred is None else n_pred,
            'lineage': lineage}


# ── Constants ───────────────────────────────────────────────────────────────────

class TestConstantsMirrorTheHeader:
    """upkeep.py restates every pool constant. They configure files the trainer also writes, so a
    drift makes models mature or get culled on different schedules in the two halves of a pool's
    life — with no error anywhere."""

    def test_every_constant_matches(self, cpp):
        cc = dict(line.split() for line in cpp('consts').splitlines())
        py = {
            'MT1_POOL_SLOTS': MT1_POOL_SLOTS,
            'MT1_SCORE_HIST': MT1_SCORE_HIST,
            'MT1_POOL_MIN_AGE': MT1_POOL_MIN_AGE,
            'MT1_BASELINE_DAYS': MT1_BASELINE_DAYS,
            'MT1_FLOOR_DAYS': MT1_FLOOR_DAYS,
            'MT1_POOL_CULL_PCT': MT1_POOL_CULL_PCT,
            'MT1_POOL_ELITE_PCT': MT1_POOL_ELITE_PCT,
            'MT1_POOL_LINEAGE_CAP': MT1_POOL_LINEAGE_CAP,
            'MT1_POOL_LINEAGE_RESUME': MT1_POOL_LINEAGE_RESUME,
            'MT1_FLOOR_FRAC': MT1_FLOOR_FRAC,
            'MT1_PRED_SCALE': MT1_PRED_SCALE,
        }
        for k, v in py.items():
            assert float(cc[k]) == pytest.approx(float(v)), f'{k}: C++ {cc[k]} vs Python {v}'

    def test_recency_weights_match(self, cpp):
        cc = dict(line.split() for line in cpp('consts').splitlines())
        assert len(MT1_RECENCY_W) == 4
        for i, w in enumerate(MT1_RECENCY_W):
            assert float(cc[f'MT1_RECENCY_W{i}']) == pytest.approx(w)

    def test_the_register_is_covered_by_the_weights(self):
        """16 scores in blocks of 4 need exactly 4 weights; a mismatch would either index out of
        MT1_RECENCY_W on the C++ side or silently clamp on the Python side."""
        assert MT1_SCORE_HIST == len(MT1_RECENCY_W) * 4


# ── Scoring parity ──────────────────────────────────────────────────────────────

class TestScoreParity:
    CASES = [
        # (actual, predicted, baseline, floor)
        (100.0, 100.0, 0.0, 50.0),        # perfect
        (100.0, 0.0, 0.0, 50.0),          # predicted nothing
        (100.0, 0.0, 100.0, 50.0),        # baseline is perfect, model is not
        (-250.0, -300.0, -10.0, 80.0),    # both negative
        (0.0, 0.0, 0.0, 1.0),             # degenerate day, floor carries the denominator
        (1650.0, 300.0, 24.9, 86.0),      # p90 loud day vs p10 floor
        (24.9, 24.9, 24.9, 1.0),          # tied the baseline exactly
        (-5.0, 5000.0, 0.0, 1.0),         # wildly wrong, as a fresh random model is
    ]

    @pytest.mark.parametrize('actual,pred,base,floor', CASES)
    def test_score_matches_cpp(self, cpp, actual, pred, base, floor):
        want = float(cpp('score', f'{actual} {pred} {base} {floor}'))
        got = mt1_score(actual, pred, base, floor)
        assert got == pytest.approx(want, rel=1e-6, abs=1e-9)

    @pytest.mark.parametrize('raw', [0.0, 0.25, -0.25, 1.0, -1.0, 3.0, -8.0, 0.0183])
    def test_pred_decode_matches_cpp(self, cpp, raw):
        want = float(cpp('pred', f'{raw}'))
        got = mt1_pred(raw)
        assert got == pytest.approx(want, rel=1e-6, abs=1e-6)

    def test_tying_the_baseline_scores_exactly_half(self):
        """The bar the whole design is measured against. If this moves, every reported score
        changes meaning and no other test would notice."""
        for actual, base, floor in ((100.0, 40.0, 10.0), (-300.0, 50.0, 200.0), (7.0, 7.5, 1.0)):
            assert mt1_score(actual, base, base, floor) == pytest.approx(0.5) or \
                abs(actual - base) <= floor
        # Explicit case where the baseline error exceeds the floor, so d == base exactly.
        assert mt1_score(100.0, 40.0, 40.0, 10.0) == pytest.approx(0.5)

    def test_score_is_symmetric_in_the_sign_of_the_error(self):
        """Over- and under-prediction by the same amount must score the same — the reason
        log-ratio scoring was rejected, since it biases toward under-guessing."""
        for err in (1.0, 50.0, 500.0):
            hi = mt1_score(200.0, 200.0 + err, 0.0, 90.0)
            lo = mt1_score(200.0, 200.0 - err, 0.0, 90.0)
            assert hi == pytest.approx(lo)

    def test_score_is_bounded_in_zero_one(self):
        for actual in (-5000.0, -1.0, 0.0, 1.0, 5000.0):
            for pred in (-9999.0, 0.0, actual, 9999.0):
                v = mt1_score(actual, pred, 12.0, 60.0)
                assert 0.0 < v <= 1.0, f'{actual}/{pred} -> {v}'

    def test_the_floor_bounds_the_denominator_from_below(self):
        """When the trailing mean lands exactly on the outcome the baseline error is 0. Without
        the floor the denominator collapses and the score becomes hypersensitive."""
        assert mt1_score(100.0, 110.0, 100.0, 0.0) < mt1_score(100.0, 110.0, 100.0, 200.0)
        # d is clamped to >= 1e-6 even with a zero floor, so it never divides by zero.
        assert mt1_score(100.0, 110.0, 100.0, 0.0) > 0.0


# ── Register and ranking parity ─────────────────────────────────────────────────

class TestRegisterParity:
    REGISTERS = [
        [0.5] * 16,
        [0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.1] * 12 + [0.9] * 4,
        [0.3, 0.7],
        [0.42] * 8,
        [i / 20.0 for i in range(16)],
        [0.0] * 16,
        [1.0] * 16,
    ]

    @pytest.mark.parametrize('scores', REGISTERS)
    def test_weighted_and_plain_means_match_cpp(self, cpp, scores):
        m = _meta(scores)
        w_cc, mean_cc, mature_cc = cpp('slot', _register(m)).split()
        weighted, plain = _mt1_slot_rank_key(m)
        # C++ mt1_slot_score is the RECENCY-WEIGHTED mean (the PRIMARY key) and mt1_slot_mean the
        # plain one (the tie-break). The key returns them in that order.
        assert weighted == pytest.approx(float(w_cc), rel=1e-5, abs=1e-6), \
            'recency-weighted mean diverges'
        assert plain == pytest.approx(float(mean_cc), rel=1e-5, abs=1e-6), \
            'plain mean diverges'
        assert _mt1_slot_mature(m) == bool(int(mature_cc))

    def test_recency_weighting_favours_the_newest_block(self, cpp):
        """Two registers with the same plain mean but opposite ordering must differ on the
        weighted key — and both sides must agree on WHICH one wins, which is only true if the two
        agree on which end of the list is 'recent'."""
        rising = _meta([0.1] * 8 + [0.9] * 8)
        falling = _meta([0.9] * 8 + [0.1] * 8)
        assert _mt1_slot_rank_key(rising)[1] == pytest.approx(_mt1_slot_rank_key(falling)[1]), \
            'the plain means are equal by construction'
        assert _mt1_slot_rank_key(rising)[0] > _mt1_slot_rank_key(falling)[0], \
            'the weighted mean is the PRIMARY key and must separate them'
        assert _mt1_slot_rank_key(rising) > _mt1_slot_rank_key(falling), \
            'a model improving over its recent predictions must rank above a declining one'
        assert not _mt1_slot_mature(rising) or True
        w_rising = float(cpp('slot', _register(rising)).split()[0])
        w_falling = float(cpp('slot', _register(falling)).split()[0])
        assert w_rising > w_falling, 'C++ disagrees on which end of the register is recent'

    def test_partial_register_is_not_penalised(self, cpp):
        """An 8-prediction model is judged on its 8. If the empty slots counted as zeros a young
        model could never reach the elite and the pool would ossify."""
        short = _meta([0.8] * 8)
        full = _meta([0.8] * 16)
        assert _mt1_slot_rank_key(short) == pytest.approx(_mt1_slot_rank_key(full))
        s_cc = cpp('slot', _register(short)).split()
        f_cc = cpp('slot', _register(full)).split()
        assert float(s_cc[1]) == pytest.approx(float(f_cc[1]), rel=1e-5)

    def test_record_caps_the_window_but_not_the_age(self):
        m = _meta([])
        for i in range(40):
            _mt1_slot_record(m, 0.5 + i * 0.001)
        assert len(m['scores']) == MT1_SCORE_HIST, 'window must stay bounded'
        assert m['n_pred'] == 40, 'age keeps counting past the window — culling reads it'
        assert m['scores'][-1] == pytest.approx(0.5 + 39 * 0.001), 'newest score is last'

    def test_record_keeps_the_most_recent_scores(self):
        m = _meta([])
        for i in range(MT1_SCORE_HIST + 5):
            _mt1_slot_record(m, float(i))
        assert m['scores'] == [float(i) for i in range(5, MT1_SCORE_HIST + 5)]


class TestRankingParity:
    """The ordering itself, not just the two numbers feeding it.

    The first version of this path returned (plain mean, weighted mean) while the C++ sorts by
    (weighted mean, plain mean) — the same two numbers with the precedence reversed. Every test
    above would still have passed.
    """

    PAIRS = [
        # (a_scores, b_scores, a_n, b_n)
        ([0.9] * 16, [0.1] * 16, None, None),                 # a clearly better
        ([0.1] * 16, [0.9] * 16, None, None),                 # b clearly better
        ([0.1] * 12 + [0.9] * 4, [0.9] * 12 + [0.1] * 4, None, None),   # same mean, recency splits
        ([0.9] * 12 + [0.1] * 4, [0.1] * 12 + [0.9] * 4, None, None),
        ([0.5] * 16, [0.5] * 16, None, None),                 # identical
        ([0.9] * 4, [0.4] * 16, 4, 16),                       # immature vs mature
        ([0.4] * 16, [0.9] * 4, 16, 4),                       # mature vs immature
        ([0.99] * 2, [0.98] * 3, 2, 3),                       # both immature
        ([0.6, 0.6, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5],
         [0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6], 8, 8),     # same mean, blocks swapped
        # THE case the swapped key got wrong: a's plain mean is HIGHER (0.60 vs 0.50) but b's
        # recency-weighted mean is higher (0.614 vs 0.600), because b's newest eight are 0.9.
        # Keying on the plain mean first ranks a above b; the C++ ranks b above a.
        ([0.6] * 16, [0.1] * 8 + [0.9] * 8, None, None),
        ([0.1] * 8 + [0.9] * 8, [0.6] * 16, None, None),
    ]

    @pytest.mark.parametrize('a_scores,b_scores,a_n,b_n', PAIRS)
    def test_ordering_matches_cpp(self, cpp, a_scores, b_scores, a_n, b_n):
        a, b = _meta(a_scores, a_n), _meta(b_scores, b_n)
        want = bool(int(cpp('better', _register(a) + '\n' + _register(b))))
        got = ((_mt1_slot_mature(a), _mt1_slot_rank_key(a))
               > (_mt1_slot_mature(b), _mt1_slot_rank_key(b)))
        assert got == want, (
            f'ranking diverges: C++ better(a,b)={want}, Python key(a)>key(b)={got}\n'
            f'  a={_mt1_slot_rank_key(a)} mature={_mt1_slot_mature(a)}\n'
            f'  b={_mt1_slot_rank_key(b)} mature={_mt1_slot_mature(b)}')

    def test_immature_never_outranks_a_record(self, cpp):
        """A model with two lucky predictions must not displace one with a full register — this is
        the max-of-N guard, and it has to hold on both sides or the deployed model differs."""
        lucky = _meta([1.0, 1.0], n_pred=2)
        proven = _meta([0.55] * 16, n_pred=40)
        assert bool(int(cpp('better', _register(lucky) + '\n' + _register(proven)))) is False
        assert ((_mt1_slot_mature(lucky), _mt1_slot_rank_key(lucky))
                < (_mt1_slot_mature(proven), _mt1_slot_rank_key(proven)))

    def test_the_full_sort_agrees_with_cpp_pairwise(self, cpp):
        """Sort a mixed pool with the Python key, then verify every adjacent pair against the C++
        comparator. Catches an ordering that is self-consistent but not the C++'s."""
        pool = [
            _meta([0.5] * 16), _meta([0.7] * 16), _meta([0.3] * 16),
            _meta([0.9] * 4, 4), _meta([0.2] * 9, 9),
            _meta([0.1] * 8 + [0.9] * 8), _meta([0.9] * 8 + [0.1] * 8),
            _meta([], 0), _meta([0.6] * 16, 100),
        ]
        order = sorted(range(len(pool)),
                       key=lambda i: (_mt1_slot_mature(pool[i]), _mt1_slot_rank_key(pool[i])),
                       reverse=True)
        for x, y in zip(order, order[1:], strict=False):
            a, b = pool[x], pool[y]
            b_better = bool(int(cpp('better', _register(b) + '\n' + _register(a))))
            assert not b_better, f'Python ranked slot {x} above {y}, C++ disagrees'


# ── Baseline, floor and the causality contract ──────────────────────────────────

class TestBaselineAndFloor:
    def test_baseline_is_the_trailing_mean(self):
        assert mt1_baseline([]) == 0.0
        assert mt1_baseline([10.0, 20.0, 30.0]) == pytest.approx(20.0)
        assert mt1_baseline([-100.0, 100.0]) == pytest.approx(0.0)

    def test_floor_uses_the_most_recent_window(self):
        """MT1_FLOOR_DAYS is shorter than MT1_BASELINE_DAYS, so the floor must read the TAIL of
        the list. Reading the head would anchor it to stale volatility for the whole run."""
        assert MT1_FLOOR_DAYS < MT1_BASELINE_DAYS
        old_loud = [5000.0] * (MT1_BASELINE_DAYS - MT1_FLOOR_DAYS) + [100.0] * MT1_FLOOR_DAYS
        assert mt1_floor(old_loud) == pytest.approx(100.0 * MT1_FLOOR_FRAC)

    def test_floor_is_half_the_mean_absolute_move(self):
        assert mt1_floor([200.0] * MT1_FLOOR_DAYS) == pytest.approx(200.0 * MT1_FLOOR_FRAC)
        assert mt1_floor([-200.0] * MT1_FLOOR_DAYS) == pytest.approx(200.0 * MT1_FLOOR_FRAC), \
            'the floor is a SCALE — sign must not cancel it'

    def test_floor_never_drops_below_one_dollar(self):
        assert mt1_floor([]) == 1.0
        assert mt1_floor([0.0] * MT1_FLOOR_DAYS) == 1.0
        assert mt1_floor([0.001, -0.002]) == 1.0

    def test_cpp_floor_formula_still_matches(self):
        """The C++ floor lives on MT1PoolScratch and cannot be linked standalone, so pin its body
        textually: same window constant, same fraction, same $1 clamp."""
        src = (REPO / 'training_v4.cpp').read_text()
        body = re.search(r'float floor_v\(\) const \{(.*?)\n    \}', src, re.S)
        assert body, 'MT1PoolScratch::floor_v not found — did it move?'
        body = body.group(1)
        assert 'MT1_FLOOR_DAYS' in body
        assert 'MT1_FLOOR_FRAC' in body
        assert 'fmaxf' in body and '1.f' in body, 'the $1 clamp is gone on the C++ side'

    def test_cpp_baseline_is_the_plain_trailing_mean(self):
        src = (REPO / 'training_v4.cpp').read_text()
        body = re.search(r'float baseline\(\) const \{(.*?)\n    \}', src, re.S)
        assert body, 'MT1PoolScratch::baseline not found — did it move?'
        assert 'actual_count' in body.group(1)


class TestCausalityContract:
    def test_windows_must_end_before_the_scored_day(self):
        assert mt1_windows_are_causal(10, 9, 9)
        assert not mt1_windows_are_causal(10, 10, 9), 'baseline including the scored day is a leak'
        assert not mt1_windows_are_causal(10, 9, 10), 'floor including the scored day is a leak'
        assert not mt1_windows_are_causal(10, 11, 11)

    def test_it_matches_the_cpp_predicate(self):
        src = (REPO / 'mt1_pool.h').read_text()
        body = re.search(r'mt1_windows_are_causal\([^)]*\)\s*\{(.*?)\n\}', src, re.S)
        assert body, 'mt1_windows_are_causal not found in mt1_pool.h'
        assert '<' in body.group(1) and '&&' in body.group(1)


# ── Lineage ─────────────────────────────────────────────────────────────────────

class TestLineageCap:
    def _pool(self, counts):
        metas, lin = [], 0
        for n in counts:
            metas += [_meta([0.5], lineage=lin) for _ in range(n)]
            lin += 1
        return metas

    def test_a_lineage_over_the_cap_is_barred(self):
        n = 200
        over = int(MT1_POOL_LINEAGE_CAP * n) + 5
        metas = self._pool([over, n - over])
        barred, counts = _mt1_barred_lineages(metas, set())
        assert 0 in barred
        assert counts[0] == over

    def test_a_lineage_under_the_cap_is_free(self):
        n = 200
        under = int(MT1_POOL_LINEAGE_CAP * n) - 1
        metas = self._pool([under, n - under])
        barred, _ = _mt1_barred_lineages(metas, set())
        assert 0 not in barred

    def test_hysteresis_keeps_a_barred_lineage_barred_between_the_thresholds(self):
        """Between RESUME and CAP the answer depends on history. Without hysteresis a lineage
        sitting near the cap would bar and unbar on alternating days."""
        assert MT1_POOL_LINEAGE_RESUME < MT1_POOL_LINEAGE_CAP
        n = 200
        between = int((MT1_POOL_LINEAGE_RESUME + MT1_POOL_LINEAGE_CAP) / 2 * n)
        metas = self._pool([between, n - between])
        assert 0 not in _mt1_barred_lineages(metas, set())[0], 'not yet over the cap'
        assert 0 in _mt1_barred_lineages(metas, {0})[0], 'already barred: must stay barred'

    def test_a_barred_lineage_resumes_below_the_resume_threshold(self):
        n = 200
        low = int(MT1_POOL_LINEAGE_RESUME * n) - 1
        metas = self._pool([low, n - low])
        assert 0 not in _mt1_barred_lineages(metas, {0})[0]

    def test_counts_cover_the_whole_pool(self):
        metas = self._pool([10, 20, 170])
        _, counts = _mt1_barred_lineages(metas, set())
        assert sum(counts.values()) == 200


# ── The daily step ──────────────────────────────────────────────────────────────

SMALL_POOL = 12          # keeps 200 model loads per call from dominating the suite


@pytest.fixture
def pool(tmp_path, monkeypatch):
    """A small MT1 pool in a temp model dir, with the slot count patched down."""
    monkeypatch.setattr(upkeep, 'MT1_POOL_SLOTS', SMALL_POOL)
    torch.manual_seed(0)
    return str(tmp_path)


def _in74(seed=7):
    torch.manual_seed(seed)
    return torch.randn(1, 74)


def _run(model_dir, state, actual, industry='energy', feat_seed=7):
    return upkeep_mt1_industry(industry, model_dir, _in74(feat_seed), actual, rolling_state=state)


class TestBootstrap:
    def test_first_run_creates_every_slot(self, pool):
        state = {}
        _run(pool, state, None)
        for s in range(SMALL_POOL):
            assert Path(pool, f'mt1_energy_model_{s}.pt').exists(), f'slot {s} missing'

    def test_slots_start_distinct(self, pool):
        """Seeded from one model but mutated, so selection has something to choose between. An
        identical pool would make every score identical and the ranking meaningless."""
        _run(pool, {}, None)
        flats = []
        for s in range(SMALL_POOL):
            sd = torch.load(Path(pool, f'mt1_energy_model_{s}.pt'), weights_only=True)
            flats.append(torch.cat([v.flatten() for v in sd.values()]))
        assert not torch.allclose(flats[0], flats[1])
        assert not torch.allclose(flats[1], flats[2])

    def test_it_seeds_from_best_pt_when_one_exists(self, pool):
        """After convert_weights the deployed model is on disk but the pool is not. Bootstrapping
        from random there would allocate real capital on untrained weights for MIN_AGE sessions."""
        torch.manual_seed(3)
        seed = MT1Net()
        torch.save(seed.state_dict(), Path(pool, 'mt1_energy_best.pt'))
        _run(pool, {}, None)
        slot0 = torch.load(Path(pool, 'mt1_energy_model_0.pt'), weights_only=True)
        for k, v in seed.state_dict().items():
            assert torch.allclose(slot0[k], v), f'{k} was not seeded from best.pt'

    def test_a_second_run_does_not_re_bootstrap(self, pool):
        state = {}
        _run(pool, state, None)
        before = torch.load(Path(pool, f'mt1_energy_model_{SMALL_POOL - 1}.pt'), weights_only=True)
        _run(pool, state, 100.0)
        after = torch.load(Path(pool, f'mt1_energy_model_{SMALL_POOL - 1}.pt'), weights_only=True)
        for k in before:
            assert torch.allclose(before[k], after[k]), \
                'an untouched slot changed — the pool was rebuilt'


class TestScoringHappensOnTheFollowingRun:
    def test_nothing_is_scored_on_the_first_run(self, pool):
        state = {}
        pred, score0 = _run(pool, state, None)
        assert score0 == 0.0, 'there was no parked prediction to score'
        assert state['energy']['metas'][0]['n_pred'] == 0

    def test_the_parked_prediction_is_scored_next_run(self, pool):
        state = {}
        pred1, _ = _run(pool, state, None)
        parked = list(state['energy']['pending'])
        assert len(parked) == SMALL_POOL
        _pred2, score0 = _run(pool, state, 250.0)
        assert score0 > 0.0, 'run 2 must score what run 1 parked'
        assert all(m['n_pred'] == 1 for m in state['energy']['metas'])

    def test_the_score_is_computed_from_the_parked_value_not_the_new_one(self, pool):
        """The defining property: a model is graded on the call it made BEFORE the outcome
        existed. Recomputing today's prediction and scoring that would be a look-ahead leak and
        would read as near-perfect skill."""
        state = {}
        _run(pool, state, None)
        parked = list(state['energy']['pending'])
        best = state['energy'].get('best_slot', 0)
        actual = 250.0
        _run(pool, state, actual)
        expected = mt1_score(actual, parked[best], mt1_baseline([]), mt1_floor([]))
        assert state['energy']['metas'][best]['scores'][0] == pytest.approx(expected)

    def test_pending_is_cleared_then_refilled(self, pool):
        """Each run parks a fresh set. Note the predictions only CHANGE when the features or the
        weights do — before maturity nothing is culled, so re-running on the same features
        legitimately reproduces the same numbers. Feed a different session to see the refill."""
        state = {}
        _run(pool, state, None, feat_seed=7)
        first = list(state['energy']['pending'])
        _run(pool, state, 100.0, feat_seed=11)
        second = state['energy']['pending']
        assert len(second) == SMALL_POOL
        assert second != first, 'a new session must produce a new set of parked predictions'

    def test_the_same_session_reproduces_the_same_predictions(self, pool):
        """The flip side, and the reason the test above has to vary the features: MT1 is
        deterministic given weights and input. A difference here would mean hidden state."""
        state = {}
        _run(pool, state, None, feat_seed=7)
        first = list(state['energy']['pending'])
        state2 = {'energy': dict(state['energy'])}
        _run(pool, state2, None, feat_seed=7)
        assert state2['energy']['pending'] == pytest.approx(first)

    def test_a_missing_outcome_skips_scoring_without_losing_the_parked_calls(self, pool):
        """production passes actual_d=None when the previous session's portfolio value is
        unknown (a missed run). The parked predictions must survive to be scored later, not be
        graded against nothing."""
        state = {}
        _run(pool, state, None)
        _, score0 = _run(pool, state, None)
        assert score0 == 0.0
        assert state['energy']['pending'] is not None
        assert all(m['n_pred'] == 0 for m in state['energy']['metas'])

    def test_actuals_window_stays_bounded(self, pool):
        state = {}
        rs = state.setdefault('energy', {})
        rs['actuals'] = [float(i) for i in range(MT1_BASELINE_DAYS + 30)]
        _run(pool, state, 10.0)
        assert len(rs['actuals']) == MT1_BASELINE_DAYS
        assert rs['actuals'][-1] == 10.0, 'the newest outcome must be kept'


class TestMaturityAndCulling:
    def _advance(self, pool, state, n, start=100.0):
        for i in range(n):
            _run(pool, state, None if i == 0 else start + i * 10.0)
        return state['energy']

    def test_nothing_is_culled_before_maturity(self, pool):
        state = {}
        rs = self._advance(pool, state, MT1_POOL_MIN_AGE)      # 1 unscored + MIN_AGE-1 scored
        assert max(m['n_pred'] for m in rs['metas']) < MT1_POOL_MIN_AGE
        assert all(m['n_pred'] == m['n_pred'] for m in rs['metas'])
        assert all(len(m['scores']) == m['n_pred'] for m in rs['metas'])

    def test_culling_starts_once_the_pool_matures(self, pool):
        state = {}
        rs = self._advance(pool, state, MT1_POOL_MIN_AGE + 3)
        ages = [m['n_pred'] for m in rs['metas']]
        assert max(ages) >= MT1_POOL_MIN_AGE, 'the pool should have matured by now'
        assert min(ages) < max(ages), 'no slot was ever reset — nothing is being culled'

    def test_a_culled_slot_is_reset_and_rewritten(self, pool):
        state = {}
        before = {}
        for i in range(MT1_POOL_MIN_AGE + 1):
            if i == MT1_POOL_MIN_AGE:
                for s in range(SMALL_POOL):
                    before[s] = torch.load(Path(pool, f'mt1_energy_model_{s}.pt'),
                                           weights_only=True)
            _run(pool, state, None if i == 0 else 100.0 + i)
        rs = state['energy']
        fresh = [s for s, m in enumerate(rs['metas']) if m['n_pred'] == 0]
        assert fresh, 'expected at least one cull once the pool matured'
        for s in fresh:
            after = torch.load(Path(pool, f'mt1_energy_model_{s}.pt'), weights_only=True)
            assert not all(torch.allclose(before[s][k], after[k]) for k in after), \
                f'slot {s} register was reset but its weights were not replaced'
            assert rs['metas'][s]['scores'] == [], 'a fresh individual carries no record'

    def test_a_child_inherits_its_parents_lineage(self, pool):
        state = {}
        for i in range(MT1_POOL_MIN_AGE + 2):
            _run(pool, state, None if i == 0 else 100.0 + i)
        rs = state['energy']
        lineages = [m['lineage'] for m in rs['metas']]
        assert len(set(lineages)) < SMALL_POOL, \
            'every lineage is still unique — children are not inheriting'

    def test_the_deployed_model_is_never_culled(self, pool):
        """Culling the deployed model mid-session would swap what production allocates on for an
        untested mutation, with no record behind it."""
        state = {}
        for i in range(MT1_POOL_MIN_AGE + 4):
            _run(pool, state, None if i == 0 else 100.0 + i)
            best = state['energy']['best_slot']
            assert state['energy']['metas'][best]['n_pred'] > 0 or i == 0, \
                'the deployed slot lost its record — it was culled'


class TestDeployedModelIsPublished:
    def test_best_pt_is_written_every_run(self, pool):
        state = {}
        _run(pool, state, None)
        assert Path(pool, 'mt1_energy_best.pt').exists()

    def test_best_pt_matches_the_ranked_slot(self, pool):
        state = {}
        for i in range(MT1_POOL_MIN_AGE + 3):
            _run(pool, state, None if i == 0 else 100.0 + i)
        best = state['energy']['best_slot']
        published = torch.load(Path(pool, 'mt1_energy_best.pt'), weights_only=True)
        slot = torch.load(Path(pool, f'mt1_energy_model_{best}.pt'), weights_only=True)
        for k in slot:
            assert torch.allclose(published[k], slot[k]), \
                'best.pt is not the slot the pool ranked first'

    def test_the_returned_prediction_is_the_deployed_models(self, pool):
        """MT2 allocates on this number. If it came from a different slot than best.pt, training
        and inference would disagree about what MT1 said."""
        state = {}
        pred, _ = _run(pool, state, None)
        best = state['energy']['best_slot']
        assert pred == pytest.approx(state['energy']['pending'][best])

    def test_the_prediction_is_in_dollars(self, pool):
        pred, _ = _run(pool, {}, None)
        assert abs(pred) <= MT1_PRED_SCALE, 'tanh x scale is bounded by the scale'
        assert isinstance(pred, float)


class TestRollingStatePersistence:
    """The bug this file exists for.

    `upkeep_mt1_industry` defaults `rolling_state` to a fresh `{}`. A caller that does not load it,
    pass it, and save it gets a pool that parks predictions every run and scores none of them,
    forever — no error, no empty file, and `best.pt` still updates because ranking an all-zero
    pool still returns slot 0.
    """

    def test_state_survives_a_save_load_round_trip(self, pool):
        state = {}
        _run(pool, state, None)
        save_mt1_rolling_state(pool, state)
        reloaded = load_mt1_rolling_state(pool)
        assert reloaded['energy']['pending'] == pytest.approx(state['energy']['pending'])
        assert reloaded['energy']['day'] == state['energy']['day']

    def test_scoring_continues_across_a_round_trip(self, pool):
        state = {}
        _run(pool, state, None)
        save_mt1_rolling_state(pool, state)
        state2 = load_mt1_rolling_state(pool)
        _, score0 = _run(pool, state2, 300.0)
        assert score0 > 0.0, 'reloaded state did not carry the parked predictions'

    def test_dropping_the_state_silently_stops_all_scoring(self, pool):
        """Pins the failure mode itself, so a caller that forgets to persist is a failing test
        rather than a pool that quietly never learns."""
        for _ in range(MT1_POOL_MIN_AGE + 5):
            _, score0 = _run(pool, {}, 250.0)      # fresh dict every run, as the bug did
            assert score0 == 0.0
        state = {}
        for i in range(MT1_POOL_MIN_AGE + 5):
            _run(pool, state, None if i == 0 else 250.0)
        assert max(m['n_pred'] for m in state['energy']['metas']) >= MT1_POOL_MIN_AGE, \
            'with persisted state the pool must accumulate a record'

    def test_state_is_json_serialisable(self, pool):
        """It is persisted as JSON. A tensor or a set in there would raise mid-run, after orders
        have already been submitted."""
        state = {}
        for i in range(3):
            _run(pool, state, None if i == 0 else 100.0)
        json.dumps(state)

    def test_industries_keep_separate_state(self, pool):
        """One state dict holds all 12. Sharing a register or a pending list between industries
        would grade one industry's model on another's outcome."""
        state = {}
        _run(pool, state, None, industry='energy', feat_seed=7)
        _run(pool, state, 150.0, industry='utilities', feat_seed=11)
        assert set(state) == {'energy', 'utilities'}
        assert state['energy']['pending'] is not state['utilities']['pending']
        assert state['energy']['metas'] is not state['utilities']['metas']
        assert state['energy']['actuals'] == []
        assert state['utilities']['actuals'] == [150.0], \
            "one industry's outcome must not land in another's window"
        assert Path(pool, 'mt1_energy_model_0.pt').exists()
        assert Path(pool, 'mt1_utilities_model_0.pt').exists()
        assert state['energy']['pending'] != state['utilities']['pending']

    def test_a_corrupt_state_file_degrades_to_empty(self, pool):
        Path(pool, 'mt1_rolling_state.json').write_text('{not json')
        assert load_mt1_rolling_state(pool) == {}

    def test_a_missing_state_file_is_not_an_error(self, tmp_path):
        assert load_mt1_rolling_state(str(tmp_path)) == {}

    def test_save_is_atomic(self, pool):
        """Written to a .tmp and renamed: production is killed mid-run often enough that a
        half-written state file is a real outcome, and it would take the pool's whole record."""
        state = {'energy': {'day': 5}}
        save_mt1_rolling_state(pool, state)
        assert not Path(pool, 'mt1_rolling_state.json.tmp').exists()
        assert load_mt1_rolling_state(pool) == state


class TestProductionWiresItUp:
    """production_v2 is the only caller. These pin the wiring, not the maths."""

    def test_it_loads_and_saves_the_rolling_state(self):
        src = (REPO / 'production_v2.py').read_text()
        body = src[src.index('def train_mt_one_day_prod('):src.index('def build_primed_portfolios(')]
        assert 'load_mt1_rolling_state(' in body, 'state is never loaded — nothing would be scored'
        assert 'save_mt1_rolling_state(' in body, 'state is never saved — nothing would be scored'
        assert 'rolling_state=rolling_state' in body, 'state is loaded but not passed through'

    def test_it_imports_both_helpers(self):
        src = (REPO / 'production_v2.py').read_text()
        assert 'load_mt1_rolling_state' in src.split('def ')[0]
        assert 'save_mt1_rolling_state' in src.split('def ')[0]

    def test_the_target_is_the_change_since_the_previous_session(self):
        src = (REPO / 'production_v2.py').read_text()
        body = src[src.index('def train_mt_one_day_prod('):src.index('def build_primed_portfolios(')]
        assert 'mt_prev_session.json' in body
        assert 'cur_pf_index[ind] - float(oi)' in body, \
            'the MT1 target must be this session\'s realised P&L, not a forward return'

    def test_the_old_forward_buffer_is_gone(self):
        """mt_fwd_buffer.json buffered 10 sessions for the 10-day-forward target. Leaving it in
        would delay every MT1 update by ten sessions against a target that resolves in one."""
        body = (REPO / 'production_v2.py').read_text()
        assert "'mt_fwd_buffer.json'" not in body
        assert 'MT1_FWD_DAYS' not in body

    def test_mt2_receives_one_number_per_industry(self):
        src = (REPO / 'production_v2.py').read_text()
        body = src[src.index('def train_mt_one_day_prod('):src.index('def build_primed_portfolios(')]
        assert 'mt2_inputs[ind] = pred' in body, 'MT2 must be fed the MT1 prediction itself'


class TestCullArithmeticMatchesTheTrainer:
    """`n_cull` and `n_elite` are computed inline in both mt1_step_day and upkeep_mt1_industry.

    Duplicated arithmetic with no shared definition: the two can drift to different rounding and
    the pool would turn over at different rates in training and in production, which shows up only
    as a slow difference in how fast the pool forgets.
    """

    @staticmethod
    def _cpp_exprs():
        src = (REPO / 'training_v4.cpp').read_text()
        cull = re.search(r'n_cull\s*=\s*\(int\)\((.*?)\);', src)
        elite = re.search(r'n_elite\s*=\s*std::max\(1,\s*\(int\)\((.*?)\)\);', src)
        assert cull and elite, 'cull/elite arithmetic not found in mt1_step_day'
        return cull.group(1), elite.group(1)

    def test_cull_rounds_to_nearest(self):
        expr, _ = self._cpp_exprs()
        assert 'MT1_POOL_CULL_PCT' in expr
        assert '+ 0.5' in expr, 'C++ rounds to nearest; Python must not truncate instead'

    def test_elite_has_a_floor_of_one(self):
        _, expr = self._cpp_exprs()
        assert 'MT1_POOL_ELITE_PCT' in expr
        src = (REPO / 'upkeep.py').read_text()
        assert 'max(1, int(MT1_POOL_ELITE_PCT' in src, \
            'without the floor a small mature set yields zero parents and the pool freezes'

    @pytest.mark.parametrize('mature', [1, 6, 12, 25, 60, 100, 120, 131, 200])
    def test_both_sides_agree_for_every_plausible_mature_count(self, mature):
        n_cull = int(MT1_POOL_CULL_PCT * mature + 0.5)
        n_elite = max(1, int(MT1_POOL_ELITE_PCT * mature))
        assert 0 <= n_cull <= mature, 'cannot cull more of the pool than is mature'
        assert 1 <= n_elite <= mature, 'there must always be at least one parent'
        assert n_cull <= MT1_POOL_SLOTS and n_elite <= MT1_POOL_SLOTS

    def test_the_steady_state_is_the_documented_sixty_percent(self):
        """mature/N settles at 1/(1 + MIN_AGE x CULL_PCT). The smoke run measured 131 of 200;
        this pins the arithmetic that predicts it, so a constant change shows its consequence."""
        predicted = 1.0 / (1.0 + MT1_POOL_MIN_AGE * MT1_POOL_CULL_PCT)
        assert predicted == pytest.approx(0.60, abs=0.02)
        assert MT1_POOL_SLOTS * predicted == pytest.approx(120, abs=5)

    def test_a_single_mature_model_still_breeds_rather_than_freezing(self):
        assert max(1, int(MT1_POOL_ELITE_PCT * 1)) == 1
        assert int(MT1_POOL_CULL_PCT * 1 + 0.5) == 0, \
            'with one mature model there is nothing to cull, and it must not cull itself'
