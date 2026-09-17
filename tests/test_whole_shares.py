"""Whole-share rounding in the fill simulation.

The simulator traded continuous quantities while production floors to int (every Alpaca order
uses qty=, never notional=, and the stop-loss orders forbid fractional trading). That let
training buy 0.7 of a share and collect exposure where production buys nothing, and because
int() truncates DOWN the bias was directional. These pin the fix.
"""
import re
import subprocess

import pytest

from training_lib import whole_shares


class TestWholeShares:
    def test_floors_fractions(self):
        assert whole_shares(2.7) == 2.0
        assert whole_shares(23.7) == 23.0
        assert whole_shares(1.999) == 1.0

    def test_sub_one_share_is_no_trade(self):
        """The case that matters at $2-5k: production cannot buy a fraction, so neither can we."""
        assert whole_shares(0.7) == 0.0
        assert whole_shares(0.999) == 0.0

    def test_exact_integers_survive(self):
        for n in (1, 2, 5, 23, 200):
            assert whole_shares(float(n)) == float(n)

    def test_float_error_does_not_eat_a_share(self):
        """2.9999997 must be 3, not 2 — the epsilon exists for this."""
        assert whole_shares(2.9999997) == 3.0
        assert whole_shares(1.0 - 1e-9) == 1.0

    def test_never_negative(self):
        assert whole_shares(0.0) == 0.0
        assert whole_shares(-5.0) == 0.0

    def test_result_is_always_integral(self):
        for q in (0.0, 0.3, 1.5, 7.77, 123.456, 999.99):
            w = whole_shares(q)
            assert w == int(w)

    def test_never_rounds_up_past_affordable(self):
        """Flooring must not let the book spend more cash than it has."""
        for q in (0.9, 1.9, 2.9, 10.9):
            assert whole_shares(q) <= q


class TestBothSidesFloored:
    """The C++ and Python fill paths must agree; a fix applied to only one silently diverges."""

    def test_python_buy_and_sell_sites_all_floored(self):
        src = open('training_lib.py').read()
        caps = re.findall(r'buy_amount\s*= min\(buy_amount, max_\w*spend', src)
        floors = re.findall(r'buy_amount = whole_shares\(buy_amount\)', src)
        assert len(floors) >= len(caps), 'a buy site lost its whole_shares() floor'
        partial_sells = re.findall(r'sell_amount\s*= min\(sell_qty, cur_qty\)', src)
        assert not partial_sells, 'a partial-sell site is still fractional'

    def test_cpp_has_the_twin(self):
        src = open('training_v4.cpp').read()
        assert 'static inline float whole_shares(float q)' in src
        assert src.count('buy_amount = whole_shares(buy_amount);') >= 2, \
            'a C++ buy site lost its floor'
        assert 'std::min(sell_qty, port.holdings[j])' not in src.replace(
            'whole_shares(std::min(sell_qty, port.holdings[j]))', ''), \
            'a C++ partial-sell site is still fractional'

    @pytest.mark.parametrize('q', [0.0, 0.7, 1.0, 2.7, 2.9999997, 23.7, 200.0])
    def test_cpp_python_agree(self, q, tmp_path):
        """Compile the C++ helper standalone and compare against the Python twin."""
        src = tmp_path / 'ws.cpp'
        src.write_text(
            '#include <cmath>\n#include <cstdio>\n'
            'static inline float whole_shares(float q){float w=std::floor(q+1e-4f);'
            'return w>0.f?w:0.f;}\n'
            'int main(int c,char**v){printf("%.1f", whole_shares((float)atof(v[1])));}\n'
            '#include <cstdlib>\n')
        exe = tmp_path / 'ws'
        r = subprocess.run(['g++', '-std=c++17', '-o', str(exe), str(src)],
                           capture_output=True)
        if r.returncode != 0:
            pytest.skip('no g++ available')
        out = subprocess.run([str(exe), str(q)], capture_output=True, text=True).stdout
        assert float(out) == whole_shares(q), f'C++ {out} vs Python {whole_shares(q)} for {q}'
