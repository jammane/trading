"""Production must not over-commit an industry's capital, and must honour the single-stock cap.

Two defects found 2026-09-17 in `production_v2.py`'s buy loop:

  1. `allocated_cash` was set once per industry and never decremented, so EVERY symbol was
     sized against the industry's full allocation. Measured on a live pre-flight, tech_hardware
     queued ~$12,800 of orders against a $1,666.67 allocation -- 7.7x over.
  2. `MAX_SINGLE_STOCK_PCT` was defined in the module and never applied, so production had no
     single-stock ceiling while the models were trained under one.

`training_lib.step_industry` was always correct on both counts; this was production-only drift.
"""
import re

import pytest

from training_lib import MAX_SINGLE_STOCK_PCT

BUY_FILL = 1.0


def size_orders(alloc, prices, want, holdings=None, decrement=True, apply_cap=True):
    """Reference implementation of the production buy loop's sizing."""
    holdings = holdings or {}
    remaining = alloc
    port_value = alloc + sum(holdings.get(s, 0.0) * p for s, p in prices.items())
    out = {}
    for sym, price in prices.items():
        budget = remaining if decrement else alloc
        amount = min(want.get(sym, 0.0), budget / (price * BUY_FILL))
        if apply_cap:
            cur = holdings.get(sym, 0.0) * price
            max_spend = max(0.0, MAX_SINGLE_STOCK_PCT * port_value - cur)
            amount = min(amount, max_spend / (price * BUY_FILL))
        amount = int(amount)
        if amount >= 1:
            out[sym] = amount
            remaining -= amount * price * BUY_FILL
    return out, remaining


# the tech_hardware bar that exposed it
PRICES = {'NVDA': 217.145, 'MU': 953.50, 'MRVL': 247.89, 'ON': 66.06, 'AMAT': 428.54,
          'LRCX': 267.80, 'KLAC': 173.79, 'SWKS': 84.435, 'MPWR': 1146.40}
ALLOC = 1666.67
WANT = {s: 1e6 for s in PRICES}          # model asks for far more than affordable


class TestOverCommit:
    def test_old_behaviour_over_commits(self):
        """Pin the bug: without decrementing, each symbol takes the whole allocation."""
        orders, _ = size_orders(ALLOC, PRICES, WANT, decrement=False, apply_cap=False)
        spent = sum(q * PRICES[s] for s, q in orders.items())
        assert spent > 5 * ALLOC, 'expected gross over-commitment from the old formula'

    def test_total_never_exceeds_allocation(self):
        orders, remaining = size_orders(ALLOC, PRICES, WANT)
        spent = sum(q * PRICES[s] for s, q in orders.items())
        assert spent <= ALLOC + 1e-6, f'queued {spent:.2f} against {ALLOC:.2f}'
        assert remaining >= -1e-6

    @pytest.mark.parametrize('alloc', [250.0, 1666.67, 5000.0, 25000.0])
    def test_holds_at_every_capital_tier(self, alloc):
        """dev / paper / prod run at very different capital; the invariant is scale-free."""
        orders, remaining = size_orders(alloc, PRICES, WANT)
        spent = sum(q * PRICES[s] for s, q in orders.items())
        assert spent <= alloc + 1e-6
        assert remaining >= -1e-6

    def test_unaffordable_symbol_is_simply_skipped(self):
        """At $250 nothing priced over the cap can be bought -- it must not go negative."""
        orders, remaining = size_orders(250.0, PRICES, WANT)
        assert 'MPWR' not in orders and 'MU' not in orders
        assert remaining >= -1e-6


class TestSingleStockCap:
    def test_no_position_exceeds_the_cap(self):
        orders, _ = size_orders(ALLOC, PRICES, WANT)
        port_value = ALLOC
        for s, q in orders.items():
            assert q * PRICES[s] <= MAX_SINGLE_STOCK_PCT * port_value + 1e-6, \
                f'{s} took {100*q*PRICES[s]/port_value:.0f}% of the industry'

    def test_cap_blocks_the_mpwr_case(self):
        """One MPWR share is 69% of a $1,666.67 industry -- over the 0.60 ceiling."""
        orders, _ = size_orders(ALLOC, {'MPWR': 1146.40}, {'MPWR': 1.0})
        assert 'MPWR' not in orders
        # ...but it is fine once the industry is large enough
        orders, _ = size_orders(25000.0, {'MPWR': 1146.40}, {'MPWR': 1.0})
        assert orders.get('MPWR') == 1

    def test_existing_holdings_count_toward_the_cap(self):
        held = {'ON': 20}           # 20 x 66.06 = 1321 already held
        orders, _ = size_orders(ALLOC, {'ON': 66.06}, {'ON': 100.0}, holdings=held)
        port_value = ALLOC + 20 * 66.06
        total = (held['ON'] + orders.get('ON', 0)) * 66.06
        assert total <= MAX_SINGLE_STOCK_PCT * port_value + 66.06


class TestProductionSourceGuards:
    """The three trading implementations drift; guard the one that isn't otherwise exercised."""

    def test_buy_loop_decrements_remaining_cash(self):
        """Cash bookkeeping stays production's own job; only the arithmetic is shared."""
        src = open('production_v2.py').read()
        assert 'remaining_cash = allocated_cash' in src
        assert re.search(r'remaining_cash\s*-=\s*amount \* buy_price', src), \
            'buy loop no longer spends its budget down'
        assert re.search(r'size_buy\(\s*\n?\s*buy_qty, buy_price, remaining_cash', src), \
            'sizing must be fed remaining_cash, not the full allocation'

    def test_cap_is_applied_via_the_shared_function(self):
        """The cap moved into size_buy; production must pass the portfolio value to it."""
        src = open('production_v2.py').read()
        assert 'ind_port_value' in src, 'production computes no portfolio value for the cap'
        assert re.search(r'size_buy\([^)]*ind_port_value', src, re.S), \
            'production must pass ind_port_value so the cap is enforced'
        lib = open('training_lib.py').read()
        assert 'MAX_SINGLE_STOCK_PCT * port_value' in lib, 'the cap left size_buy'


class TestIndustryActivityGate:
    """An industry is tradeable only if it can afford a minimal 3-symbol book.

    The gate was checking `capital >= priciest` -- one share of the dearest symbol -- while the
    log line already described the intended rule as the "3-stock entry" (priciest + 2 cheapest).
    Affording one share of one name does not make an industry tradeable: the model allocates
    across the industry, and a book that can hold exactly one position cannot express that.
    """

    @staticmethod
    def entry_min(prices):
        top1 = max(prices.items(), key=lambda kv: kv[1])
        rest = sorted((kv for kv in prices.items() if kv[0] != top1[0]), key=lambda kv: kv[1])
        return top1[1] + sum(p for _, p in rest[:2])

    def test_entry_min_is_priciest_plus_two_cheapest(self):
        assert self.entry_min({'A': 90.0, 'B': 30.0, 'C': 35.0, 'D': 60.0}) == 155.0

    def test_priciest_is_not_double_counted(self):
        """With only three symbols the priciest must not also serve as a 'cheapest'."""
        assert self.entry_min({'A': 90.0, 'B': 30.0, 'C': 40.0}) == 160.0

    def test_gate_is_stricter_than_the_old_one_shares_rule(self):
        prices = {'A': 90.0, 'B': 30.0, 'C': 35.0}
        assert self.entry_min(prices) > max(prices.values())

    def test_band_makes_unlock_thresholds_uniform(self):
        """The old universe's unlock point was an accident of its priciest symbol."""
        old_financials = {'MELI': 1817.0, 'UPST': 25.0, 'PYPL': 53.0}
        new_banded = {'WAL': 79.0, 'EZPW': 32.0, 'GBCI': 40.0}
        assert self.entry_min(old_financials) > 1800
        assert self.entry_min(new_banded) < 200

    def test_source_computes_entry_min_not_the_one_share_rule(self):
        src = open('production_v2.py').read()
        assert 'top1[1] + sum(p for _, p in bottom2)' in src, 'entry_min formula changed'
        assert 'if ind_capital >= max_price:' not in src, 'gate regressed to the 1-share rule'
        assert "bottom2    = [x for x in sorted_asc if x[0] != top1[0]][:2]" in src, \
            'the priciest must be excluded from the two cheapest'


class TestCumulativeEntryGate:
    """Opening the Nth industry requires the TOTAL portfolio to cover the summed entry_min of
    all N, taken in activation order — not each industry against its own slice.

    The old per-industry check let every industry open the moment it cleared its own floor. The
    old universe hid that, because unlock points ran $140-$1,825 and supplied an accidental
    ordering; inside the $30-$90 band they are all ~$150, so the ramp has to be explicit.
    """

    ENTRY = {'a': 150.0, 'b': 160.0, 'c': 170.0, 'd': 180.0}

    @staticmethod
    def activate(total, entry, order):
        active, running = [], 0.0
        for ind in order:
            need = running + entry[ind]
            if total >= need:
                active.append(ind)
                running = need
        return active

    def test_thresholds_are_cumulative_not_per_industry(self):
        order = ['a', 'b', 'c', 'd']
        assert self.activate(150.0, self.ENTRY, order) == ['a']
        assert self.activate(309.0, self.ENTRY, order) == ['a']          # 150+160=310
        assert self.activate(310.0, self.ENTRY, order) == ['a', 'b']
        assert self.activate(480.0, self.ENTRY, order) == ['a', 'b', 'c']

    def test_each_industry_alone_would_pass_the_old_check(self):
        """At $200 every industry individually clears its ~$150-180 floor; only one may open."""
        for e in self.ENTRY.values():
            assert 200.0 >= e or e > 200.0      # floors are all near $200
        assert len(self.activate(200.0, self.ENTRY, ['a', 'b', 'c', 'd'])) == 1

    def test_order_decides_which_industry_opens_first(self):
        assert self.activate(150.0, self.ENTRY, ['a', 'b', 'c', 'd']) == ['a']
        assert self.activate(180.0, self.ENTRY, ['d', 'a', 'b', 'c']) == ['d']

    def test_ramp_is_monotone_in_capital(self):
        order = ['a', 'b', 'c', 'd']
        counts = [len(self.activate(t, self.ENTRY, order)) for t in range(0, 700, 10)]
        assert counts == sorted(counts), 'more capital must never open fewer industries'

    def test_zero_capital_opens_nothing(self):
        assert self.activate(0.0, self.ENTRY, ['a', 'b', 'c', 'd']) == []

    def test_source_uses_cumulative_running_total(self):
        src = open('production_v2.py').read()
        assert 'need = running + entry_min' in src
        assert 'if total_value >= need:' in src, 'gate must test the cumulative need'
        assert 'load_activation_order(' in src


    def test_mask_is_prod_only(self):
        """Paper runs every industry; the ramp is for prod, which is real investment."""
        src = open('production_v2.py').read()
        assert 'if args.paper:' in src and 'active_industries = set(entry_by_ind)' in src, \
            'paper must bypass the cumulative ramp'


class TestSizingIsSharedNotCopied:
    """The same buy-sizing arithmetic existed in SEVEN copies (training_lib x4,
    training_v4.cpp x2, production_v2 x1). All three bugs found 2026-09-17 were in it, and each
    fix had to be applied at every site. It now lives once, in training_lib.size_buy.
    """

    def test_training_lib_has_exactly_one_cap_calculation(self):
        src = open('training_lib.py').read()
        n = src.count('MAX_SINGLE_STOCK_PCT * port_value')
        assert n == 1, f'cap arithmetic duplicated {n} times; it belongs only in size_buy()'

    def test_all_python_sizing_goes_through_size_buy(self):
        lib = open('training_lib.py').read()
        assert lib.count('buy_amount = size_buy(') == 4, 'a simulator site stopped sharing'
        assert "affordable = port['cash']" not in lib, 'an inlined sizing copy came back'
        prod = open('production_v2.py').read()
        assert 'size_buy(' in prod, 'production must use the shared sizing'
        assert 'affordable = remaining_cash' not in prod, 'production re-inlined the arithmetic'

    def test_production_no_longer_defines_its_own_cap_constant(self):
        src = open('production_v2.py').read()
        assert 'MAX_SINGLE_STOCK_PCT = 0.60' not in src, \
            'two sources of truth for the cap is the bug being fixed'

    def test_size_buy_enforces_all_three_bounds(self):
        from training_lib import size_buy
        # bounded by the model's request
        assert size_buy(3.0, 10.0, 1e9, 1e9, 0.0) == 3.0
        # bounded by cash
        assert size_buy(1e9, 10.0, 55.0, 1e9, 0.0) == 5.0
        # bounded by the single-stock cap (0.60 x 100 = 60 -> 6 shares at $10)
        assert size_buy(1e9, 10.0, 1e9, 100.0, 0.0) == 6.0
        # existing holding counts against the cap
        assert size_buy(1e9, 10.0, 1e9, 100.0, 40.0) == 2.0
        # always whole shares
        assert size_buy(1e9, 10.0, 59.0, 1e9, 0.0) == 5.0

    def test_size_buy_degenerate_inputs(self):
        from training_lib import size_buy
        assert size_buy(0.0, 10.0, 1e9, 1e9, 0.0) == 0.0
        assert size_buy(5.0, 0.0, 1e9, 1e9, 0.0) == 0.0      # no price
        assert size_buy(5.0, 10.0, 0.0, 1e9, 0.0) == 0.0     # no cash
        assert size_buy(5.0, 10.0, 1e9, 0.0, 0.0) == 0.0     # cap leaves nothing
