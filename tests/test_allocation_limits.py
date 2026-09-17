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
        src = open('production_v2.py').read()
        assert 'remaining_cash = allocated_cash' in src
        assert re.search(r'remaining_cash\s*-=\s*amount \* buy_price', src), \
            'buy loop no longer spends its budget down'
        assert 'affordable = remaining_cash /' in src, \
            'buy sizing must read remaining_cash, not allocated_cash'

    def test_cap_is_actually_applied(self):
        src = open('production_v2.py').read()
        uses = [ln for ln in src.splitlines()
                if 'MAX_SINGLE_STOCK_PCT' in ln and not ln.strip().startswith('#')]
        assert len(uses) >= 2, 'MAX_SINGLE_STOCK_PCT is defined but never applied'


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

    def test_source_gates_on_entry_min(self):
        src = open('production_v2.py').read()
        assert 'if ind_capital >= entry_min:' in src, 'gate regressed to the 1-share rule'
        assert 'if ind_capital >= max_price:' not in src
