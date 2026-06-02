"""Tests for the trading universe (industry/symbol definitions)."""

import pytest

from universe import ALL_SYMBOLS, INDUSTRIES, INDUSTRY_NAMES

EXPECTED_INDUSTRY_COUNT = 12
EXPECTED_SYMBOLS_PER_INDUSTRY = 12
EXPECTED_TOTAL_SYMBOLS = EXPECTED_INDUSTRY_COUNT * EXPECTED_SYMBOLS_PER_INDUSTRY

EXPECTED_INDUSTRY_NAMES = {
    'tech_hardware', 'tech_software_ai', 'financials', 'consumer_discretionary',
    'consumer_services', 'health_care', 'industrials', 'consumer_staples',
    'energy', 'utilities', 'real_estate', 'materials',
}


class TestIndustries:
    def test_industry_count(self):
        assert len(INDUSTRIES) == EXPECTED_INDUSTRY_COUNT, \
            f"Expected {EXPECTED_INDUSTRY_COUNT} industries, got {len(INDUSTRIES)}"

    def test_expected_industry_names_present(self):
        assert set(INDUSTRIES.keys()) == EXPECTED_INDUSTRY_NAMES

    def test_symbols_per_industry(self):
        for name, symbols in INDUSTRIES.items():
            assert len(symbols) == EXPECTED_SYMBOLS_PER_INDUSTRY, \
                f"{name}: expected {EXPECTED_SYMBOLS_PER_INDUSTRY} symbols, got {len(symbols)}"

    def test_no_duplicate_symbols_within_industry(self):
        for name, symbols in INDUSTRIES.items():
            assert len(symbols) == len(set(symbols)), \
                f"{name} has duplicate symbols: {[s for s in symbols if symbols.count(s) > 1]}"

    def test_no_duplicate_symbols_across_industries(self):
        seen = {}
        for name, symbols in INDUSTRIES.items():
            for sym in symbols:
                assert sym not in seen, \
                    f"Symbol '{sym}' appears in both '{seen[sym]}' and '{name}'"
                seen[sym] = name

    def test_all_symbols_are_strings(self):
        for name, symbols in INDUSTRIES.items():
            for sym in symbols:
                assert isinstance(sym, str), f"{name}: {sym!r} is not a string"

    def test_all_symbols_non_empty(self):
        for name, symbols in INDUSTRIES.items():
            for sym in symbols:
                assert sym.strip(), f"{name}: found empty/whitespace symbol"

    def test_all_symbols_uppercase(self):
        for name, symbols in INDUSTRIES.items():
            for sym in symbols:
                assert sym == sym.upper(), \
                    f"{name}: '{sym}' is not uppercase"

    def test_all_symbols_alphanumeric(self):
        for name, symbols in INDUSTRIES.items():
            for sym in symbols:
                assert sym.isalpha() or sym.isalnum(), \
                    f"{name}: '{sym}' contains unexpected characters"

    def test_total_symbol_count(self):
        assert len(ALL_SYMBOLS) == EXPECTED_TOTAL_SYMBOLS, \
            f"Expected {EXPECTED_TOTAL_SYMBOLS} total symbols, got {len(ALL_SYMBOLS)}"

    def test_all_symbols_list_matches_industries(self):
        flat = [sym for syms in INDUSTRIES.values() for sym in syms]
        assert ALL_SYMBOLS == flat

    def test_industry_names_list_matches_industries(self):
        assert INDUSTRY_NAMES == list(INDUSTRIES.keys())

    def test_no_none_symbols(self):
        for name, symbols in INDUSTRIES.items():
            for sym in symbols:
                assert sym is not None, f"{name}: found None symbol"

    @pytest.mark.parametrize("industry", list(EXPECTED_INDUSTRY_NAMES))
    def test_each_industry_has_correct_count(self, industry):
        assert len(INDUSTRIES[industry]) == EXPECTED_SYMBOLS_PER_INDUSTRY
