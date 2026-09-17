"""stock_data is shared across environments that may hold different universes.

dev / paper / prod can legitimately differ — a staged rollout means prod trades a proven set while
dev tests a new one and paper validates in between. But all three read and write the SAME
stock_data/ directory (the droplet symlinks trading-ht's to trading's), and the weekly cleanup cron
runs from ONE worktree.

On 2026-09-17 dev had been re-normalized to 144 new symbols while paper still held the old ones.
The cron, running from the paper worktree, would have deleted all 102 freshly downloaded symbols
mid-training-run. Cleanup must union across every environment, and must refuse to act when it
cannot see them all.
"""
import json
import subprocess

import universe
from universe import ENVIRONMENT_BRANCHES, union_across_environments


class TestUnionAcrossEnvironments:
    def test_every_environment_is_consulted(self):
        _, per_env, _ = union_across_environments()
        for env in ENVIRONMENT_BRANCHES:
            assert env in per_env, f'{env} not consulted'
        assert 'working' in per_env, 'the working tree may be ahead of every branch'

    def test_union_contains_every_readable_environment(self):
        union, per_env, _ = union_across_environments()
        for env, syms in per_env.items():
            if syms:
                assert syms <= union, f"{env}'s symbols are not all in the union"

    def test_union_is_not_smaller_than_any_single_environment(self):
        union, per_env, _ = union_across_environments()
        for env, syms in per_env.items():
            if syms:
                assert len(union) >= len(syms)

    def test_branches_actually_resolve(self):
        """If a branch stops resolving the union silently shrinks — which is the bug."""
        _, per_env, unreadable = union_across_environments()
        assert per_env['working'], 'the working tree universe must always be readable'
        for env in unreadable:
            assert env != 'working'


class TestFailsSafe:
    """Deleting on a partial view is the failure being prevented, so an unreadable environment
    must stop the deletion rather than quietly narrow the keep-set."""

    def test_unreadable_environment_is_reported_not_ignored(self, monkeypatch):
        monkeypatch.setattr(universe, '_universe_from_branch',
                            lambda b: None if b == 'main' else {'AAA', 'BBB'})
        _, per_env, unreadable = universe.union_across_environments()
        assert 'prod' in unreadable
        assert per_env['prod'] is None

    def test_union_still_includes_the_readable_ones(self, monkeypatch):
        monkeypatch.setattr(universe, '_universe_from_branch',
                            lambda b: None if b == 'main' else {'AAA', 'BBB'})
        union, _, _ = universe.union_across_environments()
        assert {'AAA', 'BBB'} <= union

    def test_cleanup_refuses_to_delete_when_blind(self):
        src = open('cleanup_stock_data.py').read()
        assert 'REFUSING TO DELETE' in src
        assert 'raise SystemExit(2)' in src, 'must exit non-zero so cron surfaces it'

    def test_cleanup_uses_the_union_not_one_universe(self):
        src = open('cleanup_stock_data.py').read()
        assert 'union_across_environments' in src
        assert 'from universe import ALL_SYMBOLS' not in src, \
            'cleanup must not key off a single environment'


class TestTheActualHazard:
    def test_divergent_universes_do_not_delete_each_other(self, monkeypatch):
        """The 2026-09-17 shape: prod on the old set, dev on the new, 42 shared."""
        old = {f'OLD{i}' for i in range(102)} | {f'SHARED{i}' for i in range(42)}
        new = {f'NEW{i}' for i in range(102)} | {f'SHARED{i}' for i in range(42)}
        monkeypatch.setattr(universe, '_universe_from_branch',
                            lambda b: old if b == 'main' else new)
        monkeypatch.setattr(universe, 'ALL_SYMBOLS', sorted(new))
        union, _, unreadable = universe.union_across_environments()
        assert not unreadable
        assert old <= union and new <= union
        assert len(union) == 246, 'union must keep both environments whole'

    def test_branch_map_matches_the_documented_structure(self):
        assert ENVIRONMENT_BRANCHES == {'prod': 'main', 'paper': 'paper', 'dev': 'dev'}

    def test_branch_universes_are_real_json(self):
        for env, branch in ENVIRONMENT_BRANCHES.items():
            r = subprocess.run(['git', 'show', f'{branch}:universe.json'],
                               capture_output=True, text=True)
            if r.returncode != 0:
                continue
            data = json.loads(r.stdout)
            inds = data['industries'] if isinstance(data, dict) and 'industries' in data else data
            syms = [s for v in inds.values() for s in v]
            assert len(syms) == len(set(syms)), f'{branch} has duplicate symbols'
            assert len(syms) == 144, f'{branch} has {len(syms)} symbols, expected 144'


class TestDownloadMatchesCleanup:
    """download and cleanup must agree on the keep-set, or the difference goes stale.

    cleanup keeps the union; if download fetched only one environment's universe, the symbols in
    the gap would never be updated, trip find_stale_symbols, and make download_daily exit non-zero
    every day — turning a correct cleanup into a daily false alarm.
    """

    def test_download_fetches_the_union(self):
        src = open('download_daily.py').read()
        assert 'union_across_environments' in src
        assert 'from universe import ALL_SYMBOLS' not in src, \
            'download would fetch one environment while cleanup keeps all of them'

    def test_both_scripts_use_the_same_source(self):
        dl = open('download_daily.py').read()
        cl = open('cleanup_stock_data.py').read()
        assert 'union_across_environments' in dl and 'union_across_environments' in cl, \
            'the keep-set and the fetch-set must come from one function'

    def test_download_warns_when_an_environment_is_unreadable(self):
        """download cannot refuse — it must still fetch — but it must say what it is missing."""
        src = open('download_daily.py').read()
        assert 'could not read' in src and 'may go stale' in src
