"""production_v2.trading_lock — the lock the C++ trainer pauses on.

The trainer polls this file between training days (trade_lock_active() in training_v4.cpp). The
contract that matters: it exists for the whole cycle, line 1 is the holder's PID so a dead holder
can be detected, and it is ALWAYS removed — a stranded lock would idle training until the 30-minute
staleness timeout on every subsequent check.
"""
import os

import pytest

import production_v2


@pytest.fixture
def lock_path(tmp_path, monkeypatch):
    p = tmp_path / 'sub' / 'trading_active.lock'
    monkeypatch.setattr(production_v2, 'TRADING_LOCK_PATH', str(p))
    return p


class TestTradingLock:
    def test_created_and_removed(self, lock_path):
        assert not lock_path.exists()
        with production_v2.trading_lock('paper', 'acct0'):
            assert lock_path.exists(), 'lock must exist for the whole cycle'
        assert not lock_path.exists(), 'lock must be released on normal exit'

    def test_first_line_is_the_pid(self, lock_path):
        with production_v2.trading_lock('paper', 'acct0'):
            first = lock_path.read_text().splitlines()[0]
        assert first.strip() == str(os.getpid()), 'the trainer reads line 1 as the holder PID'

    def test_second_line_identifies_the_holder(self, lock_path):
        with production_v2.trading_lock('prod', 'acct1'):
            lines = lock_path.read_text().splitlines()
        assert len(lines) >= 2
        assert 'prod' in lines[1] and 'acct1' in lines[1]

    def test_released_on_exception(self, lock_path):
        # A crash mid-cycle must not strand the lock; otherwise training idles until the timeout.
        with pytest.raises(RuntimeError), production_v2.trading_lock('paper', 'acct0'):
            assert lock_path.exists()
            raise RuntimeError('boom')
        assert not lock_path.exists()

    def test_parent_directory_created(self, lock_path):
        assert not lock_path.parent.exists()
        with production_v2.trading_lock('paper', 'acct0'):
            assert lock_path.parent.is_dir()

    def test_unwritable_path_does_not_block_trading(self, monkeypatch, capsys):
        # Failing to take the lock must never stop a production run — the only cost is that
        # training keeps competing for CPU.
        monkeypatch.setattr(production_v2, 'TRADING_LOCK_PATH', '/proc/nope/trading.lock')
        ran = False
        with production_v2.trading_lock('paper', 'acct0'):
            ran = True
        assert ran, 'the cycle must proceed even if the lock cannot be written'
        assert 'could not write trading lock' in capsys.readouterr().out

    def test_default_path_matches_the_trainer(self):
        # training_v4.cpp hardcodes the same default in g_trade_lock. Absolute, because the two
        # processes run from different worktrees.
        assert production_v2.TRADING_LOCK_PATH.startswith('/')
