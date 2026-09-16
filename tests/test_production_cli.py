"""production_v2 CLI surface.

These flags are operational contracts — they appear in the crontab and in the paper-trading
rollout — so a rename or removal should fail here rather than at 5:05 PM on a weekday.

--help exits inside argparse before any credential handling, so this needs no Alpaca keys.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope='module')
def help_text():
    r = subprocess.run([sys.executable, 'production_v2.py', '--help'],
                       cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


@pytest.mark.parametrize('flag', ['--paper', '--account', '--capital', '--withdraw',
                                  '--no-orders', '--flat-allocation'])
def test_flag_present(help_text, flag):
    assert flag in help_text


def test_no_orders_documents_that_it_still_trains(help_text):
    # The whole point of --no-orders is that the upkeep step STILL runs; if the help ever
    # describes it as a plain no-op, the catch-up procedure in PASS_SEEDING/rollout is wrong.
    # rindex, not index: '--no-orders' appears first in the wrapped usage line at the top,
    # where there is no description to check.
    i = help_text.rindex('--no-orders')
    blurb = help_text[i:i + 400].lower()
    assert 'upkeep' in blurb or 'train' in blurb


def test_source_guards_both_mutating_paths():
    # --no-orders must suppress BOTH submission and stop-order cancellation. Cancelling a live
    # stop while submitting no replacement would strip protection from a real position.
    src = (ROOT / 'production_v2.py').read_text()
    assert 'if args.no_orders:' in src, 'submission path not guarded'
    assert 'not args.no_orders' in src, 'stop-cancellation path not guarded'
    assert src.count('args.no_orders') >= 2
