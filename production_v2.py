"""
production_v2.py — Live and paper trading engine for the trained StockNN / MasterNN models.

Fetches today's market data from yfinance, runs model inference, submits orders
to Alpaca, and performs one upkeep evolution step using yesterday's data as input.

Usage:
    python production_v2.py --paper              # paper trading mode (no order submission)
    python production_v2.py --paper --output     # paper trading + submit orders to Alpaca
    python production_v2.py --output             # live trading
    python production_v2.py --paper --withdraw 500.0
"""

import argparse
import contextlib
import json
import math
import os
import random
from datetime import datetime

import keyring
import torch
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, OrderType, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLossRequest,
    StopOrderRequest,
)

import committee_alloc
import training_lib
import upkeep_cpp
from fees import BUY_FILL, SELL_FILL, _sell_net
from models import MasterNN, StockNN, stock_close_pos, stock_close_vs_wap
from universe import INDUSTRIES
from upkeep import (
    load_mt1_rolling_state,
    run_mt_inference,
    save_mt1_rolling_state,
    upkeep_industry,
    upkeep_living_bn,
    upkeep_mt1_industry,
    upkeep_mt2,
)
from version import VERSION

# MAX_SINGLE_STOCK_PCT lives in training_lib and is applied inside size_buy(). It was
# duplicated here and never used, which is exactly how production lost the cap.

MODEL_DIR = 'models'  # legacy; superseded by --account in main()
STOCK_DATA_DIR = 'stock_data'

# ── Trading lock (v0.6.5.0) ────────────────────────────────────────────────────
# The droplet has 2 cores. A training run uses ~150% CPU, and when it competes with a production
# cycle the loser slows by 3-5x — measured: training dropped from 34 s/day to 1-3 min/day while
# two short jobs ran alongside it. Production is the time-sensitive side (it fetches, decides and
# submits orders), so the trainer yields to it.
#
# This writes a lock the C++ trainer polls between training days; see trade_lock_active() in
# training_v4.cpp, which must agree with the DEFAULT PATH BELOW. Absolute because the two run from
# different worktrees (/root/trading vs /root/trading-ht), so a relative path would never match.
# /run is tmpfs, so a reboot cannot leave a stale lock behind.
TRADING_LOCK_PATH = os.environ.get('TRADING_LOCK_PATH', '/run/trading/trading_active.lock')


@contextlib.contextmanager
def trading_lock(mode, account):
    """Hold the lock for the whole cycle so the trainer pauses; always release it."""
    path = TRADING_LOCK_PATH
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(f'{os.getpid()}\n')
            f.write(f'production_v2.py {mode} {account} started '
                    f'{datetime.now().isoformat(timespec="seconds")}\n')
        print(f'Trading lock held: {path} (pid {os.getpid()}) — training will pause')
    except OSError as e:
        # Never block a production run because the lock could not be written; the cost is only
        # that training keeps competing for CPU.
        print(f'Warning: could not write trading lock {path}: {e} — continuing without it')
        path = None
    try:
        yield
    finally:
        if path:
            try:
                os.remove(path)
                print(f'Trading lock released: {path}')
            except OSError as e:
                print(f'Warning: could not remove trading lock {path}: {e}')


def load_state(model_dir):
    """Load trading state from {model_dir}/state.json, or return default initial state if absent."""
    state_file = os.path.join(model_dir, 'state.json')
    try:
        if os.path.exists(state_file):
            with open(state_file) as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading state file {state_file}: {e}")
    industries = INDUSTRIES
    symbols = [sym for syms in industries.values() for sym in syms]
    return {'industries': industries, 'histories': {sym: [] for sym in symbols}, 'cash': 20000.0, 'holdings': {sym: 0.0 for sym in symbols}}

def save_state(state, model_dir):
    """Persist the current trading state to {model_dir}/state.json."""
    state_file = os.path.join(model_dir, 'state.json')
    try:
        os.makedirs(model_dir, exist_ok=True)
        with open(state_file, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        print(f"Error saving state file {state_file}: {e}")

def compute_total_portfolio_value(cash, holdings, day_data, histories):
    """Return total portfolio value: cash plus all holdings marked at close prices."""
    total_value = cash
    for sym, qty in holdings.items():
        if qty > 0:
            # Use day's close price if available, otherwise use latest history close
            price = day_data.get(sym, {}).get('close', histories.get(sym, [[]])[-1][1] if histories.get(sym) else 0.0)
            if price > 0:
                total_value += qty * price
    return total_value

def load_activation_order(model_dir, entry_by_ind):
    """Industry priority for the cumulative entry gate: which industries to fund first.

    Read from {model_dir}/activation_order.json — a list of industry names, best first. It is
    meant to come from estimated per-industry returns measured on a full dev run, refreshed when
    a new run completes.

    IMPORTANT: derive it from several passes, not one. Measured 2026-09-17 on a 5-pass run, a
    single pass agreed with the 5-pass mean in only 2 of 12 positions, and one industry swung 3.9x
    between replicates of identical days. Coarse bands are the most the data supports.

    Falls back to cheapest-entry-first, which unlocks the most industries per dollar and
    reproduces the ordering the old price spread used to supply by accident.
    """
    path = os.path.join(model_dir, 'activation_order.json')
    known = list(entry_by_ind)
    try:
        with open(path) as f:
            order = json.load(f)
        order = [i for i in order if i in entry_by_ind]
        missing = [i for i in known if i not in order]
        if missing:
            print(f"  activation_order.json omits {missing} — appended by cheapest entry")
            missing.sort(key=lambda i: entry_by_ind[i][0])
            order += missing
        return order
    except FileNotFoundError:
        print("  no activation_order.json — falling back to cheapest-entry-first")
    except (OSError, json.JSONDecodeError) as e:
        print(f"  activation_order.json unreadable ({e}) — falling back to cheapest-entry-first")
    return sorted(known, key=lambda i: entry_by_ind[i][0])


def update_owners_file(total_value, model_dir):
    """Update the total_value field in {model_dir}/owners.json."""
    owners_file = os.path.join(model_dir, 'owners.json')
    try:
        os.makedirs(model_dir, exist_ok=True)
        if not os.path.exists(owners_file):
            owners_data = {"total value": total_value}
            with open(owners_file, 'w') as f:
                json.dump(owners_data, f)
            print(f"Created owners.json with total value: {total_value}")
        else:
            with open(owners_file) as f:
                owners_data = json.load(f)
            owners_data["total value"] = total_value
            with open(owners_file, 'w') as f:
                json.dump(owners_data, f)
            print(f"Updated owners.json with total value: {total_value}")
    except Exception as e:
        print(f"Error updating owners.json: {e}")

def _master_state_path(model_dir):
    return os.path.join(model_dir, 'master_state.json')

def _load_master_state(model_dir, industry_list):
    """Load persisted ind_value_history, mkt_val_history, zero_counts, and MT2 norm stats."""
    path = _master_state_path(model_dir)
    try:
        with open(path) as f:
            state = json.load(f)
        hist     = state.get('ind_value_history', {ind: [] for ind in industry_list})
        mkt_hist = state.get('mkt_val_history',   {ind: [] for ind in industry_list})
        zcnt     = state.get('zero_counts',        {ind: 0  for ind in industry_list})
    except (OSError, json.JSONDecodeError) as e:
        # Falling back to empty history silently restarts the MT1/MT2 feature ring from cold,
        # which looks like a fresh account rather than a failure. Say so.
        print(f"Warning: master state {path} unreadable ({e}) — starting histories cold")
        hist     = {ind: [] for ind in industry_list}
        mkt_hist = {ind: [] for ind in industry_list}
        zcnt     = {ind: 0  for ind in industry_list}
    return hist, mkt_hist, zcnt

def _save_master_state(model_dir, ind_value_history, mkt_val_history, zero_counts):
    try:
        with open(_master_state_path(model_dir), 'w') as f:
            json.dump({
                'ind_value_history': ind_value_history,
                'mkt_val_history':   mkt_val_history,
                'zero_counts':       zero_counts,
            }, f)
    except OSError as e:
        print(f"Warning: could not save master state: {e}")

def load_weighted_model(model_class, model_dir, prefix):
    """
    Load the best model for a given prefix.
    Primary:  {prefix}_best.pt  (written by training via copy_best_model)
    Fallback: random weights (untrained)
    """
    best_path = f"{model_dir}/{prefix}_best.pt"
    if os.path.exists(best_path):
        try:
            model = model_class()
            model.load_state_dict(torch.load(best_path, map_location='cpu', weights_only=True))
            return model
        except Exception as e:
            print(f"Error loading best model {best_path}: {e}")
    print(f"WARNING: no trained model found at {best_path} — using random weights")
    return model_class()

def train_industry_one_day_prod(industry, symbols, yesterday_data, primed_portfolio,
                                model_dir, today_data=None, seq_flags=None,
                                intraday_bars=None):
    """
    Upkeep training for one industry (burst refinement now correctly enabled).
    yesterday_data: OHLCV for the prediction day — model input features.
    today_data:     OHLCV for the fill day — actual fill prices.
    """
    try:
        return upkeep_industry(
            industry, symbols, model_dir, primed_portfolio,
            yesterday_data, next_day_data=today_data,
            seq_flags=seq_flags, intraday_bars=intraday_bars,
        )
    except Exception as e:
        print(f"Error in upkeep_industry {industry}: {e}")
        return None


def train_mt_one_day_prod(industries, model_dir, mkt_val_history, pf_val_history,
                          slot0_deltas, mkt_ret, mt1_outputs_prev=None):
    """
    Upkeep training for MT1 (12 single-output pools) and MT2 — mirrors the C++ trainer.

    mkt_val_history: {ind: [cumulative_market_index]}  — market feature source (compounding).
    pf_val_history:  {ind: [cumulative_slot0_portfolio_index]} — portfolio feature source.
    slot0_deltas:    {ind: today's deployed slot-0 portfolio return} — the MT2 same-day objective.

    MT1 input per industry = 74 = [37 market ‖ 37 portfolio]. MT1 target = that industry's
    realised P&L for THIS session, in dollars: the change in its cumulative portfolio index since
    the previous run.

    The 10-deep mt_fwd_buffer.json is gone with the 10-day-forward target it was waiting on. What
    remains is one number per industry — the previous session's portfolio index — because a
    next-session target is known one session later, not ten. The prediction each model parked on
    the previous run is scored on this one, which is what makes every score out-of-sample without
    any snapshot machinery.
    """
    from training_lib import build_master_features
    industry_list = list(industries.keys())

    today888 = build_master_features(mkt_val_history, pf_val_history, industry_list)
    cur_pf_index = {}
    for ind in industry_list:
        h = pf_val_history.get(ind, [])
        cur_pf_index[ind] = float(h[-1]) if h else 25000.0  # IND_STARTING_CASH cold-start

    prev_path = os.path.join(model_dir, 'mt_prev_session.json')
    prev_pf = {}
    if os.path.exists(prev_path):
        try:
            with open(prev_path) as _f:
                prev_pf = json.load(_f).get('pf_index', {})
        except (OSError, json.JSONDecodeError) as e:
            # One missed session: MT1 skips scoring this run and resumes on the next.
            print(f"Warning: {prev_path} unreadable ({e}) — MT1 scoring skipped this run")
            prev_pf = {}

    # rolling_state carries the parked predictions, each slot's score register, lineages and the
    # trailing target window ACROSS RUNS. Without persisting it every run would start with nothing
    # parked, so no model would ever be scored and the pool would never evolve.
    rolling_state = load_mt1_rolling_state(model_dir)

    mt2_inputs: dict = {}
    for i, ind in enumerate(industry_list):
        in74_t = today888[:, i * 74:(i + 1) * 74]
        oi = prev_pf.get(ind)
        actual_d = (cur_pf_index[ind] - float(oi)) if oi else None
        try:
            pred, _score0 = upkeep_mt1_industry(ind, model_dir, in74_t, actual_d,
                                                rolling_state=rolling_state)
            mt2_inputs[ind] = pred
        except Exception as e:                                   # noqa: BLE001 - one industry
            print(f"Error in upkeep_mt1_industry {ind}: {e}")
            mt2_inputs[ind] = 0.0

    save_mt1_rolling_state(model_dir, rolling_state)

    # Living conditional model: re-evaluate the state percentages and N against recent StockNN
    # performance. Read-only for now -- nothing allocates on it until it shows an edge, and the
    # measured edge so far is not distinguishable from zero. It costs one JSON write per run.
    try:
        _living = upkeep_living_bn(
            model_dir,
            {ind: (cur_pf_index[ind] - float(prev_pf[ind])) if prev_pf.get(ind) else None
             for ind in industry_list},
            industry_list)
        _dec = [r for r in _living if r['group'] == 'decel']
        if _dec:
            print("living_bn: " + "  ".join(
                f"{r['state']} P(cont)={r['p_continue']:.3f} N={r['streak_N']:.2f}" for r in _dec))
    except Exception as e:                                       # noqa: BLE001
        print(f"Error in upkeep_living_bn: {e}")

    mt2_perf = {ind: float(slot0_deltas.get(ind, 0.0)) for ind in industry_list}
    try:
        upkeep_mt2(model_dir, mt2_inputs, mt2_perf, industry_list)
    except Exception as e:                                       # noqa: BLE001
        print(f"Error in upkeep_mt2: {e}")

    with open(prev_path, 'w') as _f:
        json.dump({'pf_index': cur_pf_index}, _f)


def build_primed_portfolios(trading_client, industries, allocations, zero_counts=None):
    """
    Fetch real Alpaca account state and build per-industry primed portfolios
    plus a master primed portfolio for single-day retraining.

    Each industry's upkeep training portfolio uses a flat 40% cash allocation so
    all 12 industry models evolve with equal budgets regardless of master's current
    tier predictions — this keeps models healthy across all sectors.
    The tier-based dollar allocation (allocations dict) is used for actual order
    generation, not for seeding upkeep training.

    Returns:
        ind_primed:    {industry: {'cash': float, 'holdings': {sym: float}}}
        master_primed: {'cash': float, 'holdings': {ind: float}, 'zero_counts': {ind: int}}
    """
    try:
        account    = trading_client.get_account()
        total_cash = float(account.cash)
    except Exception as e:
        print(f"Error fetching Alpaca account: {e}")
        total_cash = 0.0

    alpaca_qty = {}
    try:
        positions = trading_client.get_all_positions()
        for pos in positions:
            alpaca_qty[pos.symbol] = float(pos.qty)
    except Exception as e:
        print(f"Error fetching Alpaca positions: {e}")

    ind_primed = {}
    for ind, syms in industries.items():
        ind_primed[ind] = {
            'cash':     total_cash * 0.40,
            'holdings': {sym: alpaca_qty.get(sym, 0.0) for sym in syms},
        }

    master_holdings = {ind: allocations.get(ind, 0.0) for ind in industries}
    industry_list   = list(industries.keys())

    master_primed = {
        'cash':        total_cash * (1.0 - 0.40),
        'holdings':    master_holdings,
        'zero_counts': {ind: (zero_counts or {}).get(ind, 0) for ind in industry_list},
    }

    return ind_primed, master_primed

def load_stock_data(symbols):
    """Load rolling 15-day OHLCV histories from stock_data/ for each symbol."""
    histories = {}
    for sym in symbols:
        try:
            file_path = f"{STOCK_DATA_DIR}/{sym}.json"
            if os.path.exists(file_path):
                with open(file_path) as f:
                    data = json.load(f)
                # Drop non-finite bars rather than feeding NaN into inference. yfinance
                # writes NaN for a missing session; one such bar (SCCO 2026-08-11) reached
                # the trainer's valuation and surfaced as prod=$-2147483648. download_daily
                # now purges them at the source, but this path reads whatever is on disk.
                histories[sym] = [
                    [d['open'], d['close'], d['high'], d['low'], d['volume']]
                    for d in data.get('days', [])[-15:]
                    if all(isinstance(d.get(k), (int, float)) and math.isfinite(d[k])
                           and d[k] > 0 for k in ('open', 'high', 'low', 'close'))]
            else:
                histories[sym] = []
        except Exception as e:
            print(f"Error loading stock data for {sym}: {e}")
            histories[sym] = []
    return histories

def save_stock_data(symbol, day_data, date=None):
    """Append one session's OHLCV to stock_data/<symbol>.json without truncating, under the
    session's own date (`date`; today if not given).

    Full history is preserved so the C++ trainer and download_daily.py can
    use the same files. download_daily.py trims to MAX_HISTORY_DAYS on each run.
    """
    try:
        file_path = f"{STOCK_DATA_DIR}/{symbol}.json"
        existing_data = []
        if os.path.exists(file_path):
            with open(file_path) as f:
                existing_data = json.load(f).get('days', [])
        today_str = date or datetime.today().strftime('%Y-%m-%d')
        new_data_point = {
            'date': today_str,
            'open': day_data['open'],
            'close': day_data['close'],
            'high': day_data['high'],
            'low': day_data['low'],
            'volume': day_data['volume']
        }
        # Dedup: replace the entry for today if it already exists (e.g. re-run)
        existing_data = [d for d in existing_data if d['date'] != today_str]
        existing_data.append(new_data_point)
        with open(file_path, 'w') as f:
            json.dump({'days': existing_data}, f)
    except Exception as e:
        print(f"Error saving stock data for {symbol}: {e}")

def last_bar_dates(symbols):
    """{sym: date of its newest bar in stock_data}; a symbol with no file or no bars is absent."""
    out = {}
    for sym in symbols:
        try:
            with open(f"{STOCK_DATA_DIR}/{sym}.json") as f:
                days = json.load(f).get('days', [])
            if days:
                out[sym] = days[-1]['date']
        except Exception:
            pass
    return out


def suspended_symbols(symbols, last_dates):
    """Symbols that stopped arriving: newest bar older than the universe's newest session, or no
    data at all. -> (session, {sym: its last bar date or None}).

    A symbol that stops downloading is the first sign it is no longer tradable (halted,
    delisted, renamed). Its trades are suspended -- no buys, sells or stops are placed for it,
    and its held shares stay held -- while every other symbol trades as usual. It resumes by
    itself the first session its data arrives again. One missed session is enough: an order on
    a bar that is not today's would be priced off a stale day."""
    session = max(last_dates.values(), default=None)
    return session, {s: last_dates.get(s) for s in symbols
                     if session is None or last_dates.get(s, '') < session}


LOG_DIR = 'logs'  # overridden in main() via --account/--paper: logs/ACCOUNT/paper|prod

def log_request_id(request_id, context, success=True):
    """Persist Alpaca X-Request-ID to {LOG_DIR}/request_ids.log for support tracing."""
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, 'request_ids.log')
    entry = json.dumps({
        'ts': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'request_id': request_id,
        'context': context,
        'success': success,
    })
    with open(path, 'a') as f:
        f.write(entry + '\n')

def log_data_failure(sym, reason, mode):
    """Append a JSON entry to {LOG_DIR}/data_fetch_failures.log and print an alert."""
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, 'data_fetch_failures.log')
    entry = json.dumps({
        'ts': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'mode': mode,
        'symbol': sym,
        'reason': reason,
    })
    with open(path, 'a') as f:
        f.write(entry + '\n')
    print(f"ALERT: data fetch failed for {sym} ({reason}) — logged to {path}")


def fetch_intraday_data(symbols):
    """
    Download 1-minute bars for today's session for each symbol.
    Returns {sym: [{'open', 'high', 'low', 'close', 'volume'}, ...]} ordered by time.
    Used post-close to reconstruct the actual intraday high/low sequence for fill simulation.
    """
    intraday = {}
    try:
        tickers = yf.Tickers(' '.join(symbols))
        for sym in symbols:
            try:
                bars = tickers.tickers[sym].history(period='1d', interval='1m')
                if not bars.empty:
                    intraday[sym] = [
                        {'open':   float(r['Open']),
                         'high':   float(r['High']),
                         'low':    float(r['Low']),
                         'close':  float(r['Close']),
                         'volume': float(r['Volume'])}
                        for _, r in bars.iterrows()
                    ]
            except Exception as e:
                print(f"Warning: could not fetch intraday data for {sym}: {e}")
    except Exception as e:
        print(f"Warning: intraday batch fetch failed: {e}")
    return intraday


def get_seq_flags_from_intraday(symbols, intraday_data, day_data):
    """
    Derive the actual intraday high/low sequence per symbol from 1-minute bars.
    Returns {sym: bool} where True = low reached before high (low-first),
    False = high reached before low (high-first).
    Falls back to random 50/50 for any symbol without intraday data.
    """
    seq_flags = {}
    for sym in symbols:
        bars = intraday_data.get(sym, [])
        if not bars:
            seq_flags[sym] = random.random() < 0.5
            continue

        daily_high = day_data.get(sym, {}).get('high', 0.0)
        daily_low  = day_data.get(sym, {}).get('low',  float('inf'))

        high_idx = None
        low_idx  = None
        for i, bar in enumerate(bars):
            if high_idx is None and bar['high'] >= daily_high:
                high_idx = i
            if low_idx is None and bar['low'] <= daily_low:
                low_idx = i
            if high_idx is not None and low_idx is not None:
                break

        if high_idx is None or low_idx is None:
            seq_flags[sym] = random.random() < 0.5
        else:
            seq_flags[sym] = low_idx <= high_idx  # True = low-first

    return seq_flags


def compute_industry_current_values(industries, holdings, histories):
    """
    Compute each industry's current total value using yesterday's close prices
    (last entry in histories). Used by master for liquidation sizing pre-market.
    Returns {industry: (holdings_value, 0.0)} — cash not tracked per-industry
    in production; master uses total_cash separately.
    """
    ind_values = {}
    for ind, syms in industries.items():
        hold_val = 0.0
        for sym in syms:
            qty   = holdings.get(sym, 0.0)
            hist  = histories.get(sym, [])
            price = float(hist[-1][1]) if hist else 0.0   # index 1 = close
            hold_val += qty * price
        ind_values[ind] = (hold_val, 0.0)
    return ind_values


def run_master_allocation(master_model, industries, mkt_val_history, pf_val_history, zero_counts,
                          total_cash, model_dir=None, flat=False):
    """
    Run master inference to get tier classification and capital allocation.
    Mutates zero_counts in place.

    mkt_val_history: {ind: [cumulative_market_index]}  — market feature source.
    pf_val_history:  {ind: [cumulative_slot0_portfolio_index]} — portfolio feature source (Part C).
    Priority: flat override → MT2 (mt2_best.pt exists) → legacy MasterNN (market-only 444).
    Returns (allocations, tier_map, mt1_outputs).
      allocations:  {ind: dollar_amount}
      tier_map:     {ind: 0-3}
      mt1_outputs:  {ind: predicted_next_session_pnl_dollars} or None for legacy path
    """
    from training_lib import _build_ind_features37, decode_master_tiers, tiers_to_alloc
    industry_list = list(industries.keys())
    tier_map      = {ind: 0 for ind in industry_list}
    mt1_outputs   = None

    # Flat override (--flat-allocation): equal weight to every industry, MT1/MT2 skipped.
    # Deliberately NOT expressed as a fallback. Two traps make the "no models" path unusable
    # as a stand-in for flat:
    #   1. With no MT2 and no MasterNN, tier_map stays all-zero and tiers_to_alloc returns
    #      $0.00 for every industry (n_pos == 0) — an empty book, not an even one.
    #   2. tiers_to_alloc re-ranks into terciles, so even a uniform positive tier_map comes
    #      back weighted 1.0/1.5/2.25, not equal. The split has to be done explicitly.
    # zero_counts is cleared rather than incremented: 3 consecutive tier-0 readings liquidate
    # an industry's holdings, so leaving it to accumulate would sell the book off on day 3.
    if flat:
        n     = len(industry_list)
        share = total_cash / n if n else 0.0
        tier_map    = {ind: 1 for ind in industry_list}
        allocations = {ind: share for ind in industry_list}
        for ind in industry_list:
            zero_counts[ind] = 0
        print(f"Flat allocation: {n} industries x ${share:,.2f} "
              f"(${total_cash:,.2f} total) — MT1/MT2 inference skipped")
        return allocations, tier_map, None

    # MT2 path — preferred if mt2_best.pt exists (MT2 consumes raw MT1 activations, no norm stats).
    mt2_path = os.path.join(model_dir, 'mt2_best.pt') if model_dir else None
    if mt2_path and os.path.exists(mt2_path):
        try:
            allocations, tier_map, mt1_outputs = run_mt_inference(
                model_dir, industries, mkt_val_history, pf_val_history, zero_counts, total_cash)
            return allocations, tier_map, mt1_outputs
        except Exception as e:
            print(f"Warning: MT2 inference failed ({e}) — falling back to MasterNN")

    # Legacy MasterNN fallback — market-only 444 features (MasterNN predates the dual MT1 input).
    if master_model is not None:
        try:
            feats = [f for ind in industry_list
                     for f in _build_ind_features37(mkt_val_history.get(ind, []))]
            today_t = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out = master_model(today_t)
            tier_map = decode_master_tiers(out, industry_list)
        except Exception as e:
            print(f"Error running master model: {e}")

    for ind in industry_list:
        if tier_map[ind] == 0:
            zero_counts[ind] = zero_counts.get(ind, 0) + 1
        else:
            zero_counts[ind] = 0

    allocations = tiers_to_alloc(tier_map, industry_list, total_cash)
    return allocations, tier_map, mt1_outputs


def compute_liquidation_orders(industries, ind_current_values, alloc_prop,
                                total_account_value, total_cash):
    """
    Determine which industries need to liquidate holdings to fund reallocation.

    Industries where master wants less capital than currently held must sell
    down to the target, but never below 2% floor of total account value.
    Liquidation priority: weakest predicted first (lowest alloc_prop).

    Returns {ind: dollar_amount_to_liquidate}
    """
    industry_list = list(industries.keys())
    floor_value   = total_account_value * 0.02
    liq_orders    = {}

    # Current total value per industry (holdings only — cash is pooled)
    ind_hold_val  = {ind: ind_current_values.get(ind, (0.0, 0.0))[0]
                     for ind in industry_list}

    # Target holdings value per industry
    # (target allocation × total_cash — this is what master wants deployed)
    ind_target    = {ind: alloc_prop.get(ind, 0.0) * total_cash
                     for ind in industry_list}

    # Industries that need to shrink: current holdings > target AND above floor
    shrink = [
        ind for ind in industry_list
        if ind_hold_val[ind] > ind_target[ind]
        and ind_hold_val[ind] > floor_value
    ]

    # Sort: weakest predicted first, spread proportionally among ties
    shrink.sort(key=lambda ind: (alloc_prop.get(ind, 0.0), -(ind_hold_val[ind] - ind_target[ind])))

    for ind in shrink:
        needed   = ind_hold_val[ind] - ind_target[ind]
        max_liq  = max(0.0, ind_hold_val[ind] - floor_value)
        liq_amt  = min(needed, max_liq)
        if liq_amt > 1e-6:
            liq_orders[ind] = liq_amt

    return liq_orders


def generate_liquidation_sells(industry, symbols, liquidation_target,
                                holdings, day_data, histories, orders):
    """
    Generate sell orders for an industry to meet a liquidation target.
    Sells lowest-conviction holdings first (smallest current position value).
    Stops when target met or no more holdings above floor.
    Appends sell orders to the shared orders list.
    Returns dollar amount actually liquidated.
    """
    if liquidation_target <= 1e-6:
        return 0.0

    # Value each current holding at yesterday's close (pre-market)
    sym_values = []
    for sym in symbols:
        qty   = holdings.get(sym, 0.0)
        hist  = histories.get(sym, [])
        price = float(hist[-1][1]) if hist else 0.0
        if qty > 0 and price > 0:
            sym_values.append((sym, qty, price, qty * price))

    # Sort lowest value first (lowest conviction = sell first)
    sym_values.sort(key=lambda x: x[3])

    remaining   = liquidation_target
    liquidated  = 0.0

    for sym, qty, price, val in sym_values:
        if remaining <= 1e-6:
            break
        sell_val    = min(val, remaining)
        sell_qty    = int(min(sell_val / price, qty))
        if sell_qty >= 1:
            # Use today's close if available, else yesterday's
            exec_price = day_data.get(sym, {}).get('close', price)
            orders.append({'symbol': sym, 'action': 'sell',
                           'quantity': sell_qty, 'price': exec_price})
            proceeds   = _sell_net(sell_qty, exec_price)
            remaining -= proceeds
            liquidated += proceeds

    return liquidated


def stocknn_outputs(model, symbols, histories, day_data, holdings, cash):
    """The industry model's (12, 4) output at today's close for a book holding `holdings` with
    `cash` to spend: per symbol buy_qty, buy_price_frac, sell_all_price_frac, sell_qty."""
    # Rolling stats for new input features
    sym_stats = {}
    for sym in symbols:
        h = histories.get(sym, [])
        if len(h) >= 2:
            closes = [r[1] for r in h]
            vols   = [r[4] for r in h]
            hi15   = max(r[2] for r in h)
            lo15   = min(r[3] for r in h)
            avg_c  = sum(closes)/len(closes)
            avg_v  = sum(vols)/len(vols) if sum(vols) > 0 else 1.0
            dvols  = [r[0]*r[4] for r in h]
            avg_dv = sum(dvols)/len(dvols) if sum(dvols) > 0 else 1.0
            std_c  = (sum((c-avg_c)**2 for c in closes)/len(closes))**0.5
            sym_stats[sym] = {'hi15': hi15, 'lo15': lo15,
                              'avg_c': avg_c if avg_c > 0 else 1.0,
                              'avg_v': avg_v, 'avg_dv': avg_dv,
                              'volatility': std_c/avg_c if avg_c > 0 else 0.0}
        else:
            sym_stats[sym] = {'hi15': 1.0, 'lo15': 0.0, 'avg_c': 1.0,
                              'avg_v': 1.0, 'avg_dv': 1.0, 'volatility': 0.0}

    # Industry aggregates for cross-correlation (15 values per timestep)
    num_past = min(len(histories.get(sym, [])) for sym in symbols)
    ind_agg_per_t = []
    for t in range(15):
        if t >= num_past:
            ind_agg_per_t.append([0.0]*15)
        else:
            dl = [histories[sym][-(t+1)][5:] for sym in symbols
                  if len(histories.get(sym,[])) > t]
            if dl:
                tr   = list(zip(*dl, strict=True))
                aggs = []
                for tp in tr:
                    aggs += [max(tp), min(tp), sum(tp)/len(tp)]
                ind_agg_per_t.append(aggs[:15])
            else:
                ind_agg_per_t.append([0.0]*15)

    # history_t: (1,15,60) — OHLCV only, oldest first
    history_rows = []
    for t in range(14, -1, -1):
        row = []
        for sym in symbols:
            h = histories.get(sym, [])
            row += list(h[-(t+1)][:5]) if len(h) > t else [0.0] * 5
        history_rows.append(row)
    history_t = torch.tensor(history_rows, dtype=torch.float32).unsqueeze(0)  # (1,15,60)

    # today_t: (1,232) — full features for current day
    state_vec = [cash] + [holdings.get(sym, 0.0) for sym in symbols]
    today_row = []
    today_dl  = []
    for sym in symbols:
        st    = sym_stats[sym]
        d     = day_data.get(sym, {})
        raw_t = [d.get('open', 0), d.get('close', 0), d.get('high', 0),
                 d.get('low', 0), d.get('volume', 0)]
        h     = histories.get(sym, [])
        prev  = h[-1][:5] if h else None
        dlt_t = [raw_t[i] - prev[i] for i in range(5)] if prev else [0.0] * 5
        rng   = max(st['hi15'] - st['lo15'], 1e-9)
        today_row += (raw_t + dlt_t +
                      [(raw_t[1] - st['lo15']) / rng,
                       raw_t[1] / st['avg_c'],
                       st['volatility'],
                       raw_t[4] / st['avg_v'],
                       (raw_t[0] * raw_t[4]) / st['avg_dv'],
                       stock_close_pos(raw_t[0], raw_t[2],
                                       raw_t[3], raw_t[1]),
                       stock_close_vs_wap(raw_t[0], raw_t[2],
                                          raw_t[3], raw_t[1])])
        today_dl.append(dlt_t)
    if today_dl:
        tr = list(zip(*today_dl, strict=True))
        for tp in tr: today_row += [max(tp), min(tp), sum(tp)/len(tp)]
    else:
        today_row += [0.0] * 15
    today_row += state_vec
    today_t = torch.tensor(today_row, dtype=torch.float32).unsqueeze(0)  # (1,232)

    with torch.no_grad():
        out = model(history_t, today_t)
    out  = out.view(12, 4)   # (buy_qty, buy_price, sell_all_price, sell_qty)
    return out.tolist()


def to_cents(price, how):
    """A price Alpaca accepts: whole cents at $1 and over, 1/100 cent below. 'down' for a buy
    limit and 'up' for a sell limit, so rounding never makes an order worse than the model's price;
    'near' for a stop. Alpaca rejects every other increment ("sub-penny increment does not fulfill
    minimum pricing criteria") -- which is what every order of the first paper session did."""
    step = 0.01 if price >= 1.0 else 0.0001
    n = price / step
    k = math.floor(n + 1e-6) if how == 'down' else math.ceil(n - 1e-6) if how == 'up' else round(n)
    return round(k * step, 4)


def plan_industry_orders(symbols, sug, holdings, day_data, allocated_cash, holds,
                         sell_only=False, stops=None):
    """One industry's orders from StockNN's suggestions: sells, then buys, then stops.

    sug:   {sym: (buy_qty, buy_price_frac, sell_all_price_frac, sell_qty)}, the model's output
    holds: {sym: runs left} -- symbols under the new-symbol hold get no orders
    stops: {sym: stop price} protecting held shares -- the open stops the caller cancelled, or the
           stop recorded for a position whose stop expired

    Every price is rounded to an increment Alpaca accepts (to_cents). A buy carries its stop-loss
    in the same order (`stop`, submitted as one OTO order: the stop goes live when the buy fills),
    because Alpaca rejects a separate sell stop while a buy for the same symbol is open ("potential
    wash trade"). For the same reason a symbol the model sells today is not also bought today.

    Held shares keep their protection: the caller cancels a held symbol's stops before selling
    (Alpaca reserves the shares a stop covers), so a GTC stop is re-placed, at its old price, for
    whatever the day's sells leave held -- unless the symbol is being bought today, when the open
    buy would make it a wash trade and the held shares go unprotected for that session. A symbol
    with no recorded stop gets none invented.

    sell_only=True is the loser zone: StockNN runs as usual and its sells go in, but nothing is
    bought, so the industry drains at the model's own sell rate. It is NOT liquidation.

    -> list of order dicts in submission order."""
    stops = stops or {}
    orders, left, sold, bought = [], {}, set(), set()
    for sym in symbols:
        if sym not in day_data or holds.get(sym, 0) > 0:
            continue
        buy_qty, buy_price_frac, sell_all_price_frac, sell_qty = sug[sym]
        cur_qty        = holdings.get(sym, 0.0)
        low            = day_data[sym]['low']
        high           = day_data[sym]['high']
        close          = day_data[sym]['close']
        span           = max(high - low, 1e-9)
        sell_all_price = low + sell_all_price_frac * span

        # Sell qty at market (open price)
        if sell_qty > 1e-6 and cur_qty > 1e-6:
            amount = int(min(sell_qty, cur_qty))
            if amount >= 1:
                orders.append({'symbol': sym, 'action': 'sell',
                               'quantity': amount, 'price': to_cents(close, 'up')})
                cur_qty -= amount
                sold.add(sym)

        # Sell all at limit price
        if cur_qty >= 1 and low <= sell_all_price <= high:
            orders.append({'symbol': sym, 'action': 'sell',
                           'quantity': int(cur_qty), 'price': to_cents(sell_all_price, 'up')})
            cur_qty = 0.0
            sold.add(sym)
        left[sym] = cur_qty

    if not sell_only:
        # ── Buys after sells: stop anchored to buy_price ─────────────
        # allocated_cash is the industry's budget for the WHOLE loop, so it has to be
        # spent down as orders are queued. Reading it fresh per symbol sized every
        # position against the full allocation: measured 2026-09-17, tech_hardware
        # queued ~$12,800 against a $1,666.67 allocation (ON 99%, LRCX 96%, SWKS 96%,
        # KLAC 94%, NVDA 91% ... each of the full budget). training_lib.step_industry
        # spends its cash down correctly; this path did not.
        remaining_cash = allocated_cash
        # Industry portfolio value for the single-stock cap, on the same basis the model
        # was fed (allocated_cash + holdings), mirroring training_lib.step_industry.
        ind_port_value = allocated_cash + sum(
            holdings.get(s, 0.0) * day_data.get(s, {}).get('close', 0.0)
            for s in symbols)
        for sym in symbols:
            if sym not in day_data or holds.get(sym, 0) > 0 or sym in sold:
                continue
            buy_qty, buy_price_frac, sell_all_price_frac, sell_qty = sug[sym]
            low       = day_data[sym]['low']
            high      = day_data[sym]['high']
            span      = max(high - low, 1e-9)
            raw_price = low + buy_price_frac * span

            if buy_qty > 1e-6 and raw_price > 0 and low <= raw_price <= high:
                buy_price = to_cents(raw_price, 'down')
                # Shared with the simulator: size_buy is the ONE place the request /
                # cash / single-stock-cap arithmetic lives.
                amount = int(training_lib.size_buy(
                    buy_qty, buy_price, remaining_cash, ind_port_value,
                    holdings.get(sym, 0.0) * buy_price))
                if amount >= 1:
                    orders.append({'symbol': sym, 'action': 'buy', 'quantity': amount,
                                   'price': buy_price, 'stop': to_cents(buy_price * 0.9, 'near')})
                    remaining_cash -= amount * buy_price * BUY_FILL
                    bought.add(sym)

    for sym in symbols:
        if left.get(sym, 0.0) >= 1 and sym in stops and sym not in bought:
            orders.append({'symbol': sym, 'action': 'stop_loss',
                           'quantity': int(left[sym]), 'price': to_cents(stops[sym], 'near')})
    return orders


# ── Daily training with the C++ trainer (--cpp-upkeep) ───────────────────────────
# The daily cycle is: download the day's bars -> train one day -> convert the StockNN elites ->
# run the models -> place tomorrow's orders. Training is training_v4_cpp --upkeep, the same code
# that trained the race (see upkeep_cpp.py), on the books this run recorded the session before.
# Its state lives in <model_dir>/cpp; <model_dir>/champion is the seed it starts from.
UPKEEP_BINARY = os.environ.get('TRAINER_BIN', 'build/training_v4_cpp')
# The race entries the committees vote with -- the winner committee (TANH32 grad, MT1 grdC,
# MT1 evo, MT1S-rep evo, MT1S-stk grad) and the loser committee (TANH32 grad, MT1S-stk evo,
# ALLOC bayes mean), picked on the v5 champion composite. Only these are trained daily; the
# trainer also always runs MT1C grad and MT1 evo, which it reads by index.
UPKEEP_RACE_ONLY = ("TANH32 grad", "MT1 grdC", "MT1 evo", "MT1S-rep evo", "MT1S-stk grad",
                    "MT1S-stk evo", "ALLOC bayes mean")


def convert_stocknn(state_dir, model_dir):
    """The trained StockNN elites, .bin -> .pt, for this run's inference."""
    from convert_weights import STOCK_LAYER_DEFS, convert_industry
    for ind in INDUSTRIES:
        convert_industry(ind, state_dir, model_dir, STOCK_LAYER_DEFS, StockNN, ind)


def run_daily_training(model_dir):
    """Train the day the previous run placed orders on, then convert. -> True if the models
    changed. A failure leaves the previous models in place: a bad training day must not stop the
    orders, and the trainer refuses (rc 3) a day it has already trained, so a re-run is safe."""
    state_dir = os.path.join(model_dir, 'cpp')
    seed_dir = os.path.join(model_dir, 'champion')
    book = os.path.join(state_dir, 'book_next.txt')
    if not os.path.exists(book):
        print("Daily training: no book recorded by a previous run yet -- nothing to train")
        return False
    if not os.path.isdir(seed_dir):
        print(f"Daily training: ERROR seed directory {seed_dir} is missing -- not training")
        return False
    log = os.path.join(LOG_DIR, 'upkeep_cpp.log')
    print(f"Daily training: training_v4_cpp --upkeep (log {log}) ...")
    rc = upkeep_cpp.run_upkeep(UPKEEP_BINARY, book, state_dir, seed_dir, UPKEEP_RACE_ONLY, log)
    if rc == 3:
        print("Daily training: that day is already trained; models unchanged")
        return False
    if rc != 0:
        print(f"Daily training: ERROR rc={rc} (see {log}); trading on the previous models")
        return False
    st = upkeep_cpp.read_state(state_dir)
    upkeep_cpp.archive_logs(os.path.join(state_dir, 'log'), os.path.join(LOG_DIR, 'training'),
                            st['date'])
    convert_stocknn(state_dir, model_dir)
    print(f"Daily training: {st['date']} trained, StockNN elites converted, logs in "
          f"{os.path.join(LOG_DIR, 'training')}")
    return True


def save_upkeep_book(model_dir, trading_client, industries, allocations, sell_only, orders,
                     day_data):
    """Record today's real books -- what each industry holds (Alpaca), the cash it was given and
    the stops queued -- dated today, for tomorrow's training step. A sell-only industry was given
    no cash. CN_BOOK's base, each industry's starting book, is fixed on the first write."""
    state_dir = os.path.join(model_dir, 'cpp')
    os.makedirs(state_dir, exist_ok=True)
    try:
        positions = {p.symbol: float(p.qty) for p in trading_client.get_all_positions()}
    except Exception as e:
        print(f"Daily training: ERROR reading positions ({e}); no book recorded for tomorrow")
        return
    stops = {}
    for o in orders:
        p = o['price'] if o['action'] == 'stop_loss' else o.get('stop')
        if p:
            stops[o['symbol']] = max(stops.get(o['symbol'], 0.0), float(p))
    books, value = {}, {}
    for ind, syms in industries.items():
        cash = 0.0 if ind in sell_only else float(allocations.get(ind, 0.0))
        hold = {s: positions.get(s, 0.0) for s in syms}
        books[ind] = {'cash': cash, 'holdings': hold,
                      'stop_prices': {s: stops.get(s, 0.0) for s in syms}}
        value[ind] = cash + sum(q * day_data.get(s, {}).get('close', 0.0) for s, q in hold.items())
    base_path = os.path.join(state_dir, 'book_base.json')
    if os.path.exists(base_path):
        with open(base_path) as f:
            base = json.load(f)
    else:
        even = sum(value.values()) / max(1, len(value))
        base = {ind: (v if v > 0 else even) or training_lib.IND_STARTING_CASH
                for ind, v in value.items()}
        with open(base_path, 'w') as f:
            json.dump(base, f, indent=1)
    upkeep_cpp.write_book(os.path.join(state_dir, 'book_next.txt'),
                          datetime.today().strftime('%Y-%m-%d'), books, base)
    print(f"Daily training: today's books recorded for tomorrow ({len(books)} industries)")


PENDING_DECISION = 'pending_decision.json'
COMMITTEE_LOG = 'committee_daily.csv'


def ensure_book_base(state_dir, industries, holdings, day_data, cash):
    """{ind: starting book}, the base CN_BOOK is read against, fixed the first time it is needed.
    Under --committee each industry starts from an even share of the account's cash plus what
    it holds, so every industry begins at the same scale whatever the first day's vote was."""
    path = os.path.join(state_dir, 'book_base.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    share = cash / max(1, len(industries))
    base = {ind: share + sum(holdings.get(s, 0.0) * day_data.get(s, {}).get('close', 0.0)
                             for s in syms)
            for ind, syms in industries.items()}
    base = {ind: v if v > 0 else training_lib.IND_STARTING_CASH for ind, v in base.items()}
    os.makedirs(state_dir, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(base, f, indent=1)
    return base


def committee_decision(model_dir, industries, histories, day_data, holdings, cash):
    """The committees' allocation for today (committee_alloc). Each industry's race inputs are
    built from its real holdings, an even share of the cash (the allocation is what is being
    decided, so it cannot be an input), and its deployed StockNN's output on that book.
    -> dict(sell_only, winners, allocations, preds)"""
    state_dir = os.path.join(model_dir, 'cpp')
    seed_dir = os.path.join(model_dir, 'champion')
    st = upkeep_cpp.read_state(state_dir)
    if st is None:
        raise RuntimeError(f"no daily-training state in {state_dir}")
    inds = list(industries)
    share = cash / len(inds)
    base = ensure_book_base(state_dir, industries, holdings, day_data, cash)
    inputs = {}
    for ind, syms in industries.items():
        model = load_weighted_model(StockNN, model_dir, ind)
        if model is None:
            print(f"Committee: no StockNN for {ind}; it sits out the vote")
            continue
        intent = stocknn_outputs(model, syms, histories, day_data, holdings, share)
        bars = []
        for sym in syms:
            d = day_data.get(sym)
            okb = d is not None and all(math.isfinite(d.get(k, float('nan'))) and d[k] > 0
                                        for k in ('open', 'high', 'low', 'close'))
            bars.append((d['open'], d['high'], d['low'], d['close']) if okb else None)
        inputs[ind] = committee_alloc.industry_inputs(
            bars, [holdings.get(sym, 0.0) for sym in syms], share, intent, base[ind])
    feats = training_lib.build_master_features(st['mkt'], st['pf'], inds)[0].numpy()
    in74 = {ind: feats[i * 74:(i + 1) * 74] for i, ind in enumerate(inds)}
    bayes_mu = {ind: v[0] for ind, v in st['bayes'].items()}
    preds = committee_alloc.predict(inds, inputs, in74, bayes_mu, state_dir, seed_dir)
    d = committee_alloc.decide(inds, preds, cash)
    d['preds'] = preds
    print(f"Committee (models through {st['date']}): sell-only {', '.join(d['sell_only'])}; "
          f"tilt toward {', '.join(d['winners'])}")
    for ind in inds:
        tag = 'SELL-ONLY' if ind in d['sell_only'] else 'winner' if ind in d['winners'] else ''
        print(f"  {ind:24} ${d['allocations'][ind]:>10.2f}  {tag}")
    return d


def settle_pending(model_dir, trading_client, day_data):
    """Settle the previous run's decision against what the account did since, appending its rows
    to committee_daily.csv. A decision from today (a re-run) is left for tomorrow."""
    path = os.path.join(model_dir, 'cpp', PENDING_DECISION)
    if not os.path.exists(path):
        return
    with open(path) as f:
        pending = json.load(f)
    if pending['date'] >= datetime.today().strftime('%Y-%m-%d'):
        return
    try:
        positions = {p.symbol: float(p.qty) for p in trading_client.get_all_positions()}
        equity = float(trading_client.get_account().equity)
        req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=500,
                               after=datetime.fromisoformat(pending['time']))
        fills = [(o.symbol, 'buy' if o.side == OrderSide.BUY else 'sell', float(o.filled_qty),
                  float(o.filled_avg_price))
                 for o in trading_client.get_orders(req)
                 if o.filled_qty and float(o.filled_qty) > 0 and o.filled_avg_price]
    except Exception as e:
        print(f"Committee log: ERROR reading the account ({e}); {pending['date']} left unsettled")
        return
    closes = {s: d['close'] for s, d in day_data.items()}
    real = committee_alloc.industry_returns(pending, closes, positions, fills)
    acct_bp = (equity / pending['equity'] - 1.0) * 1e4 if pending['equity'] > 0 else float('nan')
    log = os.path.join(LOG_DIR, COMMITTEE_LOG)
    n = 0
    if os.path.exists(log):
        with open(log) as f:
            n = sum(1 for line in f if ',REALIZED,' in line)
    committee_alloc.append_rows(log, pending['industries'],
                                committee_alloc.settle_rows(pending, real, acct_bp, n))
    os.replace(path, path + '.settled')
    print(f"Committee log: {pending['date']} settled ({len(fills)} fills, account "
          f"{acct_bp:+.1f} bp) -> {log}")


def record_decision(model_dir, industries, decision, holdings, day_data, histories, cash,
                    equity, suspended, no_orders):
    """Today's decision, for the next run to settle. Written before any order is submitted, so
    every fill after `time` belongs to it. equity: the account's, as Alpaca reports it, so the
    next run's PORTFOLIO return compares like with like. A suspended symbol's close is its last
    bar's."""
    inds = list(industries)
    closes = {}
    for syms in industries.values():
        for s in syms:
            if s in day_data:
                closes[s] = day_data[s]['close']
            elif histories.get(s):
                closes[s] = histories[s][-1][1]
    pending = dict(date=datetime.today().strftime('%Y-%m-%d'),
                   time=datetime.now().astimezone().isoformat(),
                   industries=inds, symbols={i: list(industries[i]) for i in inds},
                   allocations=decision['allocations'], sell_only=decision['sell_only'],
                   winners=decision['winners'],
                   preds={e: [None if not math.isfinite(x) else float(x) for x in v]
                          for e, v in decision['preds'].items()},
                   holdings={s: q for s, q in holdings.items() if q}, closes=closes,
                   cash=cash, equity=equity, suspended=sorted(suspended), no_orders=no_orders)
    path = os.path.join(model_dir, 'cpp', PENDING_DECISION)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + '.tmp', 'w') as f:
        json.dump(pending, f, indent=1)
    os.replace(path + '.tmp', path)


def main():
    """Run one daily trading cycle: fetch data, infer, submit orders, retrain."""
    parser = argparse.ArgumentParser(description="Run stock trading system v2.")
    parser.add_argument('--paper',   action='store_true', help='Use Alpaca paper trading API (real paper orders, real paper portfolio)')
    parser.add_argument('--account', default='acct0', help='Account identifier (e.g. acct0); derives models/ACCOUNT/paper|prod and logs/ACCOUNT/paper|prod')
    parser.add_argument('--capital', type=float, default=None, help='Cap total deployed capital regardless of Alpaca account balance')
    parser.add_argument('--withdraw', type=float, help='Amount to withdraw from portfolio')
    parser.add_argument('--no-orders', action='store_true',
                        help='Run the full cycle — fetch, allocate, decide, and train the daily '
                             'upkeep step — but submit NOTHING to Alpaca and cancel nothing. For '
                             'catching the models up on missed sessions without trading.')
    parser.add_argument('--flat-allocation', action='store_true',
                        help='Allocate capital evenly across all industries, skipping MT1/MT2 '
                             'inference. Use while MT1/MT2 are unconfirmed.')
    parser.add_argument('--sell-only', default='', metavar='IND[,IND...]',
                        help='Industries in sell-only mode (the loser zone): StockNN still runs '
                             'and its sells are placed, but nothing is bought, so the industry '
                             'drains at the model\'s own sell rate. Not liquidation. Their '
                             'allocation is not spent.')
    parser.add_argument('--committee', action='store_true',
                        help='Allocate with the race committees: the loser committee puts 4 '
                             'industries in sell-only, the winner committee tilts the other 8 '
                             '(t = 0.5). Requires --cpp-upkeep, whose state it reads.')
    parser.add_argument('--cpp-upkeep', action='store_true',
                        help='Daily training with the C++ trainer: before deciding, train the day '
                             'the previous run placed orders on (training_v4_cpp --upkeep) and '
                             'convert the StockNN elites; replaces the Python upkeep step.')
    args = parser.parse_args()
    if args.committee and (args.flat_allocation or not args.cpp_upkeep):
        parser.error('--committee needs --cpp-upkeep and excludes --flat-allocation')

    subtype   = 'paper' if args.paper else 'prod'
    model_dir = os.path.join('models', args.account, subtype)
    global LOG_DIR
    LOG_DIR   = os.path.join('logs', args.account, subtype)
    training_lib.DUMP_DIR = os.path.join(LOG_DIR, 'data_dump')
    print(f"[production_v2] v{VERSION}  account={args.account}  mode={subtype}")

    # Held for the whole cycle, including the upkeep training step at the end, so the C++ trainer
    # stays paused until this process is completely done.
    with trading_lock(subtype, args.account):
        # Retrieve API keys
        api_key = os.environ.get('ALPACA_API_KEY')
        secret_key = os.environ.get('ALPACA_SECRET_KEY')
        if not api_key or not secret_key:
            try:
                api_key = keyring.get_password("trading", "ALPACA_API_KEY")
                secret_key = keyring.get_password("trading", "ALPACA_SECRET_KEY")
            except Exception as e:
                print(f"Error retrieving Alpaca API keys from keyring: {e}")
        if not api_key or not secret_key:
            print("ALPACA_API_KEY or ALPACA_SECRET_KEY not found in environment or keyring")
            return

        # Initialize Alpaca client
        try:
            trading_client = TradingClient(api_key, secret_key, paper=args.paper)
        except Exception as e:
            print(f"Error initializing Alpaca client: {e}")
            return

        # Load state
        state = load_state(model_dir)
        industries = state['industries']
        sell_only = {x.strip() for x in args.sell_only.split(',') if x.strip()}
        unknown = sell_only - set(industries)
        if unknown:
            print(f"--sell-only: unknown industries {sorted(unknown)}; known: {sorted(industries)}")
            return
        histories = state['histories']
        # The account is the book: cash and positions come from Alpaca, never from state.json,
        # whose defaults ($20,000, nothing held) describe no real account.
        try:
            cash = float(trading_client.get_account().cash)
            holdings = {p.symbol: float(p.qty) for p in trading_client.get_all_positions()}
        except Exception as e:
            print(f"ERROR reading the Alpaca account ({e}); not trading on an unknown book")
            return
        state['cash'], state['holdings'] = cash, holdings
        print(f"Account: ${cash:,.2f} cash, {len(holdings)} position(s)")

        # Detect symbols new to the universe since last run → 5-day order hold.
        # The hold suppresses paper/prod order submission only; training still runs normally.
        _known = state.get('known_symbols', [])
        if _known:
            _current_syms = set(sym for syms in industries.values() for sym in syms)
            _new_syms = _current_syms - set(_known)
            if _new_syms:
                _holds = state.setdefault('new_symbol_holds', {})
                for sym in sorted(_new_syms):
                    if sym not in _holds:
                        _holds[sym] = 5
                        print(f"New symbol {sym}: 5-day trading hold applied "
                              f"(trains normally, no paper/prod orders until hold expires)")
        state['known_symbols'] = [sym for syms in industries.values() for sym in syms]

        if args.capital is not None:
            cash = min(cash, args.capital)
            print(f"Capital capped at ${args.capital:.2f} (account cash: ${state['cash']:.2f})")

        # Load today's stock data using yfinance
        symbols = [sym for syms in industries.values() for sym in syms]

        # Snapshot yesterday's last OHLCV *before* saving today's data.
        # Upkeep training uses yesterday as model-input features and today as fill prices,
        # matching the offline training's day-N-predict / day-N+1-fill pattern.
        _yh = load_stock_data(symbols)
        yesterday_data: dict = {}
        for _sym in symbols:
            _h = _yh.get(_sym, [])
            if _h:
                _last = _h[-1]
                yesterday_data[_sym] = {
                    'open': _last[0], 'close': _last[1],
                    'high': _last[2], 'low': _last[3], 'volume': _last[4]
                }

        mode = 'paper' if args.paper else 'prod'
        day_data = {}
        try:
            tickers = yf.Tickers(' '.join(symbols))
            for sym in symbols:
                try:
                    ticker = tickers.tickers[sym]
                    hist = ticker.history(period='1d', interval='1d')
                    if not hist.empty:
                        # Saved under the bar's own date: a halted symbol returns its LAST bar,
                        # and stamping that with today's date would hide that it stopped.
                        bar_date = hist.index[-1].date().isoformat()
                        day_data[sym] = {
                            'open': float(hist['Open'].iloc[0]),
                            'close': float(hist['Close'].iloc[0]),
                            'high': float(hist['High'].iloc[0]),
                            'low': float(hist['Low'].iloc[0]),
                            'volume': float(hist['Volume'].iloc[0])
                        }
                        save_stock_data(sym, day_data[sym], date=bar_date)
                    else:
                        log_data_failure(sym, 'empty response', mode)
                except Exception as e:
                    log_data_failure(sym, str(e), mode)
        except Exception as e:
            print(f"Error fetching data from yfinance: {e}")

        # Load historical data
        histories = load_stock_data(symbols)

        # Suspend the symbols whose data stopped arriving; everything else trades as usual.
        session, suspended = suspended_symbols(symbols, last_bar_dates(symbols))
        for sym in suspended:
            day_data.pop(sym, None)
        state['suspended'] = suspended
        if suspended:
            print(f"SUSPENDED (no bar for {session}; no orders, holdings kept): "
                  + ", ".join(f"{s} (last {d or 'never'})" for s, d in sorted(suspended.items())))
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(os.path.join(LOG_DIR, 'suspended.log'), 'a') as f:
                f.write(f"{session} " + " ".join(f"{s}:{d or 'never'}"
                                                 for s, d in sorted(suspended.items())) + "\n")

        # Daily training (--cpp-upkeep): today's bars are in stock_data now, so the day the last
        # run placed orders on can be trained against them before anything is decided today.
        if args.cpp_upkeep:
            run_daily_training(model_dir)
        if args.committee:
            settle_pending(model_dir, trading_client, day_data)

        # Fetch 1-minute intraday bars for today so the fill simulation uses the
        # actual intraday high/low sequence rather than a random 50/50 coin flip.
        print("Fetching intraday data for fill simulation ...")
        intraday_data = fetch_intraday_data(symbols)
        seq_flags = get_seq_flags_from_intraday(symbols, intraday_data, day_data)
        n_low_first  = sum(1 for v in seq_flags.values() if v)
        n_high_first = sum(1 for v in seq_flags.values() if not v)
        print(f"Intraday sequence: {n_low_first} low-first, {n_high_first} high-first "
              f"({len(intraday_data)} symbols with 1-min data)")

        # ── Master: predict tier allocations (MT2 preferred, MasterNN fallback) ──
        industry_list     = list(industries.keys())
        decision = None
        if args.committee:
            try:
                decision = committee_decision(model_dir, industries, histories, day_data,
                                              holdings, cash)
            except Exception as e:
                print(f"Committee: ERROR ({e}); allocating flat today, nothing sell-only")
        if decision is not None:
            allocations = decision['allocations']
            sell_only |= set(decision['sell_only'])
            zero_counts = {ind: 0 for ind in industry_list}      # never liquidate
        else:
            ind_value_history, mkt_val_history, zero_counts = _load_master_state(model_dir,
                                                                                 industry_list)
            master = None if args.flat_allocation or args.committee else \
                load_weighted_model(MasterNN, model_dir, 'master')
            allocations, tier_map, _mt1_inf_outputs = run_master_allocation(
                master, industries, mkt_val_history, ind_value_history, zero_counts, cash,
                model_dir=model_dir, flat=args.flat_allocation or args.committee)

        # Liquidate industries with 3+ consecutive tier-0 predictions
        ind_current_values = compute_industry_current_values(industries, holdings, histories)
        liquidation_orders = {}
        for ind in industries:
            if zero_counts.get(ind, 0) >= 3:
                hold_v = ind_current_values.get(ind, (0.0, 0.0))[0]
                if hold_v > 1e-6:
                    liquidation_orders[ind] = hold_v

        if liquidation_orders:
            liq_str = ', '.join(f"{ind}=${v:.2f}" for ind, v in liquidation_orders.items())
            print(f"Master liquidation orders (3x tier-0): {liq_str}")

        orders = []

        # Liquidate any positions in symbols that have been removed from the universe.
        # Queued as market orders so they fill regardless of intraday price.
        _universe_syms = set(sym for syms in industries.values() for sym in syms)
        _orphaned = {sym: qty for sym, qty in holdings.items()
                     if sym not in _universe_syms and qty >= 1}
        if _orphaned:
            print(f"Orphaned positions to liquidate (symbols no longer in universe): "
                  f"{list(_orphaned.keys())}")
            _orf_prices: dict = {}
            try:
                _orf_dl = yf.download(list(_orphaned.keys()), period='1d',
                                      auto_adjust=True, progress=False)
                for _sym in _orphaned:
                    try:
                        _orf_prices[_sym] = float(_orf_dl['Close'][_sym].iloc[-1])
                    except Exception:
                        _orf_prices[_sym] = 0.0
            except Exception as _e:
                print(f"  Warning: could not fetch orphaned prices: {_e}")
            for _sym, _qty in _orphaned.items():
                _price = _orf_prices.get(_sym, 0.0)
                orders.append({'symbol': _sym, 'action': 'liquidate',
                                'quantity': int(_qty), 'price': _price})
                print(f"  {_sym}: queued market sell {int(_qty)} shares "
                      f"(last known ~${_price:.2f})")

        # ── Industry activity: CUMULATIVE entry gate, walked in activation order ─
        #
        # Two rules, both of which the previous version got wrong:
        #
        #  1. entry_min is the 3-stock entry (priciest + 2 cheapest), not one share of the
        #     priciest. Affording a single share of the dearest name does not make an industry
        #     tradeable -- the model allocates across the whole industry.
        #  2. The gate is CUMULATIVE against the TOTAL portfolio (cash + holdings), not each
        #     industry against its own slice. Opening the Nth industry requires covering the
        #     summed entry_min of all N, taken in activation order. Checking each industry
        #     independently let every one open the moment it cleared its own floor, which is
        #     not a ramp -- and the old universe hid that, because unlock points ran from $140
        #     to $1,825 and so supplied an accidental ordering. Inside the $30-$90 band every
        #     industry unlocks at roughly the same figure, so the order must now be explicit.
        active_industries = set()
        inactive_log      = []
        total_value       = compute_total_portfolio_value(cash, holdings, day_data, histories)

        entry_by_ind = {}
        for ind, syms in industries.items():
            prices_today = {s: day_data.get(s, {}).get('close', 0.0) for s in syms if day_data.get(s, {}).get('close', 0.0) > 0}
            if not prices_today:
                continue
            sorted_asc = sorted(prices_today.items(), key=lambda x: x[1])
            top1       = max(prices_today.items(), key=lambda x: x[1])
            # The two cheapest OTHER than the priciest, so a 3-symbol industry cannot count the
            # same name twice.
            bottom2    = [x for x in sorted_asc if x[0] != top1[0]][:2]
            entry_by_ind[ind] = (top1[1] + sum(p for _, p in bottom2), [top1] + bottom2)

        # PROD ONLY. Paper runs every industry — its job is to prove the models work across the
        # whole universe in near-real conditions on money that is not real. The cumulative ramp
        # exists because prod is real investment starting from one industry and earning its way
        # into more; applying it to paper would just shrink the thing paper is meant to test.
        if args.paper:
            active_industries = set(entry_by_ind)
            order = []
        else:
            order = load_activation_order(model_dir, entry_by_ind)
        running = 0.0
        for ind in order:
            entry_min, top3 = entry_by_ind[ind]
            need = running + entry_min
            if total_value >= need:
                active_industries.add(ind)
                running = need
            else:
                top3_str = ', '.join(f"{s}=${p:.0f}" for s, p in top3)
                inactive_log.append(
                    f"{ind}: portfolio ${total_value:.0f}, need ${need:.0f} cumulative "
                    f"(+${entry_min:.0f} for this industry; 3-stock entry: {top3_str})")
        if inactive_log:
            print("Inactive industries (below 1-share floor):")
            for msg in inactive_log:
                print(f"  {msg}")

        # ── Industries: liquidation sells first, then normal trading ─────────
        for industry, symbols in industries.items():
            allocated_cash    = allocations.get(industry, 0.0)
            liquidation_target = liquidation_orders.get(industry, 0.0)

            # Step 1: generate liquidation sell orders if master requires it
            if liquidation_target > 1e-6:
                liquidated = generate_liquidation_sells(
                    industry, symbols, liquidation_target,
                    holdings, day_data, histories, orders)
                # Freed cash flows back to master pool — reflected in next day's Alpaca sync
                print(f"[{industry}] Liquidated ${liquidated:.2f} of ${liquidation_target:.2f} requested")

            # Sell-only (the loser zone): no budget to buy with, so the model sees the cash the
            # real book has. Its holdings still need their sells even when the entry gate
            # would close the industry, since the gate is about affording a way IN.
            if industry in sell_only:
                allocated_cash = 0.0
                held = any(holdings.get(sym, 0.0) >= 1 for sym in symbols)
                print(f"[{industry}] SELL-ONLY: sells only, no buys"
                      + ("" if held else " (nothing held)"))
                if not held:
                    continue
            # Step 2: skip trading if industry is below the 1-share capital floor
            elif industry not in active_industries:
                continue

            # Step 3: normal industry model inference — 232-feature input, 48 outputs
            model = load_weighted_model(StockNN, model_dir, industry)
            if model is None:
                print(f"Model for {industry} not found.")
                continue
            try:
                out = stocknn_outputs(model, symbols, histories, day_data, holdings,
                                      allocated_cash)
                sug = {sym: out[j] for j, sym in enumerate(symbols)}

                # Cancel existing GTC stops on held symbols before any sell (Alpaca reserves the
                # shares an open stop covers). Skipped under --no-orders: cancelling a live stop
                # while submitting no replacement would strip protection from a real position,
                # which is the one destructive thing this mode must not do. The prices are kept
                # so a sell-only industry can re-protect what it still holds.
                _sym_holds = state.get('new_symbol_holds', {})
                stops = {}
                for sym in symbols:
                    if sym not in day_data or _sym_holds.get(sym, 0) > 0:
                        continue
                    if holdings.get(sym, 0.0) > 0 and not args.no_orders:
                        try:
                            get_orders_request = GetOrdersRequest(
                                status=QueryOrderStatus.OPEN, symbols=[sym])
                            existing_orders = trading_client.get_orders(get_orders_request)
                            for o in existing_orders:
                                if o.order_type == OrderType.STOP:
                                    if o.stop_price is not None:
                                        stops[sym] = max(stops.get(sym, 0.0), float(o.stop_price))
                                    trading_client.cancel_order_by_id(o.id)
                        except Exception as e:
                            print(f"Error canceling existing orders for {sym}: {e}")
                    # A buy's stop is an OTO leg that expires with the day order, so a position
                    # bought yesterday has no open stop today: fall back to the stop recorded
                    # for it, so it is re-placed as a GTC stop.
                    if holdings.get(sym, 0.0) >= 1 and sym not in stops \
                            and sym in state.get('stops', {}):
                        stops[sym] = state['stops'][sym]

                orders += plan_industry_orders(symbols, sug, holdings, day_data, allocated_cash,
                                               _sym_holds, sell_only=industry in sell_only,
                                               stops=stops)

            except Exception as e:
                print(f"Error processing industry {industry}: {e}")

        # Update histories with today's data and calculate deltas
        for sym in histories:
            if sym in day_data:
                d = day_data[sym]
                raw = [d['open'], d['close'], d['high'], d['low'], d['volume']]
                if sym in histories and histories[sym]:
                    prev = histories[sym][-1][:5]
                    deltas = [r - p for r, p in zip(raw, prev, strict=True)]
                else:
                    deltas = [0.0] * 5
                if sym not in histories:
                    histories[sym] = []
                histories[sym].append(raw + deltas)
                if len(histories[sym]) > 15:
                    histories[sym].pop(0)

        # Handle withdrawal requests by selling stocks if necessary
        if args.withdraw is not None:
            withdraw_amount = args.withdraw
            if cash < withdraw_amount:
                needed = withdraw_amount - cash
                last_close = {sym: histories[sym][-1][1] if sym in histories and histories[sym] else 0 for sym in histories}
                values = {sym: holdings.get(sym, 0.0) * last_close.get(sym, 0) for sym in last_close}
                sorted_syms = sorted(values, key=values.get, reverse=True)
                current_needed = needed
                for sym in sorted_syms:
                    if current_needed <= 0:
                        break
                    v = values[sym]
                    if v > 0:
                        qty = holdings[sym]
                        price = last_close[sym]
                        to_sell = min(qty, current_needed / (price * SELL_FILL))
                        orders.append({'symbol': sym, 'action': 'sell', 'quantity': to_sell, 'price': price})
                        current_needed -= _sell_net(to_sell, price)

        # Decrement trading holds for new symbols; remove when they reach zero
        _holds = state.get('new_symbol_holds', {})
        for sym in list(_holds.keys()):
            _holds[sym] -= 1
            if _holds[sym] <= 0:
                del _holds[sym]
                print(f"Trading hold for {sym} has expired — orders now enabled")
        state['new_symbol_holds'] = _holds

        # Zero out orphaned holdings now that liquidation orders have been queued
        for sym in _orphaned:
            holdings.pop(sym, None)
        if _orphaned:
            print(f"Cleared orphaned holdings from state: {list(_orphaned.keys())}")

        # Remember each held or bought symbol's stop for the next run (see the fallback above).
        # A held position left without a stop today (a new buy for it is open) keeps the one it
        # had. A --no-orders run placed nothing, so it records nothing new.
        remembered = {sym: p for sym, p in state.get('stops', {}).items()
                      if holdings.get(sym, 0.0) >= 1}
        for o in ([] if args.no_orders else orders):
            if o['action'] == 'stop_loss':
                remembered[o['symbol']] = o['price']
            elif o['action'] == 'buy' and o.get('stop'):
                remembered[o['symbol']] = o['stop']
        state['stops'] = remembered

        # Save updated state
        state['histories'] = histories
        save_state(state, model_dir)

        # Update owners.json after processing day's data and potential withdrawals
        total_portfolio_value = compute_total_portfolio_value(cash, holdings, day_data, histories)
        update_owners_file(total_portfolio_value, model_dir)

        if args.cpp_upkeep:
            # The C++ step trained before inference (run_daily_training); what is left is to
            # record today's books, which tomorrow's run trains on.
            save_upkeep_book(model_dir, trading_client, industries, allocations, sell_only,
                             orders, day_data)
        else:
            # Retrain models using real Alpaca state as portfolio seed.
            # yesterday_data = model-input features (day N); day_data = fill prices (day N+1).
            # This matches the offline training's day-N-predict / day-N+1-fill pattern.
            ind_primed, master_primed = build_primed_portfolios(
                trading_client, industries, allocations, zero_counts)
            industry_top_scores = {}
            for industry, symbols in industries.items():
                if os.path.exists(f"{model_dir}/{industry}_best.pt"):
                    result = train_industry_one_day_prod(
                        industry, symbols, yesterday_data, ind_primed[industry], model_dir,
                        today_data=day_data, seq_flags=seq_flags, intraday_bars=intraday_data)
                    if result is not None:
                        industry_top_scores[industry] = result

            # Compute equal-weight market close-to-close return per industry from today's data.
            # Uses yesterday_data (prev close) and day_data (today close).
            mkt_ret: dict = {}
            for ind, syms in industries.items():
                total, valid = 0.0, 0
                for sym in syms:
                    prev = yesterday_data.get(sym, {}).get('close', 0.0)
                    cur  = day_data.get(sym, {}).get('close', 0.0)
                    if prev > 1e-3 and cur > 1e-3:
                        total += cur / prev - 1.0; valid += 1
                mkt_ret[ind] = total / valid if valid > 0 else 0.0

            # Update market cumulative index (starts at IND_STARTING_CASH per session)
            IND_START = 25_000.0
            for ind in industry_list:
                hist_v = mkt_val_history.get(ind, [])
                prev_v = hist_v[-1] if hist_v else IND_START
                hist_v.append(prev_v * (1.0 + mkt_ret.get(ind, 0.0)))
                mkt_val_history[ind] = hist_v[-92:]  # cap at IND_HIST_CAP=92

            # Append today's PORTFOLIO index (Part C: compound the deployed slot-0 return so a 10-day
            # forward ratio is a meaningful cumulative return, matching the C++ ind_val_hist), and record
            # today's same-day slot-0 delta (the MT2 objective). industry_top_scores[ind] = (baseline, slot0).
            slot0_deltas = {}
            for ind in industry_list:
                if ind in industry_top_scores:
                    baseline, slot0_v = industry_top_scores[ind]
                    slot0_ret = (slot0_v / baseline - 1.0) if baseline > 1e-6 else 0.0
                    slot0_deltas[ind] = slot0_ret
                    hist_v = ind_value_history.get(ind, [])
                    prev_v = hist_v[-1] if hist_v else IND_START
                    hist_v.append(prev_v * (1.0 + slot0_ret))
                    ind_value_history[ind] = hist_v[-92:]  # cap at IND_HIST_CAP=92
            _save_master_state(model_dir, ind_value_history, mkt_val_history, zero_counts)

            # MT1/MT2 upkeep — gate on ≥15 days of history (MT1 needs features; MT2 needs ≥5 MT1 warmup).
            min_real_days = min(
                (len(ind_value_history.get(ind, [])) for ind in industry_list),
                default=0,
            )
            if min_real_days >= 15:
                print(f"Running MT1/MT2 upkeep ({min_real_days} history days) ...")
                train_mt_one_day_prod(
                    industries, model_dir, mkt_val_history, ind_value_history,
                    slot0_deltas, mkt_ret)
            elif os.path.exists(f"{model_dir}/master_best.pt"):
                # Legacy MasterNN upkeep during transition period (≤15 real days)
                from training_lib import train_master_one_day
                try:
                    train_master_one_day(industries, master_primed, model_dir,
                                         ind_value_history,
                                         industry_top_scores=industry_top_scores)
                except Exception as e:
                    print(f"Error in legacy master upkeep: {e}")
                print(f"MT upkeep deferred: {min_real_days}/15 days of real history")
            else:
                print(f"MT upkeep deferred: {min_real_days}/15 days of real history")

        # Submit orders to Alpaca (paper or live depending on --paper flag).
        # --no-orders stops here: everything above (allocation, decisions) and the upkeep training
        # step below still run, so the models advance a day without anything reaching the market.
        if decision is not None and not args.no_orders:
            try:
                equity = float(trading_client.get_account().equity)
            except Exception as e:
                print(f"Committee log: ERROR reading equity ({e}); estimating it")
                equity = compute_total_portfolio_value(cash, holdings, day_data, histories)
            record_decision(model_dir, industries, decision, holdings, day_data, histories, cash,
                            equity, suspended, False)
        mode = 'paper' if args.paper else 'live'
        if args.no_orders:
            print(f"--no-orders: {len(orders)} order(s) computed, NONE submitted")
            for o in orders[:20]:
                print(f"    would {o['action']:<9} {o['symbol']:<6} qty={o.get('quantity')}"
                      f" price={o.get('price')}")
            if len(orders) > 20:
                print(f"    ... and {len(orders) - 20} more")
            orders = []
        for order in orders:
            try:
                if order['action'] == 'buy':
                    # One OTO order: the stop-loss goes live only when the buy fills. A separate
                    # sell stop beside an open buy is rejected by Alpaca as a wash trade.
                    stop = order.get('stop')
                    order_req = LimitOrderRequest(
                        symbol=order['symbol'],
                        qty=order['quantity'],
                        side=OrderSide.BUY,
                        type=OrderType.LIMIT,
                        time_in_force=TimeInForce.DAY,
                        limit_price=order['price'],
                        **({'order_class': OrderClass.OTO,
                            'stop_loss': StopLossRequest(stop_price=stop)} if stop else {})
                    )
                    result = trading_client.submit_order(order_req)
                    log_request_id(str(result.id), f"{mode} buy {order['symbol']} qty={order['quantity']}")
                elif order['action'] == 'sell':
                    order_req = LimitOrderRequest(
                        symbol=order['symbol'],
                        qty=order['quantity'],
                        side=OrderSide.SELL,
                        type=OrderType.LIMIT,
                        time_in_force=TimeInForce.DAY,
                        limit_price=order['price']
                    )
                    result = trading_client.submit_order(order_req)
                    log_request_id(str(result.id), f"{mode} sell {order['symbol']} qty={order['quantity']}")
                elif order['action'] == 'stop_loss':
                    order_req = StopOrderRequest(
                        symbol=order['symbol'],
                        qty=order['quantity'],
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.GTC,
                        stop_price=order['price']
                    )
                    result = trading_client.submit_order(order_req)
                    log_request_id(str(result.id), f"{mode} stop_loss {order['symbol']} qty={order['quantity']}")
                elif order['action'] == 'liquidate':
                    order_req = MarketOrderRequest(
                        symbol=order['symbol'],
                        qty=order['quantity'],
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )
                    result = trading_client.submit_order(order_req)
                    log_request_id(str(result.id), f"{mode} liquidate {order['symbol']} qty={order['quantity']}")
            except Exception as e:
                rid = getattr(getattr(e, 'response', None), 'headers', {})
                rid = rid.get('x-request-id', '') if hasattr(rid, 'get') else ''
                if rid:
                    log_request_id(rid, f"{mode} {order['action']} {order['symbol']} FAILED: {e}", success=False)
                print(f"Error submitting order for {order['symbol']}: {e}")

if __name__ == '__main__':
    main()
