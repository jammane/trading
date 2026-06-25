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
import json
import os
import random
from datetime import datetime

import keyring
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, MarketOrderRequest, StopOrderRequest

import training_lib
from fees import BUY_FILL, FINRA_TAF_MAX, FINRA_TAF_PER_SHARE, SEC_FEE_RATE, SELL_FILL, _sell_net
from models import MasterNN, MT1NN, MT2NN, StockNN
from universe import INDUSTRIES
from version import VERSION
from upkeep import (
    load_mt2_norm_stats,
    run_mt_inference,
    save_mt2_norm_stats,
    upkeep_industry,
    upkeep_mt1_industry,
    upkeep_mt2,
)

MAX_SINGLE_STOCK_PCT = 0.60   # max fraction of industry cash in one stock

MODEL_DIR = 'models'  # legacy; superseded by --account in main()
STOCK_DATA_DIR = 'stock_data'

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
    """Load persisted ind_value_history, zero_counts, and MT2 norm stats."""
    path = _master_state_path(model_dir)
    try:
        with open(path) as f:
            state = json.load(f)
        hist = state.get('ind_value_history', {ind: [] for ind in industry_list})
        zcnt = state.get('zero_counts',       {ind: 0  for ind in industry_list})
    except Exception:
        hist = {ind: [] for ind in industry_list}
        zcnt = {ind: 0  for ind in industry_list}
    norm_stats = load_mt2_norm_stats(model_dir)
    return hist, zcnt, norm_stats

def _save_master_state(model_dir, ind_value_history, zero_counts):
    try:
        with open(_master_state_path(model_dir), 'w') as f:
            json.dump({'ind_value_history': ind_value_history, 'zero_counts': zero_counts}, f)
    except Exception as e:
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


def train_mt_one_day_prod(industries, model_dir, ind_value_history,
                          norm_stats, industry_top_scores, mt1_outputs_prev=None):
    """
    Upkeep training for MT1 (12 pools) and MT2.

    ind_value_history: already updated for today before this is called.
    industry_top_scores: {ind: (baseline_score, slot0_score)} from today's industry runs.
    mt1_outputs_prev: MT1 slot0 outputs from production inference this cycle (optional,
                      used to seed the upkeep in37 without re-running build_master_features).

    Returns {ind: (conf, delta, range_hw)} — MT1 slot0 outputs after upkeep.
    """
    from training_lib import build_master_features
    industry_list = list(industries.keys())

    # actual_perf per industry from today's upkeep results
    actual_perf: dict = {}
    for ind in industry_list:
        if industry_top_scores and ind in industry_top_scores:
            baseline, slot0 = industry_top_scores[ind]
            actual_perf[ind] = (slot0 / baseline - 1.0) if baseline > 0 else 0.0
        else:
            actual_perf[ind] = 0.0

    today444 = build_master_features(ind_value_history, industry_list)

    mt1_slot0_outputs: dict = {}
    for i, ind in enumerate(industry_list):
        in37_t = today444[:, i * 37:(i + 1) * 37]
        try:
            _, _, conf, delta, range_hw = upkeep_mt1_industry(
                ind, model_dir, in37_t, actual_perf[ind])
            mt1_slot0_outputs[ind] = (conf, delta, range_hw)
        except Exception as e:
            print(f"Error in upkeep_mt1_industry {ind}: {e}")
            mt1_slot0_outputs[ind] = (0.5, 0.0, 0.02)

    try:
        upkeep_mt2(model_dir, mt1_slot0_outputs, actual_perf, industry_list)
    except Exception as e:
        print(f"Error in upkeep_mt2: {e}")

    return mt1_slot0_outputs


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
                histories[sym] = [[d['open'], d['close'], d['high'], d['low'], d['volume']] for d in data.get('days', [])[-15:]]
            else:
                histories[sym] = []
        except Exception as e:
            print(f"Error loading stock data for {sym}: {e}")
            histories[sym] = []
    return histories

def save_stock_data(symbol, day_data):
    """Append today's OHLCV to stock_data/<symbol>.json without truncating.

    Full history is preserved so the C++ trainer and download_daily.py can
    use the same files. download_daily.py trims to MAX_HISTORY_DAYS on each run.
    """
    try:
        file_path = f"{STOCK_DATA_DIR}/{symbol}.json"
        existing_data = []
        if os.path.exists(file_path):
            with open(file_path) as f:
                existing_data = json.load(f).get('days', [])
        today_str = datetime.today().strftime('%Y-%m-%d')
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


def run_master_allocation(master_model, industries, ind_value_history, zero_counts,
                          total_cash, norm_stats=None, model_dir=None):
    """
    Run master inference to get tier classification and capital allocation.
    Mutates zero_counts in place.

    Priority: MT2 (mt2_best.pt exists) → legacy MasterNN → equal allocation.
    Returns (allocations, tier_map, mt1_outputs).
      allocations:  {ind: dollar_amount}
      tier_map:     {ind: 0-3}
      mt1_outputs:  {ind: (conf, delta, range_hw)} or None for legacy path
    """
    from training_lib import build_master_features, decode_master_tiers, tiers_to_alloc
    industry_list = list(industries.keys())
    tier_map      = {ind: 0 for ind in industry_list}
    mt1_outputs   = None

    # MT2 path — preferred if mt2_best.pt exists and norm_stats are available
    mt2_path = os.path.join(model_dir, 'mt2_best.pt') if model_dir else None
    if mt2_path and os.path.exists(mt2_path) and norm_stats is not None:
        try:
            allocations, tier_map, mt1_outputs = run_mt_inference(
                model_dir, industries, ind_value_history, norm_stats, zero_counts, total_cash)
            return allocations, tier_map, mt1_outputs
        except Exception as e:
            print(f"Warning: MT2 inference failed ({e}) — falling back to MasterNN")

    # Legacy MasterNN fallback
    if master_model is not None:
        try:
            today_t = build_master_features(ind_value_history, industry_list)
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


def main():
    """Run one daily trading cycle: fetch data, infer, submit orders, retrain."""
    parser = argparse.ArgumentParser(description="Run stock trading system v2.")
    parser.add_argument('--paper',   action='store_true', help='Use Alpaca paper trading API (real paper orders, real paper portfolio)')
    parser.add_argument('--account', default='acct0', help='Account identifier (e.g. acct0); derives models/ACCOUNT/paper|prod and logs/ACCOUNT/paper|prod')
    parser.add_argument('--capital', type=float, default=None, help='Cap total deployed capital regardless of Alpaca account balance')
    parser.add_argument('--withdraw', type=float, help='Amount to withdraw from portfolio')
    args = parser.parse_args()

    subtype   = 'paper' if args.paper else 'prod'
    model_dir = os.path.join('models', args.account, subtype)
    global LOG_DIR
    LOG_DIR   = os.path.join('logs', args.account, subtype)
    training_lib.DUMP_DIR = os.path.join(LOG_DIR, 'data_dump')
    print(f"[production_v2] v{VERSION}  account={args.account}  mode={subtype}")

    # `if True:` preserves the existing indentation scope; all logic below runs unconditionally.
    if True:
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
        histories = state['histories']
        cash = state['cash']
        holdings = state['holdings']

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
                        day_data[sym] = {
                            'open': float(hist['Open'].iloc[0]),
                            'close': float(hist['Close'].iloc[0]),
                            'high': float(hist['High'].iloc[0]),
                            'low': float(hist['Low'].iloc[0]),
                            'volume': float(hist['Volume'].iloc[0])
                        }
                        save_stock_data(sym, day_data[sym])
                    else:
                        log_data_failure(sym, 'empty response', mode)
                except Exception as e:
                    log_data_failure(sym, str(e), mode)
        except Exception as e:
            print(f"Error fetching data from yfinance: {e}")

        # Load historical data
        histories = load_stock_data(symbols)

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
        all_symbols_flat  = [sym for syms in industries.values() for sym in syms]
        industry_list     = list(industries.keys())
        ind_value_history, zero_counts, norm_stats = _load_master_state(model_dir, industry_list)
        master = load_weighted_model(MasterNN, model_dir, 'master')
        allocations, tier_map, _mt1_inf_outputs = run_master_allocation(
            master, industries, ind_value_history, zero_counts, cash,
            norm_stats=norm_stats, model_dir=model_dir)

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

        # ── Industry activity flags: skip trading if capital < priciest stock ─
        active_industries = set()
        inactive_log      = []
        for ind, syms in industries.items():
            ind_capital  = allocations.get(ind, 0.0)
            prices_today = {s: day_data.get(s, {}).get('close', 0.0) for s in syms if day_data.get(s, {}).get('close', 0.0) > 0}
            if not prices_today:
                continue
            sorted_asc   = sorted(prices_today.items(), key=lambda x: x[1])
            top1         = max(prices_today.items(), key=lambda x: x[1])
            bottom2      = sorted_asc[:2]
            entry_min    = top1[1] + sum(p for _, p in bottom2)  # priciest + 2 cheapest
            top3         = [top1] + bottom2
            max_price    = top3[0][1]
            if ind_capital >= max_price:
                active_industries.add(ind)
            else:
                top3_str = ', '.join(f"{s}=${p:.0f}" for s, p in top3)
                inactive_log.append(
                    f"{ind}: have ${ind_capital:.0f}, need ${max_price:.0f} to enter "
                    f"(3-stock entry=${entry_min:.0f}: {top3_str})")
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

            # Step 2: skip trading if industry is below the 1-share capital floor
            if industry not in active_industries:
                continue

            # Step 3: normal industry model inference — new 208-feature input, 48 outputs
            model = load_weighted_model(StockNN, model_dir, industry)
            if model is None:
                print(f"Model for {industry} not found.")
                continue
            try:
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
                            tr   = list(zip(*dl))
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

                # today_t: (1,208) — full features for current day
                state_vec = [allocated_cash] + [holdings.get(sym, 0.0) for sym in symbols]
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
                                   (raw_t[0] * raw_t[4]) / st['avg_dv']])
                    today_dl.append(dlt_t)
                if today_dl:
                    tr = list(zip(*today_dl))
                    for tp in tr: today_row += [max(tp), min(tp), sum(tp)/len(tp)]
                else:
                    today_row += [0.0] * 15
                today_row += state_vec
                today_t = torch.tensor(today_row, dtype=torch.float32).unsqueeze(0)  # (1,208)

                with torch.no_grad():
                    out = model(history_t, today_t)
                out  = out.view(12, 4)   # (buy_qty, buy_price, sell_all_price, sell_qty)
                sug  = {sym: out[j].tolist() for j, sym in enumerate(symbols)}

                # ── Sells first: cancel existing stops, execute sells ────────
                _sym_holds = state.get('new_symbol_holds', {})
                for j, sym in enumerate(symbols):
                    if sym not in day_data:
                        continue
                    if _sym_holds.get(sym, 0) > 0:
                        continue  # new symbol hold active — no orders this run
                    buy_qty, buy_price_frac, sell_all_price_frac, sell_qty = sug[sym]
                    cur_qty        = holdings.get(sym, 0.0)
                    low            = day_data[sym]['low']
                    high           = day_data[sym]['high']
                    close          = day_data[sym]['close']
                    span           = max(high - low, 1e-9)
                    sell_all_price = low + sell_all_price_frac * span

                    # Cancel existing GTC stop orders
                    if cur_qty > 0:
                        try:
                            get_orders_request = GetOrdersRequest(
                                status=QueryOrderStatus.OPEN, symbols=[sym])
                            existing_orders = trading_client.get_orders(get_orders_request)
                            for o in existing_orders:
                                if o.order_type == OrderType.STOP:
                                    trading_client.cancel_order_by_id(o.id)
                        except Exception as e:
                            print(f"Error canceling existing orders for {sym}: {e}")

                    # Sell qty at market (open price)
                    if sell_qty > 1e-6 and cur_qty > 1e-6:
                        amount = int(min(sell_qty, cur_qty))
                        if amount >= 1:
                            orders.append({'symbol': sym, 'action': 'sell',
                                           'quantity': amount, 'price': close})
                            cur_qty -= amount

                    # Sell all at limit price
                    if cur_qty >= 1 and low <= sell_all_price <= high:
                        orders.append({'symbol': sym, 'action': 'sell',
                                       'quantity': int(cur_qty), 'price': sell_all_price})
                        cur_qty = 0.0

                # ── Buys after sells: stop anchored to buy_price ─────────────
                for j, sym in enumerate(symbols):
                    if sym not in day_data:
                        continue
                    if _sym_holds.get(sym, 0) > 0:
                        continue  # new symbol hold active — no orders this run
                    buy_qty, buy_price_frac, sell_all_price_frac, sell_qty = sug[sym]
                    low       = day_data[sym]['low']
                    high      = day_data[sym]['high']
                    span      = max(high - low, 1e-9)
                    buy_price    = low + buy_price_frac * span
                    stop_loss_at = buy_price * 0.9   # GTC stop at 10% below entry

                    if buy_qty > 1e-6 and buy_price > 0 and low <= buy_price <= high:
                        affordable = allocated_cash / (buy_price * BUY_FILL)
                        amount     = int(min(buy_qty, affordable))
                        if amount >= 1:
                            orders.append({'symbol': sym, 'action': 'buy',
                                           'quantity': amount, 'price': buy_price})
                            orders.append({'symbol': sym, 'action': 'stop_loss',
                                           'quantity': amount, 'price': stop_loss_at})

            except Exception as e:
                print(f"Error processing industry {industry}: {e}")

        # Update histories with today's data and calculate deltas
        for sym in histories:
            if sym in day_data:
                d = day_data[sym]
                raw = [d['open'], d['close'], d['high'], d['low'], d['volume']]
                if sym in histories and histories[sym]:
                    prev = histories[sym][-1][:5]
                    deltas = [r - p for r, p in zip(raw, prev)]
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

        # Save updated state
        state['histories'] = histories
        save_state(state, model_dir)

        # Update owners.json after processing day's data and potential withdrawals
        total_portfolio_value = compute_total_portfolio_value(cash, holdings, day_data, histories)
        update_owners_file(total_portfolio_value, model_dir)

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

        # Append today's industry slot0 values and persist master state for next day.
        for ind in industry_list:
            if ind in industry_top_scores:
                ind_value_history[ind].append(industry_top_scores[ind][1])
        _save_master_state(model_dir, ind_value_history, zero_counts)

        # MT1/MT2 upkeep — gate on ≥15 days of real history (same guard as old master).
        # MT1 upkeep needs build_master_features input; MT2 upkeep needs ≥5 MT1 warmup.
        # Sentinel filter: per-industry values ≠ 3000 (40% of $3k paper account = $1200).
        min_real_days = min(
            (sum(1 for v in ind_value_history.get(ind, []) if v != 3000.0)
             for ind in industry_list),
            default=0,
        )
        if min_real_days >= 15:
            print(f"Running MT1/MT2 upkeep ({min_real_days} real history days) ...")
            train_mt_one_day_prod(
                industries, model_dir, ind_value_history,
                norm_stats, industry_top_scores)
            save_mt2_norm_stats(model_dir, norm_stats)
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

        # Submit orders to Alpaca (paper or live depending on --paper flag)
        mode = 'paper' if args.paper else 'live'
        for order in orders:
            try:
                if order['action'] == 'buy':
                    order_req = LimitOrderRequest(
                        symbol=order['symbol'],
                        qty=order['quantity'],
                        side=OrderSide.BUY,
                        type=OrderType.LIMIT,
                        time_in_force=TimeInForce.DAY,
                        limit_price=order['price']
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
