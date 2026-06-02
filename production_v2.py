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
import torch
import torch.nn as nn
import torch.nn.functional as F
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, StopOrderRequest

from fees import BUY_FILL, FINRA_TAF_MAX, FINRA_TAF_PER_SHARE, SEC_FEE_RATE, SELL_FILL, _sell_net
from models import MasterNN, StockNN
from universe import INDUSTRIES

MAX_SINGLE_STOCK_PCT = 0.60   # max fraction of industry cash in one stock

MODEL_DIR = 'models'
STATE_FILE = 'state.json'
STOCK_DATA_DIR = 'stock_data'
OWNERS_DIR = 'owners'
OWNERS_FILE = os.path.join(OWNERS_DIR, 'owners.json')

def load_state():
    """Load trading state from state.json, or return default initial state if absent."""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading state file {STATE_FILE}: {e}")
    industries = INDUSTRIES
    symbols = [sym for syms in industries.values() for sym in syms]
    return {'industries': industries, 'histories': {sym: [] for sym in symbols}, 'cash': 20000.0, 'holdings': {sym: 0.0 for sym in symbols}}

def save_state(state):
    """Persist the current trading state to state.json."""
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        print(f"Error saving state file {STATE_FILE}: {e}")

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

def update_owners_file(total_value, paper):
    """Update the total_value field in owners/owners.json (skipped in paper mode)."""
    if not paper:
        try:
            if not os.path.exists(OWNERS_DIR):
                os.makedirs(OWNERS_DIR)
                print(f"Created owners directory: {OWNERS_DIR}")
            if not os.path.exists(OWNERS_FILE):
                owners_data = {"total value": total_value}
                with open(OWNERS_FILE, 'w') as f:
                    json.dump(owners_data, f)
                print(f"Created owners.json with total value: {total_value}")
            else:
                with open(OWNERS_FILE) as f:
                    owners_data = json.load(f)
                owners_data["total value"] = total_value
                with open(OWNERS_FILE, 'w') as f:
                    json.dump(owners_data, f)
                print(f"Updated owners.json with total value: {total_value}")
        except Exception as e:
            print(f"Error updating owners.json: {e}")

# Compute cash allocation based on predicted performance
def compute_alloc_from_predicted(predicted, industry_list):
    """
    Decode MasterNN output (1, 36) or (36,) into allocation and liquidation signals.

    predicted: tensor
      [:12]  softmax allocation weights (sum to 1)
      [12:24] sigmoid liquidation depth   (0=hold, 1=liquidate to floor)
      [24:36] sigmoid liquidation trigger (0=skip, 1=execute)

    Returns:
      alloc_prop:  {ind: fraction}  40% cap, 2% floor enforced
      liq_depth:   {ind: float}     0-1
      liq_trigger: {ind: float}     0-1
    """
    n     = len(industry_list)
    floor = 0.02
    cap   = 0.40

    # Flatten to 1D if needed
    p = predicted.squeeze()
    vals     = p.tolist() if hasattr(p, 'tolist') else list(p)
    weights  = vals[:12]   if len(vals) >= 12 else vals + [1.0/n]*(12-len(vals))
    depths   = vals[12:24] if len(vals) >= 24 else [0.5]*12
    triggers = vals[24:36] if len(vals) >= 36 else [0.5]*12

    # Hybrid allocation: top industry gets full cap; rest filled proportionally.
    w_map   = dict(zip(industry_list, weights))
    top_ind = max(w_map, key=w_map.get)

    alloc_prop          = {ind: floor for ind in industry_list}
    alloc_prop[top_ind] = cap

    others  = [ind for ind in industry_list if ind != top_ind]
    premium = 1.0 - cap - floor * len(others)

    if premium > 1e-9 and others:
        uncapped = others[:]
        while premium > 1e-9 and uncapped:
            tw = sum(w_map[ind] for ind in uncapped)
            if tw <= 0:
                break
            proposed     = {ind: alloc_prop[ind] + premium * w_map[ind] / tw for ind in uncapped}
            newly_capped = [ind for ind in uncapped if proposed[ind] >= cap]
            if not newly_capped:
                for ind in uncapped:
                    alloc_prop[ind] = proposed[ind]
                break
            for ind in newly_capped:
                alloc_prop[ind] = cap
                uncapped.remove(ind)
            premium = 1.0 - sum(alloc_prop.values())

    liq_depth   = {ind: depths[i]   for i, ind in enumerate(industry_list)}
    liq_trigger = {ind: triggers[i] for i, ind in enumerate(industry_list)}

    return alloc_prop, liq_depth, liq_trigger

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

# Train an industry model on a single day's data (called from production daily run)
def train_industry_one_day_prod(industry, symbols, yesterday_data, primed_portfolio,
                                model_dir, today_data=None, seq_flags=None):
    """
    Upkeep training for one industry.
    yesterday_data: OHLCV for the prediction day (yesterday) — model input features.
    today_data:     OHLCV for the fill day (today) — actual market prices for execution.
                    Pass this so training matches offline (predict day N, fill day N+1).
    seq_flags:      {sym: bool} actual intraday high/low sequence (True=low-first).
                    Derived from today's 1-minute bars.  Falls back to random 50/50 if None.
    """
    from training_v2 import train_industry_one_day
    try:
        return train_industry_one_day(industry, symbols, yesterday_data, primed_portfolio,
                                      model_dir, next_day_data=today_data,
                                      seq_flags=seq_flags)
    except Exception as e:
        print(f"Error training industry {industry}: {e}")
        return None


def _flat_cos_history_path(model_dir):
    """Return the path to the rolling flat_cos history JSON file for the master model."""
    return os.path.join(model_dir, 'master_flat_cos_history.json')

def _load_flat_cos_history(model_dir):
    """Load up to 15 recent flat_cos values from disk; returns [] on any failure."""
    path = _flat_cos_history_path(model_dir)
    try:
        with open(path) as f:
            return json.load(f).get('history', [])[-15:]
    except Exception:
        return []

def _save_flat_cos_history(model_dir, history):
    """Persist the last 15 flat_cos values to disk, silently ignoring write errors."""
    try:
        with open(_flat_cos_history_path(model_dir), 'w') as f:
            json.dump({'history': list(history)[-15:]}, f)
    except Exception as e:
        print(f"Warning: could not save flat_cos history: {e}")


def train_master_one_day_prod(yesterday_data, industries, primed_portfolio, model_dir,
                               industry_top_scores=None):
    """
    Upkeep training for the master model.
    yesterday_data: OHLCV for the prediction day.  industry_top_scores come from
                    train_industry_one_day_prod calls (today's actual top-slot values).
    """
    from training_v2 import train_master_one_day
    try:
        fc_history = _load_flat_cos_history(model_dir)
        flat_cos   = train_master_one_day(yesterday_data, industries, primed_portfolio, model_dir,
                                          industry_top_scores=industry_top_scores,
                                          flat_cos_history=fc_history)
        if flat_cos is not None:
            fc_history.append(flat_cos)
            _save_flat_cos_history(model_dir, fc_history)
    except Exception as e:
        print(f"Error training master: {e}")


def build_primed_portfolios(trading_client, industries, allocations):
    """
    Fetch real Alpaca account state and build per-industry primed portfolios
    plus a master primed portfolio for single-day retraining.

    Returns:
        ind_primed:    {industry: {'cash': float, 'holdings': {sym: float}}}
        master_primed: {'cash': float, 'holdings': {ind: float}}
    """
    try:
        account    = trading_client.get_account()
        total_cash = float(account.cash)
    except Exception as e:
        print(f"Error fetching Alpaca account: {e}")
        total_cash = 0.0

    # Fetch current positions from Alpaca
    alpaca_qty = {}
    try:
        positions = trading_client.get_all_positions()
        for pos in positions:
            alpaca_qty[pos.symbol] = float(pos.qty)
    except Exception as e:
        print(f"Error fetching Alpaca positions: {e}")

    # Each industry gets 40% of cash + its actual symbol holdings
    ind_primed = {}
    for ind, syms in industries.items():
        ind_primed[ind] = {
            'cash':     total_cash * 0.40,
            'holdings': {sym: alpaca_qty.get(sym, 0.0) for sym in syms},
        }

    # Master gets remaining cash (after industry allocations) + industry allocation units
    # Represent each industry holding as the dollar value allocated to it
    master_holdings = {}
    for ind in industries:
        master_holdings[ind] = allocations.get(ind, 0.0)   # dollar amount allocated

    master_primed = {
        'cash':     total_cash * (1.0 - 0.40),   # cash not deployed to industries
        'holdings': master_holdings,
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
    """Append today's OHLCV to stock_data/<symbol>.json, retaining the last 15 days."""
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
        existing_data.append(new_data_point)
        with open(file_path, 'w') as f:
            json.dump({'days': existing_data[-15:]}, f)
    except Exception as e:
        print(f"Error saving stock data for {symbol}: {e}")

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


def run_master_allocation(master_model, industries, histories, holdings,
                          total_cash, total_account_value):
    """
    Run master inference using new 229-feature input.
    Returns (allocations, alloc_prop, liq_depth, liq_trigger).

    allocations:   {ind: dollar_amount} cash to deploy
    alloc_prop:    {ind: fraction}
    liq_depth:     {ind: 0-1} how deeply to liquidate (0=to target, 1=to floor)
    liq_trigger:   {ind: 0-1} whether to liquidate (>0.5 = yes)
    """
    industry_list = list(industries.keys())
    n_ind         = len(industry_list)
    alloc_prop    = {ind: 1.0 / n_ind for ind in industry_list}
    liq_depth     = {ind: 0.0 for ind in industry_list}
    liq_trigger   = {ind: 0.0 for ind in industry_list}

    if master_model is not None:
        try:
            # Per-industry rolling stats
            ind_stats = {}
            all_syms  = [sym for syms in industries.values() for sym in syms]
            num_past  = min(len(histories.get(sym, [])) for sym in all_syms) if all_syms else 0

            for ind, ind_syms in industries.items():
                mean_deltas = []
                for t in range(15):
                    dl = [histories[sym][-(t+1)][5:] for sym in ind_syms
                          if len(histories.get(sym,[])) > t]
                    if dl:
                        mean_deltas.append([sum(col)/len(col) for col in zip(*dl)])
                    else:
                        mean_deltas.append([0.0]*5)
                vals   = [d[0] for d in mean_deltas]
                mean_v = sum(vals)/len(vals) if vals else 0.0
                std_v  = (sum((v-mean_v)**2 for v in vals)/len(vals))**0.5 if vals else 0.0
                mean_5d = sum(vals[:5])/5 if len(vals) >= 5 else mean_v
                ind_stats[ind] = {
                    'volatility': std_v / abs(mean_v) if abs(mean_v) > 1e-9 else 0.0,
                    'momentum':   mean_5d / mean_v    if abs(mean_v) > 1e-9 else 1.0,
                    'mean_deltas': mean_deltas,
                }

            all_mean_per_day = []
            for t in range(15):
                day_means = [ind_stats[ind]['mean_deltas'][t][0] for ind in industry_list]
                all_mean_per_day.append(sum(day_means)/len(day_means) if day_means else 0.0)

            # history_t: (1,15,60) — mean OHLCV per industry, oldest first
            history_rows = []
            for t in range(14, -1, -1):
                row = []
                for ind in industry_list:
                    dl = [histories[sym][-(t+1)][:5] for sym in industries[ind]
                          if len(histories.get(sym, [])) > t]
                    if dl:
                        row += [sum(col)/len(col) for col in zip(*dl)]
                    else:
                        row += [0.0] * 5
                history_rows.append(row)
            history_t = torch.tensor(history_rows, dtype=torch.float32).unsqueeze(0)  # (1,15,60)

            # today_t: (1,229) — today's delta aggregates per industry + state
            today_ind_data = {}
            for ind in industry_list:
                dl = []
                for sym in industries[ind]:
                    h = histories.get(sym, [])
                    if len(h) >= 2:
                        raw_t = h[-1][:5]
                        prev  = h[-2][:5]
                        dl.append([raw_t[i] - prev[i] for i in range(5)])
                    elif len(h) == 1:
                        dl.append([0.0] * 5)
                if dl:
                    tr   = list(zip(*dl))
                    aggs = []
                    for tp in tr: aggs += [max(tp), min(tp), sum(tp)/len(tp)]
                    ind_mean_today = sum(row[1] for row in dl) / len(dl)
                else:
                    aggs = [0.0] * 15; ind_mean_today = 0.0
                today_ind_data[ind] = (aggs, ind_mean_today)

            all_mean_today = (sum(v[1] for v in today_ind_data.values()) / len(today_ind_data)
                              if today_ind_data else 0.0)
            cash_norm = total_cash / max(total_account_value, 1.0)
            hold_vals = []
            for ind, ind_syms in industries.items():
                hv = sum(holdings.get(sym, 0.0) * (histories[sym][-1][1] if histories.get(sym) else 0.0)
                         for sym in ind_syms)
                hold_vals.append(hv)
            state_vec = [cash_norm] + hold_vals

            today_row = []
            for ind in industry_list:
                st = ind_stats[ind]; aggs, ind_mean_today = today_ind_data[ind]
                corr = ind_mean_today / all_mean_today if abs(all_mean_today) > 1e-9 else 1.0
                today_row += aggs + [st['volatility'], st['momentum'], corr]
            today_row += state_vec
            today_t = torch.tensor(today_row, dtype=torch.float32).unsqueeze(0)  # (1,229)

            with torch.no_grad():
                out = master_model(history_t, today_t)
            alloc_prop, liq_depth, liq_trigger = compute_alloc_from_predicted(
                out.squeeze(), industry_list)
        except Exception as e:
            print(f"Error running master model: {e}")

    allocations = {ind: alloc_prop[ind] * total_cash for ind in industry_list}
    return allocations, alloc_prop, liq_depth, liq_trigger


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
    """Run one daily trading cycle: fetch data, infer, optionally submit orders, retrain."""
    parser = argparse.ArgumentParser(description="Run stock trading system v2.")
    parser.add_argument('--paper', action='store_true', help='Run in paper trading mode')
    parser.add_argument('--output', action='store_true', help='Submit orders to Alpaca')
    parser.add_argument('--model-dir', default=MODEL_DIR, help='Directory containing trained models')
    parser.add_argument('--withdraw', type=float, help='Amount to withdraw from portfolio')
    args = parser.parse_args()

    if args.paper or args.output:
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
        state = load_state()
        industries = state['industries']
        histories = state['histories']
        cash = state['cash']
        holdings = state['holdings']

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
                except Exception as e:
                    print(f"Error fetching data for {sym}: {e}")
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

        # ── Master: predict allocations + liquidation signals ──────────────
        all_symbols_flat  = [sym for syms in industries.values() for sym in syms]
        total_account_val = cash + sum(
            holdings.get(sym, 0.0) * (histories[sym][-1][1] if histories.get(sym) else 0.0)
            for sym in all_symbols_flat
        )
        master = load_weighted_model(MasterNN, args.model_dir, 'master')
        allocations, alloc_prop, liq_depth, liq_trigger = run_master_allocation(
            master, industries, histories, holdings, cash, total_account_val)

        # Build per-industry liquidation orders from master's trigger + depth signals
        floor_value = total_account_val * 0.02
        ind_current_values = compute_industry_current_values(industries, holdings, histories)
        liquidation_orders = {}
        for ind in industries:
            if liq_trigger.get(ind, 0.0) <= 0.5:
                continue
            hold_v = ind_current_values.get(ind, (0.0, 0.0))[0]
            if hold_v <= floor_value:
                continue
            target_v      = alloc_prop.get(ind, 0.0) * cash
            depth         = liq_depth.get(ind, 0.0)
            effective_tgt = target_v + (1.0 - depth) * max(0.0, hold_v - target_v)
            effective_tgt = max(effective_tgt, floor_value)
            liq_amt       = max(0.0, min(hold_v - effective_tgt, hold_v - floor_value))
            if liq_amt > 1e-6:
                liquidation_orders[ind] = liq_amt

        if liquidation_orders:
            liq_str = ', '.join(f"{ind}=${v:.2f}" for ind, v in liquidation_orders.items())
            print(f"Master liquidation orders: {liq_str}")

        orders = []

        # ── Industry activity flags: skip trading if capital < priciest stock ─
        active_industries = set()
        inactive_log      = []
        for ind, syms in industries.items():
            ind_capital  = total_account_val * alloc_prop.get(ind, 0.0)
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
            model = load_weighted_model(StockNN, args.model_dir, industry)
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
                for j, sym in enumerate(symbols):
                    if sym not in day_data:
                        continue
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

        # Save updated state
        state['histories'] = histories
        save_state(state)

        # Update owners.json after processing day's data and potential withdrawals
        total_portfolio_value = compute_total_portfolio_value(cash, holdings, day_data, histories)
        update_owners_file(total_portfolio_value, args.paper)

        # Retrain models using real Alpaca state as portfolio seed.
        # yesterday_data = model-input features (day N); day_data = fill prices (day N+1).
        # This matches the offline training's day-N-predict / day-N+1-fill pattern.
        ind_primed, master_primed = build_primed_portfolios(trading_client, industries, allocations)
        industry_top_scores = {}
        for industry, symbols in industries.items():
            if any(os.path.exists(f"{args.model_dir}/{industry}_model_{i}.pt") for i in range(10)):
                result = train_industry_one_day_prod(
                    industry, symbols, yesterday_data, ind_primed[industry], args.model_dir,
                    today_data=day_data, seq_flags=seq_flags)
                if result is not None:
                    industry_top_scores[industry] = result
        if any(os.path.exists(f"{args.model_dir}/master_model_{i}.pt") for i in range(10)):
            train_master_one_day_prod(yesterday_data, industries, master_primed, args.model_dir,
                                      industry_top_scores=industry_top_scores)

        # Submit orders to Alpaca if requested
        if args.output:
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
                        trading_client.submit_order(order_req)
                    elif order['action'] == 'sell':
                        order_req = LimitOrderRequest(
                            symbol=order['symbol'],
                            qty=order['quantity'],
                            side=OrderSide.SELL,
                            type=OrderType.LIMIT,
                            time_in_force=TimeInForce.DAY,
                            limit_price=order['price']
                        )
                        trading_client.submit_order(order_req)
                    elif order['action'] == 'stop_loss':
                        order_req = StopOrderRequest(
                            symbol=order['symbol'],
                            qty=order['quantity'],
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.GTC,
                            stop_price=order['price']
                        )
                        trading_client.submit_order(order_req)
                except Exception as e:
                    print(f"Error submitting order for {order['symbol']}: {e}")

    else:
        parser.print_help()

if __name__ == '__main__':
    main()
