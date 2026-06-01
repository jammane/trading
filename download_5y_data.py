"""
download_5y_data.py — Download historical daily OHLCV data for all 144 symbols.

Saves one JSON file per symbol under stock_data/<SYM>.json.
Rate-limited to ~50 requests/minute to stay within yfinance limits.

Usage:
    python download_5y_data.py [--years N]
"""

import argparse
import json
import os
import time

import yfinance as yf

parser = argparse.ArgumentParser(description='Download historical stock data.')
parser.add_argument('--years', type=int, default=5, help='Number of years of history to download (default: 5)')
args = parser.parse_args()

# Same industries as in your training_v2.py and production_v2.py
industries = {
    # High-beta semiconductors & hardware
    'tech_hardware':          ['NVDA','AMD', 'MU',  'SMCI','MRVL','ON',  'AMAT','LRCX','KLAC','TSM', 'SWKS','MPWR'],
    # High-beta cloud / AI software
    'tech_software_ai':       ['PLTR','SNOW','DDOG','NET', 'CRWD','ZS',  'PANW','NOW', 'ADBE','CRM', 'FTNT','OKTA'],
    # High-beta fintech + traditional finance
    'financials':             ['XYZ', 'PYPL','AFRM','UPST','MELI','COIN','GS',  'SCHW',  'C',   'COF', 'BX',   'APO'  ],
    # EVs, autos, travel — already volatile
    'consumer_discretionary': ['TSLA','RCL', 'XPEV','LI',  'APTV',   'GM',  'LEA','WYNN','BKNG','ABNB','UBER','LYFT'],
    # Streaming, social, gig economy — replaces low-vol restaurants
    'consumer_services':      ['NFLX','ROKU','SPOT','META','IAC','PINS','DASH','RBLX','TTWO','LYV',  'MTCH','WBD'],
    # Biotech / genomics — replaces large-cap pharma
    'health_care':            ['MRNA','BNTX','IMVT','CRSP','ARWR','MYGN','EXAS','INMD','HIMS','BEAM','ACAD','BMRN'],
    # Airlines + industrials — already volatile
    'industrials':            ['BA',  'GE',  'CAT', 'DE',  'DAL', 'UAL', 'XPO', 'LUV', 'ALK', 'GNRC','BTU', 'STLD' ],
    # High-beta lifestyle/consumer — replaces low-vol staples
    'consumer_staples':       ['CELH','SFM','ELF', 'LULU','DECK','YETI','NKE', 'CROX', 'DKNG','PENN','MGM', 'CZR' ],
    # Volatile E&P + services — already performing
    'energy':                 ['FANG','DVN', 'OXY', 'CTRA','AR',  'EQT', 'RRC', 'SM',  'SLB', 'COP', 'EOG', 'VLO' ],
    # Clean energy / renewables — replaces stable utility stocks
    'utilities':              ['ENPH','FSLR','SEDG','CWEN', 'VST','BE',  'BEP','DQ',  'CSIQ','JKS', 'HASI','NRG' ],
    # Homebuilders + proptech — replaces low-vol REITs
    'real_estate':            ['DHI', 'LEN', 'PHM', 'TOL', 'MTH', 'KBH', 'TPH', 'TMHC','LGIH','CSGP','Z',   'SKY' ],
    # Volatile precious-metal miners — replaces ETFs
    'materials':              ['NEM', 'AEM', 'FCX', 'SCCO','TECK', 'AA', 'SQM','WPM', 'AU',  'PAAS','GFI',  'CDE' ],
}

os.makedirs('stock_data', exist_ok=True)

all_symbols = [sym for syms in industries.values() for sym in syms]

print(f"Downloading ~{args.years} year(s) of daily data for {len(all_symbols)} symbols...")
print("This may take 10-30 minutes depending on your connection (yfinance rate limits apply).\n")

for i, sym in enumerate(all_symbols, 1):
    try:
        print(f"[{i:3d}/{len(all_symbols)}] Downloading {sym} ...")

        ticker = yf.Ticker(sym)
        hist = ticker.history(period=f"{args.years}y", interval="1d")

        if hist.empty:
            print(f"    No data returned for {sym}")
            continue

        data = []
        for date, row in hist.iterrows():
            data.append({
                "date": date.strftime('%Y-%m-%d'),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": int(row['Volume'])
            })

        data = data[-(args.years * 260 + 50):]

        with open(f"stock_data/{sym}.json", 'w') as f:
            json.dump({"days": data}, f, indent=2)

        print(f"    Saved {len(data)} days for {sym}")

        # Be respectful to avoid rate limits (yfinance can be sensitive)
        time.sleep(1.2)   # ~50 requests per minute max is usually safe

    except Exception as e:
        print(f"    Error downloading {sym}: {e}")
        time.sleep(5)  # longer pause on error

print("\n🔎 Verifying saved day counts...")
saved_counts = {}
missing_or_invalid = []

for sym in all_symbols:
    file_path = f"stock_data/{sym}.json"
    if not os.path.exists(file_path):
        missing_or_invalid.append((sym, "file missing"))
        continue

    try:
        with open(file_path) as f:
            payload = json.load(f)
        saved_counts[sym] = len(payload.get("days", []))
    except Exception as e:
        missing_or_invalid.append((sym, f"read error: {e}"))

expected_days = max(saved_counts.values(), default=0)
incomplete_symbols = sorted(
    (sym, count) for sym, count in saved_counts.items()
    if expected_days and count < expected_days
)

print("\n✅ Download complete!")
print("   All files saved in the 'stock_data/' folder.")
if expected_days:
    print(f"   Expected full count for this run: {expected_days} trading days")
else:
    print("   Expected full count for this run could not be determined.")

if not missing_or_invalid and not incomplete_symbols:
    print(f"   All {len(all_symbols)} symbols have the full expected day count.")
else:
    print("   Symbols that did not reach the full expected day count:")
    for sym, reason in missing_or_invalid:
        print(f"   - {sym}: {reason}")
    for sym, count in incomplete_symbols:
        print(f"   - {sym}: {count}/{expected_days} days")

print("   You can now run initial training:")
print("   python training_v2.py --output models")
