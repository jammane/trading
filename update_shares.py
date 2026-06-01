"""
update_shares.py — Manage owner share counts in owners/owners.json.

Usage:
    python update_shares.py --invest    <name> <amount>
    python update_shares.py --withdrawal <name> <amount>
    python update_shares.py --value
"""

import argparse
import csv
import json
import os

OWNERS_DIR = 'owners'
OWNERS_FILE = os.path.join(OWNERS_DIR, 'owners.json')
DEFAULT_PRICE_PER_SHARE = 100.0

def load_owners():
    """Load owners.json, initializing if it doesn't exist."""
    try:
        if os.path.exists(OWNERS_FILE):
            with open(OWNERS_FILE) as f:
                owners = json.load(f)
                total_shares = sum(owners.get("owners", {}).values())
                if owners["total value"] > 100 and total_shares > 1:
                    owners["price per share"] = owners["total value"] / total_shares
                return owners
        else:
            return {"total value": 0.0, "price per share": DEFAULT_PRICE_PER_SHARE, "owners": {}}
    except Exception as e:
        print(f"Error loading {OWNERS_FILE}: {e}")
        return {"total value": 0.0, "price per share": DEFAULT_PRICE_PER_SHARE, "owners": {}}

def save_owners(owners_data):
    """Save owners.json, creating directory if needed."""
    try:
        if not os.path.exists(OWNERS_DIR):
            os.makedirs(OWNERS_DIR)
            print(f"Created owners directory: {OWNERS_DIR}")
        with open(OWNERS_FILE, 'w') as f:
            json.dump(owners_data, f, indent=4)
        print(f"Updated {OWNERS_FILE}")
    except Exception as e:
        print(f"Error saving {OWNERS_FILE}: {e}")

def update_value_csv(owners_data):
    """Generate owners_value.csv in the owners directory."""
    try:
        csv_path = os.path.join(OWNERS_DIR, 'owners_value.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['total value', owners_data.get('total value', 0.0)])
            price_per_share = owners_data.get('price per share', DEFAULT_PRICE_PER_SHARE)
            writer.writerow(['price per share', price_per_share])
            writer.writerow(['owner', 'shares', 'value'])
            for name, shares in owners_data.get('owners', {}).items():
                if shares > 0:
                    writer.writerow([name, shares, shares * price_per_share])
        print(f"Generated {csv_path}")
    except Exception as e:
        print(f"Error generating {csv_path}: {e}")

def main():
    """Parse CLI args and dispatch to invest, withdraw, or CSV export."""
    parser = argparse.ArgumentParser(description="Update shares in owners.json based on investments or withdrawals.")
    parser.add_argument('--invest', nargs=2, metavar=('name', 'amount'), help='Invest dollar amount for name')
    parser.add_argument('--withdrawal', nargs=2, metavar=('name', 'amount'), help='Withdraw dollar amount for name')
    parser.add_argument('--value', action='store_true', help='Output owners_value.csv with total value, price per share, and owner details')
    args = parser.parse_args()

    if not args.invest and not args.withdrawal and not args.value:
        parser.print_help()
        return

    # Load owners.json
    owners_data = load_owners()

    # Ensure price per share exists
    if 'price per share' not in owners_data:
        owners_data['price per share'] = DEFAULT_PRICE_PER_SHARE
        print(f"Set default price per share to ${DEFAULT_PRICE_PER_SHARE}")

    price_per_share = owners_data.get('price per share', DEFAULT_PRICE_PER_SHARE)

    # Initialize owners dictionary if not present
    if 'owners' not in owners_data:
        owners_data['owners'] = {}

    # Process investment
    if args.invest:
        name, amount = args.invest
        try:
            amount = float(amount)
            if amount <= 0:
                print(f"Invalid investment amount: {amount}. Must be positive.")
                return
            shares = amount / price_per_share
            owners_data['owners'][name] = owners_data['owners'].get(name, 0.0) + shares
            print(f"Added {shares:.6f} shares for {name} (invested ${amount:.2f})")
            # Update total value
            owners_data['total value'] = owners_data.get('total value', 0.0) + amount
            save_owners(owners_data)
        except ValueError:
            print(f"Invalid investment amount: {amount}. Must be a number.")
            return

    # Process withdrawal
    if args.withdrawal:
        name, amount = args.withdrawal
        try:
            amount = float(amount)
            if amount <= 0:
                print(f"Invalid withdrawal amount: {amount}. Must be positive.")
                return
            shares = amount / price_per_share
            current_shares = owners_data['owners'].get(name, 0.0)
            if current_shares < shares:
                print(f"Insufficient shares for {name}. Requested: {shares:.6f}, Available: {current_shares:.6f}")
                return
            owners_data['owners'][name] = current_shares - shares
            if owners_data['owners'][name] <= 0:
                del owners_data['owners'][name]
            print(f"Removed {shares:.6f} shares for {name} (withdrew ${amount:.2f})")
            # Update total value
            owners_data['total value'] = max(0.0, owners_data.get('total value', 0.0) - amount)
            save_owners(owners_data)
        except ValueError:
            print(f"Invalid withdrawal amount: {amount}. Must be a number.")
            return

    # Generate CSV if requested
    if args.value:
        update_value_csv(owners_data)

if __name__ == '__main__':
    main()
