#!/bin/bash

# Exit on any error
set -e

# Prompt for API keys
echo "Yahoo Finance is used for market data and does not require an API key."

echo "Please enter your Alpaca API key:"
read -r ALPACA_API_KEY
if [ -z "$ALPACA_API_KEY" ]; then
    echo "Error: Alpaca API key cannot be empty."
    exit 1
fi

echo "Please enter your Alpaca Secret key:"
read -r ALPACA_SECRET_KEY
if [ -z "$ALPACA_SECRET_KEY" ]; then
    echo "Error: Alpaca Secret key cannot be empty."
    exit 1
fi

# Set API keys for the current session
export ALPACA_API_KEY="$ALPACA_API_KEY"
export ALPACA_SECRET_KEY="$ALPACA_SECRET_KEY"

# Persist API keys in ~/.bashrc
echo "Persisting Alpaca API keys in ~/.bashrc..."
if ! grep -q "export ALPACA_API_KEY=" ~/.bashrc; then
    echo "export ALPACA_API_KEY=\"$ALPACA_API_KEY\"" >> ~/.bashrc
else
    sed -i "s|export ALPACA_API_KEY=.*|export ALPACA_API_KEY=\"$ALPACA_API_KEY\"|" ~/.bashrc
fi
if ! grep -q "export ALPACA_SECRET_KEY=" ~/.bashrc; then
    echo "export ALPACA_SECRET_KEY=\"$ALPACA_SECRET_KEY\"" >> ~/.bashrc
else
    sed -i "s|export ALPACA_SECRET_KEY=.*|export ALPACA_SECRET_KEY=\"$ALPACA_SECRET_KEY\"|" ~/.bashrc
fi

# Refresh Fedora package metadata

echo "Refreshing Fedora package metadata..."
sudo dnf makecache

# Install Python 3.12 and build dependencies on Fedora

echo "Installing Python 3.12..."
sudo dnf install -y python3.12 python3.12-devel

# Verify Python installation
if ! command -v python3.12 &> /dev/null; then
    echo "Error: Python 3.12 installation failed."
    exit 1
fi

# Create and activate a virtual environment
echo "Creating virtual environment..."
python3.12 -m venv .venv
source .venv/bin/activate

# Verify virtual environment activation
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: Failed to activate virtual environment."
    exit 1
fi

# Upgrade pip to the latest version
echo "Upgrading pip..."
pip install --upgrade pip

# Use a project-local temp directory so large wheel downloads do not fill /tmp
export TMPDIR="$(pwd)/.tmp"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

# Install required libraries
# Use the CPU-only PyTorch wheel to avoid pulling large CUDA packages on Linux.
echo "Installing required Python libraries..."
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
pip install --no-cache-dir yfinance alpaca-py

# Verify library installations
for pkg in torch yfinance alpaca-py; do
    if ! pip show $pkg &> /dev/null; then
        echo "Error: Failed to install $pkg."
        exit 1
    fi
done

# Update PATH to include Python and virtual environment binaries
VENV_BIN="$(pwd)/.venv/bin"
if [[ ":$PATH:" != *":$VENV_BIN:"* ]]; then
    echo "Adding virtual environment binaries to PATH..."
    echo "export PATH=\"$VENV_BIN:\$PATH\"" >> ~/.bashrc
    export PATH="$VENV_BIN:$PATH"
else
    echo "Virtual environment binaries already in PATH."
fi

# Ensure Python 3.12 is accessible
if ! command -v python3.12 &> /dev/null; then
    echo "Warning: Python 3.12 not found in PATH after installation."
fi

echo "Installation complete. Python 3.12, libraries, and Alpaca API keys configured."
echo "Virtual environment activated. PATH and API keys updated in ~/.bashrc."
echo "Run 'source ~/.bashrc' in new terminal sessions to apply changes."
