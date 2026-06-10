FROM python:3.12-slim

WORKDIR /app

# Torch in its own layer (~800MB) so source-code rebuilds don't re-download it
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.11.0

# Remaining runtime dependencies
RUN pip install --no-cache-dir \
    numpy==2.4.4 \
    yfinance==1.2.1 \
    alpaca-py==0.43.2 \
    keyring==25.7.0

# Source files only — data volumes mounted at runtime
COPY models.py fees.py universe.py \
     production_v2.py \
     training_v2.py training_v3.py training_v4.py \
     inspect_trades.py download_5y_data.py swap_symbols.py ./

CMD ["python", "production_v2.py", "--paper"]
