#!/usr/bin/env bash
#
# k8s/setup.sh — first-time cluster setup for the trading deployment.
#
# Runs all steps required to bring a fresh k3s (or Kubernetes) node up:
#   1. Builds and pushes the Docker image
#   2. Applies the namespace and all persistent-volume / PVC manifests
#   3. Creates (or updates) the Alpaca credentials secret — keys are passed
#      directly to kubectl and are never written to disk
#   4. Optionally submits the stock-data download job
#
# Intentionally omitted (run manually when ready):
#   - job-training.yaml       — kubectl apply -f k8s/job-training.yaml
#   - cronjob-production.yaml — kubectl apply -f k8s/cronjob-production.yaml
#
# Prerequisites: docker and kubectl must be on PATH and configured for the
# target cluster.  Run from any directory — paths are resolved relative to
# this script's location.
#
# The script is idempotent: re-running it rebuilds the image, reconciles
# manifests, and rotates credentials without disrupting existing resources.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# ── Prerequisites ──────────────────────────────────────────────────────────────
for cmd in docker kubectl; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "Error: $cmd not found on PATH."; exit 1; }
done

# ── Step 1: Build and push Docker image ───────────────────────────────────────
echo ""
echo "==> Step 1/4: Building and pushing Docker image ..."
cd "$REPO_DIR"
docker build -t jammane80/trading:latest .
docker push jammane80/trading:latest

# ── Step 2: Namespace and persistent-storage manifests ────────────────────────
echo ""
echo "==> Step 2/4: Applying namespace and persistent-volume manifests ..."
kubectl apply -f "$SCRIPT_DIR/namespace.yaml"
kubectl apply \
    -f "$SCRIPT_DIR/pv-app-state.yaml"        -f "$SCRIPT_DIR/pvc-app-state.yaml" \
    -f "$SCRIPT_DIR/pv-stock-data.yaml"        -f "$SCRIPT_DIR/pvc-stock-data.yaml" \
    -f "$SCRIPT_DIR/pv-models-training.yaml"   -f "$SCRIPT_DIR/pvc-models-training.yaml" \
    -f "$SCRIPT_DIR/pv-models-prod.yaml"       -f "$SCRIPT_DIR/pvc-models-prod.yaml"

# ── Step 3: Alpaca credentials secret ─────────────────────────────────────────
echo ""
echo "==> Step 3/4: Alpaca credentials secret"
echo "Keys are passed directly to kubectl — never written to disk."
echo ""
read -rsp "  Alpaca API key: "     ALPACA_API_KEY;     echo ""
read -rsp "  Alpaca secret key: "  ALPACA_SECRET_KEY;  echo ""
[ -z "$ALPACA_API_KEY" ]     && { echo "Error: API key cannot be empty.";    exit 1; }
[ -z "$ALPACA_SECRET_KEY" ]  && { echo "Error: Secret key cannot be empty."; exit 1; }

# --dry-run=client + apply makes this idempotent: creates on first run,
# updates on subsequent runs without kubectl complaining about existing resources.
kubectl create secret generic alpaca-credentials \
    --namespace trading \
    --from-literal=ALPACA_API_KEY="$ALPACA_API_KEY" \
    --from-literal=ALPACA_SECRET_KEY="$ALPACA_SECRET_KEY" \
    --dry-run=client -o yaml \
  | kubectl apply -f -
echo "  Secret alpaca-credentials applied."

# ── Step 4: Stock-data download job (optional) ────────────────────────────────
echo ""
echo "==> Step 4/4: Stock data download"
echo "The download job fetches ~5 years of OHLCV data for all 144 symbols (~200 MB)."
echo "Required before training.  Safe to skip if stock_data/ is already populated."
echo ""
read -rp "Submit the download job now? [y/N] " REPLY
if [[ "${REPLY:-}" =~ ^[Yy]$ ]]; then
    # Jobs cannot be updated in place — delete any prior run before reapplying.
    kubectl delete job download-job --namespace trading --ignore-not-found
    kubectl apply -f "$SCRIPT_DIR/job-download.yaml"
    echo ""
    echo "  Download job submitted.  Monitor progress:"
    echo "    kubectl logs -n trading -f -l job-name=download-job"
else
    echo ""
    echo "  Skipped.  Run when ready:"
    echo "    kubectl apply -f k8s/job-download.yaml"
fi

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "Cluster setup complete.  Next steps when ready:"
echo "  Train:      kubectl apply -f k8s/job-training.yaml"
echo "  Production: kubectl apply -f k8s/cronjob-production.yaml"
