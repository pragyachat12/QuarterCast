#!/bin/bash
# Copies the latest trained model + feature dataset into the dashboard's
# bundled API data folder, so the live Vercel functions serve up-to-date
# predictions. Run this any time you retrain the model.
#
# Usage (from repo root):
#   bash scripts/sync_dashboard_data.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_MODEL="$REPO_ROOT/output/lgbm_model.pkl"
SRC_FEATURES="$REPO_ROOT/data/model_features.csv"
DEST_DIR="$REPO_ROOT/dashboard/api/_data"

mkdir -p "$DEST_DIR"

if [ ! -f "$SRC_MODEL" ]; then
  echo "ERROR: $SRC_MODEL not found. Run scripts/train_model.py first."
  exit 1
fi
if [ ! -f "$SRC_FEATURES" ]; then
  echo "ERROR: $SRC_FEATURES not found. Run scripts/build_features.py first."
  exit 1
fi

cp "$SRC_MODEL" "$DEST_DIR/lgbm_model.pkl"
cp "$SRC_FEATURES" "$DEST_DIR/model_features.csv"

echo "Synced model + features into $DEST_DIR"
echo "  - lgbm_model.pkl    ($(date -r "$SRC_MODEL" 2>/dev/null || stat -f %Sm "$SRC_MODEL" 2>/dev/null))"
echo "  - model_features.csv"
echo ""
echo "Remember to commit and redeploy the dashboard for changes to take effect."
