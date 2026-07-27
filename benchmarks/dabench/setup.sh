#!/usr/bin/env bash
# Fetches the InfiAgent-DABench (DA-Agent) validation set + scoring script from
# https://github.com/InfiAgent/InfiAgent into benchmarks/dabench/vendor/.
# vendor/ is gitignored — third-party benchmark data and scripts are not committed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR_DIR="$SCRIPT_DIR/vendor"
REPO_URL="https://github.com/InfiAgent/InfiAgent.git"
SUBPATH="examples/DA-Agent"

if [ -d "$VENDOR_DIR" ] && [ -n "$(ls -A "$VENDOR_DIR" 2>/dev/null)" ]; then
  echo "vendor/ already populated at $VENDOR_DIR — remove it first to re-fetch." >&2
  exit 0
fi

TMP_CLONE="$(mktemp -d)"
trap 'rm -rf "$TMP_CLONE"' EXIT

git clone --depth 1 --filter=blob:none --sparse "$REPO_URL" "$TMP_CLONE"
git -C "$TMP_CLONE" sparse-checkout set "$SUBPATH"

mkdir -p "$VENDOR_DIR"
cp -R "$TMP_CLONE/$SUBPATH/." "$VENDOR_DIR/"

echo "Fetched DA-Agent benchmark into $VENDOR_DIR:"
echo "  - $VENDOR_DIR/data/da-dev-questions.jsonl"
echo "  - $VENDOR_DIR/data/da-dev-labels.jsonl"
echo "  - $VENDOR_DIR/data/da-dev-tables/*.csv"
echo "  - $VENDOR_DIR/eval_closed_form.py"
