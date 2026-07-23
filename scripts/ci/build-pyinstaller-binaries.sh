#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <pyinstaller-spec> [<pyinstaller-spec> ...]" >&2
  exit 2
fi

# --extra ml bundles scikit-learn so the analyst's ml_* operators work in the frozen binary; the
# .spec files collect_all(sklearn/scipy/…) only when it's importable at build time.
uv sync --no-dev --group build --extra ml

for spec in "$@"; do
  uv run --no-dev --group build --extra ml pyinstaller "${spec}"
done
