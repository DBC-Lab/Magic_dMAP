#!/usr/bin/env bash
set -euo pipefail

: "${METADATA_CSV:?Set METADATA_CSV to the ABIDE-I metadata CSV}"
: "${SPLIT_ROOT:?Set SPLIT_ROOT to the output split directory}"
: "${SPLIT_MANIFEST:?Set SPLIT_MANIFEST to the output TSV manifest}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "${ROOT}/src/magic_dmap/make_site_heldout_splits.py" \
  --metadata-csv "${METADATA_CSV}" \
  --out-root "${SPLIT_ROOT}" \
  --manifest-out "${SPLIT_MANIFEST}"
