#!/usr/bin/env bash
set -euo pipefail

: "${ROI_ROOT:?Set ROI_ROOT to the five-atlas ROI .1D root}"
: "${IMAGE_ROOT:=.}"
: "${OUTPUT_HDF5:?Set OUTPUT_HDF5 to the output HDF5 path}"
: "${SOURCE_SPLIT_ROOT:?Set SOURCE_SPLIT_ROOT to the source split directory}"
: "${FILTERED_SPLIT_ROOT:?Set FILTERED_SPLIT_ROOT to the filtered split directory}"
: "${PHENO_CSV:?Set PHENO_CSV to the phenotype CSV}"
: "${HDF5_MANIFEST_OUT:?Set HDF5_MANIFEST_OUT to the HDF5 manifest path}"
: "${MISSING_OUT:?Set MISSING_OUT to the missing-subject report path}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "${ROOT}/src/magic_dmap/build_dfc_hdf5.py" \
  --roi-root "${ROI_ROOT}" \
  --neurostorm-image-root "${IMAGE_ROOT}" \
  --output-hdf5 "${OUTPUT_HDF5}" \
  --split-root "${SOURCE_SPLIT_ROOT}" \
  --filtered-split-root "${FILTERED_SPLIT_ROOT}" \
  --pheno "${PHENO_CSV}" \
  --manifest "${HDF5_MANIFEST_OUT}" \
  --missing "${MISSING_OUT}" \
  --sequence-length 20 \
  --stride-within-seq 1 \
  --stride-between-seq 20 \
  --fold-count 10 \
  --split-prefix split_cv10 \
  --fisher-z \
  --force
