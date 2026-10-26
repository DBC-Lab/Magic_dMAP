#!/usr/bin/env bash
set -euo pipefail

: "${METADATA_CSV:?Set METADATA_CSV}"
: "${HDF5:?Set HDF5}"
: "${HDF5_MANIFEST:?Set HDF5_MANIFEST}"
: "${SPLIT_ROOT:?Set SPLIT_ROOT}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
: "${LOG_ROOT:?Set LOG_ROOT}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EPOCHS="${EPOCHS:-2000}"
LR="${LR:-5e-4}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
GPU_COUNT="${GPU_COUNT:-1}"
FOLDS="${FOLDS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17}"
RUN_ATLAS_LIST=(cc400 ez)
ALL_ATLASES=(aal cc400 dosenbach160 ez ho)

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
read -r -a FOLD_ARRAY <<< "${FOLDS}"
running=0
job_index=0

for atlas in "${RUN_ATLAS_LIST[@]}"; do
  for fold in "${FOLD_ARRAY[@]}"; do
    out_dir="${OUTPUT_ROOT}/${atlas}/fold${fold}"
    mkdir -p "${out_dir}"
    gpu=$((job_index % GPU_COUNT))
    job_index=$((job_index + 1))
    if [[ "${CPU_ONLY:-0}" == "1" ]]; then
      visible_devices=""
    else
      visible_devices="${gpu}"
    fi
    echo "[run] atlas=${atlas} fold=${fold} gpu=${gpu} lr=${LR} out=${out_dir}"
    CUDA_VISIBLE_DEVICES="${visible_devices}" python -u "${ROOT}/src/magic_dmap/train.py" \
      --fold "${fold}" \
      --split-root "${SPLIT_ROOT}" \
      --metadata-csv "${METADATA_CSV}" \
      --hdf5 "${HDF5}" \
      --hdf5-manifest "${HDF5_MANIFEST}" \
      --roi-atlas "${atlas}" \
      --atlases "${ALL_ATLASES[@]}" \
      --summary-stats basic \
      --max-dfc-features 4096 \
      --output-dir "${out_dir}" \
      --epochs "${EPOCHS}" \
      --patience "${EPOCHS}" \
      --batch-size 16 \
      --lr "${LR}" \
      --covar-mode sex \
      --margin 0.5 \
      --group-size 3 \
      > "${LOG_ROOT}/${atlas}_fold${fold}.log" 2>&1 &
    running=$((running + 1))
    if (( running >= MAX_PARALLEL )); then
      wait -n
      running=$((running - 1))
    fi
  done
done
wait

ENSEMBLE_OUT="${OUTPUT_ROOT}/ensemble_cc400_ez"
python -u "${ROOT}/src/magic_dmap/ensemble.py" \
  --single-atlas-root "${OUTPUT_ROOT}" \
  --output-dir "${ENSEMBLE_OUT}" \
  --atlases cc400 ez \
  --combos cc400_ez \
  --folds "${FOLD_ARRAY[@]}" \
  --min-combo-size 2 \
  --max-combo-size 2 \
  --improvement-label MAGIC-dMAP \
  --method-note "Validation-weighted CC400/EZ probability ensemble; weights fit on validation predictions only."

echo "[done] ${ENSEMBLE_OUT}/summary_by_combo.csv"
