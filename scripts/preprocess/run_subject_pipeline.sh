#!/bin/bash
set -euo pipefail

module load fsl
module load ants

export PATH="$HOME/tools/c3d/c3d-1.4.4-Linux-gcc64/bin:$PATH"
C3D_AFFINE_TOOL="$HOME/tools/c3d/c3d-1.4.4-Linux-gcc64/bin/c3d_affine_tool"

USE_C3D="${USE_C3D:-0}"
if [ "${USE_C3D}" = "1" ] && [ -x "${C3D_AFFINE_TOOL}" ]; then
    USE_C3D=1
else
    USE_C3D=0
fi

export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

BASE_DIR="${BASE_DIR:?Set BASE_DIR to the preprocessing work directory}"
CURATED_ROOT="${CURATED_ROOT:?Set CURATED_ROOT to the curated BIDS input root}"
EXPORT_ROOT="${EXPORT_ROOT:-${BASE_DIR}/exports/neurostorm_nii}"

SUBJECT_ID="${1:-}"
if [ -z "${SUBJECT_ID}" ]; then
    echo "[Error] Usage: $0 sub-<ID>"
    exit 1
fi

SUBJECT_DIR="${CURATED_ROOT}/${SUBJECT_ID}/ses-1"
WORK_DIR="${BASE_DIR}/work/${SUBJECT_ID}/ses-1"
ANAT_DIR="${SUBJECT_DIR}/anat"
FUNC_DIR="${SUBJECT_DIR}/func"

T1W="${ANAT_DIR}/${SUBJECT_ID}_ses-1_T1w.nii.gz"
T1_BRAIN="${ANAT_DIR}/${SUBJECT_ID}_ses-1_desc-brain_T1w.nii.gz"
T1_LABEL="${ANAT_DIR}/${SUBJECT_ID}_ses-1_desc-label_dseg.nii.gz"
T1_WMSEG="${WORK_DIR}/anat/${SUBJECT_ID}_ses-1_desc-wmseg_mask.nii.gz"

BOLD_RAW="${FUNC_DIR}/${SUBJECT_ID}_ses-1_task-rest_bold.nii.gz"
BOLD_JSON="${FUNC_DIR}/${SUBJECT_ID}_ses-1_task-rest_bold.json"

HMC_DIR="${WORK_DIR}/hmc_out"
BOLD_HMC="${HMC_DIR}/${SUBJECT_ID}_ses-1_task-rest_desc_bold_hmc.nii.gz"
MCFLIRT_PREFIX="${HMC_DIR}/${SUBJECT_ID}_ses-1_task-rest_mcflirt"
MCFLIRT_MAT_DIR="${HMC_DIR}/${SUBJECT_ID}_ses-1_task-rest_mcflirt.mat"

BOLDREF="${WORK_DIR}/func/boldref.nii.gz"
BOLDREF_BRAIN="${WORK_DIR}/func/boldref_hdbetbrain.nii.gz"
BOLDREF_MASK="${WORK_DIR}/func/boldref_hdbetbrain_mask.nii.gz"

STEP5_DIR="${WORK_DIR}/step5_fmri2T1w_nmi_bbr_out"
BOLD2T1_INIT_MAT="${STEP5_DIR}/bold2T1_init.mat"
BOLD2T1_BBR_MAT="${STEP5_DIR}/bold2T1_bbr.mat"

STEP6_DIR="${WORK_DIR}/step6_T1w_to_MNI_out"
T1_TO_MNI_AFFINE="${STEP6_DIR}/t1_to_mni_0GenericAffine.mat"
T1_TO_MNI_WARP="${STEP6_DIR}/t1_to_mni_1Warp.nii.gz"
T1_LABEL_MNI="${STEP6_DIR}/t1_to_mni_label_Warped.nii.gz"

STEP7_DIR="${WORK_DIR}/step7_apply_all_to_MNI_out"
LOCAL_TMP_ROOT="${TMPDIR:-/tmp}/${USER:-user}_abide1_mni_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}_${SUBJECT_ID}"
TMP_DIR="${LOCAL_TMP_ROOT}/step7_tmp"
MNI_BRAIN="${MNI_BRAIN:-${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz}"
MNI_REF_NATIVEBOLD="${STEP7_DIR}/MNI152_T1_1mm_brain_resampled_to_bold_spacing.nii.gz"
FINAL_BOLD="${STEP7_DIR}/${SUBJECT_ID}_ses-1_task-rest_space-MNI_desc-preproc_bold.nii.gz"
FINAL_BOLDREF="${STEP7_DIR}/${SUBJECT_ID}_ses-1_task-rest_space-MNI_desc-preproc_boldref.nii.gz"
FINAL_MASK="${STEP7_DIR}/${SUBJECT_ID}_ses-1_task-rest_space-MNI_desc-brain_mask.nii.gz"
BOLD2T1_BBR_ITK="${STEP7_DIR}/bold2T1_bbr_itk.txt"

cleanup() {
    rm -rf "${LOCAL_TMP_ROOT}"
}
trap cleanup EXIT

if [ -f "${FINAL_BOLD}" ] && [ -f "${FINAL_BOLDREF}" ] && [ -f "${FINAL_MASK}" ]; then
    echo "[Info] Final outputs already exist for ${SUBJECT_ID}; skipping."
    exit 0
fi

rm -rf "${WORK_DIR}/anat" "${WORK_DIR}/func" "${WORK_DIR}/hmc_out" \
       "${WORK_DIR}/step5_fmri2T1w_nmi_bbr_out" "${WORK_DIR}/step6_T1w_to_MNI_out" \
       "${WORK_DIR}/step7_apply_all_to_MNI_out"

mkdir -p "${WORK_DIR}/anat" "${WORK_DIR}/func" "${WORK_DIR}/hmc_out" \
         "${WORK_DIR}/step5_fmri2T1w_nmi_bbr_out" "${WORK_DIR}/step6_T1w_to_MNI_out" \
         "${WORK_DIR}/step7_apply_all_to_MNI_out" "${EXPORT_ROOT}" "${TMP_DIR}"

command -v mcflirt >/dev/null 2>&1 || { echo "[Error] mcflirt not found"; exit 1; }
command -v bet >/dev/null 2>&1 || { echo "[Error] bet not found"; exit 1; }
command -v flirt >/dev/null 2>&1 || { echo "[Error] flirt not found"; exit 1; }
command -v convert_xfm >/dev/null 2>&1 || { echo "[Error] convert_xfm not found"; exit 1; }
command -v antsRegistrationSyN.sh >/dev/null 2>&1 || { echo "[Error] antsRegistrationSyN.sh not found"; exit 1; }
command -v antsApplyTransforms >/dev/null 2>&1 || { echo "[Error] antsApplyTransforms not found"; exit 1; }
command -v ResampleImageBySpacing >/dev/null 2>&1 || { echo "[Error] ResampleImageBySpacing not found"; exit 1; }
command -v fslsplit >/dev/null 2>&1 || { echo "[Error] fslsplit not found"; exit 1; }
command -v fslmerge >/dev/null 2>&1 || { echo "[Error] fslmerge not found"; exit 1; }
command -v fslcpgeom >/dev/null 2>&1 || { echo "[Error] fslcpgeom not found"; exit 1; }

for f in "${T1W}" "${T1_BRAIN}" "${T1_LABEL}" "${BOLD_RAW}" "${MNI_BRAIN}"; do
    if [ ! -f "${f}" ]; then
        echo "[Error] Missing input: ${f}"
        exit 1
    fi
done

echo "[Info] Subject     = ${SUBJECT_ID}"
echo "[Info] Subject dir = ${SUBJECT_DIR}"
echo "[Info] Work dir    = ${WORK_DIR}"
echo "[Info] Tmp dir     = ${TMP_DIR}"
echo "[Info] USE_C3D     = ${USE_C3D}"

echo "[Step 1] Extracting WM from dseg..."
fslmaths "${T1_LABEL}" -thr 3 -uthr 3 -bin "${T1_WMSEG}"

echo "[Step 2] Running HMC..."
mcflirt \
  -in "${BOLD_RAW}" \
  -out "${MCFLIRT_PREFIX}" \
  -plots \
  -mats \
  -rmsrel \
  -rmsabs \
  -refvol 0
mv "${MCFLIRT_PREFIX}.nii.gz" "${BOLD_HMC}"

echo "[Step 3] Creating boldref..."
fslmaths "${BOLD_HMC}" -Tmean "${BOLDREF}"

echo "[Step 4] Skull-stripping boldref with HD-BET..."
if command -v hd-bet >/dev/null 2>&1; then
    hd-bet \
      -i "${BOLDREF}" \
      -o "${BOLDREF_BRAIN}" \
      -device cpu \
      -mode fast \
      -tta 0
elif command -v hd_bet >/dev/null 2>&1; then
    hd_bet \
      -i "${BOLDREF}" \
      -o "${BOLDREF_BRAIN}" \
      -device cpu \
      -mode fast \
      -tta 0
else
    echo "[Warn] HD-BET unavailable; falling back to FSL BET for boldref skull stripping."
    bet "${BOLDREF}" "${BOLDREF_BRAIN%.nii.gz}" -m -f 0.3
fi

echo "[Step 5] BOLDREF -> T1w (NMI + BBR)..."
flirt \
  -in "${BOLDREF_BRAIN}" \
  -ref "${T1_BRAIN}" \
  -omat "${BOLD2T1_INIT_MAT}" \
  -out "${STEP5_DIR}/bold2T1_init.nii.gz" \
  -dof 6 \
  -cost normmi \
  -searchrx -90 90 \
  -searchry -90 90 \
  -searchrz -90 90

flirt \
  -in "${BOLDREF_BRAIN}" \
  -ref "${T1_BRAIN}" \
  -out "${STEP5_DIR}/bold2T1_bbr.nii.gz" \
  -omat "${BOLD2T1_BBR_MAT}" \
  -init "${BOLD2T1_INIT_MAT}" \
  -dof 6 \
  -cost bbr \
  -wmseg "${T1_WMSEG}" \
  -schedule "${FSLDIR}/etc/flirtsch/bbr.sch"

echo "[Step 6] T1 brain -> MNI..."
antsRegistrationSyN.sh \
  -d 3 \
  -f "${MNI_BRAIN}" \
  -m "${T1_BRAIN}" \
  -o "${STEP6_DIR}/t1_to_mni_" \
  -t s \
  -n "${SLURM_CPUS_PER_TASK:-8}"

antsApplyTransforms \
  -d 3 \
  -i "${T1_LABEL}" \
  -r "${MNI_BRAIN}" \
  -o "${T1_LABEL_MNI}" \
  -n GenericLabel \
  -t "${T1_TO_MNI_WARP}" \
  -t "${T1_TO_MNI_AFFINE}"

echo "[Step 7] Applying HMC + BBR + T1->MNI to all BOLD volumes..."
SX=$(fslval "${BOLD_RAW}" pixdim1)
SY=$(fslval "${BOLD_RAW}" pixdim2)
SZ=$(fslval "${BOLD_RAW}" pixdim3)
TR_SEC=$(fslval "${BOLD_RAW}" pixdim4)

ResampleImageBySpacing 3 \
  "${MNI_BRAIN}" \
  "${MNI_REF_NATIVEBOLD}" \
  "${SX}" "${SY}" "${SZ}" \
  0 0 0

rm -f "${TMP_DIR}"/vol_*.nii.gz
fslsplit "${BOLD_RAW}" "${TMP_DIR}/vol_" -t

if [ "${USE_C3D}" -eq 1 ]; then
    "${C3D_AFFINE_TOOL}" \
      -ref "${T1_BRAIN}" \
      -src "${BOLDREF_BRAIN}" \
      "${BOLD2T1_BBR_MAT}" \
      -fsl2ras \
      -oitk "${BOLD2T1_BBR_ITK}"
fi

for VOL in "${TMP_DIR}"/vol_*.nii.gz; do
    base=$(basename "${VOL}" .nii.gz)
    idx=${base#vol_}

    HMC_FSL="${MCFLIRT_MAT_DIR}/MAT_${idx}"
    HMC_ITK="${TMP_DIR}/hmc_${idx}_itk.txt"
    VOL2T1_FSL="${TMP_DIR}/vol2t1_${idx}.mat"
    VOL_T1="${TMP_DIR}/t1_${idx}.nii.gz"
    OUT_VOL="${TMP_DIR}/mni_${idx}.nii.gz"

    if [ ! -f "${HMC_FSL}" ]; then
        echo "[Error] Missing HMC matrix: ${HMC_FSL}"
        exit 1
    fi

    if [ "${USE_C3D}" -eq 1 ]; then
        "${C3D_AFFINE_TOOL}" \
          -ref "${BOLDREF_BRAIN}" \
          -src "${VOL}" \
          "${HMC_FSL}" \
          -fsl2ras \
          -oitk "${HMC_ITK}"

        antsApplyTransforms \
          -d 3 \
          -i "${VOL}" \
          -r "${MNI_REF_NATIVEBOLD}" \
          -o "${OUT_VOL}" \
          -n LanczosWindowedSinc \
          -t "${T1_TO_MNI_WARP}" \
          -t "${T1_TO_MNI_AFFINE}" \
          -t "${BOLD2T1_BBR_ITK}" \
          -t "${HMC_ITK}"
    else
        convert_xfm \
          -omat "${VOL2T1_FSL}" \
          -concat "${BOLD2T1_BBR_MAT}" \
          "${HMC_FSL}"

        flirt \
          -in "${VOL}" \
          -ref "${T1_BRAIN}" \
          -applyxfm \
          -init "${VOL2T1_FSL}" \
          -interp sinc \
          -out "${VOL_T1}"

        antsApplyTransforms \
          -d 3 \
          -i "${VOL_T1}" \
          -r "${MNI_REF_NATIVEBOLD}" \
          -o "${OUT_VOL}" \
          -n LanczosWindowedSinc \
          -t "${T1_TO_MNI_WARP}" \
          -t "${T1_TO_MNI_AFFINE}"
    fi
done

fslmerge -tr "${FINAL_BOLD}" "${TMP_DIR}"/mni_*.nii.gz "${TR_SEC}"
fslmaths "${FINAL_BOLD}" -Tmean "${FINAL_BOLDREF}"
fslcpgeom "${FINAL_BOLD}" "${FINAL_BOLDREF}"

if [ "${USE_C3D}" -eq 1 ]; then
    antsApplyTransforms \
      -d 3 \
      -i "${BOLDREF_MASK}" \
      -r "${MNI_REF_NATIVEBOLD}" \
      -o "${FINAL_MASK}" \
      -n GenericLabel \
      -t "${T1_TO_MNI_WARP}" \
      -t "${T1_TO_MNI_AFFINE}" \
      -t "${BOLD2T1_BBR_ITK}"
else
    MASK_T1="${TMP_DIR}/boldref_mask_in_t1.nii.gz"
    flirt \
      -in "${BOLDREF_MASK}" \
      -ref "${T1_BRAIN}" \
      -applyxfm \
      -init "${BOLD2T1_BBR_MAT}" \
      -interp nearestneighbour \
      -out "${MASK_T1}"

    antsApplyTransforms \
      -d 3 \
      -i "${MASK_T1}" \
      -r "${MNI_REF_NATIVEBOLD}" \
      -o "${FINAL_MASK}" \
      -n GenericLabel \
      -t "${T1_TO_MNI_WARP}" \
      -t "${T1_TO_MNI_AFFINE}"
fi

fslmaths "${FINAL_MASK}" -thr 0.5 -bin "${FINAL_MASK}"
fslcpgeom "${FINAL_BOLD}" "${FINAL_MASK}"

echo "[Export] Linking final MNI outputs for NeuroSTORM..."
ln -sfn "${FINAL_BOLD}" "${EXPORT_ROOT}/${SUBJECT_ID}_ses-1_task-rest_space-MNI_desc-preproc_bold.nii.gz"
ln -sfn "${FINAL_MASK}" "${EXPORT_ROOT}/${SUBJECT_ID}_ses-1_task-rest_space-MNI_desc-brain_mask.nii.gz"
if [ -f "${BOLD_JSON}" ]; then
    ln -sfn "${BOLD_JSON}" "${EXPORT_ROOT}/${SUBJECT_ID}_ses-1_task-rest_bold.json"
fi

echo "[Info] Done for ${SUBJECT_ID}"
echo "[Info] Final BOLD   = ${FINAL_BOLD}"
echo "[Info] Final BOLDREF= ${FINAL_BOLDREF}"
echo "[Info] Final MASK   = ${FINAL_MASK}"
