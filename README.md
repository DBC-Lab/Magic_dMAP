# MAGIC-dMAP

**MAGIC-dMAP** (*Multiple Atlases and Granularities—Integrating Connectivity Dynamics for Multisite Autism Prediction*) is a multisite resting-state fMRI framework for autism spectrum disorder (ASD) classification.

This repository contains the reproducible implementation of the model and data-construction pipeline used for the ABIDE-I analysis described in `abstract/Autism_Abstract_Table1_Final.docx`. The private experiment label used during development is intentionally not part of the public project name.

## What the method does

1. Preprocesses resting-state fMRI into subject-level MNI-space volumes using the supplied anatomical inputs, MCFLIRT motion correction, FSL BBR, and ANTs T1-to-MNI registration.
2. Extracts ROI time series from five atlases: AAL, CC400, Dosenbach160, EZ, and Harvard–Oxford.
3. Computes 20-frame dynamic functional-connectivity clips with a 20-frame start stride, Pearson correlation, Fisher z transformation, and lower-triangle vectorization.
4. Builds coarse whole-brain distribution summaries from all five atlases and fine edge-level statistics from one atlas per model.
5. Trains separate CC400 and EZ triplet-embedding models with parallel MLP encoders, gated fusion, sex covariate integration, and a projection MLP.
6. Produces ASD probabilities from distances to training ASD and NC embeddings, then combines CC400 and EZ probabilities using validation-only weights.

The reference analysis uses 1,020 ABIDE-I subjects (500 ASD and 520 NC) from 17 sites. The public code does not include protected or large binary data, atlas NIfTIs, checkpoints, or result directories.

## Repository layout

```text
src/magic_dmap/
  extract_roi_timeseries.py       MNI fMRI -> ROI .1D files
  build_dfc_hdf5.py               ROI .1D -> clip-level dFC HDF5
  make_site_heldout_splits.py     deterministic 17-fold site-held-out splits
  features.py                     exact coarse/fine feature construction
  train.py                        triplet embedding training and evaluation
  saliency.py                     Integrated-Gradients attribution tables
  ensemble.py                     validation-weighted CC400/EZ fusion
scripts/
  preprocess/run_subject_pipeline.sh
  run_loso.sh
  make_splits.sh
  build_features.sh
  summarize_loso_results.py       full per-site train/validation/test table
configs/
  atlas_manifest.tsv
  site_heldout_validation_manifest.tsv
  splits/split_cv10_fold{1..17}.txt
docs/
  METHODS.md
  SPLIT_PROTOCOL.md
  ATLAS_REFERENCES.md
  DATA_LAYOUT.md
```

## Installation

```bash
git clone <repository-url> MAGIC-dMAP
cd MAGIC-dMAP
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

Preprocessing additionally requires FSL, ANTs, and optionally C3D. Their installation is platform-specific and is not performed by this repository.

## Reproduction workflow

All input paths are explicit so that the repository never assumes a local filesystem layout.

```bash
# 1. Prepare curated BIDS inputs and run subject-level preprocessing.
BASE_DIR=/path/to/work \
CURATED_ROOT=/path/to/curated_bids \
EXPORT_ROOT=/path/to/mni_exports \
bash scripts/preprocess/run_subject_pipeline.sh sub-XXXXX

# 2. Extract all five atlas ROI time series without global-signal regression.
python src/magic_dmap/extract_roi_timeseries.py \
  --fmri-root /path/to/mni_exports \
  --pheno /path/to/phenotypes.csv \
  --output-root /path/to/roi_1d \
  --manifest /path/to/results/roi_manifest.csv \
  --no-gsr \
  --atlas aal=/path/to/aal.nii.gz \
  --atlas cc400=/path/to/cc400.nii.gz \
  --atlas dosenbach160=/path/to/dosenbach160.nii.gz \
  --atlas ez=/path/to/ez.nii.gz \
  --atlas ho=/path/to/ho.nii.gz

# 3. Build 20-frame, stride-20 Fisher-z dFC clips. The source split root is
#    the original NeuroSTORM CV10 split directory; it is separate from the
#    17-fold site-held-out evaluation splits.
bash scripts/build_features.sh

# 4. Generate the deterministic site-held-out splits if rebuilding them.
bash scripts/make_splits.sh

# 5. Train CC400 and EZ branches and perform the final probability fusion.
bash scripts/run_loso.sh

# 6. After training, create the complete site-level result table.
python scripts/summarize_loso_results.py \
  --fold-metrics /path/to/ensemble_cc400_ez/fold_metrics.csv \
  --split-root configs/splits \
  --metadata-csv /path/to/ABIDE_1.csv \
  --output /path/to/results/site_level_results.csv

# 7. Compute the reference Integrated-Gradients attribution tables.
python scripts/compute_saliency.py \
  --run-root /path/to/run \
  --split-root configs/splits \
  --metadata-csv /path/to/ABIDE_1.csv \
  --hdf5 /path/to/abide1_nogsr_clip_level_dfc_seq20_startstride20_aal_cc400_ez_ho_dosenbach160_fullroi_cv10.hdf5 \
  --hdf5-manifest /path/to/hdf5_manifest_feature_shapes.csv \
  --output-dir /path/to/attribution
```

The exact reference split files are also included under `configs/splits/`. For the method and required input naming, see `docs/METHODS.md` and `docs/DATA_LAYOUT.md`.

## Important reproducibility notes

- The validation site is not randomly sampled inside each fold. It follows the fixed cyclic assignment documented in `docs/SPLIT_PROTOCOL.md`.
- The public runner defaults to the reference settings: 17 folds, 2,000 epochs, learning rate `5e-4`, batch size 16, margin `0.5`, group size 3, hidden dimension 256, embedding dimension 128, dropout `0.25`, and sex as the only covariate.
- The primary ROI extraction command uses `--no-gsr`, matching the `nogsr` feature source used by the reference model.
- Incomplete trailing windows are excluded by the dFC builder; only complete 20-frame clips are used.
- ABIDE-I data and atlas files are external inputs and are not included in this repository.
- The reference analysis consumed an already-built NeuroSTORM HDF5. `build_dfc_hdf5.py` is a transparent standalone reconstruction from ROI `.1D` files; use the original HDF5 when exact numerical reproduction is required.
- The attribution implementation uses Integrated Gradients from a zero baseline on the prototype-distance ASD score, not a classifier-logit saliency score.

## Reference result

The accompanying abstract reports the reference 17-fold analysis on 1,020 subjects. Exact result files are not included in the public source repository; after a run, `summary_by_combo.csv` records the CC400/EZ ensemble metrics and `fold_metrics.csv` records fold-level metrics and validation-selected weights.

## Citation

If you use this code, cite the MAGIC-dMAP manuscript and the original atlas papers listed in `docs/ATLAS_REFERENCES.md`.
