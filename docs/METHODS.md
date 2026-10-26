# Methods

## Data and preprocessing

The reference analysis uses 1,020 ABIDE-I participants from 17 sites, including 500 ASD and 520 neurotypical controls. The preprocessing script expects curated BIDS inputs containing the original T1w image, a provided brain-extracted T1w image, a provided segmentation label image, and the raw resting-state BOLD image.

The implemented sequence is:

1. Extract the white-matter registration mask from segmentation label `3` using `fslmaths -thr 3 -uthr 3 -bin`.
2. Apply FSL MCFLIRT to the raw 4D BOLD image, saving motion matrices and motion summaries.
3. Compute the motion-corrected BOLD reference and skull-strip this reference with HD-BET (CPU fast mode, no test-time augmentation), with FSL BET as fallback.
4. Register the stripped BOLD reference to the brain-extracted T1w image using FSL FLIRT: a 6-DOF normalized mutual-information initialization followed by 6-DOF BBR using the white-matter mask and `bbr.sch`.
5. Register the brain-extracted T1w image to the FSL MNI152 1-mm brain template with `antsRegistrationSyN.sh -t s`.
6. Apply motion, BBR, and T1-to-MNI transforms to every original BOLD volume. The main output is not skull-stripped; a separate transformed BOLD brain mask is produced for ROI extraction.

The supplied T1 brain and segmentation files are treated as inputs. This script does not run N4 bias correction or create a new anatomical segmentation. Fieldmap-less distortion correction is not a separate step in this pipeline; the ANTs SyN registration is the T1-to-MNI anatomical registration used by the workflow.

## ROI extraction

The final MNI-space 4D BOLD image and its transformed 3D brain mask are used with the five atlas NIfTIs listed in `configs/atlas_manifest.tsv`. Atlas labels are resampled to the BOLD grid with nearest-neighbor interpolation when necessary. The mean signal within each ROI is written as a time-by-ROI `.1D` file.

The reference feature source is `nogsr`: the reproducibility command therefore passes `--no-gsr`. No extra temporal filtering, scrubbing, or spatial smoothing is added by the ROI extraction script.

## Dynamic functional connectivity

For each ROI time series, the builder standardizes each ROI over time, then uses complete windows of 20 frames with a start stride of 20 frames. Within each window it computes the Pearson correlation matrix, applies Fisher's z transform, and stores the lower-triangle edge vector. A trailing segment shorter than 20 frames is excluded.

The HDF5 contains five atlas groups: AAL, CC400, Dosenbach160, EZ, and Harvard–Oxford. The feature builder stores one `clip_dfc` array per atlas with shape `n_windows x n_edges` and the corresponding start frames.

## Coarse and fine features

For each window and each atlas, the coarse branch computes the mean, standard deviation, mean absolute value, 25th percentile, and 75th percentile over the sampled edge distribution. It then concatenates the across-window mean and standard deviation of those five statistics for all five atlases.

For the selected fine branch (CC400 or EZ), the model computes the across-window mean and standard deviation for each edge. The implementation uses deterministic `numpy.linspace` sampling with a maximum of 4,096 dFC features. Vectors with at most 4,096 elements are retained; larger vectors are sampled to exactly 4,096. The fine mean and standard-deviation vectors are concatenated and then capped at 4,096 before the branch encoder.

## Embedding model

Each atlas-specific model has two independent MLP encoders:

- coarse branch: five-atlas distribution summaries;
- fine branch: selected-atlas edge-level dynamic features.

Each encoder maps its input to 256 dimensions using `Linear -> LayerNorm -> GELU -> Dropout(0.25) -> Linear -> GELU`. A sigmoid gate receives the concatenated 256-dimensional branch outputs and adaptively mixes them. Sex is concatenated after fusion. The projection MLP maps the fused vector to a 128-dimensional L2-normalized embedding using `Linear(257,256) -> PReLU -> Dropout(0.25) -> Linear(256,128)`.

Training uses group-size-3 triplets, margin `0.5`, AdamW, learning rate `5e-4`, weight decay `1e-4`, batch size `16`, and up to 2,000 epochs. The checkpoint with the highest validation AUROC is retained.

For an evaluation subject, the model computes the mean Euclidean distance to training ASD embeddings and to training NC embeddings. The difference between these distances determines the predicted class and ASD probability. No test labels are used for training, model selection, or ensemble-weight selection.

## Decision-level fusion

The CC400 and EZ models produce separate ASD probabilities. For each fold, nonnegative weights summing to one are selected by minimizing validation log loss. The selected weights are then applied to the test probabilities. The final reported combination is `cc400_ez`.
