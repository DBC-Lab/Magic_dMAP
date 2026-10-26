# Attribution protocol

The reference post-hoc attribution is not part of model fitting. For each
atlas-specific model and fold:

1. Rebuild the same normalized coarse and fine inputs used for training.
2. Encode all training subjects and compute two prototypes as the sets of ASD
   and NC training embeddings.
3. Define the ASD score as mean distance to the NC embeddings minus mean
   distance to the ASD embeddings.
4. Compute Integrated Gradients from a zero baseline with respect to both
   branch inputs, using the same score and the requested validation or test
   split.
5. Average absolute and signed attributions over subjects, then map sampled
   fine features back to their source statistic (`clip_mean` or `clip_std`)
   and edge index.

`src/magic_dmap/saliency.py` writes per-fold `summary_feature_importance.csv`,
`roi_feature_importance.csv`, `edge_importance_top.csv`, and
`roi_importance.csv`, plus an aggregate ROI table. The script does not create a
voxel NIfTI automatically because voxel rendering requires the exact atlas
binary and its affine. A voxel map can be created by assigning the aggregate
`roi_importance.csv` values to the corresponding supplied atlas labels.

This definition matches the attribution used for the reference analysis:
the score is prototype-distance based and the attribution is Integrated
Gradients, rather than a gradient of a separately trained softmax classifier.
