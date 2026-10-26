# Split protocol

The reference evaluation is a 17-fold leave-one-site-out protocol. In every fold, one complete site is the test set, one complete site is the validation set, and all remaining sites are the training set. Thus, the training and validation sets are not an 8:2 random split of the non-test subjects.

The validation assignment is deterministic and was fixed before model fitting. The validation sites cycle through:

```text
NYU, UM, UCLA, LEUVEN, USM, PITT, YALE, MAX_MUN
```

If the cyclic choice equals the test site, the next cycle entry is used. This is a prespecified site-level holdout rule, not a random or performance-driven selection. The exact subject lists are included in `configs/splits/`, and their counts are recorded in `configs/site_heldout_validation_manifest.tsv`.

The split text files retain the historical `split_cv10_foldN.txt` filename convention for compatibility with the source training code. The evaluation itself has 17 folds, not 10.
