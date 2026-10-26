#!/usr/bin/env python3
"""Validation-weighted post-hoc atlas probability ensemble for MAGIC-dMAP.

This follows the older ensemble project's idea:
  1. Use atlas-specific already-trained baseline models.
  2. For each atlas combination and CV fold, learn non-negative weights on the
     validation predictions only.
  3. Apply the fold-level weights to test probabilities.

No model is retrained here. The weights are learned from validation predictions
and then applied once to the held-out test predictions.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


ATLAS_DIRS = {
    "aal": "aal",
    "cc400": "cc400",
    "ez": "ez",
    "ho": "ho",
    "dosenbach160": "dosenbach160",
}


def read_predictions(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("bids_id") or row.get("subject_id") or row.get("file_id")
            if sid is None:
                raise ValueError(f"No subject id column in {path}")
            y = int(float(row["y_true"]))
            if "prob_asd" in row:
                prob = float(row["prob_asd"])
            elif "score" in row:
                score = float(row["score"])
                prob = 1.0 / (1.0 + np.exp(-score))
            else:
                raise ValueError(f"No prob_asd/score column in {path}")
            rows.append((sid, y, prob))
    rows.sort(key=lambda x: x[0])
    return [r[0] for r in rows], np.asarray([r[1] for r in rows], dtype=np.int64), np.asarray([r[2] for r in rows], dtype=np.float64)


def probs2(prob_asd: np.ndarray) -> np.ndarray:
    prob_asd = np.clip(prob_asd, 1e-7, 1.0 - 1e-7)
    return np.stack([1.0 - prob_asd, prob_asd], axis=1)


def learn_weights(valid_probs: np.ndarray, y_valid: np.ndarray, n_samples: int, seed: int):
    n_models = valid_probs.shape[0]
    candidates = [np.ones(n_models, dtype=np.float64) / n_models]
    candidates.extend(np.eye(n_models, dtype=np.float64))
    rng = np.random.RandomState(seed)
    candidates.extend(rng.dirichlet(np.ones(n_models), size=n_samples))
    candidates = np.asarray(candidates, dtype=np.float64)
    pred = candidates @ valid_probs
    pred = np.clip(pred, 1e-7, 1.0 - 1e-7)
    y = np.asarray(y_valid, dtype=np.int64)
    losses = -(y * np.log(pred) + (1 - y) * np.log(1 - pred)).mean(axis=1)
    best_idx = int(np.argmin(losses))
    return candidates[best_idx], float(losses[best_idx])


def metric_row(combo: tuple[str, ...], fold: int, ids: list[str], y_true: np.ndarray, prob: np.ndarray, weights: np.ndarray, valid_loss: float):
    pred = (prob >= 0.5).astype(np.int64)
    try:
        auc = roc_auc_score(y_true, prob)
    except ValueError:
        auc = float("nan")
    row = {
        "combo": "_".join(combo),
        "fold": fold,
        "n_test": len(y_true),
        "test_acc": float(accuracy_score(y_true, pred)),
        "test_balacc": float(balanced_accuracy_score(y_true, pred)),
        "test_auc": float(auc),
        "valid_log_loss": valid_loss,
    }
    for atlas, weight in zip(combo, weights):
        row[f"{atlas}_weight"] = float(weight)
    return row


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def atlas_fold_root(args, atlas: str, fold: int) -> Path:
    if atlas == "cc400" and args.cc400_root is not None:
        return args.cc400_root / f"fold{fold}"
    return args.single_atlas_root / ATLAS_DIRS[atlas] / f"fold{fold}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single-atlas-root", type=Path, required=True)
    ap.add_argument("--cc400-root", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--atlases", nargs="+", required=True)
    ap.add_argument("--combos", nargs="*", default=None, help="Optional explicit combos, e.g. aal_cc400_ez cc400_ez_ho.")
    ap.add_argument("--folds", nargs="+", type=int, default=list(range(1, 11)))
    ap.add_argument("--min-combo-size", type=int, default=2)
    ap.add_argument("--max-combo-size", type=int, default=5)
    ap.add_argument("--weight-samples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--improvement-label", type=str, default="MAGIC-dMAP")
    ap.add_argument(
        "--method-note",
        type=str,
        default="Weights are learned on validation predictions only and then applied to test predictions. No model retraining.",
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    pred_rows: list[dict] = []
    missing_rows: list[dict] = []
    combos = []
    if args.combos:
        for combo in args.combos:
            parts = tuple(x for x in combo.split("_") if x)
            unknown = [x for x in parts if x not in ATLAS_DIRS]
            if unknown:
                raise ValueError(f"Unknown atlas in combo={combo}: {unknown}")
            combos.append(parts)
    else:
        for size in range(args.min_combo_size, args.max_combo_size + 1):
            combos.extend(itertools.combinations(args.atlases, size))

    for combo in combos:
        for fold in args.folds:
            valid_probs, test_probs = [], []
            valid_ids_ref = test_ids_ref = None
            y_valid_ref = y_test_ref = None
            ok = True
            for atlas in combo:
                root = atlas_fold_root(args, atlas, fold)
                valid_path = root / "valid_predictions.csv"
                test_path = root / "test_predictions.csv"
                if not valid_path.exists() or not test_path.exists():
                    missing_rows.append({"combo": "_".join(combo), "fold": fold, "atlas": atlas, "missing": str(root)})
                    ok = False
                    break
                valid_ids, y_valid, vprob = read_predictions(valid_path)
                test_ids, y_test, tprob = read_predictions(test_path)
                if valid_ids_ref is None:
                    valid_ids_ref, y_valid_ref = valid_ids, y_valid
                    test_ids_ref, y_test_ref = test_ids, y_test
                elif valid_ids != valid_ids_ref or test_ids != test_ids_ref or not np.array_equal(y_valid, y_valid_ref) or not np.array_equal(y_test, y_test_ref):
                    missing_rows.append({"combo": "_".join(combo), "fold": fold, "atlas": atlas, "missing": "prediction_subject_or_label_mismatch"})
                    ok = False
                    break
                valid_probs.append(vprob)
                test_probs.append(tprob)
            if not ok:
                continue
            valid_arr = np.stack(valid_probs, axis=0)
            test_arr = np.stack(test_probs, axis=0)
            weights, valid_loss = learn_weights(valid_arr, y_valid_ref, args.weight_samples, args.seed + fold + 100 * len(combo))
            ens_prob = np.tensordot(weights, test_arr, axes=(0, 0))
            all_rows.append(metric_row(combo, fold, test_ids_ref, y_test_ref, ens_prob, weights, valid_loss))
            for sid, y, prob in zip(test_ids_ref, y_test_ref, ens_prob):
                pred_rows.append(
                    {
                        "combo": "_".join(combo),
                        "fold": fold,
                        "subject_id": sid,
                        "y_true": int(y),
                        "prob_asd": float(prob),
                        "y_pred": int(prob >= 0.5),
                    }
                )

    summary_rows = []
    by_combo: dict[str, list[dict]] = {}
    for row in all_rows:
        by_combo.setdefault(row["combo"], []).append(row)
    for combo, rows in sorted(by_combo.items()):
        summary_rows.append(
            {
                "combo": combo,
                "n_folds": len(rows),
                "mean_test_acc": float(np.mean([r["test_acc"] for r in rows])),
                "mean_test_balacc": float(np.mean([r["test_balacc"] for r in rows])),
                "mean_test_auc": float(np.nanmean([r["test_auc"] for r in rows])),
            }
        )

    metric_fields = ["combo", "fold", "n_test", "test_acc", "test_balacc", "test_auc", "valid_log_loss"] + [f"{a}_weight" for a in args.atlases]
    write_csv(args.output_dir / "fold_metrics.csv", all_rows, metric_fields)
    write_csv(args.output_dir / "summary_by_combo.csv", summary_rows, ["combo", "n_folds", "mean_test_acc", "mean_test_balacc", "mean_test_auc"])
    write_csv(args.output_dir / "test_predictions.csv", pred_rows, ["combo", "fold", "subject_id", "y_true", "prob_asd", "y_pred"])
    write_csv(args.output_dir / "missing_inputs.csv", missing_rows, ["combo", "fold", "atlas", "missing"])
    with (args.output_dir / "method.json").open("w") as f:
        json.dump(
            {
                "improvement": args.improvement_label,
                "method": "posthoc validation-weighted probability ensemble",
                "single_atlas_root": str(args.single_atlas_root),
                "cc400_root": str(args.cc400_root) if args.cc400_root is not None else None,
                "atlases": args.atlases,
                "folds_requested": args.folds,
                "weight_samples": args.weight_samples,
                "note": args.method_note,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"[Done] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
