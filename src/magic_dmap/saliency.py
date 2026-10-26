#!/usr/bin/env python3
"""Integrated-Gradients attribution for the prototype-distance score.

The score is the mean distance to training NC embeddings minus the mean
distance to training ASD embeddings. Attribution is computed separately for
the coarse and fine inputs, averaged over the requested evaluation split, and
written as feature, edge, and ROI-node tables.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from .features import (
    DEFAULT_ATLASES,
    FmriDataset,
    build_features,
    load_metadata,
    parse_basic_quantiles,
    parse_split,
)
from .train import MAGICdMAPTriplet, collect_embeddings


def parse_folds(value: str) -> list[int]:
    result = []
    for part in value.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.extend(range(int(lo), int(hi) + 1))
        elif part.strip():
            result.append(int(part))
    return result


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def feature_sources(n_edges: int, max_features: int) -> list[tuple[str, int]]:
    first = np.arange(n_edges, dtype=np.int64)
    if n_edges > max_features:
        first = np.linspace(0, n_edges - 1, max_features, dtype=np.int64)
    positions = np.arange(first.size * 2, dtype=np.int64)
    positions = positions if positions.size <= max_features else np.linspace(0, positions.size - 1, max_features, dtype=np.int64)
    return [("clip_mean" if pos < first.size else "clip_std", int(first[pos if pos < first.size else pos - first.size])) for pos in positions]


def edge_pairs(n_edges: int) -> list[tuple[int, int]] | None:
    n = int((1 + math.sqrt(1 + 8 * n_edges)) / 2)
    if n * (n - 1) // 2 != n_edges:
        return None
    return [(i + 1, j + 1) for i in range(n) for j in range(i + 1, n)]


def score(model, summary, roi, covar, asd, nc):
    embedding = model.encode(summary, roi, covar)
    return torch.cdist(embedding, nc).mean(dim=1) - torch.cdist(embedding, asd).mean(dim=1)


def integrated_gradients(model, summary, roi, covar, asd, nc, steps: int):
    base_summary = torch.zeros_like(summary)
    base_roi = torch.zeros_like(roi)
    total_summary = torch.zeros_like(summary)
    total_roi = torch.zeros_like(roi)
    for step in range(1, steps + 1):
        alpha = step / steps
        current_summary = (base_summary + alpha * (summary - base_summary)).detach().requires_grad_(True)
        current_roi = (base_roi + alpha * (roi - base_roi)).detach().requires_grad_(True)
        value = score(model, current_summary, current_roi, covar, asd, nc).sum()
        grad_summary, grad_roi = torch.autograd.grad(value, (current_summary, current_roi))
        total_summary += grad_summary.detach()
        total_roi += grad_roi.detach()
    return (
        ((summary - base_summary) * total_summary / steps).detach().cpu().numpy()[0],
        ((roi - base_roi) * total_roi / steps).detach().cpu().numpy()[0],
    )


def normalize_datasets(features, subjects_by_id, split, covar_mode):
    train = [subjects_by_id[s] for s in split["train"] if s in subjects_by_id and subjects_by_id[s].bids_id in features]
    valid = [subjects_by_id[s] for s in split["valid"] if s in subjects_by_id and subjects_by_id[s].bids_id in features]
    test = [subjects_by_id[s] for s in split["test"] if s in subjects_by_id and subjects_by_id[s].bids_id in features]
    summary = np.stack([features[s.bids_id][0] for s in train])
    roi = np.stack([features[s.bids_id][1] for s in train])
    ages = np.asarray([s.age for s in train], dtype=np.float32)
    summary_mean, summary_std = summary.mean(0), np.maximum(summary.std(0), 1e-6)
    roi_mean, roi_std = roi.mean(0), np.maximum(roi.std(0), 1e-6)
    age_mean, age_std = float(ages.mean()), max(float(ages.std()), 1e-6)
    datasets = [
        FmriDataset(group, features, age_mean, age_std, covar_mode, summary_mean, summary_std, roi_mean, roi_std)
        for group in (train, valid, test)
    ]
    return train, datasets, summary.shape[1], roi.shape[1]


def load_model(path: Path, summary_dim: int, roi_dim: int, args, device):
    covar_dim = 1 if args.covar_mode == "sex" else 2
    model = MAGICdMAPTriplet(summary_dim, roi_dim, covar_dim, args.hidden_dim, args.embedding_dim, args.dropout).to(device)
    state = torch.load(path, map_location=device, weights_only=False).get("model")
    if state is None:
        raise ValueError(f"No model state in {path}")
    model.load_state_dict(state)
    model.eval()
    return model


def process_fold(args, atlas, fold, features, subjects_by_id, n_edges, device):
    split = parse_split(args.split_root / f"split_cv10_fold{fold}.txt")
    train_subjects, datasets, summary_dim, roi_dim = normalize_datasets(features, subjects_by_id, split, args.covar_mode)
    train_base, valid_base, test_base = datasets
    eval_base = {"train": train_base, "valid": valid_base, "test": test_base}[args.split]
    checkpoint = args.run_root / args.single_atlas_dir / atlas / f"fold{fold}" / args.checkpoint
    model = load_model(checkpoint, summary_dim, roi_dim, args, device)
    rows = collect_embeddings(model, DataLoader(train_base, batch_size=64, shuffle=False), device)
    asd = torch.stack([embedding for _, label, embedding in rows if label == 1]).to(device)
    nc = torch.stack([embedding for _, label, embedding in rows if label == 0]).to(device)
    summary_abs = np.zeros(summary_dim, dtype=np.float64)
    summary_signed = np.zeros(summary_dim, dtype=np.float64)
    roi_abs = np.zeros(roi_dim, dtype=np.float64)
    roi_signed = np.zeros(roi_dim, dtype=np.float64)
    for index in range(len(eval_base)):
        summary, roi, _, covar, _ = eval_base[index]
        summary = summary[None].to(device)
        roi = roi[None].to(device)
        covar = covar[None].to(device)
        attr_summary, attr_roi = integrated_gradients(model, summary, roi, covar, asd, nc, args.ig_steps)
        summary_abs += np.abs(attr_summary)
        summary_signed += attr_summary
        roi_abs += np.abs(attr_roi)
        roi_signed += attr_roi
    n_eval = len(eval_base)
    summary_abs /= n_eval
    summary_signed /= n_eval
    roi_abs /= n_eval
    roi_signed /= n_eval

    output = args.output_dir / atlas / f"fold{fold}"
    summary_rows = [{"feature_index": i, "importance_abs": float(summary_abs[i]), "importance_signed": float(summary_signed[i])} for i in range(summary_dim)]
    summary_rows.sort(key=lambda row: row["importance_abs"], reverse=True)
    write_csv(output / "summary_feature_importance.csv", summary_rows, ["feature_index", "importance_abs", "importance_signed"])

    pairs = edge_pairs(n_edges)
    sources = feature_sources(n_edges, args.max_dfc_features)
    feature_rows = []
    edge_accumulator = defaultdict(lambda: [0.0, 0.0, 0])
    for index, (stat, edge_index) in enumerate(sources):
        pair = pairs[edge_index] if pairs and edge_index < len(pairs) else ("", "")
        feature_rows.append({"feature_index": index, "source_stat": stat, "edge_index": edge_index, "roi_i": pair[0], "roi_j": pair[1], "importance_abs": float(roi_abs[index]), "importance_signed": float(roi_signed[index])})
        edge_accumulator[edge_index][0] += float(roi_abs[index])
        edge_accumulator[edge_index][1] += float(roi_signed[index])
        edge_accumulator[edge_index][2] += 1
    feature_rows.sort(key=lambda row: row["importance_abs"], reverse=True)
    write_csv(output / "roi_feature_importance.csv", feature_rows, ["feature_index", "source_stat", "edge_index", "roi_i", "roi_j", "importance_abs", "importance_signed"])

    edge_rows = []
    roi_scores = defaultdict(float)
    for edge_index, (absolute, signed, count) in edge_accumulator.items():
        pair = pairs[edge_index] if pairs and edge_index < len(pairs) else ("", "")
        edge_rows.append({"edge_index": edge_index, "roi_i": pair[0], "roi_j": pair[1], "importance_abs": absolute / count, "importance_signed": signed / count})
        if pair[0] != "":
            roi_scores[pair[0]] += absolute / count
            roi_scores[pair[1]] += absolute / count
    edge_rows.sort(key=lambda row: row["importance_abs"], reverse=True)
    write_csv(output / "edge_importance_top.csv", edge_rows[: args.top_edges], ["edge_index", "roi_i", "roi_j", "importance_abs", "importance_signed"])
    write_csv(output / "roi_importance.csv", [{"roi_id": roi_id, "importance_abs": value} for roi_id, value in sorted(roi_scores.items(), key=lambda item: item[1], reverse=True)], ["roi_id", "importance_abs"])


def aggregate(args, atlas):
    paths = sorted((args.output_dir / atlas).glob("fold*/roi_importance.csv"))
    totals = defaultdict(list)
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                totals[row["roi_id"]].append(float(row["importance_abs"]))
    rows = [{"roi_id": key, "n_folds": len(values), "importance_abs": float(np.mean(values))} for key, values in totals.items()]
    rows.sort(key=lambda row: row["importance_abs"], reverse=True)
    write_csv(args.output_dir / atlas / "aggregate_roi_importance.csv", rows, ["roi_id", "n_folds", "importance_abs"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--single-atlas-dir", default=".")
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--hdf5-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=parse_folds, default=parse_folds("1-17"))
    parser.add_argument("--roi-atlases", nargs="+", default=["cc400", "ez"])
    parser.add_argument("--atlases", nargs="+", default=DEFAULT_ATLASES)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--max-dfc-features", type=int, default=4096)
    parser.add_argument("--summary-stats", choices=["basic", "enhanced"], default="basic")
    parser.add_argument("--basic-quantiles", type=parse_basic_quantiles, default=(25.0, 75.0))
    parser.add_argument("--covar-mode", choices=["sex", "age_sex"], default="sex")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument("--top-edges", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    subjects = load_metadata(args.metadata_csv, args.hdf5_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for atlas in args.roi_atlases:
        features = build_features(args.hdf5, subjects, args.atlases, atlas, args.max_dfc_features, args.summary_stats, args.basic_quantiles)
        with h5py.File(args.hdf5, "r") as handle:
            n_edges = next(int(handle["patients"][subject.h5_id][atlas]["clip_dfc"].shape[1]) for subject in subjects.values() if subject.h5_id in handle["patients"] and atlas in handle["patients"][subject.h5_id])
        for fold in args.folds:
            checkpoint = args.run_root / args.single_atlas_dir / atlas / f"fold{fold}" / args.checkpoint
            if checkpoint.exists():
                process_fold(args, atlas, fold, features, subjects, n_edges, device)
        aggregate(args, atlas)


if __name__ == "__main__":
    main()
