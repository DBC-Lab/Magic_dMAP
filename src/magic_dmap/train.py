#!/usr/bin/env python3
"""MAGIC-dMAP subject-level triplet learning for ABIDE-I.

This keeps the baseline inputs unchanged:
- branch1: five-atlas global dFC summary
- branch2: CC400 ROI-level dFC
- covariate: sex by default

The model uses triplet learning: anchor and positive share the same label, while
negative has the opposite label. With --group-size > 3, the same anchor-positive
pair is contrasted against multiple opposite-class negatives.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

try:
    from .features import (
        DEFAULT_ATLASES,
        BranchEncoder,
        FmriDataset,
        build_features,
        load_metadata,
        parse_basic_quantiles,
        parse_split,
    )
except ImportError:
    from features import (
        DEFAULT_ATLASES,
        BranchEncoder,
        FmriDataset,
        build_features,
        load_metadata,
        parse_basic_quantiles,
        parse_split,
    )


class TripletDataset(Dataset):
    def __init__(self, base: FmriDataset, triplets_per_epoch: int, seed: int, group_size: int = 3):
        self.base = base
        self.triplets_per_epoch = triplets_per_epoch
        self.group_size = group_size
        self.rng = random.Random(seed)
        labels = [s.label for s in base.subjects]
        self.by_label = {
            0: [i for i, y in enumerate(labels) if y == 0],
            1: [i for i, y in enumerate(labels) if y == 1],
        }
        if self.group_size < 3:
            raise ValueError("group_size must be at least 3")

    def __len__(self) -> int:
        return self.triplets_per_epoch

    def __getitem__(self, idx: int):
        anchor_label = self.rng.choice([0, 1])
        negative_label = 1 - anchor_label
        anchor_idx, positive_idx = self.rng.sample(self.by_label[anchor_label], 2)
        negative_indices = [self.rng.choice(self.by_label[negative_label]) for _ in range(self.group_size - 2)]
        indices = [anchor_idx, positive_idx] + negative_indices
        summaries, rois, covars = [], [], []
        for sample_idx in indices:
            summary, roi, _, covar, _ = self.base[sample_idx]
            summaries.append(summary)
            rois.append(roi)
            covars.append(covar)
        return torch.stack(summaries, dim=0), torch.stack(rois, dim=0), torch.stack(covars, dim=0)


class MAGICdMAPTriplet(nn.Module):
    def __init__(
        self,
        summary_dim: int,
        roi_dim: int,
        covar_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.summary_encoder = BranchEncoder(summary_dim, hidden_dim, dropout=dropout)
        self.roi_encoder = BranchEncoder(roi_dim, hidden_dim, dropout=dropout)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim + covar_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def encode(self, summary, roi, covars):
        fs = self.summary_encoder(summary)
        fr = self.roi_encoder(roi)
        gate = self.gate(torch.cat([fs, fr], dim=1))
        fused = gate * fs + (1.0 - gate) * fr
        emb = self.projector(torch.cat([fused, covars], dim=1))
        return F.normalize(emb, p=2, dim=1)

    def forward(self, a_s, a_r, a_c, p_s, p_r, p_c, n_s, n_r, n_c):
        anchor = self.encode(a_s, a_r, a_c)
        positive = self.encode(p_s, p_r, p_c)
        negative = self.encode(n_s, n_r, n_c)
        return anchor, positive, negative


@torch.no_grad()
def collect_embeddings(model, loader, device):
    model.eval()
    rows = []
    for summary, roi, y, cov, bids_id in loader:
        summary = summary.to(device)
        roi = roi.to(device)
        cov = cov.to(device)
        emb = model.encode(summary, roi, cov).detach().cpu()
        for i in range(len(y)):
            rows.append((bids_id[i], int(y[i]), emb[i]))
    return rows


def evaluate(model, train_base, eval_base, device):
    train_loader = DataLoader(train_base, batch_size=32, shuffle=False, num_workers=0)
    eval_loader = DataLoader(eval_base, batch_size=32, shuffle=False, num_workers=0)
    train_emb = collect_embeddings(model, train_loader, device)
    eval_emb = collect_embeddings(model, eval_loader, device)
    asd = [(fid, emb) for fid, y, emb in train_emb if y == 1]
    ctrl = [(fid, emb) for fid, y, emb in train_emb if y == 0]
    file_ids, y_true, y_pred, scores = [], [], [], []
    for fid, y, emb in eval_emb:
        asd_d = float(np.mean([F.pairwise_distance(emb.view(1, -1), e.view(1, -1)).item() for _, e in asd]))
        ctrl_d = float(np.mean([F.pairwise_distance(emb.view(1, -1), e.view(1, -1)).item() for _, e in ctrl]))
        score = ctrl_d - asd_d
        file_ids.append(fid)
        y_true.append(y)
        y_pred.append(int(score >= 0))
        scores.append(score)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = float("nan")
    return acc, auc, file_ids, y_true, y_pred, scores


def score_to_prob(score: float) -> tuple[float, float]:
    prob_asd = 1.0 / (1.0 + math.exp(-float(score)))
    return 1.0 - prob_asd, prob_asd


def multi_negative_triplet_loss(embeddings: torch.Tensor, margin: float) -> torch.Tensor:
    anchor = embeddings[:, 0, :]
    positive = embeddings[:, 1, :]
    negatives = embeddings[:, 2:, :]
    positive_dist = F.pairwise_distance(anchor, positive).unsqueeze(1)
    negative_dist = torch.norm(anchor.unsqueeze(1) - negatives, p=2, dim=2)
    return F.relu(positive_dist - negative_dist + margin).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--split-root", type=Path, required=True)
    ap.add_argument("--metadata-csv", type=Path, required=True)
    ap.add_argument("--hdf5", type=Path, required=True)
    ap.add_argument("--hdf5-manifest", type=Path, required=True)
    ap.add_argument("--roi-atlas", choices=DEFAULT_ATLASES, default="cc400")
    ap.add_argument("--atlases", nargs="+", default=DEFAULT_ATLASES)
    ap.add_argument("--max-dfc-features", type=int, default=4096)
    ap.add_argument("--summary-stats", choices=["basic", "enhanced"], default="basic")
    ap.add_argument("--basic-quantiles", type=parse_basic_quantiles, default=(25.0, 75.0))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--patience", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--group-size", type=int, default=3, help="3=triplet; 4/5/6 add multiple opposite-class negatives.")
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--embedding-dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--covar-mode", choices=["sex", "age_sex"], default="sex")
    ap.add_argument(
        "--norm-reference",
        choices=["all", "control"],
        default="all",
        help="Feature normalization reference. 'control' uses only training controls as the normative space.",
    )
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--resume-from", type=Path, default=None)
    args = ap.parse_args()

    random.seed(args.seed + args.fold)
    np.random.seed(args.seed + args.fold)
    torch.manual_seed(args.seed + args.fold)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split = parse_split(args.split_root / f"split_cv10_fold{args.fold}.txt")
    subjects_by_bids = load_metadata(args.metadata_csv, args.hdf5_manifest)
    features = build_features(
        args.hdf5,
        subjects_by_bids,
        args.atlases,
        args.roi_atlas,
        args.max_dfc_features,
        args.summary_stats,
        args.basic_quantiles,
    )
    missing = sorted(set(split["train"] + split["valid"] + split["test"]) - set(features))
    if missing:
        (args.output_dir / "missing_subjects.txt").write_text("\n".join(missing) + "\n")

    train_subjects = [subjects_by_bids[s] for s in split["train"] if s in subjects_by_bids and s in features]
    valid_subjects = [subjects_by_bids[s] for s in split["valid"] if s in subjects_by_bids and s in features]
    test_subjects = [subjects_by_bids[s] for s in split["test"] if s in subjects_by_bids and s in features]
    if not train_subjects or not valid_subjects or not test_subjects:
        raise SystemExit("Empty train/valid/test after filtering.")

    train_summary = np.stack([features[s.bids_id][0] for s in train_subjects], axis=0)
    train_roi = np.stack([features[s.bids_id][1] for s in train_subjects], axis=0)
    if args.norm_reference == "control":
        reference_subjects = [s for s in train_subjects if s.label == 0]
        if not reference_subjects:
            raise SystemExit("No training controls available for --norm-reference control.")
    else:
        reference_subjects = train_subjects
    reference_summary = np.stack([features[s.bids_id][0] for s in reference_subjects], axis=0)
    reference_roi = np.stack([features[s.bids_id][1] for s in reference_subjects], axis=0)
    summary_mean = reference_summary.mean(axis=0)
    summary_std = np.maximum(reference_summary.std(axis=0), 1e-6)
    roi_mean = reference_roi.mean(axis=0)
    roi_std = np.maximum(reference_roi.std(axis=0), 1e-6)
    ages = np.array([s.age for s in train_subjects], dtype=np.float32)
    age_mean, age_std = float(ages.mean()), float(ages.std())

    train_base = FmriDataset(train_subjects, features, age_mean, age_std, args.covar_mode, summary_mean, summary_std, roi_mean, roi_std)
    valid_base = FmriDataset(valid_subjects, features, age_mean, age_std, args.covar_mode, summary_mean, summary_std, roi_mean, roi_std)
    test_base = FmriDataset(test_subjects, features, age_mean, age_std, args.covar_mode, summary_mean, summary_std, roi_mean, roi_std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    covar_dim = 1 if args.covar_mode == "sex" else 2
    model = MAGICdMAPTriplet(
        train_summary.shape[1],
        train_roi.shape[1],
        covar_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_auc = -math.inf
    best_epoch = -1
    stale = 0
    start_epoch = 0
    metrics_rows = []
    metrics_path = args.output_dir / "epoch_metrics.csv"
    if metrics_path.exists():
        with metrics_path.open(newline="") as f:
            for r in csv.DictReader(f):
                metrics_rows.append(
                    {
                        "epoch": int(r["epoch"]),
                        "train_loss": float(r["train_loss"]),
                        "valid_acc": float(r["valid_acc"]),
                        "valid_auc": float(r["valid_auc"]),
                    }
                )
    if (args.output_dir / "best.pt").exists():
        ckpt = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        _, prev_best_auc, *_ = evaluate(model, train_base, valid_base, device)
        if not math.isnan(prev_best_auc):
            best_auc = prev_best_auc
            best_epoch = int(ckpt.get("epoch", -1))
    if args.resume_from is not None:
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt.get("optimizer", optimizer.state_dict()))
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        print(f"[Resume] loaded {args.resume_from} at epoch={ckpt.get('epoch', 'NA')}; start_epoch={start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        triplet_ds = TripletDataset(
            train_base,
            triplets_per_epoch=max(len(train_base), 400),
            seed=args.seed + args.fold * 1000 + epoch,
            group_size=args.group_size,
        )
        loader = DataLoader(triplet_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        losses = []
        for summaries, rois, covars in loader:
            summaries = summaries.to(device)
            rois = rois.to(device)
            covars = covars.to(device)
            optimizer.zero_grad(set_to_none=True)
            batch_size, group_size = summaries.shape[:2]
            embeddings = model.encode(
                summaries.reshape(batch_size * group_size, summaries.shape[-1]),
                rois.reshape(batch_size * group_size, rois.shape[-1]),
                covars.reshape(batch_size * group_size, covars.shape[-1]),
            ).reshape(batch_size, group_size, -1)
            loss = multi_negative_triplet_loss(embeddings, args.margin)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        valid_acc, valid_auc, *_ = evaluate(model, train_base, valid_base, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_acc": valid_acc, "valid_auc": valid_auc}
        metrics_rows.append(row)
        with metrics_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "valid_acc", "valid_auc"])
            writer.writeheader()
            writer.writerows(metrics_rows)
        print(f"epoch={epoch} train_loss={row['train_loss']:.6f} valid_acc={valid_acc:.6f} valid_auc={valid_auc:.6f}", flush=True)
        monitor = valid_auc if not math.isnan(valid_auc) else valid_acc
        if monitor > best_auc:
            best_auc = monitor
            best_epoch = epoch
            stale = 0
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, args.output_dir / "best.pt")
        else:
            stale += 1
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, args.output_dir / "last.pt")
        if stale >= args.patience:
            break

    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "valid_acc", "valid_auc"])
        writer.writeheader()
        writer.writerows(metrics_rows)

    ckpt = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    valid_acc, valid_auc, valid_ids, valid_y, valid_pred, valid_scores = evaluate(model, train_base, valid_base, device)
    test_acc, test_auc, test_ids, y_true, y_pred, scores = evaluate(model, train_base, test_base, device)
    with (args.output_dir / "final_test_metrics.txt").open("w") as f:
        f.write("method=MAGIC-dMAP_triplet_subject_embedding\n")
        f.write(f"method=multi_object_subject_level_learning_group_size_{args.group_size}\n")
        f.write(f"best_epoch={best_epoch}\n")
        f.write("evaluated_checkpoint=best.pt\n")
        f.write(f"evaluated_epoch={int(ckpt.get('epoch', -1))}\n")
        f.write(f"valid_acc={valid_acc:.6f}\n")
        f.write(f"valid_AUROC={valid_auc:.6f}\n")
        f.write(f"test_acc={test_acc:.6f}\n")
        f.write(f"test_AUROC={test_auc:.6f}\n")
        f.write(f"n_train={len(train_subjects)}\n")
        f.write(f"n_valid={len(valid_subjects)}\n")
        f.write(f"n_test={len(test_subjects)}\n")
        f.write(f"norm_reference={args.norm_reference}\n")
        f.write(f"n_norm_reference={len(reference_subjects)}\n")
        f.write(f"margin={args.margin}\n")
        f.write(f"group_size={args.group_size}\n")
        f.write(f"hidden_dim={args.hidden_dim}\n")
        f.write(f"embedding_dim={args.embedding_dim}\n")
        f.write(f"dropout={args.dropout}\n")
        f.write(f"input=fMRI_dFC_functional_summary_all_atlases+{args.roi_atlas}_sampled_dFC+{args.covar_mode}\n")
        f.write(f"hdf5={args.hdf5}\n")
        f.write(f"roi_atlas={args.roi_atlas}\n")
        f.write(f"atlases={','.join(args.atlases)}\n")
        f.write(f"max_dfc_features={args.max_dfc_features}\n")
        f.write(f"summary_stats={args.summary_stats}\n")
        f.write(f"basic_quantiles={args.basic_quantiles[0]:g},{args.basic_quantiles[1]:g}\n")
    for name, ids, yy, pred, sc in [
        ("valid_predictions.csv", valid_ids, valid_y, valid_pred, valid_scores),
        ("test_predictions.csv", test_ids, y_true, y_pred, scores),
    ]:
        with (args.output_dir / name).open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["bids_id", "y_true", "y_pred", "score", "prob_control", "prob_asd"])
            for row in zip(ids, yy, pred, sc):
                pc, pa = score_to_prob(row[3])
                writer.writerow([*row, pc, pa])
    print(f"FINAL valid_acc={valid_acc:.6f} valid_AUROC={valid_auc:.6f} test_acc={test_acc:.6f} test_AUROC={test_auc:.6f} best_epoch={best_epoch}", flush=True)


if __name__ == "__main__":
    main()
