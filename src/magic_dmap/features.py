#!/usr/bin/env python3
"""Feature construction and model components used by MAGIC-dMAP.

The two feature branches are:
- branch 1: subject-specific functional summary statistics from all atlas dFCs
- branch 2: atlas-specific sampled dFC features as the ROI/parcellation branch

The input comes from the existing ABIDE1 clip-level dFC HDF5 produced for the
NeuroSTORM dFC experiments. No sMRI labels are used here.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


DEFAULT_ATLASES = ["aal", "cc400", "dosenbach160", "ez", "ho"]


@dataclass(frozen=True)
class Subject:
    bids_id: str
    h5_id: str
    label: int
    age: float
    sex: float


def parse_split(path: Path) -> dict[str, list[str]]:
    sections = {"train": [], "valid": [], "test": []}
    current = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        key = line.lower()
        if key.startswith("train"):
            current = "train"
            continue
        if key.startswith("valid") or key.startswith("val"):
            current = "valid"
            continue
        if key.startswith("test"):
            current = "test"
            continue
        if current is not None:
            sections[current].append(line.removeprefix("sub-"))
    return sections


def load_metadata(csv_path: Path, hdf5_manifest: Path) -> dict[str, Subject]:
    h5_by_bids: dict[str, str] = {}
    with hdf5_manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            bids = row["bids_subject"].removeprefix("sub-")
            h5_by_bids[bids] = row["file_id"]

    subjects: dict[str, Subject] = {}
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 10:
                continue
            diagnosis = row[1]
            label = 1 if diagnosis == "PATIENT" else 0
            try:
                age = float(row[7])
                sex = float(row[8])
                file_num = str(int(row[9]))
            except ValueError:
                continue
            h5_id = h5_by_bids.get(file_num)
            if h5_id is None:
                continue
            subjects[file_num] = Subject(file_num, h5_id, label, age, sex)
    return subjects


def parse_basic_quantiles(value: str) -> tuple[float, float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--basic-quantiles must be two comma-separated percentiles, e.g. 10,90")
    try:
        low, high = (float(parts[0]), float(parts[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--basic-quantiles values must be numeric") from exc
    if not (0.0 <= low <= 100.0 and 0.0 <= high <= 100.0 and low < high):
        raise argparse.ArgumentTypeError("--basic-quantiles must satisfy 0 <= low < high <= 100")
    return low, high


def _dist_stats_basic(arr: np.ndarray, quantiles: tuple[float, float] = (25.0, 75.0)) -> np.ndarray:
    low, high = np.percentile(arr, quantiles)
    return np.array(
        [
            float(arr.mean()),
            float(arr.std()),
            float(np.mean(np.abs(arr))),
            float(low),
            float(high),
        ],
        dtype=np.float32,
    )


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(values.mean())


def _dist_stats_enhanced(arr: np.ndarray) -> np.ndarray:
    p5, p10, p25, p50, p75, p90, p95 = np.percentile(arr, [5, 10, 25, 50, 75, 90, 95])
    positive = arr[arr > 0]
    negative = arr[arr < 0]
    return np.array(
        [
            float(arr.mean()),
            float(arr.std()),
            float(np.mean(np.abs(arr))),
            float(p5),
            float(p10),
            float(p25),
            float(p50),
            float(p75),
            float(p90),
            float(p95),
            float(p75 - p25),
            float(arr.max() - arr.min()),
            float(np.max(np.abs(arr))),
            float(np.mean(arr > 0)),
            float(np.mean(arr < 0)),
            _safe_mean(positive),
            _safe_mean(negative),
            _safe_mean(np.abs(positive)),
            _safe_mean(np.abs(negative)),
        ],
        dtype=np.float32,
    )


def _dist_stats(arr: np.ndarray, mode: str, basic_quantiles: tuple[float, float] = (25.0, 75.0)) -> np.ndarray:
    if mode == "basic":
        return _dist_stats_basic(arr, basic_quantiles)
    if mode == "enhanced":
        return _dist_stats_enhanced(arr)
    raise ValueError(f"Unknown summary_stats mode={mode}")


def _sample_vector(vec: np.ndarray, max_features: int) -> np.ndarray:
    vec = vec.astype(np.float32, copy=False)
    if vec.size <= max_features:
        return vec
    idx = np.linspace(0, vec.size - 1, max_features, dtype=np.int64)
    return vec[idx]


def _read_sampled_clips(dataset, max_features: int) -> np.ndarray:
    n_features = int(dataset.shape[1])
    if n_features <= max_features:
        return np.asarray(dataset, dtype=np.float32)
    idx = np.linspace(0, n_features - 1, max_features, dtype=np.int64)
    return np.asarray(dataset[:, idx], dtype=np.float32)


def build_features(
    h5_path: Path,
    subjects: dict[str, Subject],
    atlases: list[str],
    roi_atlas: str,
    max_dfc_features: int,
    summary_stats: str,
    basic_quantiles: tuple[float, float] = (25.0, 75.0),
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    features: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with h5py.File(h5_path, "r") as h5:
        patients = h5["patients"]
        for bids_id, subject in subjects.items():
            if subject.h5_id not in patients:
                continue
            subj = patients[subject.h5_id]
            summary_parts = []
            for atlas in atlases:
                if atlas not in subj:
                    continue
                clips = _read_sampled_clips(subj[atlas]["clip_dfc"], max_dfc_features)
                clip_stats = np.stack([_dist_stats(c, summary_stats, basic_quantiles) for c in clips], axis=0)
                summary_parts.append(clip_stats.mean(axis=0))
                summary_parts.append(clip_stats.std(axis=0))
            if roi_atlas not in subj:
                continue
            roi = _read_sampled_clips(subj[roi_atlas]["clip_dfc"], max_dfc_features)
            roi_vec = np.concatenate([roi.mean(axis=0), roi.std(axis=0)], axis=0)
            roi_vec = _sample_vector(roi_vec, max_dfc_features)
            features[bids_id] = (
                np.concatenate(summary_parts, axis=0).astype(np.float32),
                roi_vec.astype(np.float32),
            )
    return features


class FmriDataset(Dataset):
    def __init__(
        self,
        subjects: list[Subject],
        features: dict[str, tuple[np.ndarray, np.ndarray]],
        age_mean: float,
        age_std: float,
        covar_mode: str,
        summary_mean: np.ndarray,
        summary_std: np.ndarray,
        roi_mean: np.ndarray,
        roi_std: np.ndarray,
    ):
        self.subjects = [s for s in subjects if s.bids_id in features]
        self.features = features
        self.age_mean = age_mean
        self.age_std = max(age_std, 1e-6)
        self.covar_mode = covar_mode
        self.summary_mean = summary_mean
        self.summary_std = summary_std
        self.roi_mean = roi_mean
        self.roi_std = roi_std

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int):
        subject = self.subjects[idx]
        summary, roi = self.features[subject.bids_id]
        summary = (summary - self.summary_mean) / self.summary_std
        roi = (roi - self.roi_mean) / self.roi_std
        age = (subject.age - self.age_mean) / self.age_std
        sex = subject.sex - 1.0
        if self.covar_mode == "none":
            covars = []
        elif self.covar_mode == "sex":
            covars = [sex]
        elif self.covar_mode == "age_sex":
            covars = [age, sex]
        else:
            raise ValueError(f"Unknown covar_mode={self.covar_mode}")
        return (
            torch.from_numpy(summary.astype(np.float32)),
            torch.from_numpy(roi.astype(np.float32)),
            torch.tensor(subject.label, dtype=torch.long),
            torch.tensor(covars, dtype=torch.float32),
            subject.bids_id,
        )


class BranchEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)
