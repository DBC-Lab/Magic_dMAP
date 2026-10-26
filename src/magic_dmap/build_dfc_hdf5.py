#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the clip-level dynamic functional-connectivity HDF5 used by MAGIC-dMAP.

This script is standalone by design. It writes a new HDF5 and filtered splits
under an isolated experiment directory, and it does not change the active
NeuroSTORM data loader or training code.

The model consumes ROI-level dFC only; NeuroSTORM frame tensors are therefore
optional and are not required by this HDF5 builder.
"""

from __future__ import print_function

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


DERIVATIVES = ("aal", "cc400", "ez", "ho", "dosenbach160")
SECTION_NAMES = ("train_subjects", "val_subjects", "test_subjects")
FEATURE_BLOCKS = ("clip_dfc",)


def parse_split(path):
    sections = {name: [] for name in SECTION_NAMES}
    current = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line in sections:
            current = line
            continue
        if current is None:
            raise ValueError("Subject before section header in {}".format(path))
        sections[current].append(line)
    return sections


def write_split(path, sections):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for section in SECTION_NAMES:
            f.write(section + "\n")
            for subject in sections[section]:
                f.write(subject + "\n")
            if section != SECTION_NAMES[-1]:
                f.write("\n")


def load_pheno(pheno_path):
    pheno = pd.read_csv(pheno_path)
    if {"subject_id", "dx", "sex", "age", "site_id"}.issubset(set(pheno.columns)):
        pheno = pheno.copy()
        pheno["SUB_ID_STR"] = pheno["subject_id"].astype(str).str.replace("^sub-", "", regex=True)
        pheno["bids_subject"] = pheno["subject_id"].astype(str)
        pheno["FILE_ID"] = pheno["bids_subject"]
        pheno["DX_GROUP_ZERO_BASED"] = pheno["dx"].astype(int)
        pheno["SEX"] = pheno["sex"].astype(int)
        pheno["AGE_AT_SCAN"] = pheno["age"].astype(float)
        pheno["SITE_CLEAN"] = pheno["site_id"].astype(str).str.replace(r"_[0-9]$", "", regex=True)
        return pheno
    pheno = pheno[pheno["FILE_ID"] != "no_filename"].copy()
    pheno["SUB_ID_STR"] = pheno["SUB_ID"].astype(int).astype(str)
    pheno["bids_subject"] = pheno["SUB_ID_STR"].apply(lambda s: "sub-{}".format(s))
    pheno["DX_GROUP_ZERO_BASED"] = pheno["DX_GROUP"].astype(int) - 1
    pheno["SITE_CLEAN"] = pheno["SITE_ID"].astype(str).str.replace(r"_[0-9]$", "", regex=True)
    return pheno


def load_subject_allowlist(path):
    if path is None:
        return None
    subjects = set()
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("sub-"):
            line = "sub-{}".format(line)
        subjects.add(line)
    return subjects


def roi_identifier(row, source):
    if source == "file_id":
        return row["FILE_ID"]
    if source == "bids_subject":
        if "bids_subject" not in row.index:
            return row.name
        return row["bids_subject"]
    raise ValueError("Unknown roi id source: {}".format(source))


def roi_path(roi_root, derivative, identifier):
    return roi_root / "rois_{}".format(derivative) / "{}_rois_{}.1D".format(identifier, derivative)


def frame_count(image_root, bids_subject):
    img_dir = image_root / "img" / bids_subject
    if not img_dir.exists():
        return 0
    return sum(1 for p in img_dir.iterdir() if p.name.startswith("frame_") and p.suffix == ".pt")


def available_subjects(pheno, roi_root, image_root, roi_id_source, derivatives, require_neurostorm_frames):
    rows = []
    available = set()
    for _, row in pheno.iterrows():
        roi_id = roi_identifier(row, roi_id_source)
        missing = [d for d in derivatives if not roi_path(roi_root, d, roi_id).exists()]
        n_frames = frame_count(image_root, row["bids_subject"])
        has_images = n_frames > 0
        rows.append({
            "bids_subject": row["bids_subject"],
            "file_id": row["FILE_ID"],
            "missing_derivatives": ";".join(missing),
            "has_all_derivatives": len(missing) == 0,
            "n_neurostorm_frames": n_frames,
            "has_neurostorm_frames": has_images,
        })
        if not missing and (has_images or not require_neurostorm_frames):
            available.add(row["bids_subject"])
    return available, rows


def zscore_time_by_roi(timeseries_time_by_roi):
    data = np.asarray(timeseries_time_by_roi, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (data - mean) / std


def lower_triangle(corr):
    idx = np.tril_indices(corr.shape[0], k=-1)
    return corr[idx].astype(np.float32, copy=False)


def corr_edges(z_time_by_roi):
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.nan_to_num(np.corrcoef(z_time_by_roi, rowvar=False))
    return lower_triangle(corr)


def neurostorm_clip_starts(n_timepoints, sequence_length, stride_within_seq, stride_between_seq):
    sample_duration = sequence_length * stride_within_seq
    if n_timepoints < sample_duration:
        return []
    start_stride = max(round(stride_between_seq), 1)
    session_duration = n_timepoints - sample_duration + 1
    return list(range(0, session_duration, start_stride))


def compute_clip_level_fc(
    timeseries_time_by_roi,
    sequence_length,
    stride_within_seq,
    stride_between_seq,
    fisher_z,
    subject_fc_normalization="none",
):
    z = zscore_time_by_roi(timeseries_time_by_roi)
    n_timepoints = z.shape[0]
    starts = neurostorm_clip_starts(n_timepoints, sequence_length, stride_within_seq, stride_between_seq)
    clips = []
    for start in starts:
        idx = range(start, start + sequence_length * stride_within_seq, stride_within_seq)
        clips.append(corr_edges(z[list(idx)]))
    if not clips:
        return np.zeros((0, 0), dtype=np.float32), np.asarray([], dtype=np.int64)
    clip_edges = np.vstack(clips).astype(np.float32, copy=False)
    if fisher_z:
        eps = np.float32(1e-7)
        clip_edges = np.arctanh(np.clip(clip_edges, -1.0 + eps, 1.0 - eps))
    clip_edges = np.nan_to_num(clip_edges).astype(np.float32, copy=False)
    if subject_fc_normalization == "mean_center":
        clip_edges = clip_edges - clip_edges.mean(dtype=np.float64).astype(np.float32)
    elif subject_fc_normalization == "zscore":
        mean = clip_edges.mean(dtype=np.float64).astype(np.float32)
        std = clip_edges.std(dtype=np.float64).astype(np.float32)
        if std == 0:
            std = np.float32(1.0)
        clip_edges = (clip_edges - mean) / std
    elif subject_fc_normalization != "none":
        raise ValueError("Unknown subject_fc_normalization: {}".format(subject_fc_normalization))
    return clip_edges.astype(np.float32, copy=False), np.asarray(starts, dtype=np.int64)


def build_patients(h5, pheno, roi_root, image_root, subject_set, args):
    patients = h5.require_group("patients")
    pheno_by_subject = pheno.set_index("bids_subject")
    written = 0
    feature_rows = []
    for bids_subject in sorted(subject_set):
        row = pheno_by_subject.loc[bids_subject]
        file_id = row["FILE_ID"]
        roi_id = roi_identifier(row, args.roi_id_source)
        patient = patients.require_group(file_id)
        patient.attrs["id"] = file_id
        patient.attrs["bids_subject"] = bids_subject
        patient.attrs["roi_id"] = roi_id
        patient.attrs["y"] = int(row["DX_GROUP_ZERO_BASED"])
        patient.attrs["site"] = row["SITE_CLEAN"]
        patient.attrs["sex"] = int(row["SEX"])
        patient.attrs["age"] = float(row["AGE_AT_SCAN"])
        patient.attrs["n_neurostorm_frames"] = int(frame_count(image_root, bids_subject))
        for derivative in args.derivatives:
            if derivative in patient:
                del patient[derivative]
            data = np.loadtxt(str(roi_path(roi_root, derivative, roi_id)))
            feature, starts = compute_clip_level_fc(
                data,
                args.sequence_length,
                args.stride_within_seq,
                args.stride_between_seq,
                args.fisher_z,
                args.subject_fc_normalization,
            )
            group = patient.create_group(derivative)
            ds = group.create_dataset("clip_dfc", data=feature, compression="gzip", compression_opts=4)
            ds.attrs["feature_blocks"] = ",".join(FEATURE_BLOCKS)
            ds.attrs["target_level"] = "clip"
            ds.attrs["sequence_length"] = int(args.sequence_length)
            ds.attrs["stride_within_seq"] = int(args.stride_within_seq)
            ds.attrs["stride_between_seq"] = int(args.stride_between_seq)
            ds.attrs["sample_duration"] = int(args.sequence_length * args.stride_within_seq)
            ds.attrs["start_stride"] = int(max(round(args.stride_between_seq), 1))
            ds.attrs["start_stride_semantics"] = "dense_frame_stride"
            ds.attrs["fisher_z"] = bool(args.fisher_z)
            ds.attrs["subject_fc_normalization"] = args.subject_fc_normalization
            group.create_dataset("start_frames", data=starts, compression="gzip", compression_opts=4)
            feature_rows.append({
                "file_id": file_id,
                "bids_subject": bids_subject,
                "roi_id": roi_id,
                "derivative": derivative,
                "n_clips": int(len(starts)),
                "feature_length": int(feature.shape[1]) if feature.ndim == 2 and feature.shape[0] else 0,
                "first_start": int(starts[0]) if len(starts) else -1,
                "last_start": int(starts[-1]) if len(starts) else -1,
            })
        written += 1
    return written, feature_rows


def split_filename(split_prefix, fold_idx):
    return "{}_fold{}.txt".format(split_prefix, fold_idx)


def write_experiments(h5, pheno, split_root, filtered_split_root, subject_set, fold_count, split_prefix, derivatives):
    pheno_by_subject = pheno.set_index("bids_subject")
    exps = h5.require_group("experiments")
    exps.attrs["split_source"] = str(split_root)
    exps.attrs["filtered_split_root"] = str(filtered_split_root)
    exps.attrs["filter_note"] = "Original ABIDE1 CV{} splits with subjects lacking required ROI derivatives removed; voxel-frame filtering is optional.".format(fold_count)
    manifest_rows = []
    missing_rows = []

    for derivative in derivatives:
        exp = exps.require_group("{}_clip_level".format(derivative))
        exp.attrs["derivative"] = derivative
        exp.attrs["fold_count"] = fold_count
        for fold_idx in range(1, fold_count + 1):
            split_file = split_root / split_filename(split_prefix, fold_idx)
            sections = parse_split(split_file)
            filtered_sections = {}
            mapped_sections = {}
            for section in SECTION_NAMES:
                kept = [s for s in sections[section] if s in subject_set]
                dropped = [s for s in sections[section] if s not in subject_set]
                filtered_sections[section] = kept
                mapped_sections[section] = [pheno_by_subject.loc[s, "FILE_ID"] for s in kept]
                for subject in dropped:
                    missing_rows.append({
                        "fold": fold_idx,
                        "split_section": section,
                        "bids_subject": subject,
                        "reason": "missing_required_roi_or_pheno_or_optional_voxel",
                    })

            if derivative == derivatives[0]:
                write_split(filtered_split_root / split_filename(split_prefix, fold_idx), filtered_sections)

            fold = exp.require_group(str(fold_idx))
            for h5_name, section in (("train", "train_subjects"), ("valid", "val_subjects"), ("test", "test_subjects")):
                if h5_name in fold:
                    del fold[h5_name]
                fold.create_dataset(h5_name, data=np.array(mapped_sections[section], dtype="S"))

            manifest_rows.append({
                "derivative": derivative,
                "experiment": "{}_clip_level".format(derivative),
                "fold": fold_idx,
                "source_split_file": str(split_file),
                "filtered_split_file": str(filtered_split_root / split_filename(split_prefix, fold_idx)),
                "train_n": len(mapped_sections["train_subjects"]),
                "valid_n": len(mapped_sections["val_subjects"]),
                "test_n": len(mapped_sections["test_subjects"]),
                "total_n": sum(len(v) for v in mapped_sections.values()),
            })
    return manifest_rows, missing_rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi-root", type=Path, required=True)
    parser.add_argument("--neurostorm-image-root", type=Path, required=True)
    parser.add_argument("--output-hdf5", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--filtered-split-root", type=Path, required=True)
    parser.add_argument("--pheno", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--missing", type=Path, required=True)
    parser.add_argument("--subject-allowlist", type=Path, default=None)
    parser.add_argument("--roi-id-source", choices=["file_id", "bids_subject"], default="file_id")
    parser.add_argument("--derivatives", nargs="+", default=list(DERIVATIVES), choices=list(DERIVATIVES))
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--stride-within-seq", type=int, default=1)
    parser.add_argument("--stride-between-seq", type=int, default=20)
    parser.add_argument("--fold-count", type=int, default=10)
    parser.add_argument("--split-prefix", default="split_cv10")
    parser.add_argument("--fisher-z", action="store_true")
    parser.add_argument("--subject-fc-normalization", choices=["none", "mean_center", "zscore"], default="none")
    parser.add_argument("--require-neurostorm-frames", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.sequence_length < 2:
        raise ValueError("--sequence-length must be at least 2")
    if args.stride_within_seq < 1:
        raise ValueError("--stride-within-seq must be at least 1")
    if args.stride_between_seq <= 0:
        raise ValueError("--stride-between-seq must be positive")

    pheno = load_pheno(args.pheno)
    allowlist = load_subject_allowlist(args.subject_allowlist)
    available, availability_rows = available_subjects(
        pheno,
        args.roi_root,
        args.neurostorm_image_root,
        args.roi_id_source,
        tuple(args.derivatives),
        args.require_neurostorm_frames,
    )
    split_subjects = set()
    for fold_idx in range(1, args.fold_count + 1):
        sections = parse_split(args.split_root / split_filename(args.split_prefix, fold_idx))
        for section in SECTION_NAMES:
            split_subjects.update(sections[section])
    subject_set = available & split_subjects & set(pheno["bids_subject"])
    if allowlist is not None:
        subject_set = subject_set & allowlist

    args.output_hdf5.parent.mkdir(parents=True, exist_ok=True)
    if args.force and args.output_hdf5.exists():
        args.output_hdf5.unlink()

    with h5py.File(str(args.output_hdf5), "a") as h5:
        h5.attrs["roi_root"] = str(args.roi_root)
        h5.attrs["roi_id_source"] = args.roi_id_source
        h5.attrs["neurostorm_image_root"] = str(args.neurostorm_image_root)
        h5.attrs["subject_policy"] = (
            "intersection_of_pheno_cv{}_splits_required_roi_derivatives"
            + ("_and_voxel_frames" if args.require_neurostorm_frames else "_roi_only_no_voxel_frame_requirement")
        ).format(args.fold_count)
        h5.attrs["feature_family"] = "clip_level_dynamic_fc"
        h5.attrs["feature_blocks"] = ",".join(FEATURE_BLOCKS)
        h5.attrs["dfc_aux_target_level"] = "clip"
        h5.attrs["sequence_length"] = int(args.sequence_length)
        h5.attrs["stride_within_seq"] = int(args.stride_within_seq)
        h5.attrs["stride_between_seq"] = int(args.stride_between_seq)
        h5.attrs["sample_duration"] = int(args.sequence_length * args.stride_within_seq)
        h5.attrs["start_stride"] = int(max(round(args.stride_between_seq), 1))
        h5.attrs["start_stride_semantics"] = "dense_frame_stride"
        h5.attrs["dynamic_fc_fisher_z"] = bool(args.fisher_z)
        h5.attrs["subject_fc_normalization"] = args.subject_fc_normalization
        h5.attrs["derivatives"] = ",".join(args.derivatives)
        h5.attrs["fold_count"] = int(args.fold_count)
        h5.attrs["split_prefix"] = args.split_prefix
        h5.attrs["require_neurostorm_frames"] = bool(args.require_neurostorm_frames)
        if args.subject_allowlist is not None:
            h5.attrs["subject_allowlist"] = str(args.subject_allowlist)
        written, feature_rows = build_patients(h5, pheno, args.roi_root, args.neurostorm_image_root, subject_set, args)
        manifest_rows, missing_rows = write_experiments(
            h5,
            pheno,
            args.split_root,
            args.filtered_split_root,
            subject_set,
            args.fold_count,
            args.split_prefix,
            tuple(args.derivatives),
        )

    write_csv(
        args.manifest,
        manifest_rows,
        ["derivative", "experiment", "fold", "source_split_file", "filtered_split_file", "train_n", "valid_n", "test_n", "total_n"],
    )
    write_csv(args.missing, missing_rows, ["fold", "split_section", "bids_subject", "reason"])
    write_csv(
        args.manifest.with_name(args.manifest.stem + "_availability.csv"),
        availability_rows,
        ["bids_subject", "file_id", "missing_derivatives", "has_all_derivatives", "n_neurostorm_frames", "has_neurostorm_frames"],
    )
    write_csv(
        args.manifest.with_name(args.manifest.stem + "_feature_shapes.csv"),
        feature_rows,
        ["file_id", "bids_subject", "roi_id", "derivative", "n_clips", "feature_length", "first_start", "last_start"],
    )

    print("[Info] Output HDF5:", args.output_hdf5)
    print("[Info] Filtered NeuroSTORM splits:", args.filtered_split_root)
    print("[Info] Target level: clip")
    print("[Info] sequence_length:", args.sequence_length)
    print("[Info] stride_within_seq:", args.stride_within_seq)
    print("[Info] stride_between_seq:", args.stride_between_seq)
    print("[Info] sample_duration:", args.sequence_length * args.stride_within_seq)
    print("[Info] start_stride:", max(round(args.stride_between_seq), 1))
    print("[Info] Fisher z:", args.fisher_z)
    print("[Info] Subject FC normalization:", args.subject_fc_normalization)
    print("[Info] Subjects written:", written)


if __name__ == "__main__":
    main()
