#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract ROI time series from MNI-space ABIDE-I fMRI.

This script is intentionally additive: by default it writes only under
neurostorm_cv5_roi_compare/output and never modifies the downloaded CPAC .1D
files or the NeuroSTORM processed volume data.
"""

from __future__ import print_function

import argparse
import csv
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to


DERIVATIVES = ("aal", "cc400", "ez", "ho", "dosenbach160")


def parse_atlas_args(values):
    atlas_paths = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--atlas must be given as name=/path/to/atlas.nii.gz")
        name, path = value.split("=", 1)
        name = name.strip()
        if name not in DERIVATIVES:
            raise ValueError("Unsupported atlas name: {}".format(name))
        atlas_paths[name] = Path(path)
    missing = [name for name in DERIVATIVES if name not in atlas_paths]
    if missing:
        raise ValueError("Missing atlas paths for: {}".format(", ".join(missing)))
    return atlas_paths


def load_subject_table(pheno_path):
    pheno = pd.read_csv(pheno_path)
    pheno = pheno[pheno["FILE_ID"] != "no_filename"].copy()
    pheno["SUB_ID_STR"] = pheno["SUB_ID"].astype(int).astype(str)
    pheno["bids_subject"] = pheno["SUB_ID_STR"].apply(lambda s: "sub-{}".format(s))
    return pheno


def bids_bold_path(fmri_root, bids_subject):
    return fmri_root / "{}_ses-1_task-rest_space-MNI_desc-preproc_bold.nii.gz".format(bids_subject)


def bids_mask_path(fmri_root, bids_subject):
    return fmri_root / "{}_ses-1_task-rest_space-MNI_desc-brain_mask.nii.gz".format(bids_subject)


def resample_labels_to_fmri(atlas_img, fmri_img):
    if atlas_img.shape == fmri_img.shape[:3] and np.allclose(atlas_img.affine, fmri_img.affine, atol=1e-3):
        return atlas_img.get_fdata(dtype=np.float32).astype(np.int32)
    target = (fmri_img.shape[:3], fmri_img.affine)
    return resample_from_to(atlas_img, target, order=0).get_fdata(dtype=np.float32).astype(np.int32)


def atlas_positive_labels(atlas_img):
    labels = np.rint(atlas_img.get_fdata(dtype=np.float32)).astype(np.int32)
    unique = np.unique(labels)
    unique = unique[unique > 0]
    if unique.size == 0:
        raise ValueError("Atlas has no positive ROI labels")
    return unique


def prepare_label_index(labels, brain_mask, target_labels):
    labels = np.asarray(labels, dtype=np.int32)
    if brain_mask.shape != labels.shape:
        raise ValueError("Brain mask shape {} does not match atlas labels shape {}".format(brain_mask.shape, labels.shape))
    all_labels = np.asarray(target_labels, dtype=np.int32)

    labels = np.where(brain_mask, labels, 0)
    present = np.unique(labels)
    present = present[present > 0]
    if present.size == 0:
        raise ValueError("Atlas has no positive ROI labels after resampling")
    label_to_index = np.full(int(all_labels.max()) + 1, -1, dtype=np.int32)
    label_to_index[all_labels] = np.arange(all_labels.size, dtype=np.int32)
    flat = labels.reshape(-1)
    valid = (flat > 0) & (flat < label_to_index.shape[0])
    mapped = label_to_index[flat[valid]]
    keep = mapped >= 0
    valid_indices = np.flatnonzero(valid)[keep]
    valid = np.zeros(flat.shape, dtype=bool)
    valid[valid_indices] = True
    roi_index = mapped[keep]
    counts = np.bincount(roi_index, minlength=all_labels.size).astype(np.float32)
    return valid, roi_index, counts, all_labels


def mean_roi_timeseries(fmri_data, valid_mask, roi_index, counts):
    flat = np.asarray(fmri_data, dtype=np.float32).reshape(-1, fmri_data.shape[-1])
    roi_voxels = flat[valid_mask]
    sums = np.zeros((counts.shape[0], roi_voxels.shape[1]), dtype=np.float32)
    for frame in range(roi_voxels.shape[1]):
        sums[:, frame] = np.bincount(roi_index, weights=roi_voxels[:, frame], minlength=counts.shape[0])
    out = np.zeros_like(sums)
    nonzero = counts > 0
    out[nonzero] = sums[nonzero] / counts[nonzero, None]
    return out


def regress_global_signal(roi_ts, global_signal):
    """Return ROI x time residuals after intercept + global signal regression."""
    g = np.asarray(global_signal, dtype=np.float64)
    x = np.column_stack([np.ones(g.shape[0], dtype=np.float64), g])
    betas = np.linalg.pinv(x).dot(roi_ts.T.astype(np.float64))
    fitted = x.dot(betas).T
    return (roi_ts.astype(np.float64) - fitted).astype(np.float32)


def load_global_signal(fmri_data, mask_path):
    mask = load_brain_mask(mask_path)
    if not mask.any():
        raise ValueError("Empty brain mask: {}".format(mask_path))
    return fmri_data[mask, :].mean(axis=0)


def load_brain_mask(mask_path):
    mask_img = nib.load(str(mask_path))
    mask = mask_img.get_fdata(dtype=np.float32)
    if mask.ndim == 4:
        mask = mask[..., 0]
    return mask > 0


def output_path(output_root, derivative, file_id):
    return output_root / "rois_{}".format(derivative) / "{}_rois_{}.1D".format(file_id, derivative)


def write_1d(path, roi_by_time):
    path.parent.mkdir(parents=True, exist_ok=True)
    # CPAC ROI .1D files are time x ROI, whitespace separated.
    np.savetxt(str(path), roi_by_time.T, fmt="%.8g")


def process_subject(row, fmri_root, atlas_imgs, atlas_label_sets, atlas_cache, output_root, gsr, overwrite):
    bids_subject = row["bids_subject"]
    file_id = row["FILE_ID"]
    bold_path = bids_bold_path(fmri_root, bids_subject)
    mask_path = bids_mask_path(fmri_root, bids_subject)
    if not bold_path.exists():
        return [{"bids_subject": bids_subject, "file_id": file_id, "derivative": "", "status": "missing_bold", "path": str(bold_path), "n_time": "", "n_roi": ""}]
    if not mask_path.exists():
        return [{"bids_subject": bids_subject, "file_id": file_id, "derivative": "", "status": "missing_mask", "path": str(mask_path), "n_time": "", "n_roi": ""}]

    planned = [output_path(output_root, derivative, file_id) for derivative in DERIVATIVES]
    if all(path.exists() for path in planned) and not overwrite:
        return [
            {"bids_subject": bids_subject, "file_id": file_id, "derivative": derivative, "status": "exists", "path": str(path), "n_time": "", "n_roi": ""}
            for derivative, path in zip(DERIVATIVES, planned)
        ]

    fmri_img = nib.load(str(bold_path))
    fmri_data = fmri_img.get_fdata(dtype=np.float32)
    if fmri_data.ndim != 4:
        raise ValueError("Expected 4D fMRI: {}".format(bold_path))
    brain_mask = load_brain_mask(mask_path)
    if not brain_mask.any():
        return [{"bids_subject": bids_subject, "file_id": file_id, "derivative": "", "status": "empty_mask", "path": str(mask_path), "n_time": "", "n_roi": ""}]
    global_signal = load_global_signal(fmri_data, mask_path) if gsr else None

    rows = []
    for derivative in DERIVATIVES:
        out = output_path(output_root, derivative, file_id)
        if out.exists() and not overwrite:
            rows.append({"bids_subject": bids_subject, "file_id": file_id, "derivative": derivative, "status": "exists", "path": str(out), "n_time": "", "n_roi": ""})
            continue
        if derivative not in atlas_cache:
            labels = resample_labels_to_fmri(atlas_imgs[derivative], fmri_img)
            atlas_cache[derivative] = prepare_label_index(labels, brain_mask, atlas_label_sets[derivative])
        valid, roi_index, counts, unique = atlas_cache[derivative]
        roi_ts = mean_roi_timeseries(fmri_data, valid, roi_index, counts)
        if gsr:
            roi_ts = regress_global_signal(roi_ts, global_signal)
        write_1d(out, roi_ts)
        rows.append({
            "bids_subject": bids_subject,
            "file_id": file_id,
            "derivative": derivative,
            "status": "written",
            "path": str(out),
            "n_time": roi_ts.shape[1],
            "n_roi": roi_ts.shape[0],
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmri-root", type=Path, required=True)
    parser.add_argument("--pheno", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--atlas", action="append", required=True, help="Atlas mapping, e.g. aal=/path/aal.nii.gz. Repeat for all five atlases.")
    parser.add_argument("--no-gsr", action="store_true", help="Skip global signal regression. Default is GSR.")
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional BIDS subject IDs or FILE_IDs to process.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    atlas_paths = parse_atlas_args(args.atlas)
    for name, path in atlas_paths.items():
        if not path.exists():
            raise FileNotFoundError("{} atlas not found: {}".format(name, path))
    atlas_imgs = {name: nib.load(str(path)) for name, path in atlas_paths.items()}
    atlas_label_sets = {name: atlas_positive_labels(img) for name, img in atlas_imgs.items()}

    pheno = load_subject_table(args.pheno)
    if args.subjects:
        requested = set(args.subjects)
        pheno = pheno[(pheno["bids_subject"].isin(requested)) | (pheno["FILE_ID"].isin(requested))].copy()

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["bids_subject", "file_id", "derivative", "status", "path", "n_time", "n_roi"]
    all_rows = []
    for _, row in pheno.sort_values("SUB_ID").iterrows():
        atlas_cache = {}
        all_rows.extend(process_subject(row, args.fmri_root, atlas_imgs, atlas_label_sets, atlas_cache, args.output_root, not args.no_gsr, args.overwrite))

    with args.manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    statuses = {}
    for row in all_rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    print("[Info] ROI root:", args.output_root)
    print("[Info] Manifest:", args.manifest)
    print("[Info] Status counts:", statuses)


if __name__ == "__main__":
    main()
