#!/usr/bin/env python3
"""Create a transparent per-site table for the 17-fold reference evaluation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_metadata(path: Path) -> dict[str, tuple[str, int]]:
    with path.open(newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        if "subject_id" in sample.splitlines()[0].lower():
            rows = csv.DictReader(handle)
            result = {}
            for row in rows:
                sid = (row.get("subject_id") or row.get("SUB_ID") or "").removeprefix("sub-")
                site = row.get("site_id") or row.get("SITE_ID") or ""
                dx = row.get("dx") or row.get("DX_GROUP") or row.get("diagnosis") or ""
                if not sid or not site:
                    continue
                label = int(dx) - 1 if str(dx).isdigit() and int(dx) in (1, 2) else int(str(dx).upper() in {"PATIENT", "ASD"})
                result[sid] = (site.upper(), label)
            return result

        result = {}
        for row in csv.reader(handle):
            if len(row) < 11:
                continue
            try:
                sid = str(int(row[9]))
            except ValueError:
                continue
            result[sid] = (row[10].strip().upper(), int(row[1] == "PATIENT"))
        return result


def read_split(path: Path) -> dict[str, list[str]]:
    sections = {"train": [], "valid": [], "test": []}
    current = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        lower = line.lower()
        if lower.startswith("train"):
            current = "train"
        elif lower.startswith("valid") or lower.startswith("val"):
            current = "valid"
        elif lower.startswith("test"):
            current = "test"
        elif current and line:
            sections[current].append(line.removeprefix("sub-"))
    return sections


def read_metrics(path: Path, combo: str) -> dict[int, dict[str, str]]:
    with path.open(newline="") as handle:
        result = {}
        for row in csv.DictReader(handle):
            if row.get("combo") == combo:
                result[int(row["fold"])] = row
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-metrics", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--combo", default="cc400_ez")
    args = parser.parse_args()

    metadata = read_metadata(args.metadata_csv)
    metrics = read_metrics(args.fold_metrics, args.combo)
    rows = []
    for fold in sorted(metrics):
        split = read_split(args.split_root / f"split_cv10_fold{fold}.txt")
        test_sites = {metadata[sid][0] for sid in split["test"] if sid in metadata}
        if len(test_sites) != 1:
            raise ValueError(f"fold {fold} does not have exactly one test site: {test_sites}")
        test_labels = [metadata[sid][1] for sid in split["test"] if sid in metadata]
        row = metrics[fold]
        rows.append(
            {
                "fold": fold,
                "test_site": next(iter(test_sites)),
                "n_train": len(split["train"]),
                "n_valid": len(split["valid"]),
                "n_test": len(split["test"]),
                "test_asd": sum(test_labels),
                "test_nc": len(test_labels) - sum(test_labels),
                "test_acc": row.get("test_acc", ""),
                "test_balacc": row.get("test_balacc", ""),
                "test_auc": row.get("test_auc", ""),
                "valid_log_loss": row.get("valid_log_loss", ""),
                "cc400_weight": row.get("cc400_weight", ""),
                "ez_weight": row.get("ez_weight", ""),
            }
        )

    fields = list(rows[0]) if rows else ["fold", "test_site"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
