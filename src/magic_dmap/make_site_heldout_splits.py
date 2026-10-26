#!/usr/bin/env python3
"""Create ABIDE1 fullroi1020 LOSO splits with a held-out validation site.

For each test site, the test site is completely excluded from training and
validation. One large remaining site is used as validation to make model
selection closer to the final leave-one-site-out setting.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


SITE_ORDER = [
    "CALTECH",
    "CMU",
    "KKI",
    "LEUVEN",
    "MAX_MUN",
    "NYU",
    "OHSU",
    "OLIN",
    "PITT",
    "SBL",
    "SDSU",
    "STANFORD",
    "TRINITY",
    "UCLA",
    "UM",
    "USM",
    "YALE",
]

VALID_SITE_CYCLE = ["NYU", "UM", "UCLA", "LEUVEN", "USM", "PITT", "YALE", "MAX_MUN"]


def write_split(path: Path, train: list[str], valid: list[str], test: list[str]) -> None:
    with path.open("w") as f:
        f.write("train_subjects\n")
        for sid in train:
            f.write(f"sub-{sid}\n")
        f.write("\nvalid_subjects\n")
        for sid in valid:
            f.write(f"sub-{sid}\n")
        f.write("\ntest_subjects\n")
        for sid in test:
            f.write(f"sub-{sid}\n")


def counts(subjects: list[str], labels: dict[str, int]) -> tuple[int, int]:
    asd = sum(labels[sid] == 1 for sid in subjects)
    ctrl = sum(labels[sid] == 0 for sid in subjects)
    return asd, ctrl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-csv", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--manifest-out", type=Path, required=True)
    args = ap.parse_args()

    site_subjects: dict[str, list[str]] = defaultdict(list)
    labels: dict[str, int] = {}
    with args.metadata_csv.open(newline="") as f:
        for row in csv.reader(f):
            if len(row) < 11:
                continue
            try:
                sid = str(int(row[9]))
            except ValueError:
                continue
            site = row[10].strip().upper()
            if site not in SITE_ORDER:
                continue
            label = 1 if row[1] == "PATIENT" else 0
            site_subjects[site].append(sid)
            labels[sid] = label

    missing = [site for site in SITE_ORDER if site not in site_subjects]
    if missing:
        raise SystemExit(f"Missing sites: {missing}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for fold, test_site in enumerate(SITE_ORDER, start=1):
        valid_site = VALID_SITE_CYCLE[(fold - 1) % len(VALID_SITE_CYCLE)]
        if valid_site == test_site:
            valid_site = VALID_SITE_CYCLE[fold % len(VALID_SITE_CYCLE)]

        test = sorted(site_subjects[test_site])
        valid = sorted(site_subjects[valid_site])
        train = sorted(
            sid
            for site in SITE_ORDER
            if site not in {test_site, valid_site}
            for sid in site_subjects[site]
        )
        split_path = args.out_root / f"split_cv10_fold{fold}.txt"
        write_split(split_path, train, valid, test)

        tr_asd, tr_ctrl = counts(train, labels)
        va_asd, va_ctrl = counts(valid, labels)
        te_asd, te_ctrl = counts(test, labels)
        rows.append(
            {
                "fold": fold,
                "heldout_site": test_site,
                "validation_site": valid_site,
                "split_file": str(split_path),
                "n_train": len(train),
                "train_asd": tr_asd,
                "train_control": tr_ctrl,
                "n_valid": len(valid),
                "valid_asd": va_asd,
                "valid_control": va_ctrl,
                "n_test": len(test),
                "test_asd": te_asd,
                "test_control": te_ctrl,
            }
        )

    fieldnames = [
        "fold",
        "heldout_site",
        "validation_site",
        "split_file",
        "n_train",
        "train_asd",
        "train_control",
        "n_valid",
        "valid_asd",
        "valid_control",
        "n_test",
        "test_asd",
        "test_control",
    ]
    with args.manifest_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(args.manifest_out)


if __name__ == "__main__":
    main()
