#!/usr/bin/env python3
"""
Select 400 balanced patients per task with a custom 70/15/15 train/val/test split.

Draws 200 positive + 200 negative unique patients from across all EHRSHOT
splits (train/val/test combined), ignoring the original EHRSHOT split
assignments. Assigns a fresh custom split:
  - train: 140 pos + 140 neg = 280 patients
  - val:    30 pos +  30 neg =  60 patients
  - test:   30 pos +  30 neg =  60 patients
  - total : 400 patients

A patient's label polarity is determined by majority vote across all their
prediction events (ties count as positive).

Outputs a JSON list of {"patient_id": int, "split": str} objects.
serialize.py reads this to override EHRSHOT splits and skip balancing.

Supports guo_* tasks (boolean label "True"/"False"). For lab_* tasks the
value parsing would need adjustment; those are not targeted here.

Usage:
  python select_patients.py \\
    --task guo_readmission \\
    --labels_dir /path/to/EHRSHOT_ASSETS/benchmark \\
    --output_dir  data/selected_patients
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from config.tasks import SEED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _majority_label(values: list) -> bool:
    """True if more than half of prediction events are positive (ties → True)."""
    n_true = sum(1 for v in values if v)
    return n_true >= len(values) / 2


def _parse_bool(raw: str) -> bool:
    return raw.strip() == "True"


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def select_patients(
    task: str,
    labels_dir: str,
    output_dir: str,
    n_per_class: int = 200,
    seed: int = SEED,
) -> list:
    label_file = Path(labels_dir) / task / "labeled_patients.csv"
    if not label_file.exists():
        raise FileNotFoundError(f"Label file not found: {label_file}")

    # Collect all prediction-event labels per patient
    patient_labels: dict = defaultdict(list)
    with open(label_file) as f:
        for row in csv.DictReader(f):
            pid = int(row["patient_id"])
            patient_labels[pid].append(_parse_bool(row["value"]))

    # Stratify by majority label
    pos_patients = []
    neg_patients = []
    for pid, vals in patient_labels.items():
        if _majority_label(vals):
            pos_patients.append(pid)
        else:
            neg_patients.append(pid)

    rng = random.Random(seed)
    rng.shuffle(pos_patients)
    rng.shuffle(neg_patients)

    n_pos = min(n_per_class, len(pos_patients))
    n_neg = min(n_per_class, len(neg_patients))

    if n_pos < n_per_class:
        print(f"WARNING: only {n_pos} positive patients available (wanted {n_per_class})")
    if n_neg < n_per_class:
        print(f"WARNING: only {n_neg} negative patients available (wanted {n_per_class})")

    sel_pos = pos_patients[:n_pos]
    sel_neg = neg_patients[:n_neg]

    # Assign custom 70/15/15 splits (balanced within each split)
    # pos: 70% train, 15% val, 15% test
    def _split_list(lst: list, r_train=0.70, r_val=0.15):
        n = len(lst)
        n_train = round(n * r_train)
        n_val = round(n * r_val)
        return lst[:n_train], lst[n_train:n_train + n_val], lst[n_train + n_val:]

    pos_train, pos_val, pos_test = _split_list(sel_pos)
    neg_train, neg_val, neg_test = _split_list(sel_neg)

    selected = []
    for pid in pos_train + neg_train:
        selected.append({"patient_id": pid, "split": "train"})
    for pid in pos_val + neg_val:
        selected.append({"patient_id": pid, "split": "val"})
    for pid in pos_test + neg_test:
        selected.append({"patient_id": pid, "split": "test"})

    rng.shuffle(selected)

    n_train = sum(1 for s in selected if s["split"] == "train")
    n_val   = sum(1 for s in selected if s["split"] == "val")
    n_test  = sum(1 for s in selected if s["split"] == "test")
    print(
        f"{task}: {len(selected)} patients selected  "
        f"({n_pos} pos / {n_neg} neg)\n"
        f"  train={n_train}  val={n_val}  test={n_test}"
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / f"{task}_selected_patients.json"
    with open(out_path, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"  -> {out_path}")
    return selected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True,
                   help="Task name, e.g. guo_readmission")
    p.add_argument("--labels_dir", required=True,
                   help="EHRSHOT_ASSETS/benchmark directory")
    p.add_argument("--output_dir", required=True,
                   help="Where to write {task}_selected_patients.json")
    p.add_argument("--n_per_class", type=int, default=200,
                   help="Max patients per class (default 200)")
    return p.parse_args()


def main():
    args = parse_args()
    select_patients(args.task, args.labels_dir, args.output_dir, args.n_per_class)


if __name__ == "__main__":
    main()
