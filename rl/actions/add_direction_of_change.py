#!/usr/bin/env python3
"""
Action: Add a direction-of-change field for the rubric field with the highest
individual AUROC.

For each rubric field, a LogisticRegression is trained using only that field's
label-encoded value as a feature (no GPU, no LLM). The field with the highest
test AUROC is selected.

A new field  {SOURCE_FIELD}_DIRECTION  is then added immediately after the
source field in both the rubric template and every filled rubric record.

The direction is computed per record by comparing the source field's numeric
value to the same patient's immediately preceding prediction event (sorted by
prediction_time). Patients with only one event receive "N/A".

Direction encoding:
  Increasing  — current value > previous value
  Decreasing  — current value < previous value
  Stable      — current value == previous value
  N/A         — no prior event for this patient, or non-numeric field value

Modifies in place:
  - rubric_dir/{task}/rubric.json
  - rubricified_dir/{task}/{split}.json

Usage:
  python3 add_direction_of_change.py \
      --task guo_readmission \
      --rubric_dir   data/rubric \
      --rubricified_dir data/rubric/rubricified
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

from rl.base_action import BaseAction
from rl.parsing import parse_rubric_fields, remove_field, replace_field_value, add_field_after
from rl.state import RubricState

SPLITS = ("train", "val", "test")
NUM_RE   = re.compile(r'-?\d+\.?\d*')


def load_records(rubricified_dir: Path, task: str) -> dict[str, list[dict]]:
    split_records = {}
    for split in SPLITS:
        path = rubricified_dir / task / f"{split}.json"
        if path.exists():
            split_records[split] = json.load(open(path))
    return split_records


def parse_fields(records: list[dict]) -> list[dict]:
    rows = []
    for r in records:
        fields = parse_rubric_fields(r["rubricified_text"])
        rows.append({
            "label": int(r["label"]),
            "patient_id": r["patient_id"],
            "prediction_time": r["prediction_time"],
            "fields": fields,
        })
    return rows


def compute_individual_aurocs(train_rows, test_rows) -> dict[str, float]:
    all_fields = sorted({k for r in train_rows + test_rows for k in r["fields"]})
    aurocs = {}
    train_labels = [r["label"] for r in train_rows]
    test_labels  = [r["label"] for r in test_rows]

    for field in all_fields:
        train_vals = [r["fields"].get(field, "No data") for r in train_rows]
        test_vals  = [r["fields"].get(field, "No data") for r in test_rows]
        if len(set(train_vals)) < 2:
            aurocs[field] = 0.5
            continue
        le = LabelEncoder()
        le.fit(train_vals + test_vals)
        X_train = le.transform(train_vals).reshape(-1, 1)
        X_test  = le.transform(test_vals).reshape(-1, 1)
        clf = LogisticRegression(max_iter=1000, C=0.01)
        clf.fit(X_train, train_labels)
        prob = clf.predict_proba(X_test)[:, 1]
        try:
            aurocs[field] = roc_auc_score(test_labels, prob)
        except ValueError:
            aurocs[field] = 0.5
    return aurocs


def _extract_number(value: str) -> float | None:
    m = NUM_RE.search(value)
    return float(m.group()) if m else None


def compute_directions(split_records: dict[str, list[dict]], field: str) -> dict[tuple, str]:
    """
    Returns a mapping of (patient_id, prediction_time) -> direction string.
    Builds a per-patient timeline across all splits.
    """
    # Collect all events across splits
    all_events: list[dict] = []
    for records in split_records.values():
        for r in records:
            fields = parse_rubric_fields(r["rubricified_text"])
            all_events.append({
                "patient_id": r["patient_id"],
                "prediction_time": r["prediction_time"],
                "value": fields.get(field, ""),
            })

    # Sort each patient's events chronologically
    by_patient: dict = defaultdict(list)
    for ev in all_events:
        by_patient[ev["patient_id"]].append(ev)
    for events in by_patient.values():
        events.sort(key=lambda e: e["prediction_time"])

    directions: dict[tuple, str] = {}
    for patient_id, events in by_patient.items():
        for i, ev in enumerate(events):
            key = (ev["patient_id"], ev["prediction_time"])
            if i == 0:
                directions[key] = "N/A"
                continue
            curr_num = _extract_number(ev["value"])
            prev_num = _extract_number(events[i - 1]["value"])
            if curr_num is None or prev_num is None:
                directions[key] = "N/A"
            elif curr_num > prev_num:
                directions[key] = "Increasing"
            elif curr_num < prev_num:
                directions[key] = "Decreasing"
            else:
                directions[key] = "Stable"

    return directions


def add_field_to_rubric(rubric_instructions: str, source_field: str,
                         new_field: str) -> str:
    # Find the source field's bullet line and insert the new field immediately after
    pattern = re.compile(
        r'(^\s*\*\s*\*\*' + re.escape(source_field) + r':\*\*[^\n]*)',
        re.MULTILINE,
    )
    replacement = (
        r'\1\n'
        f'*   **{new_field}:** '
        '[Increasing / Decreasing / Stable / N/A — '
        f'direction of change vs. prior event for {source_field}. '
        'N/A if no prior event exists for this patient.]'
    )
    result = pattern.sub(replacement, rubric_instructions, count=1)
    return result


class AddDirectionOfChange(BaseAction):
    @property
    def name(self) -> str:
        return "add_direction_of_change"

    def apply(self, state: RubricState) -> RubricState:
        new_state = state.copy()

        if "train" not in new_state.records or "test" not in new_state.records:
            raise ValueError("train and test splits required")

        train_rows = parse_fields(new_state.records["train"])
        test_rows  = parse_fields(new_state.records["test"])
        aurocs = compute_individual_aurocs(train_rows, test_rows)
        source_field = max(aurocs, key=lambda f: aurocs[f])
        new_field = f"{source_field}_DIRECTION"

        directions = compute_directions(new_state.records, source_field)

        new_state.rubric["rubric_instructions"] = add_field_to_rubric(
            new_state.rubric["rubric_instructions"], source_field, new_field
        )
        for recs in new_state.records.values():
            for r in recs:
                key = (r["patient_id"], r["prediction_time"])
                direction = directions.get(key, "N/A")
                r["rubricified_text"] = add_field_after(
                    r["rubricified_text"], source_field, new_field, direction
                )

        new_state.rubric["_last_action"] = {
            "action": self.name,
            "source_field": source_field,
            "new_field": new_field,
            "auroc": aurocs[source_field],
        }
        return new_state


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--rubric_dir", required=True)
    p.add_argument("--rubricified_dir", required=True)
    args = p.parse_args()

    rubric_dir      = Path(args.rubric_dir)
    rubricified_dir = Path(args.rubricified_dir)

    state = RubricState.from_disk(args.task, rubric_dir, rubricified_dir)
    if "train" not in state.records or "test" not in state.records:
        print("ERROR: train and test splits required", file=sys.stderr)
        sys.exit(1)

    action = AddDirectionOfChange()
    new_state = action.apply(state)

    meta = new_state.rubric.get("_last_action", {})
    print(f"Highest individual AUROC field: {meta.get('source_field')} (AUROC={meta.get('auroc', 0):.4f})")
    print(f"New field: {meta.get('new_field')}")
    for split, recs in new_state.records.items():
        print(f"  {split}: {len(recs)} records updated")

    new_state.to_disk(rubric_dir, rubricified_dir)
    print(f"Done. Added '{meta.get('new_field')}' and saved to disk.")


if __name__ == "__main__":
    main()
