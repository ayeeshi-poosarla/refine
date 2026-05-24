#!/usr/bin/env python3
"""
Action: Remove the lower-AUROC field among the most correlated pair.

All rubric fields are label-encoded and pairwise Pearson correlations are
computed across every record (train + val + test). The pair with the highest
absolute correlation is identified as redundant. Individual AUROCs are then
computed for each field in that pair (LogReg trained on train, evaluated on
test). The field with the lower AUROC is removed from the rubric template and
every filled record.

Modifies in place:
  - rubric_dir/{task}/rubric.json
  - rubricified_dir/{task}/{split}.json

Usage:
  python3 remove_redundant_pair.py \\
      --task guo_readmission \\
      --rubric_dir   data/rubric \\
      --rubricified_dir data/rubric/rubricified
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

from rl.base_action import BaseAction
from rl.parsing import parse_rubric_fields, remove_field, replace_field_value, add_field_after
from rl.state import RubricState

SPLITS = ("train", "val", "test")


def parse_records(records: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Parse field dicts and labels from rubricified_text, keyed by split."""
    parsed: dict[str, list[dict]] = {}
    for split, recs in records.items():
        parsed[split] = [
            {
                "label":  int(r["label"]),
                "fields": parse_rubric_fields(r["rubricified_text"]),
            }
            for r in recs
        ]
    return parsed


def compute_pairwise_correlations(parsed: dict[str, list[dict]]) -> dict[tuple[str, str], float]:
    """Pearson |r| between every pair of label-encoded fields across all splits."""
    all_rows = [row for rows in parsed.values() for row in rows]
    all_fields = sorted({k for row in all_rows for k in row["fields"]})
    if len(all_fields) < 2:
        raise ValueError("Need at least 2 rubric fields to find a correlated pair")

    # Encode each field globally so the mapping is consistent
    encoded: dict[str, np.ndarray] = {}
    for f in all_fields:
        vals = [row["fields"].get(f, "No data") for row in all_rows]
        encoded[f] = LabelEncoder().fit_transform(vals).astype(float)

    correlations: dict[tuple[str, str], float] = {}
    for i, fa in enumerate(all_fields):
        for fb in all_fields[i + 1:]:
            r = np.corrcoef(encoded[fa], encoded[fb])[0, 1]
            correlations[(fa, fb)] = 0.0 if np.isnan(r) else abs(float(r))
    return correlations


def find_most_correlated_pair(correlations: dict[tuple[str, str], float]) -> tuple[str, str]:
    return max(correlations, key=lambda k: correlations[k])


def compute_individual_aurocs(parsed: dict[str, list[dict]],
                               fields: tuple[str, str]) -> dict[str, float]:
    """Train-on-train, evaluate-on-test single-field LogReg AUROC for each field."""
    if "train" not in parsed or "test" not in parsed:
        raise ValueError("train and test splits required for AUROC computation")

    train_rows = parsed["train"]
    test_rows  = parsed["test"]
    train_labels = [r["label"] for r in train_rows]
    test_labels  = [r["label"] for r in test_rows]

    aurocs: dict[str, float] = {}
    for f in fields:
        train_vals = [r["fields"].get(f, "No data") for r in train_rows]
        test_vals  = [r["fields"].get(f, "No data") for r in test_rows]
        if len(set(train_vals)) < 2:
            aurocs[f] = 0.5
            continue
        le = LabelEncoder()
        le.fit(train_vals + test_vals)
        X_train = le.transform(train_vals).reshape(-1, 1)
        X_test  = le.transform(test_vals).reshape(-1, 1)
        clf = LogisticRegression(max_iter=1000, C=0.01)
        clf.fit(X_train, train_labels)
        prob = clf.predict_proba(X_test)[:, 1]
        try:
            aurocs[f] = roc_auc_score(test_labels, prob)
        except ValueError:
            aurocs[f] = 0.5
    return aurocs


def remove_field_from_rubric(rubric_instructions: str, field: str) -> str:
    pattern = re.compile(
        r'^\s*\*\s*\*\*' + re.escape(field) + r':\*\*[^\n]*\n?',
        re.MULTILINE,
    )
    return pattern.sub("", rubric_instructions)


class RemoveRedundantPair(BaseAction):
    @property
    def name(self) -> str:
        return "remove_redundant_pair"

    def apply(self, state: RubricState) -> RubricState:
        new_state = state.copy()

        parsed = parse_records(new_state.records)
        correlations = compute_pairwise_correlations(parsed)
        pair = find_most_correlated_pair(correlations)
        correlation = correlations[pair]

        aurocs = compute_individual_aurocs(parsed, pair)
        # Remove whichever field in the pair has the lower AUROC
        to_remove = min(pair, key=lambda f: aurocs[f])
        to_keep   = pair[0] if to_remove == pair[1] else pair[1]

        new_state.rubric["rubric_instructions"] = remove_field_from_rubric(
            new_state.rubric["rubric_instructions"], to_remove
        )
        for recs in new_state.records.values():
            for r in recs:
                r["rubricified_text"] = remove_field(r["rubricified_text"], to_remove)

        new_state.rubric["_last_action"] = {
            "action":      self.name,
            "pair":        list(pair),
            "correlation": correlation,
            "aurocs":      aurocs,
            "removed":     to_remove,
            "kept":        to_keep,
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
    if not state.records:
        print(f"ERROR: no rubricified records found for {args.task}", file=sys.stderr)
        sys.exit(1)

    action    = RemoveRedundantPair()
    new_state = action.apply(state)

    meta = new_state.rubric.get("_last_action", {})
    pair = meta.get("pair", [])
    print(f"Most correlated pair: {pair[0]} & {pair[1]}  (|r|={meta.get('correlation', 0):.4f})")
    aurocs = meta.get("aurocs", {})
    for f in pair:
        print(f"  {f}: AUROC={aurocs.get(f, float('nan')):.4f}")
    print(f"Removed: {meta.get('removed')}  (kept: {meta.get('kept')})")
    for split, recs in new_state.records.items():
        print(f"  {split}: {len(recs)} records updated")

    new_state.to_disk(rubric_dir, rubricified_dir)
    print(f"Done. Removed '{meta.get('removed')}' and saved to disk.")


if __name__ == "__main__":
    main()
