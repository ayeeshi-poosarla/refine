#!/usr/bin/env python3
"""
Action: Split the rubric field with the highest Shannon entropy into two
finer-grained bins (LOW / HIGH).

The field whose values are most uniformly distributed across patients is
selected (highest entropy). Its unique values are partitioned into two bins:

  Numeric fields    — split at the median of all observed values.
                      Values ≤ median → LOW, values > median → HIGH.
  Categorical fields — sort unique values alphabetically, assign the lower
                       half → LOW and the upper half → HIGH.

The original field is replaced in-place by {FIELD}_BIN in both the rubric
template and every filled rubric record.

Modifies in place:
  - rubric_dir/{task}/rubric.json
  - rubricified_dir/{task}/{split}.json

Usage:
  python3 split_highest_entropy_field.py \\
      --task guo_readmission \\
      --rubric_dir   data/rubric \\
      --rubricified_dir data/rubric/rubricified
"""

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rl.base_action import BaseAction
from rl.state import RubricState

SPLITS = ("train", "val", "test")
FIELD_RE = re.compile(r'\*\*([A-Z_]+):\*\*\s*(.+)')
NUM_RE   = re.compile(r'-?\d+\.?\d*')


def compute_entropy(values: list[str]) -> float:
    counts = Counter(values)
    total = len(values)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def collect_field_values(records: dict[str, list[dict]]) -> dict[str, list[str]]:
    field_vals: dict[str, list[str]] = defaultdict(list)
    for recs in records.values():
        for r in recs:
            for field, value in FIELD_RE.findall(r["rubricified_text"]):
                field_vals[field].append(value.strip())
    return dict(field_vals)


def find_highest_entropy_field(field_vals: dict[str, list[str]]) -> str:
    entropies = {f: compute_entropy(vals) for f, vals in field_vals.items() if vals}
    return max(entropies, key=lambda f: entropies[f])


def compute_bin_mapping(values: list[str]) -> dict[str, str]:
    """Map each unique string value to 'LOW' or 'HIGH'.

    Uses a numeric median split when every unique value contains a number;
    falls back to an alphabetical split for categorical fields.
    """
    unique_vals = sorted(set(values))

    numeric_map: dict[str, float] = {}
    for v in unique_vals:
        m = NUM_RE.search(v)
        if m:
            numeric_map[v] = float(m.group())

    if len(numeric_map) == len(unique_vals):
        all_nums = sorted(numeric_map[v] for v in values)
        median_val = all_nums[len(all_nums) // 2]
        return {v: "LOW" if numeric_map[v] <= median_val else "HIGH"
                for v in unique_vals}

    midpoint = max(1, len(unique_vals) // 2)
    return {v: ("LOW" if i < midpoint else "HIGH")
            for i, v in enumerate(unique_vals)}


def replace_field_in_rubric(rubric_instructions: str, field: str,
                             new_field: str, bin_map: dict[str, str]) -> str:
    low_vals  = sorted(v for v, b in bin_map.items() if b == "LOW")
    high_vals = sorted(v for v, b in bin_map.items() if b == "HIGH")
    description = (
        f"[LOW if value is in {{{', '.join(low_vals)}}}; "
        f"HIGH if value is in {{{', '.join(high_vals)}}}]"
    )
    pattern = re.compile(
        r'^\s*\*\s*\*\*' + re.escape(field) + r':\*\*[^\n]*\n?',
        re.MULTILINE,
    )
    return pattern.sub(f'*   **{new_field}:** {description}\n', rubric_instructions, count=1)


def replace_field_in_text(rubricified_text: str, field: str,
                           new_field: str, bin_label: str) -> str:
    pattern = re.compile(
        r'^\s*\*?\s*\*\*' + re.escape(field) + r':\*\*[^\n]*\n?',
        re.MULTILINE,
    )
    return pattern.sub(f'*   **{new_field}:** {bin_label}\n', rubricified_text, count=1)


class SplitHighestEntropyField(BaseAction):
    @property
    def name(self) -> str:
        return "split_highest_entropy_field"

    def apply(self, state: RubricState) -> RubricState:
        new_state = state.copy()

        field_vals = collect_field_values(new_state.records)
        if not field_vals:
            raise ValueError("No rubric fields found in records")

        source_field = find_highest_entropy_field(field_vals)
        new_field    = f"{source_field}_BIN"
        entropy      = compute_entropy(field_vals[source_field])
        bin_map      = compute_bin_mapping(field_vals[source_field])

        new_state.rubric["rubric_instructions"] = replace_field_in_rubric(
            new_state.rubric["rubric_instructions"], source_field, new_field, bin_map
        )

        for recs in new_state.records.values():
            for r in recs:
                fields    = dict(FIELD_RE.findall(r["rubricified_text"]))
                raw_val   = fields.get(source_field, "").strip()
                bin_label = bin_map.get(raw_val, "LOW")
                r["rubricified_text"] = replace_field_in_text(
                    r["rubricified_text"], source_field, new_field, bin_label
                )

        new_state.rubric["_last_action"] = {
            "action":       self.name,
            "source_field": source_field,
            "new_field":    new_field,
            "entropy":      entropy,
            "bin_map":      bin_map,
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

    action    = SplitHighestEntropyField()
    new_state = action.apply(state)

    meta = new_state.rubric.get("_last_action", {})
    print(f"Highest entropy field: {meta.get('source_field')} (entropy={meta.get('entropy', 0):.4f})")
    print(f"New field: {meta.get('new_field')}")
    print(f"Bin mapping: {meta.get('bin_map')}")
    for split, recs in new_state.records.items():
        print(f"  {split}: {len(recs)} records updated")

    new_state.to_disk(rubric_dir, rubricified_dir)
    print(f"Done. Replaced '{meta.get('source_field')}' with '{meta.get('new_field')}' and saved to disk.")


if __name__ == "__main__":
    main()
