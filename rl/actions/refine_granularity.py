#!/usr/bin/env python3
"""
Action: Replace the coarsest binary field with a 3-level ordinal (none / mild / severe).

"Binary field" — a rubric field whose filled values are exactly 2 distinct
non-missing strings across all records (missing = None / N/A / No data).

"Coarsest" — the binary field with the most skewed value distribution
(lowest Shannon entropy), meaning most records collapse into one bucket and
the binary split is doing the least work.

The two existing values are mapped to the outer levels of the new ordinal:
  - The value whose records have a lower mean label → "none"
  - The value whose records have a higher mean label → "severe"
  - "mild" is introduced in the rubric template as a new in-between option;
    existing filled records receive "none" or "severe" according to the map.

Modifies in place:
  - rubric_dir/{task}/rubric.json
  - rubricified_dir/{task}/{split}.json

Usage:
  python3 refine_granularity.py \\
      --task guo_readmission \\
      --rubric_dir   data/rubric \\
      --rubricified_dir data/rubric/rubricified
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from rl.base_action import BaseAction
from rl.state import RubricState

SPLITS = ("train", "val", "test")
FIELD_RE = re.compile(r'\*\*([A-Z_]+):\*\*\s*(.+)')
MISSING_VALUES = {"None", "N/A", "No data"}


def collect_nonmissing_values(records: dict[str, list[dict]]) -> dict[str, set[str]]:
    """Return {field: set of unique non-missing values} across all splits."""
    field_vals: dict[str, set[str]] = defaultdict(set)
    for recs in records.values():
        for r in recs:
            for field, value in FIELD_RE.findall(r["rubricified_text"]):
                v = value.strip()
                if v not in MISSING_VALUES:
                    field_vals[field].add(v)
    return dict(field_vals)


def find_binary_fields(field_vals: dict[str, set[str]]) -> dict[str, set[str]]:
    """Filter to fields with exactly 2 unique non-missing values."""
    return {f: vals for f, vals in field_vals.items() if len(vals) == 2}


def compute_entropy(records: dict[str, list[dict]], field: str) -> float:
    counts: dict[str, int] = defaultdict(int)
    for recs in records.values():
        for r in recs:
            for f, value in FIELD_RE.findall(r["rubricified_text"]):
                if f == field:
                    v = value.strip()
                    if v not in MISSING_VALUES:
                        counts[v] += 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def find_coarsest_binary_field(records: dict[str, list[dict]],
                                binary_fields: dict[str, set[str]]) -> str:
    """Return the binary field with the lowest entropy (most skewed distribution)."""
    entropies = {f: compute_entropy(records, f) for f in binary_fields}
    return min(entropies, key=lambda f: (entropies[f], f))


def build_value_map(records: dict[str, list[dict]], field: str,
                    binary_vals: set[str]) -> dict[str, str]:
    """Map each binary value to 'none' or 'severe' using mean label on train records.

    The value whose patients have a higher mean label gets 'severe'; the other gets 'none'.
    Falls back to alphabetical order when train split is unavailable.
    """
    train_recs = records.get("train") or next(iter(records.values()))
    val_labels: dict[str, list[int]] = defaultdict(list)
    for r in train_recs:
        for f, value in FIELD_RE.findall(r["rubricified_text"]):
            if f == field:
                v = value.strip()
                if v not in MISSING_VALUES:
                    val_labels[v].append(int(r["label"]))

    means = {
        v: (sum(val_labels[v]) / len(val_labels[v]) if val_labels[v] else 0.5)
        for v in binary_vals
    }
    ordered = sorted(binary_vals, key=lambda v: (means[v], v))
    return {ordered[0]: "none", ordered[1]: "severe"}


def refine_field_in_rubric(rubric_instructions: str, field: str,
                            value_map: dict[str, str]) -> str:
    """Replace the field's rubric bullet with a 3-level ordinal description."""
    none_val   = next(v for v, level in value_map.items() if level == "none")
    severe_val = next(v for v, level in value_map.items() if level == "severe")
    description = (
        f"[none if {none_val}; "
        f"mild if intermediate presentation; "
        f"severe if {severe_val}]"
    )
    pattern = re.compile(
        r'^\s*\*\s*\*\*' + re.escape(field) + r':\*\*[^\n]*\n?',
        re.MULTILINE,
    )
    return pattern.sub(f'*   **{field}:** {description}\n', rubric_instructions, count=1)


def refine_field_in_text(rubricified_text: str, field: str,
                          value_map: dict[str, str]) -> str:
    """Remap the field's current binary value to its 3-level equivalent."""
    def replace(m):
        current = m.group(1).strip()
        new_val = value_map.get(current, current)
        return f'*   **{field}:** {new_val}\n'

    pattern = re.compile(
        r'^\s*\*?\s*\*\*' + re.escape(field) + r':\*\*\s*(.+)\n?',
        re.MULTILINE,
    )
    return pattern.sub(replace, rubricified_text)


class RefineGranularity(BaseAction):
    @property
    def name(self) -> str:
        return "refine_granularity"

    def apply(self, state: RubricState) -> RubricState:
        new_state = state.copy()

        field_vals    = collect_nonmissing_values(new_state.records)
        binary_fields = find_binary_fields(field_vals)
        if not binary_fields:
            raise ValueError("No binary fields found — cannot apply refine_granularity")

        target    = find_coarsest_binary_field(new_state.records, binary_fields)
        value_map = build_value_map(new_state.records, target, binary_fields[target])
        entropy   = compute_entropy(new_state.records, target)

        new_state.rubric["rubric_instructions"] = refine_field_in_rubric(
            new_state.rubric["rubric_instructions"], target, value_map
        )
        for recs in new_state.records.values():
            for r in recs:
                r["rubricified_text"] = refine_field_in_text(
                    r["rubricified_text"], target, value_map
                )

        new_state.rubric["_last_action"] = {
            "action":    self.name,
            "field":     target,
            "entropy":   entropy,
            "value_map": value_map,
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

    action    = RefineGranularity()
    new_state = action.apply(state)

    meta = new_state.rubric.get("_last_action", {})
    print(f"Coarsest binary field: {meta.get('field')} (entropy={meta.get('entropy', 0):.4f})")
    print(f"Value mapping: {meta.get('value_map')}")
    for split, recs in new_state.records.items():
        print(f"  {split}: {len(recs)} records updated")

    new_state.to_disk(rubric_dir, rubricified_dir)
    print(f"Done. Refined '{meta.get('field')}' to 3-level ordinal and saved to disk.")


if __name__ == "__main__":
    main()
