#!/usr/bin/env python3
"""
Action: Remove the rubric field with the lowest variance across all filled rubrics.

Variance is measured as the number of unique values a field takes across all
patient records (train + val + test). The field with fewest unique values carries
the least information and is removed.

Modifies in place:
  - rubric_dir/{task}/rubric.json          (rubric_instructions text)
  - rubricified_dir/{task}/{split}.json    (rubricified_text in every record)

Usage:
  python3 remove_lowest_variance_field.py \
      --task guo_readmission \
      --rubric_dir   data/rubric \
      --rubricified_dir data/rubric/rubricified
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from rl.base_action import BaseAction
from rl.parsing import parse_rubric_fields, remove_field, replace_field_value, add_field_after
from rl.state import RubricState

SPLITS = ("train", "val", "test")


def compute_unique_counts(rubricified_dir: Path, task: str) -> dict[str, int]:
    field_values: dict[str, set] = defaultdict(set)
    for split in SPLITS:
        path = rubricified_dir / task / f"{split}.json"
        if not path.exists():
            continue
        for record in json.load(open(path)):
            for field, value in parse_rubric_fields(record["rubricified_text"]).items():
                field_values[field].add(value.strip())
    return {f: len(v) for f, v in field_values.items()}


def find_lowest_variance_field(unique_counts: dict[str, int]) -> str:
    return min(unique_counts, key=lambda f: unique_counts[f])


def remove_field_from_rubric(rubric_instructions: str, field: str) -> str:
    # Matches the bullet line for the field, e.g.:
    #   *   **FIELD_NAME:** [...]
    pattern = re.compile(
        r'^\s*\*\s*\*\*' + re.escape(field) + r':\*\*[^\n]*\n?',
        re.MULTILINE,
    )
    return pattern.sub("", rubric_instructions)


class RemoveLowestVarianceField(BaseAction):
    @property
    def name(self) -> str:
        return "remove_lowest_variance_field"

    def apply(self, state: RubricState) -> RubricState:
        new_state = state.copy()

        # Compute unique counts from in-memory records
        field_values: dict[str, set] = defaultdict(set)
        for recs in new_state.records.values():
            for r in recs:
                for field, value in parse_rubric_fields(r["rubricified_text"]).items():
                    field_values[field].add(value.strip())
        unique_counts = {f: len(v) for f, v in field_values.items()}

        target = find_lowest_variance_field(unique_counts)

        new_state.rubric["rubric_instructions"] = remove_field_from_rubric(
            new_state.rubric["rubric_instructions"], target
        )
        for recs in new_state.records.values():
            for r in recs:
                r["rubricified_text"] = remove_field(r["rubricified_text"], target)

        new_state.rubric["_last_action"] = {
            "action": self.name,
            "removed_field": target,
            "unique_values": unique_counts[target],
        }
        return new_state


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--rubric_dir", required=True)
    p.add_argument("--rubricified_dir", required=True)
    args = p.parse_args()

    rubric_dir = Path(args.rubric_dir)
    rubricified_dir = Path(args.rubricified_dir)

    state = RubricState.from_disk(args.task, rubric_dir, rubricified_dir)
    if not state.records:
        print(f"ERROR: no rubricified records found for {args.task}", file=sys.stderr)
        sys.exit(1)

    action = RemoveLowestVarianceField()
    new_state = action.apply(state)

    meta = new_state.rubric.get("_last_action", {})
    print(f"Lowest variance field: {meta.get('removed_field')} ({meta.get('unique_values')} unique value(s))")
    for split, recs in new_state.records.items():
        print(f"  {split}: {len(recs)} records updated")

    new_state.to_disk(rubric_dir, rubricified_dir)
    print(f"Done. Removed '{meta.get('removed_field')}' and saved to disk.")


if __name__ == "__main__":
    main()
