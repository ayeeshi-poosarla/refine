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

SPLITS = ("train", "val", "test")
FIELD_RE = re.compile(r'\*\*([A-Z_]+):\*\*\s*(.+)')


def compute_unique_counts(rubricified_dir: Path, task: str) -> dict[str, int]:
    field_values: dict[str, set] = defaultdict(set)
    for split in SPLITS:
        path = rubricified_dir / task / f"{split}.json"
        if not path.exists():
            continue
        for record in json.load(open(path)):
            for field, value in FIELD_RE.findall(record["rubricified_text"]):
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


def remove_field_from_text(rubricified_text: str, field: str) -> str:
    pattern = re.compile(
        r'^\s*\*?\s*\*\*' + re.escape(field) + r':\*\*[^\n]*\n?',
        re.MULTILINE,
    )
    return pattern.sub("", rubricified_text)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--rubric_dir", required=True)
    p.add_argument("--rubricified_dir", required=True)
    args = p.parse_args()

    rubric_dir = Path(args.rubric_dir)
    rubricified_dir = Path(args.rubricified_dir)

    # 1. Identify the lowest-variance field
    unique_counts = compute_unique_counts(rubricified_dir, args.task)
    if not unique_counts:
        print(f"ERROR: no rubricified records found for {args.task}", file=sys.stderr)
        sys.exit(1)

    target = find_lowest_variance_field(unique_counts)
    print(f"Lowest variance field: {target} ({unique_counts[target]} unique value(s))")

    # 2. Remove from rubric template
    rubric_path = rubric_dir / args.task / "rubric.json"
    rubric = json.load(open(rubric_path))
    before = rubric["rubric_instructions"]
    rubric["rubric_instructions"] = remove_field_from_rubric(before, target)
    if rubric["rubric_instructions"] == before:
        print(f"WARNING: field {target} not found in rubric_instructions — template unchanged")
    else:
        print(f"Removed {target} from rubric template")
    with open(rubric_path, "w") as f:
        json.dump(rubric, f, indent=2)

    # 3. Remove from every filled rubric record
    total_modified = 0
    for split in SPLITS:
        path = rubricified_dir / args.task / f"{split}.json"
        if not path.exists():
            continue
        records = json.load(open(path))
        for r in records:
            new_text = remove_field_from_text(r["rubricified_text"], target)
            if new_text != r["rubricified_text"]:
                r["rubricified_text"] = new_text
                total_modified += 1
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  {split}: {len(records)} records updated")

    print(f"Done. Removed '{target}' from {total_modified} rubricified records.")
    return {"removed_field": target, "unique_values": unique_counts[target]}


if __name__ == "__main__":
    main()
