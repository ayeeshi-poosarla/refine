#!/usr/bin/env python3
"""
Action: Add a binary missingness indicator for the rubric field with the
highest missing rate across all filled rubrics.

"Missing" means the field value is exactly one of: "None", "N/A", "No data".

A new field  {SOURCE_FIELD}_IS_MISSING  is added immediately after the source
field in both the rubric template and every filled rubric record.

Value encoding:
  1  — field was missing (None / N/A / No data)
  0  — field had a real value

Modifies in place:
  - rubric_dir/{task}/rubric.json
  - rubricified_dir/{task}/{split}.json

Usage:
  python3 add_missingness_indicator.py \
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

SPLITS  = ("train", "val", "test")
FIELD_RE = re.compile(r'\*\*([A-Z_]+):\*\*\s*(.+)')
MISSING_VALUES = {"None", "N/A", "No data"}


def compute_missing_rates(rubricified_dir: Path, task: str) -> dict[str, tuple[int, int]]:
    """Returns {field: (n_missing, n_total)}."""
    field_missing: dict[str, int] = defaultdict(int)
    field_total:   dict[str, int] = defaultdict(int)
    for split in SPLITS:
        path = rubricified_dir / task / f"{split}.json"
        if not path.exists():
            continue
        for record in json.load(open(path)):
            for field, value in FIELD_RE.findall(record["rubricified_text"]):
                field_total[field] += 1
                if value.strip() in MISSING_VALUES:
                    field_missing[field] += 1
    return {f: (field_missing[f], field_total[f]) for f in field_total}


def find_highest_missing_field(rates: dict[str, tuple[int, int]]) -> str:
    return max(rates, key=lambda f: rates[f][0] / rates[f][1] if rates[f][1] else 0)


def add_field_to_rubric(rubric_instructions: str, source_field: str,
                         new_field: str) -> str:
    pattern = re.compile(
        r'(^\s*\*\s*\*\*' + re.escape(source_field) + r':\*\*[^\n]*)',
        re.MULTILINE,
    )
    replacement = (
        r'\1\n'
        f'*   **{new_field}:** '
        f'[1 if {source_field} is missing (None / N/A / No data), 0 otherwise.]'
    )
    return pattern.sub(replacement, rubric_instructions, count=1)


def add_field_to_text(rubricified_text: str, source_field: str,
                       new_field: str) -> str:
    pattern = re.compile(
        r'(^\s*\*?\s*\*\*' + re.escape(source_field) + r':\*\*\s*)(.+)',
        re.MULTILINE,
    )
    def replace(m):
        value = m.group(2).strip()
        indicator = "1" if value in MISSING_VALUES else "0"
        return m.group(0) + f'\n*   **{new_field}:** {indicator}'

    return pattern.sub(replace, rubricified_text, count=1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--rubric_dir", required=True)
    p.add_argument("--rubricified_dir", required=True)
    args = p.parse_args()

    rubric_dir      = Path(args.rubric_dir)
    rubricified_dir = Path(args.rubricified_dir)

    # 1. Find field with highest missing rate
    rates = compute_missing_rates(rubricified_dir, args.task)
    if not rates:
        print(f"ERROR: no rubricified records found for {args.task}", file=sys.stderr)
        sys.exit(1)

    source_field = find_highest_missing_field(rates)
    n_missing, n_total = rates[source_field]
    new_field = f"{source_field}_IS_MISSING"
    pct = n_missing / n_total * 100
    print(f"Highest missing rate field: {source_field} ({pct:.1f}% missing, {n_missing}/{n_total})")
    print(f"New field: {new_field}")

    # 2. Update rubric template
    rubric_path = rubric_dir / args.task / "rubric.json"
    rubric = json.load(open(rubric_path))
    rubric["rubric_instructions"] = add_field_to_rubric(
        rubric["rubric_instructions"], source_field, new_field
    )
    with open(rubric_path, "w") as f:
        json.dump(rubric, f, indent=2)
    print(f"Updated rubric template: inserted {new_field} after {source_field}")

    # 3. Update every filled rubric record
    total_modified = 0
    for split in SPLITS:
        path = rubricified_dir / args.task / f"{split}.json"
        if not path.exists():
            continue
        records = json.load(open(path))
        for r in records:
            r["rubricified_text"] = add_field_to_text(
                r["rubricified_text"], source_field, new_field
            )
            total_modified += 1
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  {split}: {len(records)} records updated")

    print(f"Done. Added '{new_field}' to {total_modified} records.")
    return {"source_field": source_field, "new_field": new_field,
            "missing_pct": pct}


if __name__ == "__main__":
    main()
