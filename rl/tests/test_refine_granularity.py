#!/usr/bin/env python3
"""Tests for refine_granularity action."""

import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from rl.actions.refine_granularity import (
    build_value_map,
    collect_nonmissing_values,
    compute_entropy,
    find_binary_fields,
    find_coarsest_binary_field,
    refine_field_in_rubric,
    refine_field_in_text,
    RefineGranularity,
)
from rl.state import RubricState

FIELD_RE = re.compile(r'\*\*([A-Z_]+):\*\*\s*(.+)')
ORDINAL_LEVELS = {"none", "mild", "severe"}


def make_record(patient_id, prediction_time, label, fields: dict) -> dict:
    text = "\n".join(f"*   **{k}:** {v}" for k, v in fields.items())
    return {
        "patient_id": patient_id,
        "prediction_time": prediction_time,
        "label": label,
        "rubricified_text": text,
    }


def make_state(records_by_split: dict, instructions: str) -> RubricState:
    rubric = {
        "task": "test_task",
        "rubric_instructions": instructions,
        "task_query": "test",
        "num_examples": 0,
        "usage": {},
    }
    return RubricState(task="test_task", rubric=rubric, records=records_by_split)


# ── Unit tests ────────────────────────────────────────────────────────────────

def test_collect_nonmissing_values_excludes_sentinels():
    records = {
        "train": [
            make_record(1, "2020-01-01", 1, {"STATUS": "Yes", "LEVEL": "None"}),
            make_record(2, "2020-01-02", 0, {"STATUS": "No",  "LEVEL": "N/A"}),
        ]
    }
    vals = collect_nonmissing_values(records)
    assert vals["STATUS"] == {"Yes", "No"}
    assert vals.get("LEVEL", set()) == set(), "Missing sentinels should be excluded"
    print("PASS test_collect_nonmissing_values_excludes_sentinels")


def test_find_binary_fields():
    field_vals = {
        "BINARY":    {"Yes", "No"},
        "TERNARY":   {"Low", "Medium", "High"},
        "CONSTANT":  {"Yes"},
        "ALSO_BIN":  {"Pos", "Neg"},
    }
    result = find_binary_fields(field_vals)
    assert set(result.keys()) == {"BINARY", "ALSO_BIN"}
    print("PASS test_find_binary_fields")


def test_compute_entropy_skewed():
    # 9 Yes, 1 No → very low entropy
    records = {
        "train": [
            make_record(i, f"2020-01-{i+1:02d}", i % 2,
                        {"F": "Yes" if i < 9 else "No"})
            for i in range(10)
        ]
    }
    e = compute_entropy(records, "F")
    assert e < 0.5, f"Expected low entropy for 9:1 split, got {e:.4f}"
    print("PASS test_compute_entropy_skewed")


def test_compute_entropy_uniform():
    records = {
        "train": [
            make_record(i, f"2020-01-{i+1:02d}", i % 2,
                        {"F": "Yes" if i % 2 else "No"})
            for i in range(10)
        ]
    }
    e = compute_entropy(records, "F")
    assert abs(e - 1.0) < 1e-6, f"Expected entropy=1.0 for 50:50 split, got {e:.4f}"
    print("PASS test_compute_entropy_uniform")


def test_find_coarsest_binary_field_picks_lowest_entropy():
    # SKEWED: 9 Yes, 1 No → low entropy
    # BALANCED: 5 Yes, 5 No → entropy=1.0
    records = {
        "train": [
            make_record(i, f"2020-01-{i+1:02d}", i % 2, {
                "SKEWED":   "Yes" if i < 9 else "No",
                "BALANCED": "Yes" if i % 2  else "No",
            })
            for i in range(10)
        ]
    }
    binary_fields = {"SKEWED": {"Yes", "No"}, "BALANCED": {"Yes", "No"}}
    result = find_coarsest_binary_field(records, binary_fields)
    assert result == "SKEWED", f"Expected SKEWED, got {result}"
    print("PASS test_find_coarsest_binary_field_picks_lowest_entropy")


def test_build_value_map_uses_label_correlation():
    # "Yes" patients all have label=1, "No" patients all have label=0
    # → "Yes" should map to "severe", "No" to "none"
    records = {
        "train": [
            make_record(1, "2020-01-01", 1, {"FLAG": "Yes"}),
            make_record(2, "2020-01-02", 1, {"FLAG": "Yes"}),
            make_record(3, "2020-01-03", 0, {"FLAG": "No"}),
            make_record(4, "2020-01-04", 0, {"FLAG": "No"}),
        ]
    }
    vmap = build_value_map(records, "FLAG", {"Yes", "No"})
    assert vmap["Yes"] == "severe", f"Expected Yes→severe, got {vmap}"
    assert vmap["No"]  == "none",   f"Expected No→none, got {vmap}"
    print("PASS test_build_value_map_uses_label_correlation")


def test_build_value_map_falls_back_to_alphabetical_on_tie():
    # Equal label means across both values → alphabetical tiebreak
    records = {
        "train": [
            make_record(1, "2020-01-01", 1, {"FLAG": "Alpha"}),
            make_record(2, "2020-01-02", 0, {"FLAG": "Alpha"}),
            make_record(3, "2020-01-03", 1, {"FLAG": "Zeta"}),
            make_record(4, "2020-01-04", 0, {"FLAG": "Zeta"}),
        ]
    }
    vmap = build_value_map(records, "FLAG", {"Alpha", "Zeta"})
    # Alpha < Zeta alphabetically → Alpha=none, Zeta=severe
    assert vmap["Alpha"] == "none"
    assert vmap["Zeta"]  == "severe"
    print("PASS test_build_value_map_falls_back_to_alphabetical_on_tie")


def test_refine_field_in_rubric_replaces_description():
    instructions = (
        "*   **OTHER:** [stays]\n"
        "*   **FLAG:** [Yes/No]\n"
    )
    vmap = {"No": "none", "Yes": "severe"}
    result = refine_field_in_rubric(instructions, "FLAG", vmap)
    assert "**FLAG:**" in result
    assert "none" in result
    assert "mild" in result
    assert "severe" in result
    assert "Yes/No" not in result
    assert "OTHER" in result
    print("PASS test_refine_field_in_rubric_replaces_description")


def test_refine_field_in_text_remaps_values():
    text = (
        "*   **OTHER:** present\n"
        "*   **FLAG:** Yes\n"
    )
    vmap = {"No": "none", "Yes": "severe"}
    result = refine_field_in_text(text, "FLAG", vmap)
    fields = dict(FIELD_RE.findall(result))
    assert fields["FLAG"] == "severe", f"Expected severe, got {fields['FLAG']}"
    assert fields["OTHER"] == "present"
    print("PASS test_refine_field_in_text_remaps_values")


def test_refine_field_in_text_preserves_missing():
    text = "*   **FLAG:** None\n"
    vmap = {"No": "none", "Yes": "severe"}
    result = refine_field_in_text(text, "FLAG", vmap)
    fields = dict(FIELD_RE.findall(result))
    # "None" is not in vmap → left unchanged
    assert fields["FLAG"] == "None"
    print("PASS test_refine_field_in_text_preserves_missing")


def test_apply_refines_coarsest_binary_field():
    # FLAG is 9:1 skewed (coarsest binary); RISK is 50:50 binary
    train = [
        make_record(i, f"2020-01-{i+1:02d}", int(i >= 9), {
            "FLAG": "Yes" if i < 9 else "No",   # 9 No-label=0, 1 Yes-label=1
            "RISK": "High" if i % 2 else "Low",
        })
        for i in range(10)
    ]
    test = [
        make_record(i + 10, f"2020-02-{i+1:02d}", i % 2, {
            "FLAG": "Yes" if i % 3 else "No",
            "RISK": "High" if i % 2 else "Low",
        })
        for i in range(6)
    ]
    instructions = (
        "*   **FLAG:** [Yes/No]\n"
        "*   **RISK:** [High/Low]\n"
    )
    state     = make_state({"train": train, "test": test}, instructions)
    new_state = RefineGranularity().apply(state)

    meta = new_state.rubric["_last_action"]
    assert meta["field"] == "FLAG", f"Expected FLAG, got {meta['field']}"
    assert set(meta["value_map"].values()) == {"none", "severe"}

    instr = new_state.rubric["rubric_instructions"]
    assert "none" in instr and "mild" in instr and "severe" in instr
    assert "RISK" in instr

    for recs in new_state.records.values():
        for r in recs:
            fields = dict(FIELD_RE.findall(r["rubricified_text"]))
            assert fields["FLAG"] in ORDINAL_LEVELS, f"Unexpected value: {fields['FLAG']}"
            assert fields["RISK"] in {"High", "Low"}
    print("PASS test_apply_refines_coarsest_binary_field")


def test_apply_does_not_mutate_input():
    records = {
        "train": [make_record(i, f"2020-01-{i+1:02d}", i % 2,
                              {"FLAG": "Yes" if i % 5 else "No", "RISK": "High"})
                  for i in range(10)],
        "test":  [make_record(i + 10, f"2020-02-{i+1:02d}", i % 2,
                              {"FLAG": "No", "RISK": "Low"})
                  for i in range(4)],
    }
    state = make_state(records, "*   **FLAG:** [Yes/No]\n*   **RISK:** [High/Low]\n")
    original_texts = [r["rubricified_text"] for r in state.records["train"]]
    original_instr = state.rubric["rubric_instructions"]

    RefineGranularity().apply(state)

    assert state.rubric["rubric_instructions"] == original_instr
    for orig, r in zip(original_texts, state.records["train"]):
        assert r["rubricified_text"] == orig
    print("PASS test_apply_does_not_mutate_input")


def test_apply_raises_when_no_binary_fields():
    records = {
        "train": [make_record(1, "2020-01-01", 1,
                              {"MULTI": "A", "ALSO": "B"})],
        "test":  [make_record(2, "2020-02-01", 0,
                              {"MULTI": "C", "ALSO": "D"})],
    }
    # train+test together have 2 unique values per field, but let's make them ternary
    records["train"].append(make_record(3, "2020-01-03", 0, {"MULTI": "C", "ALSO": "D"}))
    records["test"].append( make_record(4, "2020-02-02", 1, {"MULTI": "E", "ALSO": "F"}))
    state = make_state(records, "*   **MULTI:** [A/B/C]\n*   **ALSO:** [B/D/F]\n")
    try:
        RefineGranularity().apply(state)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("PASS test_apply_raises_when_no_binary_fields")


def test_end_to_end_via_subprocess():
    with tempfile.TemporaryDirectory() as rub_tmp, \
         tempfile.TemporaryDirectory() as rubricified_tmp:
        rub_tmp         = Path(rub_tmp)
        rubricified_tmp = Path(rubricified_tmp)
        task            = "test_task"

        # SKEWED: 8 No, 2 Yes → coarsest; RISK: balanced
        train_recs = [
            make_record(i, f"2020-01-{i+1:02d}", int(i >= 8), {
                "SKEWED": "Yes" if i >= 8 else "No",
                "RISK":   "High" if i % 2 else "Low",
            })
            for i in range(10)
        ]
        test_recs = [
            make_record(i + 10, f"2020-02-{i+1:02d}", i % 2, {
                "SKEWED": "Yes" if i % 3 else "No",
                "RISK":   "High" if i % 2 else "Low",
            })
            for i in range(6)
        ]
        for split, recs in [("train", train_recs), ("test", test_recs)]:
            d = rubricified_tmp / task
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{split}.json").write_text(json.dumps(recs))

        rubric = {
            "task": task,
            "rubric_instructions": "*   **SKEWED:** [Yes/No]\n*   **RISK:** [High/Low]\n",
            "task_query": "test", "num_examples": 0, "usage": {},
        }
        d = rub_tmp / task
        d.mkdir(parents=True, exist_ok=True)
        (d / "rubric.json").write_text(json.dumps(rubric))

        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
        result = subprocess.run(
            [sys.executable,
             str(Path(__file__).resolve().parents[1] / "actions" / "refine_granularity.py"),
             "--task", task,
             "--rubric_dir", str(rub_tmp),
             "--rubricified_dir", str(rubricified_tmp)],
            capture_output=True, text=True, env=env,
        )
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        assert "entropy" in result.stdout.lower()
        assert "SKEWED" in result.stdout

        updated_rubric = json.loads((rub_tmp / task / "rubric.json").read_text())
        instr = updated_rubric["rubric_instructions"]
        assert "none" in instr and "mild" in instr and "severe" in instr
        assert "RISK" in instr

        for split in ("train", "test"):
            updated = json.loads((rubricified_tmp / task / f"{split}.json").read_text())
            for r in updated:
                fields = dict(FIELD_RE.findall(r["rubricified_text"]))
                assert fields["SKEWED"] in ORDINAL_LEVELS, f"Bad value: {fields['SKEWED']}"

    print("PASS test_end_to_end_via_subprocess")


# ── Parsing correctness: right values from rubricified_text ──────────────────

def test_collect_nonmissing_values_multiword_values():
    # Values like "No evidence of disease" and "Stage IV" must be captured whole.
    records = {
        "train": [
            {
                "patient_id": 1, "prediction_time": "2020-01-01", "label": 1,
                "rubricified_text": (
                    "*   **DISEASE_STATUS:** Active disease\n"
                    "*   **STAGE:** Stage IV\n"
                    "*   **FLAG:** Yes\n"
                ),
            },
            {
                "patient_id": 2, "prediction_time": "2020-01-02", "label": 0,
                "rubricified_text": (
                    "*   **DISEASE_STATUS:** No evidence of disease\n"
                    "*   **STAGE:** Stage I\n"
                    "*   **FLAG:** No\n"
                ),
            },
        ]
    }
    vals = collect_nonmissing_values(records)
    assert "Active disease" in vals["DISEASE_STATUS"], \
        f"Multiword value truncated; got: {vals['DISEASE_STATUS']}"
    assert "No evidence of disease" in vals["DISEASE_STATUS"], \
        f"Multiword value truncated; got: {vals['DISEASE_STATUS']}"
    assert vals["STAGE"] == {"Stage IV", "Stage I"}, \
        f"STAGE values wrong: {vals['STAGE']}"
    assert vals["FLAG"] == {"Yes", "No"}
    print("PASS test_collect_nonmissing_values_multiword_values")


def test_collect_nonmissing_values_no_prefix_contamination():
    # RISK and READMISSION_RISK share a suffix; values must not bleed across fields.
    records = {
        "train": [
            {
                "patient_id": 1, "prediction_time": "2020-01-01", "label": 1,
                "rubricified_text": (
                    "*   **READMISSION_RISK:** High\n"
                    "*   **RISK:** Low\n"
                ),
            },
            {
                "patient_id": 2, "prediction_time": "2020-01-02", "label": 0,
                "rubricified_text": (
                    "*   **READMISSION_RISK:** Low\n"
                    "*   **RISK:** High\n"
                ),
            },
        ]
    }
    vals = collect_nonmissing_values(records)
    assert vals["READMISSION_RISK"] == {"High", "Low"}, \
        f"READMISSION_RISK captured wrong: {vals.get('READMISSION_RISK')}"
    assert vals["RISK"] == {"Low", "High"}, \
        f"RISK captured wrong: {vals.get('RISK')}"
    # No value from one field should appear under the other's key as a contamination
    # (both happen to have the same values here, but the sets are independently built)
    assert "READMISSION_RISK" in vals and "RISK" in vals
    print("PASS test_collect_nonmissing_values_no_prefix_contamination")


def test_refine_field_in_text_no_prefix_match():
    # Remapping FLAG must not touch RED_FLAG.
    text = (
        "*   **RED_FLAG:** Yes\n"
        "*   **FLAG:** No\n"
    )
    vmap = {"No": "none", "Yes": "severe"}
    result = refine_field_in_text(text, "FLAG", vmap)
    fields = dict(FIELD_RE.findall(result))
    assert fields.get("RED_FLAG") == "Yes", \
        f"RED_FLAG was modified unexpectedly: {fields}"
    assert fields.get("FLAG") == "none", \
        f"FLAG not remapped: {fields}"
    print("PASS test_refine_field_in_text_no_prefix_match")


def test_refine_field_in_text_both_values_remapped():
    # Every record — whether it holds the "none" or "severe" value — must be remapped.
    records_text = [
        "*   **RISK:** High\n*   **FLAG:** Yes\n",
        "*   **RISK:** Low\n*   **FLAG:** No\n",
        "*   **RISK:** High\n*   **FLAG:** Yes\n",
        "*   **RISK:** Low\n*   **FLAG:** No\n",
    ]
    vmap = {"No": "none", "Yes": "severe"}
    for text in records_text:
        result = refine_field_in_text(text, "FLAG", vmap)
        fields = dict(FIELD_RE.findall(result))
        assert fields["FLAG"] in ("none", "severe"), \
            f"FLAG not remapped: {fields['FLAG']!r} in {text!r}"
        assert fields["RISK"] in ("High", "Low"), "RISK modified unexpectedly"
    print("PASS test_refine_field_in_text_both_values_remapped")


# ── Rubric format correctness ─────────────────────────────────────────────────

def test_refine_field_in_rubric_exact_bullet_format():
    # The edited FLAG line must follow the exact rubric bullet format:
    #   *   **FLAG:** [none if X; mild if intermediate presentation; severe if Y]
    instructions = (
        "*   **OTHER:** [Stays unchanged]\n"
        "*   **FLAG:** [Yes/No]\n"
        "*   **ALSO:** [Also unchanged]\n"
    )
    vmap = {"No": "none", "Yes": "severe"}
    result = refine_field_in_rubric(instructions, "FLAG", vmap)

    flag_lines = [l for l in result.splitlines() if "**FLAG:**" in l]
    assert len(flag_lines) == 1, \
        f"Expected exactly 1 FLAG line, got {len(flag_lines)}: {flag_lines}"
    line = flag_lines[0]

    # Must start with '*   **FLAG:** [' and end with ']'
    assert re.match(r'^\*   \*\*FLAG:\*\* \[.+\]$', line), \
        f"Wrong bullet format: {line!r}"
    # Must contain all three ordinal levels
    assert "none" in line and "mild" in line and "severe" in line, \
        f"Missing ordinal level in: {line!r}"
    print("PASS test_refine_field_in_rubric_exact_bullet_format")


def test_refine_field_in_rubric_preserves_surrounding_line_format():
    # Lines for fields other than the target must be byte-for-byte unchanged.
    first = "*   **FIRST_FIELD:** [First description]"
    last  = "*   **LAST_FIELD:** [Last description]"
    instructions = f"{first}\n*   **FLAG:** [Yes/No]\n{last}\n"

    vmap = {"No": "none", "Yes": "severe"}
    result = refine_field_in_rubric(instructions, "FLAG", vmap)

    lines = [l for l in result.splitlines() if l.strip()]
    assert any(l == first for l in lines), \
        f"FIRST_FIELD line changed: {[l for l in lines if 'FIRST' in l]}"
    assert any(l == last for l in lines), \
        f"LAST_FIELD line changed: {[l for l in lines if 'LAST' in l]}"
    print("PASS test_refine_field_in_rubric_preserves_surrounding_line_format")


if __name__ == "__main__":
    test_collect_nonmissing_values_excludes_sentinels()
    test_find_binary_fields()
    test_compute_entropy_skewed()
    test_compute_entropy_uniform()
    test_find_coarsest_binary_field_picks_lowest_entropy()
    test_build_value_map_uses_label_correlation()
    test_build_value_map_falls_back_to_alphabetical_on_tie()
    test_refine_field_in_rubric_replaces_description()
    test_refine_field_in_text_remaps_values()
    test_refine_field_in_text_preserves_missing()
    test_apply_refines_coarsest_binary_field()
    test_apply_does_not_mutate_input()
    test_apply_raises_when_no_binary_fields()
    test_end_to_end_via_subprocess()
    test_collect_nonmissing_values_multiword_values()
    test_collect_nonmissing_values_no_prefix_contamination()
    test_refine_field_in_text_no_prefix_match()
    test_refine_field_in_text_both_values_remapped()
    test_refine_field_in_rubric_exact_bullet_format()
    test_refine_field_in_rubric_preserves_surrounding_line_format()
    print("\nAll tests passed.")
