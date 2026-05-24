#!/usr/bin/env python3
"""
Tests for rl.parsing — format-agnostic rubric field parsing and mutation.

Run from the REFINE root:
    python3 -m pytest rl/tests/test_parsing.py -v
"""

import json
import sys
from pathlib import Path

# Ensure the REFINE root is on sys.path regardless of where pytest is invoked.
_REFINE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REFINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_REFINE_ROOT))

from rl.parsing import (
    _detect_format,
    add_field_after,
    parse_rubric_fields,
    remove_field,
    rename_field,
    replace_field_value,
)

# ---------------------------------------------------------------------------
# Fixture-like constants (plain strings — no pytest fixtures needed)
# ---------------------------------------------------------------------------

READMISSION_TEXT = (
    "**PATIENT_DEMOGRAPHICS:**\n"
    "*   **AGE:** 29\n"
    "*   **SEX:** FEMALE\n"
    "*   **RACE:** White\n"
    "\n"
    "**RECENT_HOSPITAL_UTILIZATION:**\n"
    "*   **DAYS_SINCE_LAST_DISCHARGE:** 3\n"
    "*   **NUMBER_OF_ER_VISITS:** 0\n"
)

GUO_LOS_TEXT = (
    "## Clinical Evaluation Rubric\n"
    "\n"
    "**Patient_Demographics:**\n"
    "*   **Field Name:** Age_Years\n"
    "    *   **Extraction:** 72\n"
    "*   **Field Name:** Sex\n"
    "    *   **Extraction:** FEMALE\n"
    "*   **Field Name:** Race_Ethnicity\n"
    "    *   **Extraction:** Black or African American\n"
    "\n"
    "**Vitals:**\n"
    "*   **Field Name:** Heart_Rate_bpm\n"
    "    *   **Extraction:** 95\n"
)

NEW_LUPUS_TEXT = (
    "Patient_Demographics:\n"
    "- Age: 30\n"
    "- Sex: MALE\n"
    "- Race: White\n"
    "\n"
    "Lupus_Specific:\n"
    "- ANA_Status: ORDERED\n"
    "- AntiDsDNA_Status: ABSENT\n"
)

JSON_TEXT = (
    "```json\n"
    "{\n"
    '  "RUBRIC_TEMPLATE": [\n'
    '    {"FIELD_NAME": "PATIENT_AGE", "VALUE": 25},\n'
    '    {"FIELD_NAME": "PATIENT_GENDER", "VALUE": "MALE"},\n'
    '    {"FIELD_NAME": "POTASSIUM", "VALUE": "4.1 mmol/L"}\n'
    "  ]\n"
    "}\n"
    "```\n"
)

# ---------------------------------------------------------------------------
# _detect_format
# ---------------------------------------------------------------------------

def test_detect_readmission():
    assert _detect_format(READMISSION_TEXT) == "readmission"


def test_detect_guo_los():
    assert _detect_format(GUO_LOS_TEXT) == "guo_los"


def test_detect_new_lupus():
    assert _detect_format(NEW_LUPUS_TEXT) == "new_lupus"


def test_detect_json():
    assert _detect_format(JSON_TEXT) == "json"


# ---------------------------------------------------------------------------
# parse_rubric_fields — field names and values
# ---------------------------------------------------------------------------

def test_parse_readmission_field_names():
    fields = parse_rubric_fields(READMISSION_TEXT)
    assert "AGE" in fields
    assert "SEX" in fields
    assert "RACE" in fields
    assert "DAYS_SINCE_LAST_DISCHARGE" in fields
    assert "NUMBER_OF_ER_VISITS" in fields


def test_parse_readmission_field_values():
    fields = parse_rubric_fields(READMISSION_TEXT)
    assert fields["AGE"] == "29"
    assert fields["SEX"] == "FEMALE"
    assert fields["RACE"] == "White"
    assert fields["DAYS_SINCE_LAST_DISCHARGE"] == "3"


def test_parse_readmission_no_section_headers():
    """Section headers like **PATIENT_DEMOGRAPHICS:** should not appear as fields."""
    fields = parse_rubric_fields(READMISSION_TEXT)
    assert "PATIENT_DEMOGRAPHICS" not in fields
    assert "RECENT_HOSPITAL_UTILIZATION" not in fields


def test_parse_guo_los_field_names():
    fields = parse_rubric_fields(GUO_LOS_TEXT)
    assert "AGE_YEARS" in fields
    assert "SEX" in fields
    assert "RACE_ETHNICITY" in fields
    assert "HEART_RATE_BPM" in fields


def test_parse_guo_los_field_values():
    fields = parse_rubric_fields(GUO_LOS_TEXT)
    assert fields["AGE_YEARS"] == "72"
    assert fields["SEX"] == "FEMALE"
    assert fields["RACE_ETHNICITY"] == "Black or African American"
    assert fields["HEART_RATE_BPM"] == "95"


def test_parse_new_lupus_field_names():
    fields = parse_rubric_fields(NEW_LUPUS_TEXT)
    assert "AGE" in fields
    assert "SEX" in fields
    assert "RACE" in fields
    assert "ANA_STATUS" in fields
    assert "ANTIDSDNA_STATUS" in fields


def test_parse_new_lupus_field_values():
    fields = parse_rubric_fields(NEW_LUPUS_TEXT)
    assert fields["AGE"] == "30"
    assert fields["SEX"] == "MALE"
    assert fields["ANA_STATUS"] == "ORDERED"
    assert fields["ANTIDSDNA_STATUS"] == "ABSENT"


def test_parse_json_field_names():
    fields = parse_rubric_fields(JSON_TEXT)
    assert "PATIENT_AGE" in fields
    assert "PATIENT_GENDER" in fields
    assert "POTASSIUM" in fields


def test_parse_json_field_values():
    fields = parse_rubric_fields(JSON_TEXT)
    assert fields["PATIENT_AGE"] == "25"
    assert fields["PATIENT_GENDER"] == "MALE"
    assert fields["POTASSIUM"] == "4.1 mmol/L"


def test_parse_json_normalises_field_names_to_uppercase():
    text = (
        "```json\n"
        '{"RUBRIC_TEMPLATE": [{"FIELD_NAME": "patient_age", "VALUE": 40}]}\n'
        "```\n"
    )
    fields = parse_rubric_fields(text)
    assert "PATIENT_AGE" in fields
    assert fields["PATIENT_AGE"] == "40"


# ---------------------------------------------------------------------------
# remove_field
# ---------------------------------------------------------------------------

def test_remove_field_readmission():
    out = remove_field(READMISSION_TEXT, "AGE")
    fields = parse_rubric_fields(out)
    assert "AGE" not in fields
    assert "SEX" in fields
    # Other content preserved
    assert "PATIENT_DEMOGRAPHICS" in out
    assert "DAYS_SINCE_LAST_DISCHARGE" in fields


def test_remove_field_guo_los():
    out = remove_field(GUO_LOS_TEXT, "SEX")
    fields = parse_rubric_fields(out)
    assert "SEX" not in fields
    assert "AGE_YEARS" in fields
    assert "HEART_RATE_BPM" in fields


def test_remove_field_guo_los_removes_both_lines():
    """Removing a guo_los field should remove both the Field Name line and Extraction line."""
    out = remove_field(GUO_LOS_TEXT, "SEX")
    assert "**Field Name:** Sex" not in out
    assert "**Extraction:** FEMALE" not in out


def test_remove_field_new_lupus():
    out = remove_field(NEW_LUPUS_TEXT, "SEX")
    fields = parse_rubric_fields(out)
    assert "SEX" not in fields
    assert "AGE" in fields
    assert "ANA_STATUS" in fields


def test_remove_field_json():
    out = remove_field(JSON_TEXT, "PATIENT_GENDER")
    fields = parse_rubric_fields(out)
    assert "PATIENT_GENDER" not in fields
    assert "PATIENT_AGE" in fields
    assert "POTASSIUM" in fields


def test_remove_field_case_insensitive_normalisation():
    """remove_field should accept lowercase/mixed field name."""
    out = remove_field(READMISSION_TEXT, "age")
    assert "AGE" not in parse_rubric_fields(out)


# ---------------------------------------------------------------------------
# replace_field_value
# ---------------------------------------------------------------------------

def test_replace_field_value_readmission():
    out = replace_field_value(READMISSION_TEXT, "SEX", "MALE")
    fields = parse_rubric_fields(out)
    assert fields["SEX"] == "MALE"
    # Other fields untouched
    assert fields["AGE"] == "29"


def test_replace_field_value_guo_los():
    out = replace_field_value(GUO_LOS_TEXT, "SEX", "MALE")
    fields = parse_rubric_fields(out)
    assert fields["SEX"] == "MALE"
    assert fields["AGE_YEARS"] == "72"


def test_replace_field_value_new_lupus():
    out = replace_field_value(NEW_LUPUS_TEXT, "AGE", "35")
    fields = parse_rubric_fields(out)
    assert fields["AGE"] == "35"
    assert fields["SEX"] == "MALE"


def test_replace_field_value_json():
    out = replace_field_value(JSON_TEXT, "PATIENT_AGE", "30")
    fields = parse_rubric_fields(out)
    assert fields["PATIENT_AGE"] == "30"
    assert fields["PATIENT_GENDER"] == "MALE"


def test_replace_field_value_preserves_other_fields_readmission():
    out = replace_field_value(READMISSION_TEXT, "RACE", "Hispanic")
    fields = parse_rubric_fields(out)
    assert fields["RACE"] == "Hispanic"
    assert "DAYS_SINCE_LAST_DISCHARGE" in fields


# ---------------------------------------------------------------------------
# add_field_after
# ---------------------------------------------------------------------------

def test_add_field_after_readmission():
    out = add_field_after(READMISSION_TEXT, "AGE", "AGE_GROUP", "Young")
    fields = parse_rubric_fields(out)
    assert "AGE_GROUP" in fields
    assert fields["AGE_GROUP"] == "Young"
    assert fields["AGE"] == "29"


def test_add_field_after_readmission_ordering():
    """New field should appear immediately after the target field in text."""
    out = add_field_after(READMISSION_TEXT, "AGE", "AGE_GROUP", "Young")
    age_pos = out.index("**AGE:**")
    age_group_pos = out.index("**AGE_GROUP:**")
    sex_pos = out.index("**SEX:**")
    assert age_pos < age_group_pos < sex_pos


def test_add_field_after_guo_los():
    out = add_field_after(GUO_LOS_TEXT, "AGE_YEARS", "AGE_GROUP", "Senior")
    fields = parse_rubric_fields(out)
    assert "AGE_GROUP" in fields
    assert fields["AGE_GROUP"] == "Senior"
    assert fields["AGE_YEARS"] == "72"


def test_add_field_after_new_lupus():
    out = add_field_after(NEW_LUPUS_TEXT, "SEX", "SEX_CODE", "M")
    fields = parse_rubric_fields(out)
    assert "SEX_CODE" in fields
    assert fields["SEX_CODE"] == "M"
    assert fields["SEX"] == "MALE"


def test_add_field_after_json():
    out = add_field_after(JSON_TEXT, "PATIENT_AGE", "AGE_GROUP", "Adult")
    fields = parse_rubric_fields(out)
    assert "AGE_GROUP" in fields
    assert fields["AGE_GROUP"] == "Adult"
    assert fields["PATIENT_AGE"] == "25"


def test_add_field_after_json_ordering():
    """New field should appear immediately after the target in the RUBRIC_TEMPLATE list."""
    out = add_field_after(JSON_TEXT, "PATIENT_AGE", "AGE_GROUP", "Adult")
    parsed = json.loads(out.split("```json")[1].split("```")[0])
    names = [item["FIELD_NAME"] for item in parsed["RUBRIC_TEMPLATE"]]
    idx_age = names.index("PATIENT_AGE")
    idx_group = names.index("AGE_GROUP")
    assert idx_group == idx_age + 1


# ---------------------------------------------------------------------------
# rename_field (used by split_highest_entropy_field)
# ---------------------------------------------------------------------------

def test_rename_field_readmission():
    out = rename_field(READMISSION_TEXT, "SEX", "SEX_BIN", "FEMALE")
    fields = parse_rubric_fields(out)
    assert "SEX" not in fields
    assert "SEX_BIN" in fields
    assert fields["SEX_BIN"] == "FEMALE"


def test_rename_field_new_lupus():
    out = rename_field(NEW_LUPUS_TEXT, "AGE", "AGE_BIN", "LOW")
    fields = parse_rubric_fields(out)
    assert "AGE" not in fields
    assert "AGE_BIN" in fields
    assert fields["AGE_BIN"] == "LOW"


def test_rename_field_json():
    out = rename_field(JSON_TEXT, "PATIENT_AGE", "PATIENT_AGE_BIN", "LOW")
    fields = parse_rubric_fields(out)
    assert "PATIENT_AGE" not in fields
    assert "PATIENT_AGE_BIN" in fields
    assert fields["PATIENT_AGE_BIN"] == "LOW"


# ---------------------------------------------------------------------------
# Real data integration tests — load train.json and verify parse results
# ---------------------------------------------------------------------------

_DATA_BASE = _REFINE_ROOT / "data" / "rubric" / "rubricified"


def _load_first_record(task: str) -> str:
    path = _DATA_BASE / task / "train.json"
    with open(path) as f:
        records = json.load(f)
    return records[0]["rubricified_text"]


def test_real_data_guo_readmission():
    text = _load_first_record("guo_readmission")
    fields = parse_rubric_fields(text)
    assert len(fields) >= 3, f"Expected at least 3 fields, got {len(fields)}"
    # Core fields that should always be present
    assert "AGE" in fields
    assert "SEX" in fields
    # Values should not be empty
    for name, val in fields.items():
        assert val.strip() != "", f"Empty value for field {name}"


def test_real_data_guo_los():
    text = _load_first_record("guo_los")
    fields = parse_rubric_fields(text)
    assert len(fields) >= 3, f"Expected at least 3 fields, got {len(fields)}"
    assert "AGE_YEARS" in fields
    assert "SEX" in fields
    for name, val in fields.items():
        assert val.strip() != "", f"Empty value for field {name}"


def test_real_data_new_lupus():
    text = _load_first_record("new_lupus")
    fields = parse_rubric_fields(text)
    assert len(fields) >= 3, f"Expected at least 3 fields, got {len(fields)}"
    assert "AGE" in fields
    assert "SEX" in fields
    for name, val in fields.items():
        assert val.strip() != "", f"Empty value for field {name}"


def test_real_data_lab_hyperkalemia():
    text = _load_first_record("lab_hyperkalemia")
    fields = parse_rubric_fields(text)
    assert len(fields) >= 3, f"Expected at least 3 fields, got {len(fields)}"
    assert "PATIENT_AGE" in fields
    assert "PATIENT_GENDER" in fields
    for name, val in fields.items():
        assert val.strip() != "", f"Empty value for field {name}"


def test_real_data_remove_field_roundtrip_readmission():
    """remove_field on real data should leave other fields intact."""
    text = _load_first_record("guo_readmission")
    fields_before = parse_rubric_fields(text)
    field_to_remove = list(fields_before.keys())[0]
    out = remove_field(text, field_to_remove)
    fields_after = parse_rubric_fields(out)
    assert field_to_remove not in fields_after
    assert len(fields_after) == len(fields_before) - 1


def test_real_data_remove_field_roundtrip_guo_los():
    text = _load_first_record("guo_los")
    fields_before = parse_rubric_fields(text)
    field_to_remove = list(fields_before.keys())[0]
    out = remove_field(text, field_to_remove)
    fields_after = parse_rubric_fields(out)
    assert field_to_remove not in fields_after
    assert len(fields_after) == len(fields_before) - 1


def test_real_data_remove_field_roundtrip_new_lupus():
    text = _load_first_record("new_lupus")
    fields_before = parse_rubric_fields(text)
    field_to_remove = list(fields_before.keys())[0]
    out = remove_field(text, field_to_remove)
    fields_after = parse_rubric_fields(out)
    assert field_to_remove not in fields_after
    assert len(fields_after) == len(fields_before) - 1


def test_real_data_remove_field_roundtrip_lab_hyperkalemia():
    text = _load_first_record("lab_hyperkalemia")
    fields_before = parse_rubric_fields(text)
    field_to_remove = list(fields_before.keys())[0]
    out = remove_field(text, field_to_remove)
    fields_after = parse_rubric_fields(out)
    assert field_to_remove not in fields_after
    assert len(fields_after) == len(fields_before) - 1


def test_real_data_replace_field_value_readmission():
    text = _load_first_record("guo_readmission")
    fields = parse_rubric_fields(text)
    field = list(fields.keys())[0]
    out = replace_field_value(text, field, "TEST_VALUE")
    assert parse_rubric_fields(out)[field] == "TEST_VALUE"


def test_real_data_replace_field_value_guo_los():
    text = _load_first_record("guo_los")
    fields = parse_rubric_fields(text)
    field = list(fields.keys())[0]
    out = replace_field_value(text, field, "TEST_VALUE")
    assert parse_rubric_fields(out)[field] == "TEST_VALUE"


def test_real_data_replace_field_value_new_lupus():
    text = _load_first_record("new_lupus")
    fields = parse_rubric_fields(text)
    field = list(fields.keys())[0]
    out = replace_field_value(text, field, "TEST_VALUE")
    assert parse_rubric_fields(out)[field] == "TEST_VALUE"


def test_real_data_replace_field_value_lab_hyperkalemia():
    text = _load_first_record("lab_hyperkalemia")
    fields = parse_rubric_fields(text)
    field = list(fields.keys())[0]
    out = replace_field_value(text, field, "TEST_VALUE")
    assert parse_rubric_fields(out)[field] == "TEST_VALUE"


def test_real_data_add_field_after_readmission():
    text = _load_first_record("guo_readmission")
    fields = parse_rubric_fields(text)
    after_field = list(fields.keys())[0]
    out = add_field_after(text, after_field, "TEST_NEW_FIELD", "test_val")
    new_fields = parse_rubric_fields(out)
    assert "TEST_NEW_FIELD" in new_fields
    assert new_fields["TEST_NEW_FIELD"] == "test_val"


def test_real_data_add_field_after_guo_los():
    text = _load_first_record("guo_los")
    fields = parse_rubric_fields(text)
    after_field = list(fields.keys())[0]
    out = add_field_after(text, after_field, "TEST_NEW_FIELD", "test_val")
    new_fields = parse_rubric_fields(out)
    assert "TEST_NEW_FIELD" in new_fields
    assert new_fields["TEST_NEW_FIELD"] == "test_val"


def test_real_data_add_field_after_new_lupus():
    text = _load_first_record("new_lupus")
    fields = parse_rubric_fields(text)
    after_field = list(fields.keys())[0]
    out = add_field_after(text, after_field, "TEST_NEW_FIELD", "test_val")
    new_fields = parse_rubric_fields(out)
    assert "TEST_NEW_FIELD" in new_fields
    assert new_fields["TEST_NEW_FIELD"] == "test_val"


def test_real_data_add_field_after_lab_hyperkalemia():
    text = _load_first_record("lab_hyperkalemia")
    fields = parse_rubric_fields(text)
    after_field = list(fields.keys())[0]
    out = add_field_after(text, after_field, "TEST_NEW_FIELD", "test_val")
    new_fields = parse_rubric_fields(out)
    assert "TEST_NEW_FIELD" in new_fields
    assert new_fields["TEST_NEW_FIELD"] == "test_val"
