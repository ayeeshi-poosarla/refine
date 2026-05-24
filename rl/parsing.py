#!/usr/bin/env python3
"""
Format-agnostic rubric field parsing and mutation.

Supports 4 rubricified_text formats:
  - 'readmission' : **UPPERCASE_FIELD:** value  (guo_readmission)
  - 'guo_los'     : **Field Name:** name / **Extraction:** value pairs
  - 'new_lupus'   : - Field_Name: value  (dash-prefixed key-value)
  - 'json'        : ```json {"RUBRIC_TEMPLATE": [...]} ```

Public API
----------
parse_rubric_fields(text)           -> dict[str, str]
remove_field(text, field)           -> str
replace_field_value(text, field, new_value) -> str
add_field_after(text, after_field, new_field, value) -> str
"""

import json
import re

# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r'```json\s*\{', re.DOTALL)
_READMISSION_RE = re.compile(r'^\s*\*\s*\*\*[A-Z][A-Z0-9_]+:\*\*\s*.+', re.MULTILINE)
_GUO_LOS_RE = re.compile(r'^\s*\*\s*\*\*Field Name:\*\*', re.MULTILINE | re.IGNORECASE)
_NEW_LUPUS_RE = re.compile(r'^\s*-\s+\w[\w\s]*:\s+\S', re.MULTILINE)


def _detect_format(text: str) -> str:
    """Return the format name: 'readmission', 'guo_los', 'new_lupus', 'json', or 'unknown'."""
    if _JSON_BLOCK_RE.search(text):
        return 'json'
    if _GUO_LOS_RE.search(text):
        return 'guo_los'
    if _READMISSION_RE.search(text):
        return 'readmission'
    if _NEW_LUPUS_RE.search(text):
        return 'new_lupus'
    return 'unknown'


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_field_name(name: str) -> str:
    """Convert field name to UPPERCASE with underscores (no spaces/hyphens)."""
    return re.sub(r'[\s\-]+', '_', name.strip()).upper()


# ---------------------------------------------------------------------------
# Format: readmission  (**UPPERCASE_FIELD:** value)
# ---------------------------------------------------------------------------

# Matches leaf field lines: optional *, then **FIELD:** value
# Header-only lines (e.g. **SECTION:**\n) have no value after the colon.
# Use [^\S\n]* (horizontal whitespace only) so we don't span across newlines.
_RM_FIELD_RE = re.compile(r'^\s*\*?\s*\*\*([A-Z][A-Z0-9_]+):\*\*[^\S\n]*(.+)', re.MULTILINE)


def _parse_readmission(text: str) -> dict[str, str]:
    result = {}
    for name, value in _RM_FIELD_RE.findall(text):
        result[_normalise_field_name(name)] = value.strip()
    return result


def _remove_readmission(text: str, field: str) -> str:
    pattern = re.compile(
        r'^\s*\*?\s*\*\*' + re.escape(field) + r':\*\*[^\n]*\n?',
        re.MULTILINE,
    )
    return pattern.sub('', text)


def _replace_readmission(text: str, field: str, new_value: str) -> str:
    def _repl(m):
        # Preserve leading whitespace + bullet style from original
        return m.group(1) + new_value + '\n'
    pattern = re.compile(
        r'^(\s*\*?\s*\*\*' + re.escape(field) + r':\*\*[^\S\n]*).*\n?',
        re.MULTILINE,
    )
    return pattern.sub(_repl, text)


def _add_after_readmission(text: str, after_field: str, new_field: str, value: str) -> str:
    pattern = re.compile(
        r'(^\s*\*?\s*\*\*' + re.escape(after_field) + r':\*\*[^\n]*)',
        re.MULTILINE,
    )
    replacement = r'\1' + f'\n*   **{new_field}:** {value}'
    return pattern.sub(replacement, text, count=1)


# ---------------------------------------------------------------------------
# Format: guo_los  (**Field Name:** name / **Extraction:** value)
# ---------------------------------------------------------------------------

# Matches a Field Name line followed immediately by an Extraction line.
_LOS_PAIR_RE = re.compile(
    r'^\s*\*\s*\*\*Field Name:\*\*\s*(.+?)\s*\n'
    r'\s*\*\s*\*\*Extraction:\*\*\s*(.+)',
    re.MULTILINE | re.IGNORECASE,
)


def _parse_guo_los(text: str) -> dict[str, str]:
    result = {}
    for name, value in _LOS_PAIR_RE.findall(text):
        result[_normalise_field_name(name)] = value.strip()
    return result


def _remove_guo_los(text: str, field: str) -> str:
    """Remove the Field Name + Extraction pair for a given (normalised) field name."""
    # We need to match whatever the original field name looks like in text.
    # Since _normalise_field_name converts to UPPERCASE+underscores we can't
    # just do a literal match. Instead we iterate over pairs to find it.
    pairs = list(_LOS_PAIR_RE.finditer(text))
    for m in reversed(pairs):
        if _normalise_field_name(m.group(1)) == field:
            # Remove from start of match to the end of Extraction line (+ newline)
            start = m.start()
            end = m.end()
            # Include trailing newline if present
            if end < len(text) and text[end] == '\n':
                end += 1
            text = text[:start] + text[end:]
    return text


def _replace_guo_los(text: str, field: str, new_value: str) -> str:
    def _repl(m):
        if _normalise_field_name(m.group(1)) == field:
            # Reconstruct the pair with the new extraction value
            indent_fn = re.match(r'(\s*\*\s*)', m.group(0)).group(1)
            inner_indent = '    '
            fn_line = f'{indent_fn}**Field Name:** {m.group(1)}\n'
            ex_line = f'{inner_indent}*   **Extraction:** {new_value}'
            return fn_line + ex_line
        return m.group(0)
    return _LOS_PAIR_RE.sub(_repl, text)


def _add_after_guo_los(text: str, after_field: str, new_field: str, value: str) -> str:
    """Insert a new Field Name / Extraction pair after the matched pair."""
    pairs = list(_LOS_PAIR_RE.finditer(text))
    for m in reversed(pairs):
        if _normalise_field_name(m.group(1)) == after_field:
            insert_pos = m.end()
            new_pair = (
                f'\n*   **Field Name:** {new_field}\n'
                f'    *   **Extraction:** {value}'
            )
            text = text[:insert_pos] + new_pair + text[insert_pos:]
            break
    return text


# ---------------------------------------------------------------------------
# Format: new_lupus  (- Field_Name: value)
# ---------------------------------------------------------------------------

_LUPUS_FIELD_RE = re.compile(r'^-\s+([\w][\w\s\-]*?):\s+(.+)', re.MULTILINE)


def _parse_new_lupus(text: str) -> dict[str, str]:
    result = {}
    for name, value in _LUPUS_FIELD_RE.findall(text):
        result[_normalise_field_name(name)] = value.strip()
    return result


def _remove_new_lupus(text: str, field: str) -> str:
    lines = text.split('\n')
    out = []
    for line in lines:
        m = _LUPUS_FIELD_RE.match(line)
        if m and _normalise_field_name(m.group(1)) == field:
            continue
        out.append(line)
    return '\n'.join(out)


def _replace_new_lupus(text: str, field: str, new_value: str) -> str:
    lines = text.split('\n')
    out = []
    for line in lines:
        m = _LUPUS_FIELD_RE.match(line)
        if m and _normalise_field_name(m.group(1)) == field:
            # Preserve original field name spelling and dash style
            out.append(f'- {m.group(1)}: {new_value}')
        else:
            out.append(line)
    return '\n'.join(out)


def _add_after_new_lupus(text: str, after_field: str, new_field: str, value: str) -> str:
    lines = text.split('\n')
    out = []
    for line in lines:
        out.append(line)
        m = _LUPUS_FIELD_RE.match(line)
        if m and _normalise_field_name(m.group(1)) == after_field:
            out.append(f'- {new_field}: {value}')
    return '\n'.join(out)


# ---------------------------------------------------------------------------
# Format: json  (```json {"RUBRIC_TEMPLATE": [...]} ```)
# ---------------------------------------------------------------------------

_JSON_OUTER_RE = re.compile(r'(```json\s*)(\{.*?\})\s*(```)', re.DOTALL)


def _extract_json_block(text: str):
    """Return (prefix, dict_obj, suffix, pre_text, post_text) or None."""
    m = _JSON_OUTER_RE.search(text)
    if not m:
        return None
    pre = text[:m.start()]
    post = text[m.end():]
    obj = json.loads(m.group(2))
    return pre, obj, post, m.group(1), m.group(3)


def _render_json_block(pre: str, obj: dict, post: str, fence_open: str, fence_close: str) -> str:
    body = json.dumps(obj, indent=2)
    return pre + fence_open + body + '\n' + fence_close + post


def _parse_json(text: str) -> dict[str, str]:
    result = _extract_json_block(text)
    if result is None:
        return {}
    _, obj, _, _, _ = result
    items = obj.get('RUBRIC_TEMPLATE', [])
    out = {}
    for item in items:
        field = _normalise_field_name(str(item.get('FIELD_NAME', '')))
        value = item.get('VALUE', '')
        if field:
            out[field] = str(value) if not isinstance(value, list) else json.dumps(value)
    return out


def _remove_json(text: str, field: str) -> str:
    result = _extract_json_block(text)
    if result is None:
        return text
    pre, obj, post, fence_open, fence_close = result
    items = obj.get('RUBRIC_TEMPLATE', [])
    obj['RUBRIC_TEMPLATE'] = [
        it for it in items
        if _normalise_field_name(str(it.get('FIELD_NAME', ''))) != field
    ]
    return _render_json_block(pre, obj, post, fence_open, fence_close)


def _replace_json(text: str, field: str, new_value: str) -> str:
    result = _extract_json_block(text)
    if result is None:
        return text
    pre, obj, post, fence_open, fence_close = result
    for item in obj.get('RUBRIC_TEMPLATE', []):
        if _normalise_field_name(str(item.get('FIELD_NAME', ''))) == field:
            item['VALUE'] = new_value
    return _render_json_block(pre, obj, post, fence_open, fence_close)


def _add_after_json(text: str, after_field: str, new_field: str, value: str) -> str:
    result = _extract_json_block(text)
    if result is None:
        return text
    pre, obj, post, fence_open, fence_close = result
    items = obj.get('RUBRIC_TEMPLATE', [])
    new_items = []
    for item in items:
        new_items.append(item)
        if _normalise_field_name(str(item.get('FIELD_NAME', ''))) == after_field:
            new_items.append({'FIELD_NAME': new_field, 'VALUE': value})
    obj['RUBRIC_TEMPLATE'] = new_items
    return _render_json_block(pre, obj, post, fence_open, fence_close)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_rubric_fields(text: str) -> dict[str, str]:
    """Auto-detect format and return {FIELD_NAME: value}.

    Field names are normalised to UPPERCASE with underscores.
    Section headers (lines with no value) are filtered out automatically.
    """
    fmt = _detect_format(text)
    if fmt == 'readmission':
        return _parse_readmission(text)
    if fmt == 'guo_los':
        return _parse_guo_los(text)
    if fmt == 'new_lupus':
        return _parse_new_lupus(text)
    if fmt == 'json':
        return _parse_json(text)
    # Fallback: try readmission-style regex
    return _parse_readmission(text)


def remove_field(text: str, field: str) -> str:
    """Remove a field and its value line(s) from text, preserving format."""
    field = _normalise_field_name(field)
    fmt = _detect_format(text)
    if fmt == 'readmission':
        return _remove_readmission(text, field)
    if fmt == 'guo_los':
        return _remove_guo_los(text, field)
    if fmt == 'new_lupus':
        return _remove_new_lupus(text, field)
    if fmt == 'json':
        return _remove_json(text, field)
    return _remove_readmission(text, field)


def replace_field_value(text: str, field: str, new_value: str) -> str:
    """Replace the value of an existing field in-place."""
    field = _normalise_field_name(field)
    fmt = _detect_format(text)
    if fmt == 'readmission':
        return _replace_readmission(text, field, new_value)
    if fmt == 'guo_los':
        return _replace_guo_los(text, field, new_value)
    if fmt == 'new_lupus':
        return _replace_new_lupus(text, field, new_value)
    if fmt == 'json':
        return _replace_json(text, field, new_value)
    return _replace_readmission(text, field, new_value)


def add_field_after(text: str, after_field: str, new_field: str, value: str) -> str:
    """Insert a new field immediately after an existing one, using the same format."""
    after_field = _normalise_field_name(after_field)
    fmt = _detect_format(text)
    if fmt == 'readmission':
        return _add_after_readmission(text, after_field, new_field, value)
    if fmt == 'guo_los':
        return _add_after_guo_los(text, after_field, new_field, value)
    if fmt == 'new_lupus':
        return _add_after_new_lupus(text, after_field, new_field, value)
    if fmt == 'json':
        return _add_after_json(text, after_field, new_field, value)
    return _add_after_readmission(text, after_field, new_field, value)


# ---------------------------------------------------------------------------
# rename_field: replace field name and value simultaneously (for split action)
# ---------------------------------------------------------------------------

def _rename_readmission(text: str, old_field: str, new_field: str, new_value: str) -> str:
    def _repl(m):
        return f'*   **{new_field}:** {new_value}\n'
    pattern = re.compile(
        r'^\s*\*?\s*\*\*' + re.escape(old_field) + r':\*\*[^\n]*\n?',
        re.MULTILINE,
    )
    return pattern.sub(_repl, text, count=1)


def _rename_guo_los(text: str, old_field: str, new_field: str, new_value: str) -> str:
    pairs = list(_LOS_PAIR_RE.finditer(text))
    for m in reversed(pairs):
        if _normalise_field_name(m.group(1)) == old_field:
            indent_fn = re.match(r'(\s*\*\s*)', m.group(0)).group(1)
            fn_line = f'{indent_fn}**Field Name:** {new_field}\n'
            ex_line = f'    *   **Extraction:** {new_value}'
            text = text[:m.start()] + fn_line + ex_line + text[m.end():]
            break
    return text


def _rename_new_lupus(text: str, old_field: str, new_field: str, new_value: str) -> str:
    lines = text.split('\n')
    out = []
    for line in lines:
        m = _LUPUS_FIELD_RE.match(line)
        if m and _normalise_field_name(m.group(1)) == old_field:
            out.append(f'- {new_field}: {new_value}')
        else:
            out.append(line)
    return '\n'.join(out)


def _rename_json(text: str, old_field: str, new_field: str, new_value: str) -> str:
    result = _extract_json_block(text)
    if result is None:
        return text
    pre, obj, post, fence_open, fence_close = result
    for item in obj.get('RUBRIC_TEMPLATE', []):
        if _normalise_field_name(str(item.get('FIELD_NAME', ''))) == old_field:
            item['FIELD_NAME'] = new_field
            item['VALUE'] = new_value
    return _render_json_block(pre, obj, post, fence_open, fence_close)


def rename_field(text: str, old_field: str, new_field: str, new_value: str) -> str:
    """Replace old_field with new_field and set new_value, in-place (used by split action)."""
    old_field = _normalise_field_name(old_field)
    fmt = _detect_format(text)
    if fmt == 'readmission':
        return _rename_readmission(text, old_field, new_field, new_value)
    if fmt == 'guo_los':
        return _rename_guo_los(text, old_field, new_field, new_value)
    if fmt == 'new_lupus':
        return _rename_new_lupus(text, old_field, new_field, new_value)
    if fmt == 'json':
        return _rename_json(text, old_field, new_field, new_value)
    return _rename_readmission(text, old_field, new_field, new_value)
