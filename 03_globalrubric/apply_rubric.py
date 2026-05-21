#!/usr/bin/env python3
"""
Apply a task-specific rubric to every patient EHR via Gemini 2.5 Flash batch inference.

For each task, this script:
  1. Loads the rubric instructions (from create_rubric.py).
  2. Loads serialized patient records for the requested splits.
  3. Submits all records as a single Gemini inline batch job.
  4. Polls until complete, then saves rubricified JSONs per split.

Inline batch: requests are passed directly as list[InlinedRequest]; responses
come back in the same order via batch.dest.inlined_responses. No GCS or file
upload required.

Inputs:
  --rubric_dir     : Directory with {task}/rubric.json.
  --serialized_dir : Naivetext serialized dir (data/serialized/naivetext).
  --output_dir     : Where to write rubricified JSONs.
  --tasks          : Space-separated list (default: all 15).
  --splits         : Which splits to process (default: train val test).

Outputs:
  {output_dir}/{task}/{split}.json
  Each record has: patient_id, prediction_time, task, split, label,
                   rubricified_text.

Requires: GOOGLE_API_KEY environment variable.

Connects to:
  - Upstream  : create_rubric.py, 01_serialize (naivetext)
  - Downstream: create_globalrubric_sft.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from loguru import logger
from google import genai
from google.genai import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.tasks import TASKS, ALL_TASK_NAMES
from config.gemini import GeminiConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SYSTEM_PROMPT = (
    "You are a medical expert AI assistant specializing in "
    "structured clinical evaluation."
)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_transform_prompt(ehr_text: str, rubric_instructions: str,
                             task_query: str) -> str:
    return f"""You are a medical data extraction specialist. Your job is to read a patient's EHR and fill in a structured rubric template.

## Task
{task_query}

## Rubric Template (follow this exactly)

{rubric_instructions}

## Patient EHR

{ehr_text}

## Instructions
Fill in every field of the rubric template above using ONLY information from this patient's EHR. Rules:
- Follow the exact field order and section structure of the rubric.
- Be concise: use short phrases, numbers, and dates. Do not write paragraphs.
- If data for a field is not present in the EHR, write "No data".
- Do NOT add commentary, predictions, risk assessments, or conclusions.
- Do NOT include any information not found in the EHR above.

Rubric output:"""


# ---------------------------------------------------------------------------
# Resumability helpers
# ---------------------------------------------------------------------------

def _load_done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    try:
        with open(path) as f:
            data = json.load(f)
        return {(e["patient_id"], e["prediction_time"]) for e in data}
    except Exception:
        return set()


def _load_existing(path: Path) -> list:
    if not path.exists():
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Core batch routine
# ---------------------------------------------------------------------------

def apply_rubric_batch(
    task: str,
    splits: List[str],
    rubric_dir: Path,
    serialized_dir: Path,
    output_dir: Path,
    config: GeminiConfig,
) -> None:
    rubric_path = rubric_dir / task / "rubric.json"
    if not rubric_path.exists():
        logger.warning(f"No rubric for {task} at {rubric_path}, skipping")
        return

    with open(rubric_path) as f:
        rubric_data = json.load(f)
    rubric_text = rubric_data["rubric_instructions"]
    task_query = TASKS[task]

    # ---- Collect todo records across all splits ----
    # order_index maps position in flat list → (split, original record)
    order_index: List[Tuple[str, dict]] = []

    for split in splits:
        src = serialized_dir / task / f"{split}.json"
        if not src.exists():
            logger.warning(f"  {task}/{split}: no serialized file at {src}, skipping")
            continue
        with open(src) as f:
            records = json.load(f)

        out_path = output_dir / task / f"{split}.json"
        done = _load_done_keys(out_path)
        todo = [
            r for r in records
            if (r["patient_id"], r["prediction_time"]) not in done
        ]
        if not todo:
            logger.info(f"  {task}/{split}: already complete ({len(done)} done)")
        else:
            logger.info(
                f"  {task}/{split}: {len(todo)} to rubricify "
                f"({len(done)} already done)"
            )
        for r in todo:
            order_index.append((split, r))

    if not order_index:
        logger.info(f"  {task}: nothing to do")
        return

    logger.info(f"  {task}: submitting {len(order_index)} requests as inline batch")

    # ---- Build InlinedRequest list ----
    inlined_requests = []
    for split, record in order_index:
        prompt = _build_transform_prompt(
            record["serialization"], rubric_text, task_query
        )
        inlined_requests.append(
            types.InlinedRequest(
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=config.max_output_tokens,
                    temperature=config.temperature,
                ),
            )
        )

    # ---- Submit batch ----
    client = genai.Client(api_key=config.api_key)
    batch = client.batches.create(
        model=config.model,
        src=inlined_requests,
        config=types.CreateBatchJobConfig(
            display_name=f"refine-rubric-{task}",
        ),
    )
    logger.info(f"  batch job created: {batch.name}  state={batch.state.name}")

    # ---- Poll ----
    poll_interval = 60
    elapsed = 0
    while not batch.done:
        logger.info(
            f"  [{elapsed//60}m] state={batch.state.name} — "
            f"waiting {poll_interval}s ..."
        )
        time.sleep(poll_interval)
        elapsed += poll_interval
        batch = client.batches.get(name=batch.name)

    if batch.state.name != "JOB_STATE_SUCCEEDED":
        err = getattr(batch, "error", None)
        raise RuntimeError(
            f"Batch job {batch.name} ended with state {batch.state.name}. "
            f"Error: {err}"
        )

    logger.info(f"  batch complete: {batch.name}")

    # ---- Parse inline responses ----
    responses = batch.dest.inlined_responses
    if responses is None:
        raise RuntimeError(
            f"batch.dest.inlined_responses is None for job {batch.name}. "
            "Check the Gemini API batch documentation for this SDK version."
        )

    if len(responses) != len(order_index):
        logger.warning(
            f"  response count mismatch: got {len(responses)}, "
            f"expected {len(order_index)}"
        )

    # Accumulate results per split
    new_by_split: Dict[str, List[dict]] = {s: [] for s in splits}
    error_count = 0

    for i, (inlined_resp, (split, record)) in enumerate(
        zip(responses, order_index)
    ):
        if inlined_resp.error:
            text = f"[ERROR: {inlined_resp.error}]"
            error_count += 1
        else:
            try:
                text = inlined_resp.response.candidates[0].content.parts[0].text.strip()
            except Exception as e:
                text = f"[ERROR: could not extract text — {e}]"
                error_count += 1

        new_by_split[split].append({
            "patient_id": record["patient_id"],
            "prediction_time": record["prediction_time"],
            "task": record["task"],
            "split": split,
            "label": record["label"],
            "rubricified_text": text,
        })

    if error_count:
        logger.warning(f"  {error_count}/{len(order_index)} responses had errors")

    # ---- Save per-split JSONs (merge with any already-done records) ----
    for split in splits:
        if not new_by_split.get(split):
            continue
        out_path = output_dir / task / f"{split}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        existing = _load_existing(out_path)
        all_results = existing + new_by_split[split]
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        pos = sum(1 for r in all_results if r["label"])
        logger.info(
            f"  saved {task}/{split}: {len(all_results)} total "
            f"(pos={pos}, neg={len(all_results)-pos}) -> {out_path}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_dir(path_str: str) -> Path:
    p = Path(path_str)
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rubric_dir", required=True)
    p.add_argument("--serialized_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tasks", nargs="+", default=ALL_TASK_NAMES)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return p.parse_args()


def main():
    args = parse_args()
    config = GeminiConfig.from_env()
    rubric_dir = _resolve_dir(args.rubric_dir)
    serialized_dir = _resolve_dir(args.serialized_dir)
    output_dir = _resolve_dir(args.output_dir)

    for task in args.tasks:
        logger.info(f"\n{'='*60}\nApplying rubric for: {task}\n{'='*60}")
        apply_rubric_batch(
            task, args.splits,
            rubric_dir, serialized_dir, output_dir,
            config,
        )

    logger.success("All rubric transformations complete.")


if __name__ == "__main__":
    main()
