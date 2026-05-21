#!/usr/bin/env python3
"""
Apply a task-specific rubric to every patient EHR via Gemini 2.5 Flash
Vertex AI batch prediction.

For each task, this script:
  1. Loads the rubric instructions (from create_rubric.py).
  2. Loads serialized patient records for the requested splits.
  3. Uploads a JSONL request file to GCS.
  4. Submits a Vertex AI BatchPredictionJob and polls until complete.
  5. Downloads the output JSONL from GCS and saves rubricified JSONs per split.

Each request carries a _meta field ({patient_id, prediction_time, split, label,
task}) that is preserved in the Vertex AI output for result-to-record matching.

Authentication: Application Default Credentials (ADC) — automatic on this GCP VM.

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
from typing import Dict, List, Tuple

import vertexai
from google.cloud import storage
from loguru import logger
from vertexai.batch_prediction import BatchPredictionJob

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
# GCS helpers
# ---------------------------------------------------------------------------

def _parse_gcs_uri(uri: str) -> Tuple[str, str]:
    s = uri.replace("gs://", "", 1)
    bucket, _, prefix = s.partition("/")
    return bucket, prefix


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

    logger.info(f"  {task}: building {len(order_index)} batch requests")

    # ---- Build JSONL ----
    timestamp = int(time.time())
    requests_jsonl: List[str] = []
    for split, record in order_index:
        prompt = _build_transform_prompt(
            record["serialization"], rubric_text, task_query
        )
        meta = {
            "patient_id": record["patient_id"],
            "prediction_time": record["prediction_time"],
            "split": split,
            "label": record["label"],
            "task": task,
        }
        row = {
            "request": {
                "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": config.temperature,
                    "maxOutputTokens": config.max_output_tokens,
                },
            },
            "_meta": json.dumps(meta),
        }
        requests_jsonl.append(json.dumps(row))

    # ---- Upload JSONL to GCS ----
    in_uri = (
        f"gs://{config.gcs_bucket}/{config.gcs_prefix}/"
        f"input/refine_rubric_{task}_{timestamp}.jsonl"
    )
    out_prefix = (
        f"gs://{config.gcs_bucket}/{config.gcs_prefix}/"
        f"output/refine_rubric_{task}_{timestamp}"
    )

    storage_client = storage.Client()
    in_bucket_name, in_blob_path = _parse_gcs_uri(in_uri)
    storage_client.bucket(in_bucket_name).blob(in_blob_path).upload_from_string(
        "\n".join(requests_jsonl).encode("utf-8"),
        content_type="application/jsonl",
    )
    logger.info(f"  uploaded {len(order_index)} requests to {in_uri}")

    # ---- Submit Vertex AI batch job ----
    vertexai.init(project=config.project, location=config.location)
    job = BatchPredictionJob.submit(
        source_model=config.model,
        input_dataset=in_uri,
        output_uri_prefix=out_prefix,
        job_display_name=f"refine-rubric-{task}-{timestamp}",
    )
    logger.info(f"  batch job submitted: {job.resource_name}")

    # ---- Poll ----
    poll_interval = 60
    elapsed = 0
    while not job.has_ended:
        logger.info(
            f"  [{elapsed//60}m] state={job.state} — waiting {poll_interval}s ..."
        )
        time.sleep(poll_interval)
        elapsed += poll_interval
        job.refresh()

    if not job.has_succeeded:
        raise RuntimeError(
            f"Batch job {job.resource_name} failed: {job.error}"
        )
    logger.info(f"  batch complete — output at {job.output_location}")

    # ---- Download and parse output JSONL from GCS ----
    out_bucket_name, out_blob_prefix = _parse_gcs_uri(job.output_location)
    blobs = [
        b for b in storage_client.bucket(out_bucket_name).list_blobs(
            prefix=out_blob_prefix
        )
        if b.name.endswith(".jsonl")
    ]
    logger.info(f"  found {len(blobs)} output JSONL blob(s)")

    new_by_split: Dict[str, List[dict]] = {s: [] for s in splits}
    error_count = 0

    for blob in blobs:
        for line in blob.download_as_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)

            # Skip errored rows
            if obj.get("status") not in ("", None):
                error_count += 1
                logger.warning(f"  row error: {obj.get('status')}")
                continue

            meta_raw = obj.get("_meta", "{}")
            meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
            split = meta.get("split", "unknown")

            try:
                text = (
                    obj["response"]["candidates"][0]["content"]["parts"][0]["text"]
                    .strip()
                )
            except Exception as e:
                text = f"[ERROR: could not extract text — {e}]"
                error_count += 1

            new_by_split.setdefault(split, []).append({
                "patient_id": meta["patient_id"],
                "prediction_time": meta["prediction_time"],
                "task": meta["task"],
                "split": split,
                "label": meta["label"],
                "rubricified_text": text,
            })

    if error_count:
        logger.warning(f"  {error_count} responses had errors")

    # ---- Save per-split JSONs (merge with any already-done records) ----
    for split, new_results in new_by_split.items():
        if not new_results:
            continue
        out_path = output_dir / task / f"{split}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        existing = _load_existing(out_path)
        all_results = existing + new_results
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
    config = GeminiConfig.default()
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
