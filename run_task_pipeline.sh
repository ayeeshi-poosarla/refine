#!/usr/bin/env bash
# run_task_pipeline.sh — Full REFINE rubric pipeline for one task.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_task_pipeline.sh guo_readmission
#
# Prerequisites:
# Authentication via Application Default Credentials (ADC) — automatic on GCP VM.
#
# GPU steps (embedding generation) use a file lock at /tmp/refine_gpu.lock
# so two parallel tmux sessions share the single A100 without OOM.

set -euo pipefail

TASK=${1:?"Usage: bash run_task_pipeline.sh <task_name>"}

# ── Paths ──────────────────────────────────────────────────────────────────
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$REPO/data"
EHRSHOT="$HOME/data/EHRSHOT_ASSETS"
DB="$EHRSHOT/femr/extract"
LABELS="$EHRSHOT/benchmark"
SPLITS="$EHRSHOT/splits/person_id_map.csv"

SELECTED_DIR="$DATA/selected_patients"
SERIALIZED="$DATA/serialized/naivetext"
SFT_NAIVE="$DATA/sft/naivetext"
EMB_NAIVE="$DATA/embeddings/naivetext"
RUBRIC_DIR="$DATA/rubric"
RUBRICIFIED="$DATA/rubric/rubricified"
SFT_RUBRIC="$DATA/sft/global-rubric"
EMB_RUBRIC="$DATA/embeddings/global-rubric"
RESULTS="$DATA/results/global-rubric"
GPU_LOCK="/tmp/refine_gpu.lock"

mkdir -p "$DATA/logs"

log() { echo "[$(date '+%H:%M:%S')] [$TASK] $*"; }

# ── Pre-flight checks ──────────────────────────────────────────────────────
python3 - <<'PYCHECK'
import torch
if torch.cuda.is_available():
    print(f"[GPU] {torch.cuda.get_device_name(0)} — CUDA available")
else:
    print("[WARNING] CUDA not available — embedding steps will run on CPU (much slower)")
PYCHECK

# ── Step 1: Select 400 patients (200 pos + 200 neg, custom 70/15/15 splits) ─
log "=== Step 1: Select patients ==="
python3 "$REPO/select_patients.py" \
    --task "$TASK" \
    --labels_dir "$LABELS" \
    --output_dir "$SELECTED_DIR"

# ── Step 2: Serialize EHRs (only selected patients, no balancing) ───────────
log "=== Step 2: Serialize EHRs ==="
SERIALIZE_EXTRA_ARGS=()
if [[ "$TASK" == lab_* ]]; then
    SERIALIZE_EXTRA_ARGS+=(--max_records_per_patient 1)
fi
python3 "$REPO/01_serialize/serialize.py" \
    --path_to_database "$DB" \
    --path_to_labels_dir "$LABELS" \
    --path_to_splits "$SPLITS" \
    --output_dir "$SERIALIZED" \
    --mode naivetext \
    --tasks "$TASK" \
    --patient_ids_file "$SELECTED_DIR/${TASK}_selected_patients.json" \
    --workers 6 \
    --skip_existing \
    "${SERIALIZE_EXTRA_ARGS[@]}"

# ── Step 3: Create naivetext SFT ────────────────────────────────────────────
log "=== Step 3: Create naivetext SFT ==="
python3 "$REPO/02_create_sft/create_sft.py" \
    --input_dir "$SERIALIZED" \
    --output_dir "$SFT_NAIVE" \
    --tasks "$TASK"

# Naivetext representations saved:
#   $SERIALIZED/$TASK/{train,val,test}.json  — raw EHR text
#   $SFT_NAIVE/{train,val,test}/$TASK.json  — SFT conversation format

# ── Step 4: Generate naivetext embeddings [GPU] ─────────────────────────────
log "=== Step 4: Generate naivetext embeddings (GPU) ==="
(
    flock -x 200
    log "  [GPU lock acquired]"
    python3 "$REPO/05_eval/generate_embeddings.py" \
        --sft_dir "$SFT_NAIVE" \
        --output_dir "$EMB_NAIVE" \
        --tasks "$TASK" \
        --splits train val test \
        --batch_size 2
    log "  [GPU lock released]"
) 200>"$GPU_LOCK"

# ── Step 5: Build cohort (k-means on train embeddings → 40 medoids) ─────────
log "=== Step 5: Build cohort ==="
python3 "$REPO/03_globalrubric/build_cohort.py" \
    --input_dir "$SERIALIZED" \
    --output_dir "$RUBRIC_DIR" \
    --embeddings_dir "$EMB_NAIVE" \
    --tasks "$TASK"

# ── Step 6: Create rubric template (Gemini 2.5 Flash) ───────────────────────
log "=== Step 6: Create rubric template ==="
python3 "$REPO/03_globalrubric/create_rubric.py" \
    --cohort_dir "$RUBRIC_DIR" \
    --output_dir "$RUBRIC_DIR" \
    --tasks "$TASK"

# ── Step 7: Apply rubric via Gemini batch (all splits) ──────────────────────
log "=== Step 7: Apply rubric (Gemini batch) ==="
python3 "$REPO/03_globalrubric/apply_rubric.py" \
    --rubric_dir "$RUBRIC_DIR" \
    --serialized_dir "$SERIALIZED" \
    --output_dir "$RUBRICIFIED" \
    --tasks "$TASK" \
    --splits train val test

# ── Step 8: Create global-rubric SFT ────────────────────────────────────────
log "=== Step 8: Create global-rubric SFT ==="
python3 "$REPO/03_globalrubric/create_globalrubric_sft.py" \
    --input_dir "$RUBRICIFIED" \
    --output_dir "$SFT_RUBRIC" \
    --tasks "$TASK"

# ── Step 9: Generate rubric embeddings [GPU] ────────────────────────────────
log "=== Step 9: Generate rubric embeddings (GPU) ==="
(
    flock -x 200
    log "  [GPU lock acquired]"
    python3 "$REPO/05_eval/generate_embeddings.py" \
        --sft_dir "$SFT_RUBRIC" \
        --output_dir "$EMB_RUBRIC" \
        --tasks "$TASK" \
        --splits train val test \
        --batch_size 2
    log "  [GPU lock released]"
) 200>"$GPU_LOCK"

# ── Step 10: Evaluate rubric embeddings (LogReg, train→val→test AUROC) ──────
log "=== Step 10: Evaluate — LogReg AUROC on rubric embeddings ==="
python3 "$REPO/05_eval/eval_embeddings.py" \
    --embeddings_dir "$EMB_RUBRIC" \
    --output_dir "$RESULTS" \
    --tasks "$TASK"

# ── Summary ─────────────────────────────────────────────────────────────────
log "=== DONE ==="
log "Outputs:"
log "  Naivetext (raw EHR text) : $SERIALIZED/$TASK/{train,val,test}.json"
log "  Naivetext (SFT format)   : $SFT_NAIVE/{train,val,test}/$TASK.json"
log "  Naivetext (embeddings)   : $EMB_NAIVE/$TASK/{train,val,test}.npz"
log "  Rubric template          : $RUBRIC_DIR/$TASK/rubric.json"
log "  Filled rubrics           : $RUBRICIFIED/$TASK/{train,val,test}.json"
log "  AUROC results            : $RESULTS/$TASK/metrics.json"
log ""
log "  View AUROC:"
log "    python3 -c \"import json; d=json.load(open('$RESULTS/$TASK/metrics.json')); print(d)\""
