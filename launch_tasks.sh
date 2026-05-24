#!/usr/bin/env bash
# launch_tasks.sh — Start one tmux session per task.
#
# Usage:
#   bash launch_tasks.sh
#
# Sessions:
#   refine_new_lupus
#   refine_lab_hyperkalemia
#
# Both sessions share the single A100 (CUDA_VISIBLE_DEVICES=0).
# GPU-heavy steps are serialized via /tmp/refine_gpu.lock.
# Gemini API calls run in parallel across sessions.
#
# Monitor:
#   tmux attach -t refine_new_lupus
#   tmux attach -t refine_lab_hyperkalemia
#   tail -f data/logs/new_lupus.log
#   tail -f data/logs/lab_hyperkalemia.log

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Pre-flight ─────────────────────────────────────────────────────────────
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'[GPU] {torch.cuda.get_device_name(0)} (CUDA available — shared via GPU lock)')
else:
    print('[WARNING] CUDA not available — embedding steps will run on CPU')
"

# Verify ADC works
python3 -c "
import vertexai
vertexai.init(project='som-nero-plevriti-deidbdf', location='us-central1')
print('[AUTH] Vertex AI ADC OK (project=som-nero-plevriti-deidbdf)')
" || { echo 'ERROR: Vertex AI ADC check failed. Run: gcloud auth application-default login'; exit 1; }

mkdir -p "$REPO/data/logs"

# Kill existing sessions with the same name
for SESSION in refine_new_lupus refine_lab_hyperkalemia; do
    tmux kill-session -t "$SESSION" 2>/dev/null && echo "  killed existing session: $SESSION" || true
done

# ── Launch ─────────────────────────────────────────────────────────────────
tmux new-session -d -s "refine_new_lupus" \
    "export CUDA_VISIBLE_DEVICES=0; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
     bash '${REPO}/run_task_pipeline.sh' new_lupus \
     2>&1 | tee '${REPO}/data/logs/new_lupus.log'"

tmux new-session -d -s "refine_lab_hyperkalemia" \
    "export CUDA_VISIBLE_DEVICES=0; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
     bash '${REPO}/run_task_pipeline.sh' lab_hyperkalemia \
     2>&1 | tee '${REPO}/data/logs/lab_hyperkalemia.log'"

echo ""
echo "Started sessions:"
echo "  tmux attach -t refine_new_lupus"
echo "  tmux attach -t refine_lab_hyperkalemia"
echo ""
echo "Monitor logs:"
echo "  tail -f ${REPO}/data/logs/new_lupus.log"
echo "  tail -f ${REPO}/data/logs/lab_hyperkalemia.log"
echo ""
echo "Note: both sessions share GPU 0 via /tmp/refine_gpu.lock."
echo "Gemini API calls run in parallel; embedding steps run one at a time."
