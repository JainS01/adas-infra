#!/usr/bin/env bash
# judge_quickstart.sh — One-shot judge path: ingest → train → serve → verify
# No cloud account, no GPU required.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "=== ADAS Infra — Judge Quickstart ==="
echo "Working directory: ${REPO_ROOT}"

# 1. Generate synthetic data and seed the delta log
echo ""
echo "Step 1/4: Seeding delta log with synthetic data..."
uv run python scripts/seed_delta_log.py \
    --synthetic \
    --num-shards 8 \
    --num-subjects 20 \
    --samples-per-subject 5 \
    --output-dir ./data/synthetic \
    --state-dir ./state

# 2. Train for 50 steps
echo ""
echo "Step 2/4: Training FusionBaseline for 50 steps (local_mock profile)..."
uv run adas-train +profile=local_mock trainer.max_steps=50

# 3. Publish model to local MLflow registry
echo ""
echo "Step 3/4: Publishing model to local registry..."
RUN_ID=$(cat .last_run_id 2>/dev/null || echo "latest")
uv run adas-publish-model --run-id "${RUN_ID}"

# 4. Start serving + verify endpoint
echo ""
echo "Step 4/4: Starting inference server and verifying /predict endpoint..."

MODEL_PATH=$(cat .last_model_path 2>/dev/null || echo "./checkpoints/fusion.pt")
ADAS_MODEL_PATH="${MODEL_PATH}" uv run adas-serve +profile=local_mock &
SERVE_PID=$!

echo "Waiting for server startup..."
for i in $(seq 1 10); do
    if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
        echo "Server ready."
        break
    fi
    sleep 1
done

echo ""
echo "=== /predict response ==="
curl -s http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d @tests/e2e/sample.json | python3 -m json.tool

kill "${SERVE_PID}" 2>/dev/null || true

echo ""
echo "=== Judge quickstart complete! ==="
echo "Run 'uv run pytest tests/e2e -q -m e2e' to validate the full pipeline."
echo "Open http://localhost:3000 for Grafana dashboards."
