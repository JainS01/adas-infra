.PHONY: all install fmt lint type-check test test-unit test-integration test-e2e \
        obs-up obs-down judge-quickstart clean seed

PYTHON := python
UV := uv
COMPOSE := docker compose

# ── Development ──────────────────────────────────────────────────────────────
install:
	$(UV) sync

fmt:
	$(UV) run ruff format packages/ tests/ scripts/
	$(UV) run ruff check --fix packages/ tests/ scripts/

lint:
	$(UV) run ruff check packages/ tests/ scripts/
	$(UV) run bandit -r packages/ -c pyproject.toml

type-check:
	$(UV) run mypy packages/ --config-file mypy.ini

# ── Testing ───────────────────────────────────────────────────────────────────
test: test-unit test-integration

test-unit:
	$(UV) run pytest tests/unit/ -v --tb=short

test-integration:
	$(UV) run pytest tests/integration/ -v --tb=short -m "not gpu"

test-e2e:
	$(UV) run pytest tests/e2e/ -v --tb=short -m e2e --timeout=180

benchmark:
	$(UV) run pytest tests/benchmarks/ --benchmark-json=tests/benchmarks/results.json -v

# ── Observability stack ───────────────────────────────────────────────────────
obs-up:
	$(COMPOSE) up -d prometheus grafana

obs-down:
	$(COMPOSE) down

# ── Judge quickstart (no cloud, no GPU, one command) ─────────────────────────
judge-quickstart: seed
	$(UV) run adas-train +profile=local_mock trainer.max_steps=50
	$(UV) run adas-publish-model --run-id $$(cat .last_run_id 2>/dev/null || echo "latest")
	curl -sf http://localhost:8080/health || ($(UV) run adas-serve +profile=local_mock &); sleep 3
	curl -sf http://localhost:8080/predict -H "Content-Type: application/json" \
	     -d @tests/e2e/sample.json | python -m json.tool

seed:
	$(UV) run python scripts/seed_delta_log.py --synthetic --num-shards 8 \
	      --output-dir ./data/synthetic

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null; true
	rm -rf ./mlruns ./state ./data/synthetic .last_run_id
