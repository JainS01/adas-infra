#!/usr/bin/env bash
# Bootstrap a development environment.
set -euo pipefail

echo "Installing uv workspace dependencies..."
uv sync

echo "Copying .env.example to .env.local..."
if [ ! -f .env.local ]; then
    cp conf/.env.example .env.local
fi

echo "Installing pre-commit hooks..."
uv run pre-commit install

echo ""
echo "Bootstrap complete. Run 'make judge-quickstart' to test the pipeline."
