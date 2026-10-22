#!/usr/bin/env bash
# Starts the Llmzempic web UI. Extra arguments are passed through (e.g. --port 8000 --no-browser).
cd "$(dirname "$0")"
exec .venv/bin/python -m llmzempic "$@"
