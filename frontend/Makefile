	SHELL := /bin/bash

.PHONY: dev start check-uv lint

# dev: Ensure `uv` is installed, sync dependencies, and run the frontend in reload mode
dev: check-uv
	@echo "-> Syncing dependencies with uv..."
	uv sync
	@echo "-> Launching frontend (uv run uvicorn main:app)..."
	uv run uvicorn main:app --reload --host 127.0.0.1 --port 8000

# start: Run without reload (use for local production testing)
start: check-uv
	@echo "-> Syncing dependencies with uv..."
	uv sync
	@echo "-> Launching frontend (uv run uvicorn main:app)..."
	uv run uvicorn main:app --host 0.0.0.0 --port 8000

# lint: Run ruff via uv (development linting)
lint: check-uv
	@echo "-> Running ruff lint (via uv)..."
	uv run ruff check .

# check-uv: fail fast with instructions if `uv` is not installed
check-uv:
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "Error: 'uv' is not installed."; \
		echo "Install it with: pip install uv"; \
		exit 1; \
	fi
