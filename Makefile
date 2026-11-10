export OMP_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export MKL_NUM_THREADS := 1

.PHONY: setup reproduce check serve

setup:
	uv sync --frozen
	npm ci --prefix ui --cache .local/npm-cache --ignore-scripts

reproduce:
	uv run lease-lab build --through core
	uv run python scripts/export_contracts.py
	npm run contracts --prefix ui
	npm run build --prefix ui

check:
	uv run pytest
	uv run ruff check src tests scripts --select F
	uv run python scripts/contract_fixtures.py
	npm run test:contracts --prefix ui
	npm run check --prefix ui
	uv run python scripts/check_challenger.py

serve:
	uv run lease-lab serve
