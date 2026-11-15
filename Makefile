.PHONY: sync lint typecheck test ui demo serve

sync:
	uv sync --all-extras

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

typecheck:
	uv run mypy src

test:
	uv run pytest --cov=axiom_zero --cov-report=term-missing --cov-fail-under=95

ui:
	cd ui && npm ci && npm run build

demo:
	uv run axiom demo

serve:
	uv run axiom serve --port 8030
