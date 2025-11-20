.PHONY: test typecheck lint format check

test:
	uv run pytest -v

typecheck:
	uv run ty check

lint:
	uv run ruff check

format:
	uv run ruff format

check: typecheck test lint format