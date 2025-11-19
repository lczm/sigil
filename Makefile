.PHONY: test typecheck lint check

test:
	uv run pytest -v

typecheck:
	uv run ty check

lint:
	uv run ruff check

check: typecheck test lint