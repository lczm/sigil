.PHONY: test
test:
	uv run pytest -v

typecheck:
	uv run ty check