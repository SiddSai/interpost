.PHONY: install test smoke lint fmt all

install:
	pip install -e ".[dev]"

test:
	pytest -q -m "not smoke"

smoke:
	pytest -q -m smoke

lint:
	ruff check .

fmt:
	ruff check --fix .
	ruff format .

all: lint test
