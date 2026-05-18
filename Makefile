.PHONY: lint test test_integration build

lint:
	ruff check langchain_ceramic tests

test:
	pytest tests/unit_tests

test_integration:
	pytest tests/integration_tests

build:
	uv build
