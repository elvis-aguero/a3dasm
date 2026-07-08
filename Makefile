.DEFAULT_GOAL := help

PACKAGEDIR := dist
COVERAGEREPORTDIR := coverage_html_report

.PHONY: help test test-html build docs lint

help:
	@echo "Please use \`make <target>' where <target> is one of:"
	@echo "  test        Run the tests with pytest"
	@echo "  test-html   Run the tests and open the HTML coverage report"
	@echo "  build       Build the package"
	@echo "  docs        Build the documentation with mkdocs"
	@echo "  lint        Lint the code with ruff"

test:
	uv run pytest -m "not integration and not ollama"

test-html:
	pytest
	xdg-open ./$(COVERAGEREPORTDIR)/index.html

build:
	uv build

docs:
	mkdocs build

lint:
	ruff check
