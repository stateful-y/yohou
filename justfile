# Available commands for Yohou

# Show all available commands
default:
    @just --list

# Install dependencies and pre-commit
install:
    uv sync --group dev
    uvx pre-commit install

# Run tests and doctests with parallel execution
test:
    uv run pytest tests src/yohou --doctest-modules --doctest-continue-on-failure -n auto -v

# Run fast tests (excludes slow and integration tests)
test-fast:
    uv run pytest -m "not slow and not integration" -n auto -v

# Run slow tests (includes integration tests)
test-slow:
    uv run pytest -m "slow or integration" -n auto -v

# Run tests with coverage
test-cov:
    uv run pytest --cov=yohou --cov-report=html --cov-report=term -n auto

# Run docstring examples
test-docstrings:
    uv run pytest --doctest-modules --doctest-continue-on-failure --no-cov src/yohou

# Run marimo example notebook interactively
example file='':
    uv run marimo edit examples/{{file}}

# Test all example notebooks
test-examples:
    uv run pytest tests -m example -n auto -v --no-cov

# Run linters and type checkers
lint:
    uv run ruff check src tests
    uv run ty check src
    uvx rumdl check .

# Format and fix code (via pre-commit)
fix:
    uvx pre-commit run --all-files --show-diff-on-failure

# Build documentation
build:
    uv run mkdocs build --clean

# Build documentation without exporting notebooks
build-fast:
    MKDOCS_SKIP_NOTEBOOKS=1 uv run mkdocs build --clean

# Serve documentation locally
serve:
    @echo "###### Starting local server. Press Control+C to stop server ######"
    uv run mkdocs serve -a localhost:8080

# Serve documentation locally without exporting notebooks
serve-fast:
    @echo "###### Starting local server. Press Control+C to stop server ######"
    MKDOCS_SKIP_NOTEBOOKS=1 uv run mkdocs serve -a localhost:8080

# Check built docs for dead links (build first with 'just build' or 'just build-fast')
link:
    uvx linkchecker site/index.html --no-status --no-warnings --ignore-url 'material/overrides'


# Clean build artifacts
clean:
    rm -rf .nox
    rm -rf build dist *.egg-info
    rm -rf .pytest_cache .ty_cache .ruff_cache
    rm -rf htmlcov .coverage coverage.xml
    rm -rf site
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete

# Run all quality checks (fix, test)
all: fix test
