# Developer Workflow & Tools

**Purpose**: Essential commands, tools, and workflows for Yohou development.

---

## Environment Setup

### Package Manager: uv

Yohou uses **uv** (fast Python package installer/resolver) for dependency management.

```bash
# Install dev environment (syncs all dependency groups)
uv sync

# Install specific groups
uv sync --group tests
uv sync --group docs
```

### Dependency Groups (pyproject.toml)

- `dev`: Includes all groups (tests, docs, fix)
- `tests`: pytest, pytest-cov, covdefaults
- `docs`: mkdocs-material, mkdocstrings, etc.
- `fix`: pre-commit-uv, ruff, ty, interrogate
- `examples`: marimo

---

## Nox Sessions

**Critical**: Always use `uvx nox` (not plain `nox`) to leverage uv's automatic tool management.

Nox is configured to use `uv` as the default venv backend (`noxfile.py` at project root).

### Available Sessions

**Default sessions** (run with `uvx nox`): `fix`, `test`, `docs`

#### test
Run pytest with coverage, includes doctests and unit tests

```bash
uvx nox -s test

# Outputs:
# - coverage.{python}.xml
# - junit.{python}.xml
# - HTML coverage in temp dir
```

**What it does**:
- Uses `coverage run --parallel-mode` for parallel execution
- Runs unit tests: `pytest src/yohou`
- Runs doctests: `pytest --doctest-modules src/yohou`
- Combines coverage data from parallel runs
- Generates HTML and XML reports

#### fix
Run pre-commit hooks (ruff linter/formatter + ty type checking + interrogate)

```bash
uvx nox -s fix

# Runs: pre-commit run --all-files
# Includes: ruff, ruff-format, ty, interrogate, yaml checks, etc.
```

**When to use**:
- Before committing (or let git hooks run it automatically)
- After making changes to ensure code quality
- To auto-fix linting issues

#### docs
Build MkDocs documentation

```bash
uvx nox -s docs

# Output: site/ directory
```

#### serve_docs
Local docs server with live reload

```bash
uvx nox -s serve_docs

# Opens localhost:8080 with live reload
```

#### deploy_docs
Deploy to GitHub Pages

```bash
uvx nox -s deploy_docs

# Runs: mkdocs gh-deploy
```

---

## Code Quality Tools

### Ruff: Linter & Formatter

**Configuration**: `pyproject.toml`
- Line length: 100 characters
- Target: Python 3.12
- Rules: E, F, W, I (PEP 8 + imports)

**Commands**:
```bash
# Check for issues
uvx ruff check src/yohou

# Auto-fix issues
uvx ruff check --fix src/yohou

# Format code
uvx ruff format src/yohou

# Check specific file
uvx ruff check src/yohou/point_forecaster/naive.py
```

**Common issues**:
- Import ordering (automatically fixed with `--fix`)
- Unused imports (automatically removed with `--fix`)
- Line length violations (automatically fixed by `ruff format`)

### ty: Type Checker

**Critical**: Yohou uses **ty** (Rust-based type checker), NOT mypy.

**Commands**:
```bash
# Check types
uvx ty check src

# Check specific file
uvx ty check src/yohou/point_forecaster/naive.py
```

**Important notes**:
- Do NOT use mypy commands - ty uses different inference rules
- ty is NOT mypy-compliant (has different type inference behavior)
- Pre-commit enforces ty checking
- Type hints required for all public methods/functions

### interrogate: Docstring Coverage

**Requirement**: 100% docstring coverage (enforced)

**Configuration** (`pyproject.toml`):
```toml
[tool.interrogate]
fail-under = 100
exclude = ["tests", "examples", "_version.py"]
ignore-init-method = true
ignore-init-module = true
ignore-magic = true
ignore-private = true
ignore-nested-classes = true
```

**Commands**:
```bash
# Check coverage
uvx interrogate src/yohou

# Check specific file
uvx interrogate src/yohou/point_forecaster/naive.py
```

**What counts**:
- ✅ Public modules, classes, methods, functions
- ✅ Nested functions (NOT nested classes)
- ❌ Private methods (`_method`)
- ❌ Magic methods (`__init__`, `__repr__`)
- ❌ Test files, examples, `_version.py`

**Docstring style**: NumPy style (NOT Google style)

### Pre-commit Hooks

**Configuration**: `.pre-commit-config.yaml`

**Hooks**:
- check-yaml
- check-merge-conflict
- end-of-file-fixer
- trailing-whitespace
- interrogate (100% coverage)
- ruff (linter)
- ruff-format (formatter)
- ty (type checker)

**Commands**:
```bash
# Run all hooks manually
uvx nox -s fix
# OR
pre-commit run --all-files

# Run specific hook
pre-commit run ruff --all-files
pre-commit run ty --all-files

# Install git hooks (auto-runs on commit)
pre-commit install
```

**Auto-runs on commit**: Git hooks are installed by default

---

## Testing Workflows

### Quick Test Runs

```bash
# Run all tests (no coverage, fast)
uv run pytest

# Run specific test file
uv run pytest tests/decomposition/test_trend.py -v

# Run specific test
uv run pytest tests/decomposition/test_trend.py::test_polynomial_quadratic_analytical -v

# Run tests matching pattern
uv run pytest -k "polynomial" -v
```

### With Coverage

```bash
# Full coverage report (via nox)
uvx nox -s test

# View HTML coverage report
# Output location shown after nox run (temp directory)
# Example: open /tmp/nox-session-xxxxx/htmlcov/index.html
```

### Debugging Tests

```bash
# Run with debugger (pdb)
uv run pytest tests/decomposition/test_trend.py::test_polynomial_quadratic_analytical --pdb

# When pdb activates:
# n (next): Execute next line
# s (step): Step into function
# c (continue): Continue execution
# p variable: Print variable value
# l (list): Show code context
# q (quit): Exit debugger

# Inline debugging in test code
import pdb; pdb.set_trace()  # Add before the line you want to inspect
```

### Rerun Failed Tests

```bash
# Rerun only failed tests from last run
uv run pytest --lf

# Rerun failed, then all
uv run pytest --ff
```

### Doctests

```bash
# Run doctests in specific file
uv run pytest --doctest-modules src/yohou/point_forecaster/naive.py

# Run all doctests (included in nox test session)
uvx nox -s test
```

---

## CI/CD Workflows

### GitHub Actions

**Workflows** (`.github/workflows/`):

1. **tests-os-coverage.yml**: Cross-platform testing
   - Matrix: Python 3.12+ on Ubuntu/macOS/Windows
   - Uses uv for fast dependency installation
   - Runs: `uvx nox -s test`
   - Concurrent execution with auto-cancellation

2. **lint.yml**: Code quality checks
   - Runs: `uvx nox -s fix`
   - Validates ruff + ty + interrogate

3. **release.yml**: PyPI package publishing
   - Automated on version tags (e.g., `v0.1.0`)
   - Builds with hatchling + hatch-vcs
   - Publishes to PyPI with trusted publishing

### Local vs CI Differences

**Local development**:
```bash
uvx nox -s test  # uv automatically manages nox as tool
```

**CI**:
```bash
uv tool install --python-preference only-managed --python 3.12 nox
nox -s test  # Explicit nox install
```

**Why different?**: CI requires explicit tool installation, local uses uvx for automatic tool management.

---

## Common Workflows

### Creating New Forecaster

```bash
# 1. Create implementation file
# src/yohou/point_forecaster/my_forecaster.py

# 2. Create test file
# tests/point_forecaster/test_my_forecaster.py

# 3. Run quality checks
uvx ruff check --fix src/yohou/point_forecaster/my_forecaster.py
uvx ruff format src/yohou/point_forecaster/my_forecaster.py
uvx ty check src/yohou/point_forecaster/my_forecaster.py
uvx interrogate src/yohou/point_forecaster/my_forecaster.py

# 4. Run tests
uv run pytest tests/point_forecaster/test_my_forecaster.py -v

# 5. Run doctests
uv run pytest --doctest-modules src/yohou/point_forecaster/my_forecaster.py

# 6. Run all checks (before commit)
uvx nox -s fix
```

### Debugging Failed Tests

```bash
# 1. Run test with verbose output
uv run pytest tests/path/to/test.py::test_name -vv

# 2. If still unclear, add debugger
uv run pytest tests/path/to/test.py::test_name --pdb

# 3. Inspect specific variables in test code
# Add: import pdb; pdb.set_trace()

# 4. Check coverage gaps
uvx nox -s test
# Open HTML report from output
```

### Before Committing

```bash
# Run all quality checks
uvx nox -s fix

# If any issues, fix and rerun
uvx ruff check --fix .
uvx ruff format .

# Run tests to ensure nothing broke
uv run pytest

# Commit (pre-commit hooks will run automatically)
git add .
git commit -s -m "feat: add MyForecaster"
```

### Making a Release

```bash
# 1. Update version (managed by hatch-vcs from git tags)
# 2. Create tag
git tag v0.1.0
git push origin v0.1.0

# 3. GitHub Actions automatically:
#    - Builds package
#    - Runs tests
#    - Publishes to PyPI
```

---

## Examples & Marimo Notebooks

Yohou uses **marimo** for interactive examples (reactive notebook system).

### Running Examples

```bash
# Start marimo notebook server
marimo edit examples/air_passengers_tutorial.py

# Or run all cells non-interactively
marimo run examples/air_passengers_tutorial.py
```

### Marimo Patterns

**NOT traditional Jupyter notebooks** - marimo is reactive:

```python
import marimo

app = marimo.App(width="medium")

@app.cell
def _():
    # Import cells - define dependencies
    import polars as pl
    from yohou.point_forecaster import PointReductionForecaster
    return (pl, PointReductionForecaster)

@app.cell
def _(pl):
    # Data loading cell - uses imports from previous cell
    df = pl.read_csv("data.csv")
    return (df,)

@app.cell
def _(mo):
    # Interactive controls with marimo UI
    horizon_slider = mo.ui.slider(start=1, stop=24, value=12,
                                   label="Forecast Horizon")
    return (horizon_slider,)
```

**Key patterns**:
- Each cell is a function decorated with `@app.cell`
- Return values in tuples: `return (var1, var2)`
- Cells automatically re-run when dependencies change (reactive execution)
- Use `mo.ui.*` for interactive controls (sliders, dropdowns, etc.)
- Use `mo.md(r"""...""")` for markdown documentation cells
- Plotly figures render automatically without explicit `.show()`

### Example Topics

**`air_passengers_tutorial.py`**: Full forecasting workflow
- Baseline models (SeasonalNaive)
- Preprocessing pipelines (LogTransform, SeasonalDifferencing, LagTransformer)
- Interactive parameter exploration with sliders
- Hyperparameter optimization with GridSearchCV/RandomizedSearchCV
- Incremental learning with `update_predict()`
- Visualization with plotly

---

## Performance Profiling

### Time Profiling

```bash
# Basic profiling
uv run python -m cProfile -o profile.stats your_script.py

# View results
uv run python -c "import pstats; p=pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"
```

### Memory Profiling

```bash
# Install memory_profiler
uv pip install memory_profiler

# Profile script
uv run python -m memory_profiler your_script.py
```

### Polars Query Plans

```python
import polars as pl

# Lazy evaluation - see optimization plan
df.lazy().select([...]).explain()

# Visual query graph
df.lazy().select([...]).show_graph()
```

---

## Troubleshooting

### Problem: Tests fail locally but pass in CI
**Check**: Python version matches CI (3.12+)
```bash
uv python install 3.12
uv run pytest
```

### Problem: Pre-commit hooks slow
**Solution**: Skip specific hooks during development
```bash
SKIP=ty git commit -m "WIP: work in progress"

# Run all checks before pushing
uvx nox -s fix
```

### Problem: Nox creates environment but fails to install
**Solution**: Clear nox cache and retry
```bash
rm -rf .nox
uvx nox -s test
```

### Problem: Type checker (ty) errors differ from CI
**Solution**: ty version mismatch - update to match CI
```bash
uvx ty --version  # Check version
# Update pyproject.toml if needed
```

### Problem: Coverage report missing files
**Solution**: Ensure coverage.py is running from project root
```bash
# Check COVERAGE_FILE environment variable
echo $COVERAGE_FILE

# Should be: .coverage.{python_version}
```

---

## Contributing Workflow

### Branch Naming

**Pattern**: `<type>/<description>`

**Types**:
- `feature/add-fourier-forecaster`
- `fix/panel-data-bug`
- `docs/update-architecture`
- `tests/add-reduction-checks`

### Commit Messages

**Must follow**: [Conventional Commits](https://www.conventionalcommits.org/)

**Examples**:
```bash
git commit -s -m "feat: add FourierSeasonalityForecaster"
git commit -s -m "fix: handle empty panel groups in predict"
git commit -s -m "docs: update decomposition module guide"
git commit -s -m "test: add panel data tests for PolynomialTrend"
```

**Required**: Commits must be signed off (`-s` flag)

### Pull Request Checklist

- [ ] Branch name follows `<type>/<description>` pattern
- [ ] PR title follows Conventional Commits format
- [ ] All nox sessions pass (`fix`, `test`, `docs`)
- [ ] Tests added for new features
- [ ] Doctests included in public methods
- [ ] 100% docstring coverage
- [ ] Type hints added for all public APIs
- [ ] Examples updated if API changed

### Code Compatibility

**Target**: Python 3.12+
**Platforms**: Windows, macOS, Linux

**Test locally on your platform**:
```bash
# Run full test suite
uvx nox -s test

# CI will test on all platforms
```

---

## License

All contributions under **BSD License**
