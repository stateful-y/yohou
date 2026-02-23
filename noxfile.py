"""Nox sessions for Yohou."""

import nox

# Require Nox version 2024.3.2 or newer to support the 'default_venv_backend' option
nox.needs_version = ">=2024.3.2"

# Set 'uv' as the default backend for creating virtual environments
nox.options.default_venv_backend = "uv|virtualenv"

# Default sessions to run when nox is called without arguments
nox.options.sessions = ["fix", "test_fast", "serve_docs"]

# Generate list of Python versions from minimum to maximum
ALL_VERSIONS = ["3.11", "3.12", "3.13", "3.14"]
MIN_VERSION = "3.11"
MAX_VERSION = "3.14"
PYTHON_VERSIONS = [v for v in ALL_VERSIONS if v >= MIN_VERSION and v <= MAX_VERSION]


@nox.session(python=PYTHON_VERSIONS[0], venv_backend="uv")
def test_coverage(session: nox.Session) -> None:
    """Run the tests with pytest and coverage under the default Python version."""
    session.env["COVERAGE_FILE"] = f".coverage.{session.python}"
    session.env["COVERAGE_PROCESS_START"] = "pyproject.toml"

    # Install dependencies
    session.run_install(
        "uv",
        "sync",
        "--no-default-groups",
        "--group",
        "tests",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )

    # Clear all .coverage* files
    session.run("coverage", "erase")

    # Run unit tests under coverage with parallel execution
    session.run(
        "coverage",
        "run",
        "--source=src/yohou",
        "-m",
        "pytest",
        "tests",
        "-m",
        "not example",
        "-n",
        "auto",
        f"--junitxml=junit.{session.python}.xml",
        *session.posargs,
    )

    # Generate HTML and XML reports
    session.run("coverage", "html", "--ignore-errors", "-d", session.create_tmp())

    # XML report for CI
    session.run("coverage", "xml", "-o", f"coverage.{session.python}.xml")


@nox.session(python=PYTHON_VERSIONS, venv_backend="uv")
def test(session: nox.Session) -> None:
    """Run the test suite across multiple Python versions (no coverage)."""
    # Install dependencies
    session.run_install(
        "uv",
        "sync",
        "--no-default-groups",
        "--group",
        "tests",
        "--group",
        "examples",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )

    # Run unit tests and doctests with parallel execution
    session.run(
        "pytest",
        "tests",
        "src/yohou",
        "--doctest-modules",
        "--doctest-continue-on-failure",
        "-n",
        "auto",
        "-v",
        *session.posargs,
    )


@nox.session(python=PYTHON_VERSIONS, venv_backend="uv")
def test_fast(session: nox.Session) -> None:
    """Run fast tests (excludes slow and integration tests)."""
    # Install dependencies
    session.run_install(
        "uv",
        "sync",
        "--no-default-groups",
        "--group",
        "tests",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )

    # Run fast tests only with parallel execution
    # --no-cov disables coverage (from addopts) so it cannot fail this step
    session.run(
        "pytest",
        "tests",
        "--no-cov",
        "-m",
        "not slow and not integration and not example",
        "-n",
        "auto",
        "-v",
        *session.posargs,
    )


@nox.session(python=PYTHON_VERSIONS, venv_backend="uv")
def test_slow(session: nox.Session) -> None:
    """Run slow and integration tests."""
    # Install dependencies
    session.run_install(
        "uv",
        "sync",
        "--no-default-groups",
        "--group",
        "tests",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )

    # Run slow/integration tests only with parallel execution
    session.run(
        "pytest",
        "tests",
        "-m",
        "slow or integration",
        "-n",
        "auto",
        "-v",
        *session.posargs,
    )


@nox.session(venv_backend="uv")
def test_examples(session: nox.Session) -> None:
    """Run marimo notebook examples to validate they execute."""
    # Install dependencies (both tests and examples groups needed)
    session.run_install(
        "uv",
        "sync",
        "--no-default-groups",
        "--group",
        "tests",
        "--group",
        "examples",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )

    # Run example tests in parallel using pytest with pytest-xdist with no coverage
    session.run(
        "pytest",
        "tests",
        "-m",
        "example",
        "-n",
        "auto",
        "-v",
        "--no-cov",
        *session.posargs,
    )


@nox.session(venv_backend="uv")
def test_docstrings(session: nox.Session) -> None:
    """Run docstring examples with pytest."""
    # Install dependencies
    session.run_install(
        "uv",
        "sync",
        "--no-default-groups",
        "--group",
        "tests",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )

    # Run doctest on source code
    session.run(
        "pytest",
        "--doctest-modules",
        "--doctest-continue-on-failure",
        "--no-cov",
        "src/yohou",
        *session.posargs,
    )


@nox.session(venv_backend="uv")
def lint(session: nox.Session) -> None:
    """Run linters and type checkers."""
    # Install dependencies
    session.run_install(
        "uv",
        "sync",
        "--no-default-groups",
        "--group",
        "lint",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )

    # Run ruff check
    session.run("ruff", "check", "src", "tests", external=True)

    # Run ty
    session.run("ty", "check", "src", external=True)


@nox.session(venv_backend="uv")
def fix(session: nox.Session) -> None:
    """Format the code base to adhere to our styles, and complain about what we cannot do automatically."""
    # Install dependencies
    session.run_install(
        "uv",
        "sync",
        "--no-default-groups",
        "--group",
        "dev",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    # Run pre-commit
    session.run("pre-commit", "run", "--all-files", "--show-diff-on-failure", *session.posargs, external=True)


@nox.session(venv_backend="uv")
def build_docs(session: nox.Session) -> None:
    """Build the documentation."""
    # Install dependencies
    session.run_install(
        "uv",
        "sync",
        "--no-default-groups",
        "--group",
        "docs",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )

    # Build the docs (hooks automatically export notebooks and prepare site)
    session.run("mkdocs", "build", "--clean", external=True)


@nox.session(venv_backend="uv")
def serve_docs(session: nox.Session) -> None:
    """Run a development server for working on documentation."""
    # Install dependencies
    session.run_install(
        "uv",
        "sync",
        "--no-default-groups",
        "--group",
        "docs",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )

    # Serve the docs (hooks automatically export notebooks and prepare site)
    session.log("###### Starting local server. Press Control+C to stop server ######")
    session.run("mkdocs", "serve", "-a", "localhost:8080", external=True)
