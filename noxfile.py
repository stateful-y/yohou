"""Nox sessions."""

import nox

# Require Nox version 2024.3.2 or newer to support the 'default_venv_backend' option
nox.needs_version = ">=2024.3.2"

# Set 'uv' as the default backend for creating virtual environments
nox.options.default_venv_backend = "uv|virtualenv"

# Default sessions to run when nox is called without arguments
nox.options.sessions = ["fix", "tests_coverage", "docs"]


@nox.session(python=["3.12"], venv_backend="uv")
def tests_coverage(session: nox.Session) -> None:
    """Run the tests with pytest under the specified Python version."""
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

    # Clears all .coverage* files
    session.run("coverage", "erase")

    # Run unit tests under coverage
    session.run(
        "coverage",
        "run",
        "--parallel-mode",
        "--source=src/yohou",
        "-m",
        "pytest",
        "src/yohou",
        f"--junitxml=junit.{session.python}.xml",
        *session.posargs,
    )

    # Combine coverage data from parallel runs
    session.run("coverage", "combine")

    # HTML report, ignoring parse errors and without contexts
    session.run("coverage", "html", "--ignore-errors", "-d", session.create_tmp())

    # XML report for CI
    session.run("coverage", "xml", "-o", f"coverage.{session.python}.xml")


@nox.session(venv_backend="uv")
def fix(session: nox.Session) -> None:
    """Run pre-commit hooks."""
    session.run_install(
        "uv",
        "sync",
        "--no-default-groups",
        "--group",
        "fix",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session(venv_backend="uv")
def docs(session: nox.Session) -> None:
    """Build the documentation."""
    session.run_install(
        "uv",
        "sync",
        "--no-default-groups",
        "--group",
        "docs",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    # Build the docs
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
    # Build and serve the docs
    session.run("mkdocs", "build", "--clean", external=True)
    session.log("###### Starting local server. Press Control+C to stop server ######")
    session.run("mkdocs", "serve", "-a", "localhost:8080", external=True)


@nox.session(venv_backend="uv")
def deploy_docs(session: nox.Session) -> None:
    """Build fresh docs and deploy them."""
    # Install dependencies
    session.run_install(
        "uv",
        "sync",
        "--no-default-groups",
        "--group",
        "docs",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    # Deploy docs to GitHub pages
    session.run("mkdocs", "gh-deploy", "--clean", external=True)
