"""Export the Yohou example notebooks to static HTML.

Runs every marimo notebook under ``examples/`` and writes a rendered page per
notebook into ``docs/examples/<stem>/``. Exporting executes the notebook, which
dominates a docs build, so each export is cached against a hash of the
notebook's source.

Importable and runnable on its own::

    python docs/_notebooks.py

This module exists only when the project was generated with examples enabled --
the template does not emit it otherwise, so nothing here needs to guard on that.

It deliberately imports nothing from ``mkdocs``. The one place it needs mkdocs'
attention is the stem-collision warning, which reaches it by logging under the
``mkdocs`` logger tree by name; mkdocs counts those warnings and a ``--strict``
build fails on them, without this module importing anything.
"""

import hashlib
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Warnings logged under the "mkdocs" logger tree are counted by mkdocs and turn
# a --strict build red. Named, not imported: this module must stay free of
# mkdocs so it can run outside a docs build.
log = logging.getLogger("mkdocs.hooks")


# Written beside an exported notebook to record the source it was built from.
# Deliberately not a _CACHE module global: this one has to outlive the process,
# because its whole purpose is to skip work on a *later* build.
_SOURCE_HASH_FILE = ".source_hash"


def _notebook_content_hash(notebook):
    """Hash a notebook's source, to tell an unchanged one from an edited one."""
    return hashlib.sha256(notebook.read_bytes()).hexdigest()


def _is_cached(output_dir, expected_hash):
    """Whether this notebook's export is present and built from this exact source.

    Requires the rendered page *and* a matching hash. Checking the hash alone
    would reuse a directory whose html failed to write; checking the page alone
    would serve a stale render of an edited notebook forever.
    """
    hash_file = output_dir / _SOURCE_HASH_FILE
    if not (output_dir / "index.html").exists() or not hash_file.exists():
        return False
    try:
        return hash_file.read_text(encoding="utf-8").strip() == expected_hash
    except OSError:
        return False


def export(project_root):
    """Export every example notebook to ``docs/examples/<stem>/``.

    ``project_root`` is the directory holding ``examples/`` and ``docs/``.
    Raises ``RuntimeError`` if any notebook fails to execute.
    """
    examples_dir = project_root / "examples"

    if not examples_dir.exists():
        return

    # Find all marimo notebooks (recursively, excluding __marimo__ and bugs dirs)
    notebooks = [
        p
        for p in examples_dir.rglob("*.py")
        if "__marimo__" not in p.parts and "bugs" not in p.parts and "__init__" not in p.name
    ]
    if not notebooks:
        return

    # Checked before the skip below, not after: a stem collision is a property of
    # the source tree, not of the export, and check_docs -- the only place a
    # warning is fatal -- always sets MKDOCS_SKIP_NOTEBOOKS. Warning after the
    # return would fire only where nothing listens.
    #
    # The export dir is keyed on the stem alone, so two notebooks with the same
    # stem in different subdirectories write to one directory and the second
    # rmtree's the first. Both gallery cards then point at whichever won, and the
    # loser is unreachable with nothing said. The winner is filesystem-order
    # dependent: this walk is unsorted while the gallery's is sorted.
    seen_stems = {}
    for notebook in sorted(notebooks):
        first = seen_stems.setdefault(notebook.stem, notebook)
        if first is not notebook:
            log.warning(
                "notebook stem %r is used by both %s and %s; they export to the same page and only "
                "one survives. Rename one.",
                notebook.stem,
                first.relative_to(project_root),
                notebook.relative_to(project_root),
            )

    # Allow skipping slow notebook export during development
    if os.environ.get("MKDOCS_SKIP_NOTEBOOKS"):
        print("[hooks] MKDOCS_SKIP_NOTEBOOKS set, skipping notebook export")
        return

    docs_examples = project_root / "docs" / "examples"
    docs_examples.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []

    for notebook in notebooks:
        rel_path = notebook.relative_to(project_root)
        output_dir = docs_examples / notebook.stem

        # Exporting a notebook means executing it, which dominates the build.
        # Skip the ones whose source has not changed since their last export.
        content_hash = _notebook_content_hash(notebook)
        if _is_cached(output_dir, content_hash):
            print(f"[hooks] unchanged, reusing export: {rel_path}")
            continue

        # Clean previous export artifacts before re-exporting
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export static HTML (read-only view)
        static_file = output_dir / "index.html"
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "marimo",
                    "-y",
                    "-q",
                    "export",
                    "html",
                    "--no-sandbox",
                    str(notebook),
                    "-o",
                    str(static_file),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"[hooks] exported html {rel_path} -> {static_file.relative_to(project_root)}")
            # Stamp the source hash only after a successful export, so a failed
            # or interrupted run re-exports next time instead of caching a
            # half-written page.
            (output_dir / _SOURCE_HASH_FILE).write_text(content_hash, encoding="utf-8")
        except subprocess.CalledProcessError as e:
            failed.append(str(rel_path))
            print(f"[hooks] FAILED html {rel_path}: {e}", file=sys.stderr)
            if e.stderr:
                print(e.stderr, file=sys.stderr)
            continue
        except FileNotFoundError:
            print("[hooks] marimo not found, skipping notebook export", file=sys.stderr)
            break

    if failed:
        msg = f"[hooks] {len(failed)} notebook(s) had cell execution errors:\n"
        msg += "\n".join(f"  - {f}" for f in failed)
        raise RuntimeError(msg)


def main():
    """Export the notebooks for the project this file lives in."""
    export(Path(__file__).parent.parent)


if __name__ == "__main__":
    main()
