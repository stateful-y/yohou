"""Marker resolution and the API sidebar TOC, as engine-independent hosts.

This is what ``hooks.on_page_markdown`` and the ``module_toc`` half of
``hooks.on_page_content`` used to do, plus every builder they relied on, now
hosted in a Python-Markdown extension instead of mkdocs hooks. mkdocs and the
successor engine (Zensical) both drive the same Python-Markdown, so this keeps
working when the ``hooks:`` key does not.

Two facts make it possible, both settled by spike (see the change's design):

1. **Loading.** A markdown extension is loaded by *module name* at config time,
   before any hook runs, so ``docs_build/`` must be importable then.
   ``dev-mode-dirs = ["."]`` in the wheel target puts the repo root on
   ``sys.path`` in the editable install, so ``docs_build._markers`` resolves --
   with nothing added to the built wheel.
2. **Page context.** A Preprocessor is handed the ``Markdown`` instance, not the
   page. ``_current_page(md)`` recovers it the same way mkdocstrings does: the
   Zensical page provider if present, else the ``mkdocs-autorefs`` plugin's
   ``current_page``. Writing ``page.meta`` from the Preprocessor reaches the
   template intact -- the mechanism the ``module_toc`` sidebar depends on.

The ``<!-- SUBPAGES -->`` index is built off the filesystem: a markdown extension
never receives mkdocs' ``files`` collection, so it scans the index page's own
directory instead.
"""

import ast
import contextlib
import logging
import posixpath
import re
import sys
from pathlib import Path

import yaml
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor

# Loaded by path/name as an extension, ``docs_build/`` is not a package here, so
# put this file's own directory on sys.path and import the build steps and the
# shared git-ref helper as plain modules -- the pattern hooks.py used before it
# was deleted.
sys.path.insert(0, str(Path(__file__).parent))

import _api_pages  # noqa: E402
from _api_pages import (  # noqa: E402
    _get_api_name_lookup,  # only the examples/gallery path needs the name lookup
    _get_public_members,
    _get_root_members,
    _get_submodules,
    _qualified_name,
)
from _git_ref import git_ref  # only the examples playground link needs the ref  # noqa: E402

# The docs tree is a sibling of this file, so the project root is two up. A
# markdown extension is not handed mkdocs' config, so paths anchor here.
_PROJECT_ROOT = Path(__file__).parent.parent

# Warnings logged under the "mkdocs" logger tree are counted by mkdocs and turn a
# --strict build red. Every marker is silently inert when it does not resolve, so
# warning here is what makes a dead marker a build failure instead of blank space.
log = logging.getLogger("mkdocs.hooks")

# Per-build caches. The module stays imported across a `mkdocs serve` session, so
# these live for the process; `reset_caches()` clears them for tests and any
# in-process rebuild. There is no `on_config` to reset them now, and none is
# needed: `serve.py` resets the API-discovery caches when a source edit
# regenerates the pages, and the gallery keys off notebooks the preview does not
# watch, so within a build they are filled once and reused.
_GALLERY_CACHE = None
_COMPANION_INDEX_CACHE = None
_GALLERY_PAGE_CACHE = None
_NOTEBOOK_API_USAGE_CACHE = None

# Max example cards on a single API page. Most symbols are well under this; the
# cap exists for widely used helpers, where an uncapped list runs to dozens of
# cards and stops being scannable. The overflow link keeps the rest reachable.
_API_EXAMPLES_CAP = 6


def reset_caches():
    """Clear the per-build discovery caches.

    Kept for tests and any caller that rebuilds in-process. A single `mkdocs
    build` fills each cache once and never calls this; `serve.py` calls
    `_api_pages.reset_caches()` directly on a source edit.
    """
    _api_pages.reset_caches()
    global _GALLERY_CACHE, _COMPANION_INDEX_CACHE, _NOTEBOOK_API_USAGE_CACHE, _GALLERY_PAGE_CACHE  # noqa: PLW0603
    _GALLERY_CACHE = None
    _COMPANION_INDEX_CACHE = None
    _NOTEBOOK_API_USAGE_CACHE = None
    _GALLERY_PAGE_CACHE = None


def _current_page(md):
    """Return the page being rendered, or ``None`` if it cannot be found.

    Under Zensical the page provider is the preprocessor registered as
    ``rendering_context`` (``zensical.extensions.context.ContextPreprocessor``),
    which exposes ``.page`` -- a ``Page`` with ``url``/``path``/``title``/``meta``.
    An earlier version of this function probed a ``zensical_current_page`` key that
    does not exist in Zensical, so every page-context marker silently rendered
    empty at a green ``--strict`` build; read ``.page`` off ``rendering_context``
    instead. Under MkDocs there is no such seam, so reach the ``mkdocs-autorefs``
    plugin's ``current_page`` through the processors it registered on this md
    instance -- exactly how mkdocstrings gets the page.
    """
    with contextlib.suppress(KeyError, TypeError, AttributeError):
        rc = md.preprocessors["rendering_context"]
        if getattr(rc, "page", None) is not None:
            return rc.page

    for registry in (md.treeprocessors, md.inlinePatterns, md.preprocessors):
        for proc in registry:
            plugin = getattr(proc, "_plugin", None) or getattr(proc, "plugin", None)
            if plugin is not None and hasattr(plugin, "current_page"):
                return plugin.current_page
    return None


def _page_src_path(page):
    """Source path of ``page`` under either documentation engine.

    MkDocs pages carry ``page.file.src_path``; Zensical's ``Page`` has
    ``page.path``. Every marker that lists a page's siblings or resolves a
    page-relative link needs this identity, so it must read whichever shape the
    engine in use provides.
    """
    file = getattr(page, "file", None)
    if file is not None and getattr(file, "src_path", None) is not None:
        return file.src_path
    return page.path


def _mkdocs_config():
    """Read ``mkdocs.yml`` for the keys the markers need (``nav``, ``repo_url``).

    A markdown extension gets the handler/extension config, not the mkdocs
    config, so it reads the file directly. ``!!python/name:`` (pymdownx.emoji) and
    ``!ENV`` are tolerated, the same reader ``_source_links`` and ``_api_pages``
    use, so a strict loader does not raise on them.
    """
    config_file = _PROJECT_ROOT / "mkdocs.yml"
    if not config_file.exists():
        return {}

    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_multi_constructor("tag:yaml.org,2002:python/name:", lambda _loader, suffix, _node: suffix)
    _Loader.add_constructor("!ENV", lambda _loader, _node: None)
    try:
        return yaml.load(config_file.read_text(encoding="utf-8"), Loader=_Loader) or {}
    except yaml.YAMLError:
        return {}


def _site_root_prefix(page):
    """Relative path from `page`'s rendered URL back to the site root.

    Every link injected is relative, because the site may be served under a
    subpath and `use_directory_urls` makes each page its own directory. A
    hardcoded `../../` only works if the page never moves: a project is free to
    put its API index at `pages/api/index.md` rather than the template's
    `pages/reference/api.md`, and a fixed prefix silently 404s every link on it.
    """
    parts = _page_src_path(page).split("/")
    depth = len(parts) if parts[-1] != "index.md" else len(parts) - 1
    return "../" * depth


def _build_api_table_html(project_root, prefix):
    """Build an HTML <table> for the API index with DataTables init.

    Lists every public class and function across all submodules with
    Name, Type, Module, and Description columns.  The table is initialised
    with jQuery DataTables for client-side filtering and sorting.
    """
    modules = _get_submodules(project_root)

    rows = []
    scans = []
    for mod in modules:
        # Keyed on the module NAME: discovery no longer takes a source path, so
        # the single-file-or-package probe and its `.exists()` guard are gone.
        scans.append((mod["module_name"], _get_public_members(project_root, mod["module_name"])))
    # Symbols exported only from the package root belong to no submodule, so a
    # loop over submodules alone leaves them out of the table entirely.
    scans.append(("", _get_root_members(project_root)))

    for module_name, members in scans:
        module_label = _qualified_name(module_name, "").rstrip(".") or "yohou"
        # A root export has no module page, and there is nothing to link it to: this
        # pointed at pages/api/, which is only a directory of generated module pages
        # and has no index of its own, so every root export's Module cell was a 404.
        # Nothing catches that -- the cell is raw HTML, which --strict never
        # validates, and only a project with a root-only export renders one.
        module_href = f"{prefix}pages/api/{module_name}/" if module_name else None

        for cls in members["classes"]:
            qualified = _qualified_name(module_name, cls["name"])
            rows.append((cls["name"], "Class", module_label, module_href, cls["doc"], qualified))

        for func in members["functions"]:
            qualified = _qualified_name(module_name, func["name"])
            rows.append((func["name"], "Function", module_label, module_href, func["doc"], qualified))

    rows.sort(key=lambda r: r[0].lower())

    _type_badge_cls = {
        "Class": "api-badge--class",
        "Function": "api-badge--function",
    }

    tbody_lines = []
    for name, kind, module_label, module_href, desc, qualified in rows:
        href = f"{prefix}pages/api/generated/{qualified}/"
        badge_cls = _type_badge_cls.get(kind, "")
        module_cell = f'<a href="{module_href}">{module_label}</a>' if module_href else module_label
        tbody_lines.append(
            f"      <tr>"
            f'<td><a href="{href}"><code>{name}</code></a></td>'
            f'<td><span class="api-badge {badge_cls}">{kind}</span></td>'
            f"<td>{module_cell}</td>"
            f"<td>{desc}</td>"
            f"</tr>"
        )

    tbody = "\n".join(tbody_lines)
    return (
        '<div class="api-table-wrapper">\n'
        '<table id="api-table" class="display" style="width:100%">\n'
        "  <thead>\n"
        "    <tr>\n"
        "      <th>Name</th>\n"
        "      <th>Type</th>\n"
        "      <th>Module</th>\n"
        "      <th>Description</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        f"{tbody}\n"
        "  </tbody>\n"
        "</table>\n"
        "</div>\n"
        "\n"
        "<script>\n"
        "document.addEventListener('DOMContentLoaded', function() {\n"
        '  if (typeof jQuery !== "undefined" && jQuery.fn.DataTable) {\n'
        '    jQuery("#api-table").DataTable({\n'
        "      pageLength: 25,\n"
        '      order: [[0, "asc"]],\n'
        "      columns: [\n"
        "        null,\n"
        "        null,\n"
        "        null,\n"
        "        { orderable: false }\n"
        "      ],\n"
        "      language: {\n"
        '        search: "",\n'
        '        searchPlaceholder: "Filter API reference...",\n'
        '        info: "Showing _START_ to _END_ of _TOTAL_ entries",\n'
        '        lengthMenu: "Show _MENU_",\n'
        "      },\n"
        '      dom: \'<"api-controls"fl>t<"api-footer"ip>\',\n'
        "    });\n"
        "  }\n"
        "});\n"
        "</script>"
    )


def _get_gallery_items(project_root):
    """Parse ``__gallery__`` metadata from all example notebooks (cached)."""
    global _GALLERY_CACHE  # noqa: PLW0603
    if _GALLERY_CACHE is not None:
        return _GALLERY_CACHE

    examples_dir = project_root / "examples"
    if not examples_dir.exists():
        _GALLERY_CACHE = []
        return _GALLERY_CACHE

    items = []
    for notebook in sorted(examples_dir.rglob("*.py")):
        if "__marimo__" in notebook.parts or "bugs" in notebook.parts:
            continue
        if "__init__" in notebook.name:
            continue

        try:
            source = notebook.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue

        gallery = None
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__gallery__":
                        with contextlib.suppress(ValueError, TypeError):
                            gallery = ast.literal_eval(node.value)

        if not gallery or not isinstance(gallery, dict):
            continue

        stem = notebook.stem
        # The export is flat (docs/examples/<stem>/), so [View] keys on the stem.
        # The playground is not: it reconstructs the real repo path from this URL,
        # so it must carry the notebook's path relative to examples/. Building it
        # from the stem assumes a flat examples/ dir and 404s for every notebook
        # in a subdirectory -- 78 of yohou's 79. The link is generated rather than
        # authored, so mkdocs never validates it and --strict stays green.
        rel_parts = notebook.relative_to(examples_dir).with_suffix("").parts
        view_path = f"/examples/{stem}/"
        open_path = "/examples/" + "/".join(rel_parts) + "/edit/"

        # api_references: absent (None) means "infer from imports"; an empty
        # list means "this notebook belongs on no API page" -- a deliberate
        # statement, not a default.
        api_references = gallery.get("api_references")
        if api_references is not None and not isinstance(api_references, list):
            api_references = None

        items.append({
            "title": gallery.get("title", stem.replace("_", " ").title()),
            "description": gallery.get("description", ""),
            "category": gallery.get("category", ""),
            # `category` is the Diataxis kind (tutorial / how-to); `section` is
            # the topic grouping a gallery splits into once it outgrows one
            # page. They are independent: a section holds both kinds.
            "section": gallery.get("section", ""),
            "api_references": api_references,
            "companion": gallery.get("companion"),
            "view_path": view_path,
            "open_path": open_path,
            "stem": stem,
        })

    _GALLERY_CACHE = items
    return _GALLERY_CACHE


_SECTION_GALLERY_RE = re.compile(r"<!-- GALLERY:section:([\w.-]+) -->")


def _get_gallery_sections(project_root):
    """Every section name declared by a notebook, in first-seen order."""
    seen = []
    for item in _get_gallery_items(project_root):
        section = item.get("section")
        if section and section not in seen:
            seen.append(section)
    return seen


def _build_gallery_html(project_root, section=None):
    """Build gallery card grid as Material 'grid cards' markdown, grouped by category.

    ``section`` narrows the grid to the notebooks declaring that ``section``,
    which is how a gallery too big for one page splits across subpages. The
    category grouping still applies inside the section: a topic holds both
    tutorials and how-tos, and the split is by topic, not by kind.
    """
    items = _get_gallery_items(project_root)

    if section is not None:
        items = [item for item in items if item.get("section") == section]
        if not items:
            # An author renamed a section, or misspelled one. Either way the
            # page silently loses its whole card grid, so refuse to be quiet.
            known = ", ".join(_get_gallery_sections(project_root)) or "none"
            log.warning(
                "gallery section %r matches no notebook (declared sections: %s). "
                "The page requesting it renders no cards.",
                section,
                known,
            )
            return f"<!-- no gallery items in section: {section} -->\n"

    if not items:
        return "<!-- no gallery items found -->\n"

    # Group items by category, preserving order within each group
    _CATEGORY_ORDER = ["tutorial", "how-to"]
    _CATEGORY_HEADINGS = {
        "tutorial": "Tutorials",
        "how-to": "How-to Guides",
    }

    grouped: dict[str, list[dict]] = {}
    for item in items:
        cat = item.get("category") or "other"
        grouped.setdefault(cat, []).append(item)

    sections = []
    for cat in _CATEGORY_ORDER:
        group = grouped.pop(cat, [])
        if not group:
            continue
        heading = _CATEGORY_HEADINGS.get(cat, cat.title())
        cards = _build_gallery_cards(group)
        sections.append(f"## {heading}\n\n{cards}")

    # Remaining uncategorized items
    for _cat, group in grouped.items():
        cards = _build_gallery_cards(group)
        sections.append(cards)

    return "\n\n".join(sections) + "\n"


def _build_gallery_cards(items):
    """Build a Material 'grid cards' block from a list of gallery items."""
    cards = []
    for item in items:
        desc = item["description"] or "No description."
        cards.append(
            f"-   **{item['title']}**\n"
            f"\n"
            f"    ---\n"
            f"\n"
            f"    {desc}\n"
            f"\n"
            f"    [View]({item['view_path']}) · "
            f"[Open in marimo]({item['open_path']})"
        )

    return '<div class="grid cards" markdown>\n\n' + "\n\n".join(cards) + "\n\n</div>\n"


def _get_gallery_page_url(project_root):
    """URL of the page hosting the gallery, or None if there is not one.

    The gallery page is whichever page carries the ``<!-- GALLERY -->``
    placeholder -- found by looking, because that page is local-owned and a
    project is free to move it.  A hardcoded path is wrong the moment someone
    reorganises their docs, and produces a 404 with no build error.

    ``index.md`` is dropped from the URL: under ``use_directory_urls`` (mkdocs'
    default) ``pages/examples/index.md`` serves at ``/pages/examples/``, not at
    ``/pages/examples/index/``.  This link is emitted as raw HTML, so mkdocs
    never validates it and even --strict cannot see it break -- only RTD's
    post-build linkchecker catches it.
    """
    global _GALLERY_PAGE_CACHE  # noqa: PLW0603
    if _GALLERY_PAGE_CACHE is not None:
        return _GALLERY_PAGE_CACHE or None

    docs_dir = project_root / "docs"
    _GALLERY_PAGE_CACHE = ""
    if not docs_dir.exists():
        return None

    def _url_of(md):
        rel = md.relative_to(docs_dir).with_suffix("")
        parts = rel.parts[:-1] if rel.name == "index" else rel.parts
        return "/" + "".join(f"{part}/" for part in parts)

    sectioned = []
    for md in sorted(docs_dir.rglob("*.md")):
        try:
            text = md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "<!-- GALLERY -->" in text:
            _GALLERY_PAGE_CACHE = _url_of(md)
            return _GALLERY_PAGE_CACHE
        if _SECTION_GALLERY_RE.search(text):
            sectioned.append(md)

    # A gallery too big for one page has no bare <!-- GALLERY --> at all: it is a
    # directory of <!-- GALLERY:section:… --> pages behind an index. Looking only
    # for the bare marker returns None for those projects, and the caller drops
    # the "see all N examples" link on an `and gallery_url` -- so the API pages
    # for the most-used symbols, which are the ones that overflow the cap, link to
    # nothing at all. Silently: no marker is involved, so nothing warns.
    if sectioned:
        directory = sectioned[0].parent
        if all(md.parent == directory for md in sectioned) and (directory / "index.md").is_file():
            _GALLERY_PAGE_CACHE = _url_of(directory / "index.md")

    return _GALLERY_PAGE_CACHE or None


def _normalize_companion_path(path):
    """Normalize a companion path so authored variants compare equal.

    ``companion`` is hand-written, so it turns up as ``/pages/how-to/x/``,
    ``pages/how-to/x`` and ``pages/how-to/x.md``.  All three mean the same
    page; normalizing beats making the author guess the one true spelling.
    """
    return str(path).strip("/").removesuffix(".md").removesuffix("/index")


def _get_companion_index(project_root):
    """Reverse map: normalized doc page path -> notebooks naming it (cached)."""
    global _COMPANION_INDEX_CACHE  # noqa: PLW0603
    if _COMPANION_INDEX_CACHE is not None:
        return _COMPANION_INDEX_CACHE

    index: dict[str, list[dict]] = {}
    for item in _get_gallery_items(project_root):
        companion = item.get("companion")
        if companion:
            index.setdefault(_normalize_companion_path(companion), []).append(item)
    _COMPANION_INDEX_CACHE = index
    return _COMPANION_INDEX_CACHE


def _build_companion_cards_html(project_root, page_src_uri):
    """Build cards for notebooks declaring this page as their companion."""
    items = _get_companion_index(project_root).get(_normalize_companion_path(page_src_uri), [])
    if not items:
        return ""
    # The heading is emitted here rather than written into the page, so a page
    # whose companions were removed renders nothing at all instead of a heading
    # with an empty section under it.
    cards = _build_gallery_cards(sorted(items, key=lambda item: item["title"]))
    return f"## Try it interactively\n\n{cards}"


def _get_notebook_api_usage(project_root):
    """Build reverse map: qualified API name → list of gallery items that use it.

    Scans example notebooks for ``from yohou.* import …``
    statements and maps each imported name back to its fully-qualified
    API identifier.
    """
    global _NOTEBOOK_API_USAGE_CACHE  # noqa: PLW0603
    if _NOTEBOOK_API_USAGE_CACHE is not None:
        return _NOTEBOOK_API_USAGE_CACHE

    name_to_qualified = _get_api_name_lookup(project_root)

    gallery_items = _get_gallery_items(project_root)
    stem_to_item = {item["stem"]: item for item in gallery_items}

    usage: dict[str, list[dict]] = {}
    examples_dir = project_root / "examples"
    if not examples_dir.exists():
        _NOTEBOOK_API_USAGE_CACHE = {}
        return _NOTEBOOK_API_USAGE_CACHE

    for notebook in sorted(examples_dir.rglob("*.py")):
        if "__marimo__" in notebook.parts or "bugs" in notebook.parts:
            continue
        if "__init__" in notebook.name:
            continue

        stem = notebook.stem
        item = stem_to_item.get(stem)
        if item is None:
            continue

        # Declared api_references win; import scanning is the fallback.
        # Import scanning cannot tell a symbol a notebook *demonstrates* from
        # one it merely uses as scaffolding, so an author saying which is which
        # beats inference.  But a notebook that says nothing must still work --
        # a fresh project has no metadata, and the feature has to be visible
        # before anyone opts in.
        declared = item.get("api_references")
        if declared is not None:
            imported_names = set(declared)
        else:
            try:
                source = notebook.read_text(encoding="utf-8")
                tree = ast.parse(source)
            except (SyntaxError, UnicodeDecodeError):
                continue

            # Extract names imported from yohou.*
            imported_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("yohou"):
                    for alias in node.names:
                        imported_names.add(alias.name)

        for imp_name in imported_names:
            qualified = name_to_qualified.get(imp_name)
            if qualified is not None:
                usage.setdefault(qualified, []).append(item)

    _NOTEBOOK_API_USAGE_CACHE = usage
    return _NOTEBOOK_API_USAGE_CACHE


def _build_api_examples_html(project_root, qualified_name):
    """Build Material grid cards for example notebooks that use a given API object."""
    usage = _get_notebook_api_usage(project_root)
    items = usage.get(qualified_name, [])

    if not items:
        return ""

    # Deduplicate by stem
    seen: set[str] = set()
    unique_items: list[dict] = []
    for item in items:
        if item["stem"] not in seen:
            seen.add(item["stem"])
            unique_items.append(item)

    # Bound the list.  A widely used helper accumulates a card per notebook,
    # so the most-used symbols get the longest and least useful lists -- the
    # failure is inversely proportional to usefulness.  Curation alone does not
    # fix this: a genuinely central class really is used by dozens of
    # notebooks.  Sort by title so the selection is stable across builds.
    unique_items.sort(key=lambda item: item["title"])
    total = len(unique_items)
    shown = unique_items[:_API_EXAMPLES_CAP]

    # "Tutorials", not "Examples", and h3 rather than h2. This markdown is ours,
    # injected at an `<!-- EXAMPLES_FOR -->` marker, so it can simply say what it
    # means -- it used to emit `## Examples` and a post-render pass rewrote the
    # element to `<h3 id="tutorials">Tutorials</h3>`. Emitting it correctly here
    # means the toc extension sees a real heading and gives it `#tutorials` for
    # free, and it keeps the name distinct from the docstring's own Examples
    # section, which is a different thing and owns `#doc-examples`.
    html = "### Tutorials\n\nThe following example notebooks use this component:\n\n" + _build_gallery_cards(shown)
    gallery_url = _get_gallery_page_url(project_root)
    if total > _API_EXAMPLES_CAP:
        if gallery_url:
            html += f"\n[See all {total} examples in the gallery]({gallery_url})\n"
        else:
            # The symbols that overflow the cap are the most-used ones, so this
            # drops the link exactly where the remaining examples matter most --
            # and does it on an `if`, with no marker involved for the catch-all
            # to notice. yohou renders 6 of PointReductionForecaster's 45 and
            # links to none of the other 39.
            log.warning(
                "%s has %d examples but no gallery page to link the other %d to. Give the "
                "gallery index a <!-- GALLERY --> marker, or put its "
                "<!-- GALLERY:section:… --> pages behind an index.md.",
                qualified_name,
                total,
                total - _API_EXAMPLES_CAP,
            )
    return html


# ---------------------------------------------------------------------------
# Marker substitution
# ---------------------------------------------------------------------------


# Every name this extension substitutes. A comment *opening* with one of these is
# a marker; if one survives to the end of `_inject`, it was misspelled. Matched
# without a word boundary so `<!-- GALLERY:quickstart -->` and `<!-- SUBPAGES_FOO
# -->` are both caught, not just the separator-delimited ones.
#
# The net is deliberately the marker namespace and nothing else: it cannot catch
# a typo that mangles the name itself (`<!-- GALLRY -->`), because widening it to
# every upper-case comment would flag ordinary `<!-- TODO -->`s.
_MARKER_NAMES = ("API_TABLE", "SUBPAGES", "GALLERY", "COMPANION_NOTEBOOKS", "EXAMPLES_FOR")
_UNHANDLED_MARKER_RE = re.compile(r"<!--\s*(?:" + "|".join(_MARKER_NAMES) + r")[^>]*-->")


def _warn_on_unhandled_markers(markdown, src_path):
    """Warn about a marker that no substitution above recognised.

    The per-marker warnings only fire for a *well-formed* marker that resolves to
    nothing. A misspelled one is worse and was completely silent: `<!--
    GALLERY:quickstart -->` matches neither the bare nor the sectioned pattern, so
    nothing claimed it, nothing substituted it, and it shipped to the page as a
    raw comment that renders as blank space. Catching the leftovers is the only
    place a typo in the marker namespace can be noticed at all.
    """
    for match in _UNHANDLED_MARKER_RE.finditer(markdown):
        log.warning(
            "%s: unrecognised marker %s -- it renders as blank space. "
            "Known markers: <!-- API_TABLE -->, <!-- SUBPAGES -->, <!-- GALLERY -->, "
            "<!-- GALLERY:section:NAME -->, <!-- COMPANION_NOTEBOOKS -->, <!-- EXAMPLES_FOR:NAME -->.",
            src_path,
            match.group(0),
        )


def _replace_marker(markdown, marker, replacement):
    """Replace ``marker`` with ``replacement``, re-indented to the marker's column.

    A marker nested inside an indented block -- an admonition body, a list item
    -- carries leading whitespace that its replacement has to inherit. A plain
    ``str.replace`` indents only the first line, so every line after it lands at
    column 0 and silently falls out of the enclosing block: the block keeps the
    first line and the rest renders as a sibling. That failure is invisible in
    the markdown and only shows up in the built HTML, which is why it survived
    so long. Matching the indentation keeps the replacement inside whatever the
    author nested it in.
    """
    if marker not in markdown:
        return markdown

    out = []
    for line in markdown.split("\n"):
        stripped = line.strip()
        if stripped != marker:
            # A marker sharing its line with prose is substituted in place; it
            # was never nested, so there is no indentation to match.
            out.append(line.replace(marker, replacement) if marker in line else line)
            continue
        indent = line[: len(line) - len(line.lstrip())]
        if not replacement:
            continue
        if not indent:
            out.append(replacement)
            continue
        # Blank lines stay blank: trailing whitespace on an "empty" line is a
        # lint violation, and markdown does not need it to keep the block open.
        out.extend(indent + rline if rline.strip() else "" for rline in replacement.split("\n"))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Section index (<!-- SUBPAGES -->)
# ---------------------------------------------------------------------------


_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_DESCRIPTION_RE = re.compile(r"^description:\s*(.+?)\s*$", re.MULTILINE)


def _nav_entries(config):
    """Map ``src_path`` -> (position in the nav, title the nav gives it).

    The nav is the order the author chose and the order the reader sees in the
    sidebar; an index that lists its pages in a different order than the nav
    beside it reads as a different set of pages. The title is a fallback for a
    page that has no H1 of its own -- see _page_title_and_description.
    """
    entries = {}

    def walk(node, title=None):
        if isinstance(node, str):
            entries.setdefault(node, (len(entries), title))
        elif isinstance(node, list):
            for child in node:
                walk(child)
        elif isinstance(node, dict):
            for key, value in node.items():
                walk(value, key if isinstance(value, str) else None)

    walk(config.get("nav") or [])
    return entries


def _page_title_and_description(abs_path):
    """Pull a page's title and one-line summary from its own source.

    Title is the H1; summary is the frontmatter ``description`` when present,
    else the first prose paragraph. Deriving both from the page keeps the index
    honest -- there is no second copy of the title to drift out of sync.
    """
    try:
        text = Path(abs_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None, ""

    description = ""
    frontmatter = _FRONTMATTER_RE.match(text)
    if frontmatter:
        found = _DESCRIPTION_RE.search(frontmatter.group(1))
        if found:
            description = found.group(1).strip().strip("\"'")
        text = text[frontmatter.end() :]

    heading = _H1_RE.search(text)
    if not heading:
        return None, description
    title = heading.group(1).strip()

    if not description:
        body = text[heading.end() :]
        for raw_block in body.split("\n\n"):
            block = raw_block.strip()
            # Skip anything that is not prose: nested headings, markers,
            # admonitions, code fences, tables, images, lists.
            if not block or block[0] in "#<!|-*>`" or block.startswith("!!!"):
                continue
            description = " ".join(block.split())
            break

    return title, description


def _build_subpages_list(config, page, project_root):
    """List the pages this index introduces, as ``- [Title](slug.md): summary``.

    Off the filesystem: the hook version iterated mkdocs' ``files`` collection,
    which a markdown extension is never handed. It scans the index page's own
    directory instead -- direct children only, since a nested section owns its
    own index -- and reads each sibling's title and summary out of its own
    source, so there is no second copy of either to drift.
    """
    src = _page_src_path(page)
    directory = posixpath.dirname(src)
    dir_path = project_root / "docs" / directory

    siblings = []
    for candidate in sorted(dir_path.glob("*.md")):
        name = candidate.name
        candidate_src = f"{directory}/{name}" if directory else name
        if candidate_src == src or name == "index.md":
            continue
        siblings.append((candidate_src, candidate))

    if not siblings:
        log.warning("<!-- SUBPAGES --> on %s, which has no sibling pages to list.", src)
        return "<!-- no subpages -->\n"

    entries = _nav_entries(config)

    # The index enumerates sibling *files*; the sidebar comes from mkdocs.yml.
    # When the two disagree the index quietly papers over it -- an entry dropped
    # from the nav still appears here, so the page stays reachable by link while
    # vanishing from navigation. mkdocs reports not-in-nav pages at INFO, which
    # --strict does not fail on, so nothing else says a word.
    orphans = sorted(candidate_src for candidate_src, _ in siblings if candidate_src not in entries)
    if orphans:
        log.warning(
            "%s lists %s, which %s missing from the nav in mkdocs.yml -- the page is linked but "
            "unreachable by navigation.",
            src,
            ", ".join(orphans),
            "is" if len(orphans) == 1 else "are",
        )

    rows = []
    for candidate_src, candidate in siblings:
        title, description = _page_title_and_description(str(candidate))
        position, nav_title = entries.get(candidate_src, (len(entries) + 1, None))
        if title is None:
            # A page can legitimately have no H1 in its own source: a bare
            # `--8<-- "CHANGELOG.md"` include grows one only once snippets
            # expand, which is after this runs. The nav already names such a
            # page, so prefer its name over dropping the page from its own index.
            title = nav_title
        if title is None:
            log.warning(
                "%s has no H1 heading and no nav title; omitted from the %s index.",
                candidate_src,
                src,
            )
            continue
        rows.append((position, title, posixpath.basename(candidate_src), description))

    if not rows:
        return "<!-- no subpages -->\n"

    rows.sort(key=lambda row: (row[0], row[1]))
    lines = [f"- [{title}]({slug})" + (f": {desc}" if desc else "") for _, title, slug, desc in rows]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# API sidebar module TOC (page.meta, read by the api-submodule template)
# ---------------------------------------------------------------------------


def _build_module_toc(project_root, current_src_path=None, prefix=None):
    """Build the module TOC list used by the api-submodule sidebar template.

    ``current_src_path`` marks the matching entry ``active``; ``prefix`` makes
    every url site-root relative, so the TOC is correct on any page that renders
    it. Reads the generated module pages under ``docs/pages/api/`` off the
    filesystem -- they exist by the time this runs, the prebuild step generates
    them before mkdocs is invoked.
    """
    docs_dir = project_root / "docs"
    api_dir = docs_dir / "pages" / "api"

    modules = _get_submodules(project_root)
    module_toc = []

    for mod in modules:
        md_filename = f"{mod['module_name']}.md"
        md_path = api_dir / md_filename
        if not md_path.exists():
            continue

        page_url = f"{prefix}pages/api/{md_filename.replace('.md', '/')}"
        active = current_src_path == f"pages/api/{md_filename}" if current_src_path else False

        entry = {
            "title": f"yohou.{mod['module_name']}",
            "url": page_url,
            "active": active,
            "children": [],
        }

        # Parse h3 subsections from the module markdown for sidebar children
        content = md_path.read_text(encoding="utf-8")
        for m in re.finditer(r"^###\s+(.+)$", content, re.MULTILINE):
            sub_title = m.group(1).strip()
            sub_slug = re.sub(r"[^\w]+", "-", sub_title.lower()).strip("-")
            child_url = f"{page_url}#{sub_slug}" if not active else f"#{sub_slug}"
            entry["children"].append({"title": sub_title, "url": child_url, "active": False})

        module_toc.append(entry)

    return module_toc


def _set_module_toc(page):
    """Attach the API sidebar module TOC to ``page.meta`` for the template.

    The api-index / api-submodule pages declare their template in frontmatter, so
    ``page.meta`` already carries it by the time this Preprocessor runs. Keyed on
    the declared template, not on where the page sits: the index is wherever a
    project put it, and a hardcoded ``pages/reference/api.md`` leaves a relocated
    index with an empty sidebar and nothing erroring.
    """
    meta = getattr(page, "meta", None)
    if not isinstance(meta, dict):
        return
    if meta.get("template") in ("api-index.html", "api-submodule.html"):
        meta["module_toc"] = _build_module_toc(
            _PROJECT_ROOT, current_src_path=_page_src_path(page), prefix=_site_root_prefix(page)
        )


def _inject(markdown, page, config=None):
    """Resolve every marker in ``markdown`` for ``page`` and return the result.

    This is the body of the retired ``on_page_markdown`` hook. ``config`` is read
    from ``mkdocs.yml`` when not supplied (the Preprocessor path); tests pass one
    to exercise a specific nav or ``repo_url`` without touching the file.
    """
    if config is None:
        config = _mkdocs_config()
    project_root = _PROJECT_ROOT
    prefix = _site_root_prefix(page)

    # API_TABLE placeholder
    if "<!-- API_TABLE -->" in markdown:
        table = _build_api_table_html(project_root, prefix)
        markdown = markdown.replace("<!-- API_TABLE -->", table)

    # SUBPAGES placeholder
    if "<!-- SUBPAGES -->" in markdown:
        markdown = _replace_marker(markdown, "<!-- SUBPAGES -->", _build_subpages_list(config, page, project_root))

    # EXAMPLES_FOR placeholders on generated API pages
    for match in re.finditer(r"<!-- EXAMPLES_FOR:([\w.]+) -->", markdown):
        qualified = match.group(1)
        examples_html = _build_api_examples_html(project_root, qualified)
        markdown = markdown.replace(match.group(0), examples_html)

    repo_url = config.get("repo_url", "").rstrip("/")
    github_path = repo_url.removeprefix("https://")
    # Same ref as the "View on GitHub" links, from the single definition in
    # `_git_ref.git_ref`, so the two cannot point at different commits.
    playground_base = f"https://marimo.app/{github_path}/blob/{git_ref()}"

    # GALLERY:section:<name> placeholders -> one section's cards. Matched before
    # the bare marker below; the two are distinct strings, so the order is for
    # the reader, not the parser.
    for match in _SECTION_GALLERY_RE.finditer(markdown):
        section_html = _build_gallery_html(project_root, section=match.group(1))
        markdown = _replace_marker(markdown, match.group(0), section_html)

    # GALLERY placeholder
    if "<!-- GALLERY -->" in markdown:
        gallery_html = _build_gallery_html(project_root)
        markdown = markdown.replace("<!-- GALLERY -->", gallery_html)

    # COMPANION_NOTEBOOKS placeholder -> cards for notebooks naming this page.
    # Substituted here, before the URL rewrites below, so companion cards go
    # through the same [View]/[Open in marimo] resolution as gallery cards.
    #
    # The marker is optional: a notebook's `companion` is the whole declaration
    # of the association, so appending when the marker is absent makes the
    # notebook's declaration sufficient on its own, and the marker purely a
    # placement override.
    companion_html = _build_companion_cards_html(project_root, _page_src_path(page))
    if "<!-- COMPANION_NOTEBOOKS -->" in markdown:
        if not companion_html:
            # The marker is well-formed, so the catch-all below never sees it:
            # it is consumed and replaced with nothing, leaving a blank where the
            # page asked for cards. This is the one marker the template seeds by
            # default, so replacing hello.py without re-pointing its `companion`
            # empties the page and says nothing.
            log.warning(
                "%s carries <!-- COMPANION_NOTEBOOKS --> but no notebook names it as their "
                'companion, so it renders blank. Add `"companion": "%s"` to a notebook\'s '
                "__gallery__, or drop the marker.",
                _page_src_path(page),
                _page_src_path(page),
            )
        markdown = _replace_marker(markdown, "<!-- COMPANION_NOTEBOOKS -->", companion_html)
    elif companion_html:
        markdown = markdown.rstrip("\n") + "\n\n" + companion_html

    # Resolve [Open in marimo] placeholder URLs -> full marimo.app playground URLs
    markdown = re.sub(
        r"\[Open in marimo\]\(/examples/([^)]+?)/edit/\)",
        rf"[Open in marimo]({playground_base}/examples/\1.py)",
        markdown,
    )

    # Rewrite [View] to relative paths pointing to local HTML exports
    markdown = re.sub(r"\]\(/examples/", f"]({prefix}examples/", markdown)

    # Absolute doc-page links (e.g. the gallery overflow link) resolve to the
    # current page's depth, the same way [View] does.
    markdown = re.sub(r"\]\(/pages/", f"]({prefix}pages/", markdown)

    _warn_on_unhandled_markers(markdown, _page_src_path(page))

    return markdown


class _MarkerPreprocessor(Preprocessor):
    """Resolve markers and set the sidebar TOC once per page, before HTML stashing."""

    def run(self, lines):
        """Inject markers into the page, or pass it through if the page is unknown."""
        page = _current_page(self.md)
        if page is None:
            # No page context means no SUBPAGES/COMPANION resolution and no
            # prefix for URL rewrites. Passing the lines through unchanged is
            # safer than resolving against a guessed page.
            return lines
        _set_module_toc(page)
        return _inject("\n".join(lines), page).split("\n")


class MarkerExtension(Extension):
    """Register the marker Preprocessor high enough to see raw HTML comments."""

    def extendMarkdown(self, md):
        """Register at priority 100, above Python-Markdown's html_block (~20).

        The markers are HTML comments. The stock ``html_block`` preprocessor
        stashes HTML out of the stream at priority 20, so a lower priority would
        never see ``<!-- API_TABLE -->`` as text. Running first keeps them
        visible.
        """
        md.preprocessors.register(_MarkerPreprocessor(md), "docs_markers", 100)


def makeExtension(**_kwargs):
    """Entry point Python-Markdown calls when loading this by module name."""
    return MarkerExtension()
