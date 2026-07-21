"""MkDocs hooks for the Yohou documentation build.

These hooks are adapters. The work they used to do inline now lives in sibling
modules -- ``_api_pages``, ``_notebooks`` and ``_markdown_export`` -- which import
nothing from mkdocs and can therefore be run, tested and debugged without a docs
build. What stays here is the part that genuinely needs the renderer: the page
hooks, and the per-build cache reset they depend on.
"""

import ast
import logging
import os
import posixpath
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

# MkDocs loads this file by path rather than importing it as a module inside a
# package, so `docs/` has no parent package and `from ._api_pages import ...`
# cannot resolve. Putting `docs/` on sys.path lets the build steps be imported as
# plain modules.
#
# Rejected alternative: adding `docs/__init__.py`. That would make `docs/` a
# package, which changes what docs_dir scanning sees and invites a name collision
# with the project's own package. This wart is smaller and confined to one line.
sys.path.insert(0, str(Path(__file__).parent))

import _api_pages  # noqa: E402
import _markdown_export  # noqa: E402
import _notebooks  # noqa: E402
from _api_pages import (  # noqa: E402
    _get_api_name_lookup,
    _get_public_members,
    _get_root_members,
    _get_submodules,
    _qualified_name,
)

# Warnings logged under the "mkdocs" logger tree are counted by mkdocs and turn
# a --strict build red. Every marker this file understands is silently inert
# when it does not resolve -- a placeholder that renders nothing looks exactly
# like a page that never had one. Warning here is what makes a dead marker a
# build failure instead of a blank space nobody notices.
log = logging.getLogger("mkdocs.hooks")

# Module-level caches. MkDocs loads hooks as plugin instances and does not
# reload the module between builds, so these live for the whole process --
# under `mkdocs serve` an unreset cache serves stale content for the rest of the
# session. Every cache is cleared per build by on_config() below.
#
# Naming is load-bearing: the reset and its test discover caches by the
# `_CACHE` suffix, so a cache named otherwise escapes both, silently.
#
# The API discovery caches (`_SUBMODULE_CACHE`, `_API_NAME_LOOKUP_CACHE`) moved
# to `_api_pages` with the functions that own them. They are still reset every
# build -- on_config calls `_api_pages.reset_caches()` -- but they are no longer
# declared here, so a reset or a test that scans only this module would silently
# stop covering them. The registration test scans every module the hooks load.
_GLOSSARY_TERMS_CACHE = None


def _site_root_prefix(page):
    """Relative path from `page`'s rendered URL back to the site root.

    Every link this file injects is relative, because the site may be served
    under a subpath and `use_directory_urls` makes each page its own directory.
    A hardcoded `../../` only works if the page never moves: a project is free
    to put its API index at `pages/api/index.md` rather than the template's
    `pages/reference/api.md`, and a fixed prefix silently 404s every link on it.
    """
    parts = page.file.src_path.split("/")
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
        # Nothing catches that -- the cell is raw HTML from this hook, which --strict
        # never validates, and only a project with a root-only export renders one.
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


_GALLERY_CACHE = None
_COMPANION_INDEX_CACHE = None
_GALLERY_PAGE_CACHE = None


# Max example cards on a single API page.  Most symbols are well under this
# (a typical notebook demonstrates a handful of things); the cap exists for the
# widely used helpers, where an uncapped list runs to dozens of cards and stops
# being scannable.  The overflow link keeps the rest reachable.
#
# The `_CACHE` suffix on cache globals above is load-bearing: the per-build
# reset's registration test discovers caches by that suffix, so a cache named
# otherwise escapes it silently.
_API_EXAMPLES_CAP = 6


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
                        try:
                            gallery = ast.literal_eval(node.value)
                        except (ValueError, TypeError):
                            pass

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


_NOTEBOOK_API_USAGE_CACHE = None


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


# Every name this file substitutes. A comment *opening* with one of these is a
# marker; if one survives to the end of on_page_markdown, it was misspelled.
# Matched without a word boundary so `<!-- GALLERY:quickstart -->` and
# `<!-- SUBPAGES_FOO -->` are both caught, not just the separator-delimited ones.
#
# The net is deliberately the marker namespace and nothing else: it cannot catch
# a typo that mangles the name itself (`<!-- GALLRY -->`), because widening it to
# every upper-case comment would flag ordinary `<!-- TODO -->`s. The realistic
# mistake is right name, wrong syntax -- which is exactly what shipped.
_MARKER_NAMES = ("API_TABLE", "SUBPAGES", "GALLERY", "COMPANION_NOTEBOOKS", "EXAMPLES_FOR")
_UNHANDLED_MARKER_RE = re.compile(r"<!--\s*(?:" + "|".join(_MARKER_NAMES) + r")[^>]*-->")


def _warn_on_unhandled_markers(markdown, src_path):
    """Warn about a marker that no substitution above recognised.

    The per-marker warnings only fire for a *well-formed* marker that resolves to
    nothing -- an unknown gallery section, an index with no children. A
    misspelled one is worse and was completely silent: `<!-- GALLERY:quickstart
    -->` matches neither the bare nor the sectioned pattern, so nothing claimed
    it, nothing substituted it, and it shipped to the page as a raw comment that
    renders as blank space. It cannot even be reported by the code that would
    have handled it, because that code never sees it. Catching the leftovers is
    the only place a typo in the marker namespace can be noticed at all.
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


def _build_subpages_list(config, page, files):
    """List the pages this index page introduces, as ``- [Title](slug.md): summary``.

    Generated rather than hand-written: an index is the one page guaranteed to
    fall behind, because adding a page elsewhere is what makes it stale, and
    nothing fails when it does.
    """
    src = page.file.src_path
    directory = posixpath.dirname(src)

    siblings = []
    for candidate in files:
        candidate_src = getattr(candidate, "src_path", "")
        if not candidate_src.endswith(".md") or candidate_src == src:
            continue
        # Direct children only: a nested section owns its own index.
        if posixpath.dirname(candidate_src) != directory:
            continue
        if posixpath.basename(candidate_src) == "index.md":
            continue
        siblings.append(candidate)

    if not siblings:
        log.warning("<!-- SUBPAGES --> on %s, which has no sibling pages to list.", src)
        return "<!-- no subpages -->\n"

    entries = _nav_entries(config)

    # The index enumerates sibling *files*; the sidebar comes from mkdocs.yml.
    # When the two disagree the index quietly papers over it -- an entry dropped
    # from the nav still appears here, so the page stays reachable by link while
    # vanishing from navigation. mkdocs itself reports not-in-nav pages at INFO,
    # which --strict does not fail on, so nothing else says a word. A copier
    # update deleted a real nav entry exactly this way and every other guard
    # passed it.
    orphans = sorted(s.src_path for s in siblings if s.src_path not in entries)
    if orphans:
        log.warning(
            "%s lists %s, which %s missing from the nav in mkdocs.yml -- the page is linked but "
            "unreachable by navigation.",
            src,
            ", ".join(orphans),
            "is" if len(orphans) == 1 else "are",
        )

    rows = []
    for sibling in siblings:
        title, description = _page_title_and_description(sibling.abs_src_path)
        position, nav_title = entries.get(sibling.src_path, (len(entries) + 1, None))
        if title is None:
            # A page can legitimately have no H1 in its own source: a bare
            # `--8<-- "CHANGELOG.md"` include grows one only once snippets
            # expand, which is after this runs. The nav already names such a
            # page, and that name is what the sidebar shows, so prefer it over
            # dropping the page from its own index.
            title = nav_title
        if title is None:
            log.warning(
                "%s has no H1 heading and no nav title; omitted from the %s index.",
                sibling.src_path,
                src,
            )
            continue
        rows.append((position, title, posixpath.basename(sibling.src_path), description))

    if not rows:
        return "<!-- no subpages -->\n"

    rows.sort(key=lambda row: (row[0], row[1]))
    lines = [f"- [{title}]({slug})" + (f": {desc}" if desc else "") for _, title, slug, desc in rows]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# API sidebar module TOC
# ---------------------------------------------------------------------------


def _build_module_toc(config, current_src_path=None, prefix=None):
    """Build the module TOC list used by the api-submodule sidebar template.

    Parameters
    ----------
    config : dict
        MkDocs config with ``docs_dir``.
    current_src_path : str or None
        Source path of the current page (e.g. ``pages/api/hello.md``).
        When set, the matching entry gets ``active: True``.

    Returns
    -------
    list[dict]
        TOC entries with keys *title*, *url*, *children*, and optionally
        *active*.
    """
    docs_dir = Path(config["docs_dir"])
    api_dir = docs_dir / "pages" / "api"
    project_root = docs_dir.parent

    modules = _get_submodules(project_root)
    module_toc = []

    for mod in modules:
        md_filename = f"{mod['module_name']}.md"
        md_path = api_dir / md_filename
        if not md_path.exists():
            continue

        # Site-root relative, so the TOC is correct on any page that renders it.
        # The old form branched on whether the current page was the API index and
        # hardcoded that index's depth -- which silently 404s for a project that
        # keeps its index somewhere else.
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


# ---------------------------------------------------------------------------
# API page content post-processing
# ---------------------------------------------------------------------------

_GIT_REF_CACHE = None


def _get_git_ref():
    """Return the current git commit hash (short) for GitHub source links.

    Falls back to ``"main"`` when git is unavailable or the working directory
    is not a repository.  The result is cached for the lifetime of the build.
    """
    global _GIT_REF_CACHE  # noqa: PLW0603
    if _GIT_REF_CACHE is not None:
        return _GIT_REF_CACHE
    try:
        _GIT_REF_CACHE = subprocess.check_output(  # noqa: S603
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        _GIT_REF_CACHE = "main"
    return _GIT_REF_CACHE


# The See Also container. This used to be `<details class="see-also">`, emitted by
# mkdocstrings' shipped admonition template and dissolved later by a restructuring
# pass -- which is why linkification had to run first. The template now emits a
# heading plus this div, so there is nothing left to dissolve and no ordering
# constraint. Keyed on the class the override emits; change one and change both.
_SEE_ALSO_BLOCK_RE = re.compile(r'<div\s+class="[^"]*doc-admonition-see-also[^"]*"[^>]*>.*?</div>', re.DOTALL)
# An entry's name sits at the START of its line -- mkdocstrings renders one entry
# per line inside the paragraph. Anchoring here is what keeps a colon-terminated
# word in an entry's DESCRIPTION ("Target : Note: see below") from being treated
# as another entry and linked.
_SEE_ALSO_ENTRY_RE = re.compile(r"^(\s*)(<code>[^<]+</code>|[A-Za-z_][\w.]*)(\s*:)")


# A See Also entry opens a line with a name and a colon. The name may already be
# a link (mkdocstrings resolved it) or bare (the linkifier is about to). Anything
# else on its own line is a continuation of the previous entry's description.
_SEE_ALSO_ENTRY_START_RE = re.compile(
    r"""^\s*(?:
        <a\s[^>]*>.*?</a>          # already-linked name
      | <autoref[^>]*>.*?</autoref>
      | <code>[^<]+</code>
      | [A-Za-z_][\w.]*
    )\s*:""",
    re.VERBOSE,
)


def _split_see_also_entries(inner):
    """Split a numpydoc See Also paragraph into one string per entry.

    numpydoc puts one entry per source line; a long description wraps onto
    continuation lines that carry no name of their own, and those belong to the
    entry above rather than becoming entries in their own right.
    """
    entries: list[list[str]] = []
    for line in inner.split("\n"):
        if not line.strip():
            continue
        if _SEE_ALSO_ENTRY_START_RE.match(line) or not entries:
            entries.append([line.strip()])
        else:
            entries[-1].append(line.strip())
    return [" ".join(parts) for parts in entries]


def _resolve_see_also_url(name, prefix):
    """Resolve a See Also entry naming a project symbol to a URL, or None.

    A dotted name whose leading segment is not this package is external and is
    not resolved here -- see ``_external_autoref`` for why.

    Classifying by leading segment is the only rule that works: the project
    lookup is keyed by *short* name, so a dotted external name always misses it,
    and stripping the qualifier first would let ``sklearn.linear_model.Ridge``
    collide with a project symbol called ``Ridge``.

    The URL is built from *prefix* rather than a bare ``../``. A See Also block
    renders on any page carrying a docstring, not only under
    ``pages/api/generated/``, and ``../`` is the right answer from exactly one
    depth. Same reasoning as ``_site_root_prefix``, which says it plainly: a
    hardcoded prefix silently 404s every link once the page moves.
    """
    package = "yohou"
    if "." in name and name.split(".", 1)[0] != package:
        return None

    short_name = name.rsplit(".", 1)[-1]
    project_root = Path(__file__).parent.parent
    qualified = _get_api_name_lookup(project_root).get(short_name)
    return f"{prefix}pages/api/generated/{qualified}/" if qualified is not None else None


def _external_autoref(name, title):
    """Defer an external name to autorefs, which resolves it later.

    External names cannot be resolved here: mkdocstrings registers downloaded
    inventories into the autorefs URL map, but autorefs applies them in its
    ``on_env`` hook -- after ``on_page_content`` runs.  Asking for the URL now
    raises ``KeyError`` even when the inventory is configured.

    So emit autorefs' own markup and let it resolve at the right time.  The
    ``optional`` attribute is what keeps an unresolvable name quiet: autorefs
    logs it at debug level and renders the title as plain text, rather than
    recording it as an unmapped reference and warning (which would fail a
    ``--strict`` build for a docstring that names something undocumented).

    Only dotted names get here.  A bare name is not offered to autorefs: it
    could resolve to an unrelated symbol that happens to share the name in some
    configured inventory, and a wrong link is worse than no link.
    """
    return f'<autoref optional identifier="{name}">{title}</autoref>'


def _resolve_member_identifier(name):
    """Qualify a dotted ``Class.member`` See Also entry to its autoref identifier.

    A member -- a method or attribute -- has no page of its own; it is an anchor
    on its class page, so it cannot be resolved to a URL the way a class or a
    function can.  This qualifies the entry to a full identifier and hands it to
    autorefs, which does know the anchor.

    ``OtherClass.build`` becomes ``yohou.module.OtherClass.build``
    via the short-name lookup; an entry already led by the package name is
    returned unchanged (its trailing segment is a member, so
    ``_resolve_see_also_url`` did not resolve it to a page).  Returns None when
    the leading segment is neither this package nor a known project symbol, so a
    genuinely external dotted name (``sklearn.linear_model.Ridge``) falls through
    to external handling and keeps its own inventory link.
    """
    package = "yohou"
    if "." not in name:
        return None
    head, rest = name.split(".", 1)
    if head == package:
        return name
    qualified_head = _get_api_name_lookup(Path(__file__).parent.parent).get(head)
    if qualified_head is None:
        return None
    return f"{qualified_head}.{rest}"


def _link_entry(name, title, colon, entry, prefix):
    """Render one See Also entry: project link, deferred external ref, or as-is."""
    url = _resolve_see_also_url(name, prefix)
    if url:
        return f'<a href="{url}">{title}</a>{colon}'
    member_identifier = _resolve_member_identifier(name)
    if member_identifier is not None:
        return _external_autoref(member_identifier, title) + colon
    if "." in name and name.split(".", 1)[0] != "yohou":
        return _external_autoref(name, title) + colon
    return entry


# The glossary lives in the explanation quadrant by Diataxis convention. A
# project without this page simply gets no glossary linking.
_GLOSSARY_SRC_PATH = "pages/explanation/glossary.md"

# Text inside these never becomes a glossary link: code is not prose, a heading
# linking mid-title looks broken, and nesting an <a> inside an <a> is invalid.
_GLOSSARY_SKIP_TAGS = frozenset({"code", "pre", "a", "h1", "h2", "h3", "h4", "h5", "h6", "script", "style"})

# A definition-list term carrying attributes, e.g. ``Memory buffer { #memory-buffer .autolink }``.
_GLOSSARY_TERM_RE = re.compile(r"^(?!\s)(.+?)\s*\{:?\s*([^}]*)\}\s*$")


def _get_glossary_terms(project_root):
    """Map each auto-linkable glossary term to its anchor (cached).

    The glossary page is the single source of truth: terms are read from it, so
    a term and its definition cannot drift apart the way a second list in this
    file would.

    A term opts in with ``.autolink``::

        Memory buffer { #memory-buffer .autolink }
        :   The internal store of recent rows...

    Opting in is deliberate rather than automatic. A glossary defines whatever
    its authors find worth defining, including short common words -- "step",
    "pipeline", "ensemble" -- and auto-linking those wherever they appear in
    prose produces noise, not navigation. Defining a term and advertising it
    everywhere are separate editorial decisions, so they get separate syntax.
    """
    global _GLOSSARY_TERMS_CACHE  # noqa: PLW0603
    if _GLOSSARY_TERMS_CACHE is not None:
        return _GLOSSARY_TERMS_CACHE

    terms = {}
    page = project_root / "docs" / _GLOSSARY_SRC_PATH
    try:
        lines = page.read_text(encoding="utf-8").split("\n")
    except (OSError, UnicodeDecodeError):
        _GLOSSARY_TERMS_CACHE = terms
        return terms

    for i, line in enumerate(lines[:-1]):
        match = _GLOSSARY_TERM_RE.match(line)
        # The next line starting with ':' is what makes this a definition-list
        # term rather than ordinary prose that happens to end in braces.
        if not match or not lines[i + 1].lstrip().startswith(":"):
            continue
        attrs = match.group(2).split()
        if ".autolink" not in attrs:
            continue
        anchor = next((a[1:] for a in attrs if a.startswith("#")), None)
        if anchor:
            terms[match.group(1).strip().lower()] = anchor

    _GLOSSARY_TERMS_CACHE = terms
    return terms


def _glossary_link_replacer(terms, rel_glossary, linked):
    """Build the text-node rewriter that links a term's first occurrence."""
    # Longest first, so "seasonal naive forecaster" wins over "forecaster"
    # rather than being shadowed by the shorter term inside it.
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )

    def _replace(text):
        def _sub(match):
            term = match.group(1).lower()
            if term in linked:
                return match.group(0)
            linked.add(term)
            return f'<a href="{rel_glossary}/#{terms[term]}">{match.group(0)}</a>'

        return pattern.sub(_sub, text)

    return _replace


class _GlossaryLinker(HTMLParser):
    """Rewrites text nodes into glossary links, leaving markup untouched.

    Parsed rather than regexed over the whole page: a bare regex would match
    inside tag attributes and code blocks, producing broken markup from a
    document that was fine.
    """

    def __init__(self, replace):
        super().__init__(convert_charrefs=False)
        self._replace = replace
        self.result = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in _GLOSSARY_SKIP_TAGS:
            self._skip_depth += 1
        self.result.append(self.get_starttag_text())

    def handle_startendtag(self, tag, attrs):
        self.result.append(self.get_starttag_text())

    def handle_endtag(self, tag):
        if tag.lower() in _GLOSSARY_SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        self.result.append(f"</{tag}>")

    def handle_data(self, data):
        self.result.append(data if self._skip_depth else self._replace(data))

    def handle_entityref(self, name):
        self.result.append(f"&{name};")

    def handle_charref(self, name):
        self.result.append(f"&#{name};")

    def handle_comment(self, data):
        self.result.append(f"<!--{data}-->")

    def handle_decl(self, decl):
        self.result.append(f"<!{decl}>")

    def get_html(self):
        """Return the rewritten HTML."""
        return "".join(self.result)


def _linkify_glossary_terms(html, page, project_root):
    """Link the first occurrence of each glossary term on a page.

    First occurrence only: linking every "memory buffer" in a paragraph is
    noise, and the reader only needs the definition once.

    The glossary page itself is skipped -- it would link its own terms to
    themselves.
    """
    src = page.file.src_path
    if src == _GLOSSARY_SRC_PATH or not src.startswith("pages/"):
        return html

    terms = _get_glossary_terms(project_root)
    if not terms:
        return html

    # Relative, not absolute: the site may be served under a subpath, and
    # use_directory_urls means a page's own URL is a directory.
    dest_dir = posixpath.dirname(page.file.dest_path)
    rel_glossary = posixpath.relpath(posixpath.splitext(_GLOSSARY_SRC_PATH)[0], dest_dir)

    linker = _GlossaryLinker(_glossary_link_replacer(terms, rel_glossary, set()))
    linker.feed(html)
    linker.close()
    return linker.get_html()


def _linkify_see_also(html, prefix):
    """Turn the names in a rendered See Also section into links.

    Unresolvable names are left untouched: a docstring may reference a private
    helper or a concept, and none of those are build errors.

    This no longer has an ordering constraint. It used to have to run before the
    restructuring pass, which dissolved the container it matches -- and getting
    that order wrong silently degraded class-level sections while method-level
    ones kept working. The container is now emitted by the admonition template
    override and nothing consumes it.
    """

    def _process_block(block_match):
        def _linkify_line(line):
            entry_match = _SEE_ALSO_ENTRY_RE.match(line)
            if not entry_match:
                return line
            lead, token, colon = entry_match.groups()
            rest = line[entry_match.end() :]
            code_match = re.fullmatch(r"<code>([^<]+)</code>", token)
            name = code_match.group(1) if code_match else token
            title = f"<code>{name}</code>" if code_match else name
            return lead + _link_entry(name, title, colon, token + colon, prefix) + rest

        def _linkify_inner(inner):
            # Leave an author's explicit [Name][target] reference alone: they have
            # said what they mean, and autorefs resolves it later.
            if "<a " in inner or "<autoref" in inner:
                return inner
            return "\n".join(_linkify_line(line) for line in inner.split("\n"))

        def _process_container(container_match):
            tag, inner = container_match.group(1), container_match.group(2)
            if tag == "li":
                # Already one entry per item; only the names need linking.
                return f"<li>{_linkify_inner(inner)}</li>"
            entries = _split_see_also_entries(_linkify_inner(inner))
            if len(entries) < 2:
                return f"<p>{_linkify_inner(inner)}</p>"
            # numpydoc puts each entry on its own source line inside one
            # paragraph, and HTML collapses those newlines to spaces -- so every
            # entry runs together on a single line, and the more references a
            # symbol has the worse it reads. An author can dodge it by hand-
            # writing markdown bullets (yohou does, which is why its pages look
            # right and nobody else's do), but plain numpydoc is what the other
            # 145 blocks in the fleet are written in. One entry per line is what
            # the section is for.
            items = "".join(f"<li>{entry}</li>" for entry in entries)
            return f"<ul>{items}</ul>"

        # numpydoc renders See Also entries as a paragraph, one per line; an author
        # may also write them as a markdown list, which renders as <li>. Both are
        # ordinary numpydoc, so both get linked.
        return re.sub(r"<(p|li)>(.*?)</\1>", _process_container, block_match.group(0), flags=re.DOTALL)

    return _SEE_ALSO_BLOCK_RE.sub(_process_block, html)


def _dedupe_heading_ids(html):
    """Make repeated heading ids unique, keeping the first occurrence bare.

    A page documenting more than one object -- a curated reference page with
    several ``:::`` directives -- renders each of them at the same depth, so the
    section templates give every one of them the same ids: two ``#see-also``,
    two ``#parameters``. That is invalid HTML and makes the fragment ambiguous.

    The templates cannot prevent it. Each ``:::`` is rendered independently and a
    template has no way to know another object shares its page. Qualifying every
    id with the object path would fix it and break every existing deep link into
    the generated pages, which are single-object and vastly more numerous.

    So the first occurrence keeps the bare id -- preserving those links -- and
    later ones get a numeric suffix, which is what Python-Markdown's own toc
    extension does for repeated markdown headings. Pages without duplicates are
    returned unchanged, so this is inert for the single-object case.
    """
    seen = {}

    def _rewrite(match):
        prefix, hid, suffix = match.group(1), match.group(2), match.group(3)
        seen[hid] = seen.get(hid, 0) + 1
        if seen[hid] == 1:
            return match.group(0)
        return f"{prefix}{hid}_{seen[hid] - 1}{suffix}"

    return re.sub(r'(<h[1-6][^>]*\sid=")([^"]+)(")', _rewrite, html)


def _strip_redundant_section_titles(html):
    """Drop the section title the shipped mkdocstrings template still emits.

    The docstring templates render every section title as
    ``<p><span class="doc-section-title">Parameters:</span></p>``, and nothing in
    the template layer can suppress it -- ``section.title`` falls back to a
    translated string. Our dispatcher override emits a real heading for the same
    sections, so without this the page shows each title twice.

    Deliberately narrow: only the titles the dispatcher actually maps are
    removed. A section it does not map (Yields, Warns, ...) keeps its title and
    is untouched, so nothing loses its label. That set is the inverse of
    ``doc_section_slugs`` in
    ``docs/material/templates/python/material/docstring.html.jinja`` and the two
    must stay in sync -- adding a heading there without adding its title here
    renders it twice; the reverse renders it not at all.
    """
    titles = "|".join(re.escape(t) for t in ("Parameters", "Attributes", "Returns", "Raises", "Examples"))
    return re.sub(
        rf'<p>\s*<span class="doc-section-title">\s*(?:{titles}):?\s*</span>\s*</p>\s*',
        "",
        html,
    )


def _add_source_links(html, page, config):
    """Insert a "View on GitHub" link after each Source Code heading.

    This is the one part of the old restructuring pass that genuinely cannot move
    into a template: it needs ``repo_url`` and a git ref, and a mkdocstrings
    template receives the handler's options, not the mkdocs config. The heading
    itself IS template-owned (see ``class.html.jinja`` / ``function.html.jinja``);
    only the link is inserted here.

    Keyed on the heading's id rather than its text, because the text is
    translatable and the id is the thing we control.
    """
    repo_url = config.get("repo_url", "").rstrip("/")
    if not repo_url:
        return html

    # pages/api/generated/{qualified}.md -> package/module.py
    qualified = page.file.src_path.split("/")[-1].removesuffix(".md")
    parts = qualified.split(".")
    if len(parts) < 2:
        return html
    module_path = "/".join(parts[:-1])
    link = (
        f'<p class="github-source-link">'
        f'<a href="{repo_url}/blob/{_get_git_ref()}/src/{module_path}.py">View on GitHub</a></p>'
    )

    return re.sub(
        r'(<h[35][^>]*id="[^"]*source-code"[^>]*>.*?</h[35]>)',
        lambda m: m.group(1) + link,
        html,
        flags=re.DOTALL,
    )


def on_config(config):
    """Clear per-build caches.

    `on_config` is the first event on every build, including each rebuild in a
    `mkdocs serve` session -- which is the lifetime these caches need.

    Deliberately not `on_startup`: that runs once per `mkdocs` invocation, so a
    reset there fires when the caches are already empty and never again, and
    `mkdocs serve` keeps serving the first build's content.

    The API discovery caches live in `_api_pages` now. A `global` statement here
    cannot rebind another module's globals, so that module exposes its own reset
    and this hook calls it -- which also keeps the set of caches defined beside
    the functions that fill them.
    """
    _api_pages.reset_caches()

    global _GIT_REF_CACHE, _GLOSSARY_TERMS_CACHE  # noqa: PLW0603
    _GIT_REF_CACHE = None
    _GLOSSARY_TERMS_CACHE = None
    global _GALLERY_CACHE, _COMPANION_INDEX_CACHE, _NOTEBOOK_API_USAGE_CACHE, _GALLERY_PAGE_CACHE  # noqa: PLW0603
    _GALLERY_CACHE = None
    _COMPANION_INDEX_CACHE = None
    _NOTEBOOK_API_USAGE_CACHE = None
    _GALLERY_PAGE_CACHE = None
    return config


def on_page_content(html, page, config, files):
    """Post-process rendered HTML: See Also links, glossary links, API sidebar."""
    src = page.file.src_path

    # Keyed on the markup, not on where the page lives: mkdocstrings emits a See
    # Also block wherever a docstring is rendered, and a project is free to put
    # ::: directives on a curated reference page. Gating this on
    # `pages/api/generated/` left those blocks raw -- kedro-dagster's datasets
    # page rendered three entries as plain text while the same names linked fine
    # on the generated pages, which is exactly the shape of "it works where we
    # looked". --strict never sees it: this is our own HTML.
    #
    # There is no ordering constraint here any more. There used to be: a
    # restructuring pass dissolved the `<details class="see-also">` container
    # this linkifier matched, so linkifying afterwards silently did nothing for
    # class-level sections -- the majority -- while appearing to work for
    # method-level ones. The container is gone (the templates emit a heading
    # instead), so nothing downstream can consume the markup out from under it.
    html = _linkify_see_also(html, _site_root_prefix(page))

    # NOT gated on `pages/api/generated/`. The templates render wherever a `:::`
    # directive appears, including a curated reference page, so the duplicate
    # title and the repeated ids appear there too. Gating these to the generated
    # pages left a curated page showing each section title twice and carrying
    # ambiguous fragments -- the same "it works where we looked" shape that once
    # left See Also unlinked on exactly those pages. Both are no-ops on a page
    # with no mkdocstrings output.
    html = _strip_redundant_section_titles(html)
    html = _dedupe_heading_ids(html)

    # Still gated: this one derives the module path from the page's own filename
    # (`pages/api/generated/{qualified}.md`), so it is meaningless elsewhere.
    if src.startswith("pages/api/generated/"):
        html = _add_source_links(html, page, config)

    # Keyed on the template a page declares, not on where the page happens to
    # live: the index is wherever a project put it. Matching a hardcoded
    # `pages/reference/api.md` leaves a relocated index with no module_toc at
    # all -- its sidebar renders empty, and nothing errors.
    if page.meta.get("template") in ("api-index.html", "api-submodule.html"):
        page.meta["module_toc"] = _build_module_toc(config, current_src_path=src, prefix=_site_root_prefix(page))

    html = _linkify_glossary_terms(html, page, Path(__file__).parent.parent)

    return html


def on_page_markdown(markdown, page, config, files):
    """Inject dynamic content into markdown pages.

    Placeholder injection
    ---------------------
    ``<!-- API_TABLE -->``            → submodule table for API index
    ``<!-- SUBPAGES -->``             → linked list of the pages an index introduces
    ``<!-- GALLERY -->``              → flat card grid of example notebooks
    ``<!-- GALLERY:section:name -->`` → card grid for one section of the gallery
    ``<!-- COMPANION_NOTEBOOKS -->``  → cards for notebooks naming this page
    """
    project_root = Path(__file__).parent.parent
    prefix = _site_root_prefix(page)

    # API_TABLE placeholder
    if "<!-- API_TABLE -->" in markdown:
        table = _build_api_table_html(project_root, prefix)
        markdown = markdown.replace("<!-- API_TABLE -->", table)

    # SUBPAGES placeholder
    if "<!-- SUBPAGES -->" in markdown:
        markdown = _replace_marker(markdown, "<!-- SUBPAGES -->", _build_subpages_list(config, page, files))

    # EXAMPLES_FOR placeholders on generated API pages
    for match in re.finditer(r"<!-- EXAMPLES_FOR:([\w.]+) -->", markdown):
        qualified = match.group(1)
        examples_html = _build_api_examples_html(project_root, qualified)
        markdown = markdown.replace(match.group(0), examples_html)

    repo_url = config.get("repo_url", "").rstrip("/")
    github_path = repo_url.removeprefix("https://")
    git_ref = os.environ.get(
        "READTHEDOCS_GIT_COMMIT_HASH",
        os.environ.get("READTHEDOCS_GIT_IDENTIFIER", "main"),
    )
    playground_base = f"https://marimo.app/{github_path}/blob/{git_ref}"

    # GALLERY:section:<name> placeholders → one section's cards. Matched before
    # the bare marker below; the two are distinct strings, so the order is for
    # the reader, not the parser.
    for match in _SECTION_GALLERY_RE.finditer(markdown):
        section_html = _build_gallery_html(project_root, section=match.group(1))
        markdown = _replace_marker(markdown, match.group(0), section_html)

    # GALLERY placeholder
    if "<!-- GALLERY -->" in markdown:
        gallery_html = _build_gallery_html(project_root)
        markdown = markdown.replace("<!-- GALLERY -->", gallery_html)

    # COMPANION_NOTEBOOKS placeholder → cards for notebooks naming this page.
    # Substituted here, before the URL rewrites below, so companion cards go
    # through the same [View]/[Open in marimo] resolution as gallery cards.
    # Emitting resolved HTML directly would bypass those markdown-syntax
    # rewrites and ship unresolved links.
    #
    # The marker is optional. A notebook's `companion` is the whole declaration
    # of the association; requiring the target page to opt in as well means a
    # notebook can name a page that never shows it, and nothing anywhere says
    # so. Appending when the marker is absent makes the notebook's declaration
    # sufficient on its own, and the marker purely a placement override.
    companion_html = _build_companion_cards_html(project_root, page.file.src_path)
    if "<!-- COMPANION_NOTEBOOKS -->" in markdown:
        if not companion_html:
            # The marker is well-formed, so the catch-all below never sees it:
            # it is consumed and replaced with nothing, leaving a blank where the
            # page asked for cards. This is the one marker the template seeds by
            # default -- it points at hello.py, so replacing hello.py without
            # re-pointing its `companion` empties the page and says nothing.
            log.warning(
                "%s carries <!-- COMPANION_NOTEBOOKS --> but no notebook names it as their "
                'companion, so it renders blank. Add `"companion": "%s"` to a notebook\'s '
                "__gallery__, or drop the marker.",
                page.file.src_path,
                page.file.src_path,
            )
        markdown = _replace_marker(markdown, "<!-- COMPANION_NOTEBOOKS -->", companion_html)
    elif companion_html:
        markdown = markdown.rstrip("\n") + "\n\n" + companion_html

    # Resolve [Open in marimo] placeholder URLs → full marimo.app playground URLs
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

    _warn_on_unhandled_markers(markdown, page.file.src_path)

    return markdown


def on_pre_build(config):
    """Generate API submodule pages and export marimo notebooks."""
    project_root = Path(__file__).parent.parent

    _api_pages.generate(project_root)
    _notebooks.export(project_root)


def on_post_build(config):
    """Copy markdown files for LLM consumption after build completes."""
    _markdown_export.export(config["site_dir"], config["docs_dir"], Path(__file__).parent.parent)
