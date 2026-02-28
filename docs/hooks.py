"""MkDocs hooks for post-build processing."""

import ast
import fnmatch
import os
import re
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

_CATEGORY_LABELS = {
    "quickstart": "Quickstart",
    "point": "Point Forecasting",
    "interval": "Interval Forecasting",
    "compose": "Composition",
    "preprocessing": "Preprocessing",
    "stationarity": "Stationarity",
    "metrics": "Metrics",
    "model_selection": "Model Selection",
    "datasets": "Datasets",
    "plotting": "Plotting",
}

_DISCOVERY_CACHE = None
_SEE_ALSO_LOOKUP = None

_EXTERNAL_PREFIXES = ("sklearn.", "scipy.", "numpy.", "pandas.")


def _get_see_also_lookup():
    """Build lookup from short class/function names to qualified page paths."""
    global _SEE_ALSO_LOOKUP
    if _SEE_ALSO_LOOKUP is not None:
        return _SEE_ALSO_LOOKUP

    data = _get_discovery_data()
    lookup = {}

    for name, cls in data["abstract_base_classes"]:
        qualified = f"{cls.__module__}.{name}"
        lookup[name] = qualified

    for name, cls in data["estimators"]:
        qualified = f"{cls.__module__}.{name}"
        lookup[name] = qualified

    for name, cls in data["displays"]:
        qualified = f"{cls.__module__}.{name}"
        lookup[name] = qualified

    for name, func in data["functions"]:
        qualified = f"{func.__module__}.{name}"
        lookup[name] = qualified

    _SEE_ALSO_LOOKUP = lookup
    return lookup


def _resolve_see_also_url(name, lookup):
    """Resolve a See Also reference name to a relative URL, or None."""
    if any(name.startswith(prefix) for prefix in _EXTERNAL_PREFIXES):
        return None
    short_name = name.rsplit(".", 1)[-1] if "." in name else name
    if short_name in lookup:
        return f"../{lookup[short_name]}/"
    return None


def _linkify_see_also(html):
    """Replace class/function names in See Also sections with links."""
    lookup = _get_see_also_lookup()

    def _process_block(block_match):
        block = block_match.group(0)

        def _process_p(p_match):
            content = p_match.group(1)

            def _linkify_entry(entry_match):
                full = entry_match.group(0)
                code_m = re.match(r"<code>([^<]+)</code>(\s*:)", full)
                if code_m:
                    name, colon = code_m.group(1), code_m.group(2)
                    url = _resolve_see_also_url(name, lookup)
                    if url:
                        return f'<a href="{url}"><code>{name}</code></a>{colon}'
                    return full
                plain_m = re.match(r"([A-Za-z_][\w.]*?)(\s*:)", full)
                if plain_m:
                    name, colon = plain_m.group(1), plain_m.group(2)
                    url = _resolve_see_also_url(name, lookup)
                    if url:
                        return f'<a href="{url}">{name}</a>{colon}'
                    return full
                return full

            new_content = re.sub(
                r"(?:<code>[^<]+</code>|[A-Za-z_][\w.]*)\s*:",
                _linkify_entry,
                content,
            )
            return f"<p>{new_content}</p>"

        block = re.sub(r"<p>(.*?)</p>", _process_p, block, flags=re.DOTALL)
        return block

    return re.sub(
        r'<details class="see-also[^"]*"[^>]*>.*?</details>',
        _process_block,
        html,
        flags=re.DOTALL,
    )


def _discover_base_and_abstract_classes():
    """Discover Base* classes and abstract non-Base estimators.

    Returns
    -------
    base_classes : list of (name, class)
        Classes whose name starts with ``Base`` (shown as "Base Class" badge).
    abstract_estimators : list of (name, class)
        Abstract estimators that do not follow the ``Base*`` naming convention
        (e.g. ``QuantileResidual``).  These are included as regular "Class"
        entries so they are not missing from the API index.
    """
    import inspect
    import pkgutil
    from importlib import import_module
    from pathlib import Path

    from sklearn.base import BaseEstimator

    root = str(Path(__file__).parent.parent / "src" / "yohou")
    base_classes = []
    abstract_estimators = []
    seen = set()
    for _, module_name, _ in pkgutil.walk_packages(path=[root], prefix="yohou."):
        parts = module_name.split(".")
        if "tests" in parts or "testing" in parts:
            continue
        try:
            module = import_module(module_name)
        except Exception:
            continue
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if name.startswith("_") or cls.__module__ != module_name:
                continue
            if name in seen:
                continue
            is_estimator = issubclass(cls, BaseEstimator)
            is_base = name.startswith("Base") and name != "BaseEstimator"
            is_abstract = is_estimator and hasattr(cls, "__abstractmethods__") and cls.__abstractmethods__
            if is_base and (is_estimator or module_name.startswith("yohou.base.")):
                base_classes.append((name, cls))
                seen.add(name)
            elif is_abstract:
                abstract_estimators.append((name, cls))
                seen.add(name)

    base_classes.sort(key=lambda x: (x[1].__module__, x[0]))
    abstract_estimators.sort(key=lambda x: (x[1].__module__, x[0]))
    return base_classes, abstract_estimators


def _get_discovery_data():
    """Import and cache all estimators, displays, and functions."""
    global _DISCOVERY_CACHE
    if _DISCOVERY_CACHE is not None:
        return _DISCOVERY_CACHE

    from yohou.utils.discovery import all_displays, all_estimators, all_functions

    estimators = all_estimators()
    base_classes, abstract_estimators = _discover_base_and_abstract_classes()
    base_names = {name for name, _ in base_classes}
    # Base*-prefixed classes belong in the base class section only
    estimators = [(name, cls) for name, cls in estimators if name not in base_names]
    # Abstract non-Base estimators are not in all_estimators(); add as regular classes
    estimators.extend(abstract_estimators)

    _DISCOVERY_CACHE = {
        "estimators": estimators,
        "displays": all_displays(),
        "functions": all_functions(),
        "abstract_base_classes": base_classes,
    }
    return _DISCOVERY_CACHE


def _first_docstring_line(obj):
    """Extract the first non-empty line of a docstring."""
    doc = getattr(obj, "__doc__", None)
    if not doc:
        return ""
    for line in doc.strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _generate_api_pages(docs_dir):
    """Generate per-class/function .md pages under docs/pages/api/generated/."""
    generated_dir = Path(docs_dir) / "pages" / "api" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    gitignore = generated_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("# Auto-generated. Do not commit\n*\n!.gitignore\n")

    # Clean stale pages from previous builds
    for old in generated_dir.glob("*.md"):
        old.unlink()

    data = _get_discovery_data()
    count = 0

    _page_template = (
        "---\n"
        "template: api-page.html\n"
        "---\n\n"
        "# {name}\n\n"
        "::: {qualified}\n"
        "    options:\n"
        "      show_root_heading: true\n"
        "      show_source: true\n"
        "      members_order: source\n"
        "\n"
        "<!-- EXAMPLES_FOR:{qualified} -->\n"
    )

    for name, cls in data["abstract_base_classes"]:
        module = cls.__module__
        qualified = f"{module}.{name}"
        page = generated_dir / f"{qualified}.md"
        page.write_text(_page_template.format(name=name, qualified=qualified))
        count += 1

    for name, cls in data["estimators"]:
        module = cls.__module__
        qualified = f"{module}.{name}"
        page = generated_dir / f"{qualified}.md"
        page.write_text(_page_template.format(name=name, qualified=qualified))
        count += 1

    for name, cls in data["displays"]:
        module = cls.__module__
        qualified = f"{module}.{name}"
        page = generated_dir / f"{qualified}.md"
        page.write_text(_page_template.format(name=name, qualified=qualified))
        count += 1

    for name, func in data["functions"]:
        module = func.__module__
        qualified = f"{module}.{name}"
        page = generated_dir / f"{qualified}.md"
        page.write_text(_page_template.format(name=name, qualified=qualified))
        count += 1

    print(f"[hooks] generated {count} API pages in pages/api/generated/")


# Mapping from top-level module prefix to nav section label
_MODULE_NAV_LABEL = {
    "yohou.base": "yohou.base",
    "yohou.compose": "yohou.compose",
    "yohou.point": "yohou.point",
    "yohou.interval": "yohou.interval",
    "yohou.metrics": "yohou.metrics",
    "yohou.model_selection": "yohou.model_selection",
    "yohou.preprocessing": "yohou.preprocessing",
    "yohou.stationarity": "yohou.stationarity",
    "yohou.plotting": "yohou.plotting",
    "yohou.datasets": "yohou.datasets",
    "yohou.utils": "yohou.utils",
    "yohou.testing": "yohou.testing",
}


def _inject_generated_pages_into_nav(config, docs_dir):
    """Inject generated API pages into the nav under their parent module sections.

    Currently disabled: generated pages are accessed via table links on
    submodule pages to keep the sidebar clean.  The pages are built but
    excluded from nav via ``not_in_nav`` in mkdocs.yml.
    """
    return


def _build_api_table_html():
    """Build an HTML <table> for the API index with DataTables init."""
    data = _get_discovery_data()

    # Map top-level submodule to its API page slug
    _submodule_page = {
        "yohou.base": "base",
        "yohou.compose": "compose",
        "yohou.point": "point",
        "yohou.interval": "interval",
        "yohou.metrics": "metrics",
        "yohou.model_selection": "model-selection",
        "yohou.preprocessing": "preprocessing",
        "yohou.stationarity": "stationarity",
        "yohou.plotting": "plotting",
        "yohou.datasets": "datasets",
        "yohou.utils": "utils",
        "yohou.testing": "testing",
    }

    def _submodule_label_and_href(full_module: str):
        """Return (display_label, relative_href) for a full module path."""
        parts = full_module.split(".")
        # Take first two parts: yohou.<submodule>
        submodule = ".".join(parts[:2]) if len(parts) >= 2 else full_module
        page = _submodule_page.get(submodule)
        if page is not None:
            return submodule, f"{page}/"
        return submodule, None

    rows = []
    for name, cls in data["abstract_base_classes"]:
        module = cls.__module__
        qualified = f"{module}.{name}"
        desc = _first_docstring_line(cls)
        rows.append((name, "Base Class", module, desc, qualified))

    for name, cls in data["estimators"]:
        module = cls.__module__
        qualified = f"{module}.{name}"
        desc = _first_docstring_line(cls)
        rows.append((name, "Class", module, desc, qualified))

    for name, cls in data["displays"]:
        module = cls.__module__
        qualified = f"{module}.{name}"
        desc = _first_docstring_line(cls)
        rows.append((name, "Display", module, desc, qualified))

    for name, func in data["functions"]:
        module = func.__module__
        qualified = f"{module}.{name}"
        desc = _first_docstring_line(func)
        rows.append((name, "Function", module, desc, qualified))

    rows.sort(key=lambda r: r[0].lower())

    _type_badge_cls = {
        "Base Class": "api-badge--base",
        "Class": "api-badge--class",
        "Display": "api-badge--display",
        "Function": "api-badge--function",
    }

    tbody_lines = []
    for name, kind, module, desc, qualified in rows:
        href = f"generated/{qualified}/"
        label, mod_href = _submodule_label_and_href(module)
        if mod_href is not None:
            mod_cell = f'<a href="{mod_href}">{label}</a>'
        else:
            mod_cell = label
        badge_cls = _type_badge_cls.get(kind, "")
        tbody_lines.append(
            f"      <tr>"
            f'<td><a href="{href}"><code>{name}</code></a></td>'
            f'<td><span class="api-badge {badge_cls}">{kind}</span></td>'
            f"<td>{mod_cell}</td>"
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
_NOTEBOOK_API_USAGE_CACHE = None


def _get_gallery_items(project_root):
    """Parse __gallery__ metadata from all example notebooks (cached)."""
    global _GALLERY_CACHE
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

        rel = notebook.relative_to(examples_dir)
        parts = rel.parts
        category = parts[0] if len(parts) > 1 else "quickstart"
        stem = notebook.stem

        # view_path is always flat (notebooks are exported to docs/examples/{stem}/)
        view_path = f"/examples/{stem}/"

        if len(parts) > 1:
            # open_path preserves category so the marimo-playground rewrite
            # regex can reconstruct the real repo path (examples/{category}/{stem}.py)
            open_path = f"/examples/{category}/{stem}/edit/"
        else:
            open_path = f"/examples/{stem}/edit/"

        items.append({
            "title": gallery.get("title", stem.replace("_", " ").title()),
            "category": category,
            "description": gallery.get("description", ""),
            "view_path": view_path,
            "open_path": open_path,
            "stem": stem,
        })

    _GALLERY_CACHE = items
    return _GALLERY_CACHE


def _get_notebook_api_usage(project_root):
    """Build reverse map: qualified API name -> list of gallery items that use it."""
    global _NOTEBOOK_API_USAGE_CACHE
    if _NOTEBOOK_API_USAGE_CACHE is not None:
        return _NOTEBOOK_API_USAGE_CACHE

    data = _get_discovery_data()

    # Build name -> qualified_name lookup from all discovered API objects
    name_to_qualified: dict[str, str] = {}
    for name, cls in data["estimators"]:
        name_to_qualified[name] = f"{cls.__module__}.{name}"
    for name, cls in data["displays"]:
        name_to_qualified[name] = f"{cls.__module__}.{name}"
    for name, func in data["functions"]:
        name_to_qualified[name] = f"{func.__module__}.{name}"

    gallery_items = _get_gallery_items(project_root)
    # Build stem -> gallery item lookup
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

        try:
            source = notebook.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue

        # Extract all names imported from yohou.*
        imported_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("yohou"):
                for alias in node.names:
                    imported_names.add(alias.name)

        # Map imported names to qualified API names
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

    # Deduplicate by stem (in case of multiple imports from same notebook)
    seen = set()
    unique_items = []
    for item in items:
        if item["stem"] not in seen:
            seen.add(item["stem"])
            unique_items.append(item)

    cards = []
    for item in unique_items:
        desc = item["description"] or "No description."
        cat_label = _CATEGORY_LABELS.get(item["category"], item["category"].title())
        cards.append(
            f"-   **{item['title']}**\n"
            f"\n"
            f"    ---\n"
            f"\n"
            f"    <small>{cat_label}</small>\n"
            f"\n"
            f"    {desc}\n"
            f"\n"
            f"    [View]({item['view_path']}) \u00b7 "
            f"[Open in marimo]({item['open_path']})"
        )

    return (
        "## Examples\n\n"
        "The following example notebooks use this component:\n\n"
        '<div class="grid cards" markdown>\n\n' + "\n\n".join(cards) + "\n\n</div>\n"
    )


def _build_gallery_html(project_root, category=None):
    """Build gallery card grid as Material 'grid cards' markdown."""
    items = _get_gallery_items(project_root)

    if category:
        items = [i for i in items if i["category"] == category]

    if not items:
        return "<!-- no gallery items found -->\n"

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


def on_page_markdown(markdown, page, config, files):
    """Rewrite example links and inject generated API / gallery content.

    Link rewriting
    --------------
    [View] links are converted to relative paths pointing to locally exported
    static HTML notebooks.

    [Open in marimo] placeholder links are resolved to the marimo online
    playground via marimo.app. On RTD the commit SHA being built is used so
    PR previews link to the correct revision; locally, ``main`` is used.

    Placeholder injection
    ---------------------
    ``<!-- API_TABLE -->``         → searchable DataTables HTML table
    ``<!-- GALLERY -->``           → all-categories card grid
    ``<!-- GALLERY:category -->``  → single-category card grid
    """
    project_root = Path(__file__).parent.parent

    src_parts = page.file.src_path.split("/")
    depth = len(src_parts) if src_parts[-1] != "index.md" else len(src_parts) - 1
    prefix = "../" * depth

    repo_url = config.get("repo_url", "").rstrip("/")
    github_path = repo_url.removeprefix("https://")
    git_ref = os.environ.get(
        "READTHEDOCS_GIT_COMMIT_HASH",
        os.environ.get("READTHEDOCS_GIT_IDENTIFIER", "main"),
    )
    playground_base = f"https://marimo.app/{github_path}/blob/{git_ref}"

    # API_TABLE placeholder
    if "<!-- API_TABLE -->" in markdown:
        table_html = _build_api_table_html()
        markdown = markdown.replace("<!-- API_TABLE -->", table_html)

    # GALLERY placeholders: all categories
    if "<!-- GALLERY -->" in markdown:
        gallery_html = _build_gallery_html(project_root)
        markdown = markdown.replace("<!-- GALLERY -->", gallery_html)

    # Per-category: <!-- GALLERY:point -->, <!-- GALLERY:interval -->, etc.
    for match in re.finditer(r"<!-- GALLERY:(\w+) -->", markdown):
        cat = match.group(1)
        gallery_html = _build_gallery_html(project_root, category=cat)
        markdown = markdown.replace(match.group(0), gallery_html)

    # EXAMPLES_FOR placeholders on generated API pages
    for match in re.finditer(r"<!-- EXAMPLES_FOR:([\w.]+) -->", markdown):
        qualified = match.group(1)
        examples_html = _build_api_examples_html(project_root, qualified)
        markdown = markdown.replace(match.group(0), examples_html)

    # Rewrite [Open in marimo] links (after gallery injection so gallery links are included)
    markdown = re.sub(
        r"\[Open in marimo\]\(/examples/([^)]+?)/edit/\)",
        rf"[Open in marimo]({playground_base}/examples/\1.py)",
        markdown,
    )

    markdown = re.sub(r"\]\(/examples/", f"]({prefix}examples/", markdown)

    return markdown


# Numpydoc section types to surface in the TOC.
_DOC_SECTION_TITLE_SLUGS = {
    "Parameters": "parameters",
    "Attributes": "attributes",
    "Returns": "returns",
    "Raises": "raises",
    "Examples": "doc-examples",
}
_DETAIL_SECTION_SLUGS = {
    "note": ("notes", "Notes"),
    "see-also": ("see-also", "See Also"),
    "references": ("references", "References"),
}


def _make_section_heading(slug, title):
    """Build an h3 heading element for an API page section."""
    return (
        f'<h3 id="{slug}" class="doc-section-heading">{title}'
        f'<a class="headerlink" href="#{slug}" '
        f'title="Permanent link">&para;</a></h3>'
    )


def _process_api_page_content(html, page):
    """Convert numpydoc sections to h3 headings under mkdocstrings h2."""
    from mkdocs.structure.toc import AnchorLink

    is_class_page = bool(re.search(r'<h3\s+id="yohou\.', html))

    # Locate class-level content region
    h2_match = re.search(r'<h2\s+id="yohou\.', html)
    if not h2_match:
        return html
    h2_pos = h2_match.start()

    if is_class_page:
        boundary_match = re.search(r'<div\s+class="doc doc-children"', html[h2_pos:])
        boundary_pos = h2_pos + boundary_match.start() if boundary_match else len(html)
    else:
        boundary_pos = len(html)

    class_region = html[h2_pos:boundary_pos]
    sections_found = []  # (id, title) in document order

    # Convert doc-section-title spans to h3 headings
    def _span_to_h3(m):
        title = re.sub(r"<[^>]+>", "", m.group(1)).strip().rstrip(":")
        slug = _DOC_SECTION_TITLE_SLUGS.get(title)
        if slug:
            sections_found.append((slug, title))
            return _make_section_heading(slug, title)
        return m.group(0)

    new_class_region = re.sub(
        r"<p>\s*<span\s+class=\"doc-section-title\"[^>]*>(.*?)</span>\s*</p>",
        _span_to_h3,
        class_region,
    )

    # Convert <details> sections to h3 heading + unwrapped content
    for detail_cls, (slug, title) in _DETAIL_SECTION_SLUGS.items():
        detail_re = re.compile(
            rf'<details\s+class="{re.escape(detail_cls)}"[^>]*>'
            rf"\s*<summary>{re.escape(title)}</summary>"
            rf"(.*?)</details>",
            re.DOTALL,
        )
        m = detail_re.search(new_class_region)
        if m:
            heading = _make_section_heading(slug, title)
            inner = m.group(1).strip()
            new_class_region = new_class_region[: m.start()] + heading + "\n" + inner + new_class_region[m.end() :]
            sections_found.append((slug, title))

    # Convert <details class="mkdocstrings-source"> to Source Code h3
    src_re = re.compile(
        r'<details\s+class="mkdocstrings-source"[^>]*>'
        r"\s*<summary>.*?</summary>"
        r"(.*?)</details>",
        re.DOTALL,
    )
    src_m = src_re.search(new_class_region)
    if src_m:
        heading = _make_section_heading("source-code", "Source Code")
        inner = src_m.group(1).strip()
        new_class_region = new_class_region[: src_m.start()] + heading + "\n" + inner + new_class_region[src_m.end() :]
        sections_found.append(("source-code", "Source Code"))

    # Split See Also entries so each appears on its own line
    def _split_see_also_p(m):
        content = m.group(1)
        # Split on newlines that separate entries (each entry starts with
        # a link <a, a <code> tag, or a plain identifier followed by " :")
        entries = re.split(r"\n(?=<a\s|<code>|[A-Za-z_][\w.]*\s*:)", content.strip())
        if len(entries) <= 1:
            return m.group(0)
        return "\n".join(f"<p>{e.strip()}</p>" for e in entries if e.strip())

    see_also_heading = '<h3 id="see-also"'
    if see_also_heading in new_class_region:
        # Only replace <p> tags that follow the See Also heading
        sa_pos = new_class_region.index(see_also_heading)
        before_sa = new_class_region[:sa_pos]
        after_sa = new_class_region[sa_pos:]
        after_sa = re.sub(r"<p>(.*?)</p>", _split_see_also_p, after_sa, count=1, flags=re.DOTALL)
        new_class_region = before_sa + after_sa

    html = html[:h2_pos] + new_class_region + html[boundary_pos:]

    # Insert "Methods" h3 before doc-children
    if is_class_page:
        methods_heading = _make_section_heading("methods", "Methods") + "\n"
        html = re.sub(
            r'(<div\s+class="doc doc-children")',
            methods_heading + r"\1",
            html,
            count=1,
        )

    # Increase method heading levels (h3 -> h5) in doc-children
    if is_class_page:
        dc_match = re.search(r'<div\s+class="doc doc-children"', html)
        if dc_match:
            before = html[: dc_match.start()]
            after = html[dc_match.start() :]
            after = re.sub(r"<h3(\s)", r"<h5\1", after)
            after = re.sub(r"</h3>", "</h5>", after)
            html = before + after

    # Rename "Examples" h2 to "Tutorials" h3
    examples_h2 = re.search(r'<h2 id="examples">.*?</h2>', html, re.DOTALL)
    if examples_h2:
        old = examples_h2.group(0)
        new = (
            old.replace('<h2 id="examples">', '<h3 id="tutorials">')
            .replace("</h2>", "</h3>")
            .replace(">Examples<", ">Tutorials<")
            .replace("#examples", "#tutorials")
        )
        html = html.replace(old, new, 1)

    # Rebuild page.toc
    old_toc = list(page.toc)
    if old_toc:
        h1 = old_toc[0]
        old_h2s = list(h1.children)

        # The first h2 child is the mkdocstrings class/func heading
        method_items = []
        if old_h2s:
            main_h2 = old_h2s[0]
            method_items = list(main_h2.children)

        # All sections nest inside the mkdocstrings h2
        section_children = []

        # Numpydoc + detail + source code sections (level 3)
        for slug, title in sections_found:
            section_children.append(AnchorLink(title=title, id=slug, level=3))

        # Methods with individual methods nested underneath (level 3 + 4)
        if is_class_page and method_items:
            methods_entry = AnchorLink(title="Methods", id="methods", level=3)
            for mi in method_items:
                mi.level = 4
            methods_entry.children = method_items
            section_children.append(methods_entry)

        # Tutorials (level 3)
        for h2 in old_h2s[1:]:
            if h2.id in ("examples", "tutorials"):
                section_children.append(AnchorLink(title="Tutorials", id="tutorials", level=3))
                break

        if old_h2s:
            main_h2.children = section_children
            h1.children = [main_h2]
        else:
            h1.children = section_children

    return html


# Module page files in display order, mapping module label -> markdown filename.
_MODULE_PAGE_FILES = [
    ("yohou.base", "base.md"),
    ("yohou.compose", "compose.md"),
    ("yohou.point", "point.md"),
    ("yohou.interval", "interval.md"),
    ("yohou.metrics", "metrics.md"),
    ("yohou.model_selection", "model-selection.md"),
    ("yohou.preprocessing", "preprocessing.md"),
    ("yohou.stationarity", "stationarity.md"),
    ("yohou.plotting", "plotting.md"),
    ("yohou.datasets", "datasets.md"),
    ("yohou.utils", "utils.md"),
    ("yohou.testing", "testing.md"),
]


def _build_module_toc(config, current_src_path=None):
    """Build the module TOC list used by API index and submodule templates.

    Parameters
    ----------
    config : dict
        MkDocs config with ``docs_dir``.
    current_src_path : str or None
        Source path of the current page (e.g. ``pages/api/point.md``).
        When set, the matching entry gets ``active: True``.

    Returns
    -------
    list[dict]
        TOC entries with keys *title*, *url*, *children*, and optionally *active*.
    """
    docs_dir = Path(config["docs_dir"])
    api_dir = docs_dir / "pages" / "api"

    # Determine URL prefix based on whether we're on the index or a submodule page
    is_index = current_src_path is None or current_src_path == "pages/api/index.md"

    module_toc = []
    for module_label, md_filename in _MODULE_PAGE_FILES:
        md_path = api_dir / md_filename
        if not md_path.exists():
            continue

        # Compute relative URL
        if is_index:
            page_url = md_filename.replace(".md", "/")
        else:
            # From a sibling submodule page, link to adjacent page
            page_url = f"../{md_filename.replace('.md', '/')}".replace("//", "/")

        # Check if this is the currently active module
        active = current_src_path == f"pages/api/{md_filename}" if current_src_path else False

        entry = {"title": module_label, "url": page_url, "active": active, "children": []}

        # Parse h3 subsections from the module markdown
        content = md_path.read_text(encoding="utf-8")
        for m in re.finditer(r"^###\s+(.+)$", content, re.MULTILINE):
            sub_title = m.group(1).strip()
            sub_slug = re.sub(r"[^\w]+", "-", sub_title.lower()).strip("-")
            child_url = f"{page_url}#{sub_slug}" if not active else f"#{sub_slug}"
            entry["children"].append({"title": sub_title, "url": child_url, "active": False})

        module_toc.append(entry)

    return module_toc


def _process_api_index_toc(page, config):
    """Build a module TOC for the API index page stored in page.meta."""
    page.meta["module_toc"] = _build_module_toc(config, current_src_path="pages/api/index.md")


def _process_api_submodule_toc(page, config):
    """Build a module TOC for an API submodule page stored in page.meta."""
    page.meta["module_toc"] = _build_module_toc(config, current_src_path=page.file.src_path)


def on_page_content(html, page, config, files):
    """Post-process HTML: See Also links and API page TOC restructuring."""
    if '<details class="see-also' in html:
        html = _linkify_see_also(html)

    if page.file.src_path.startswith("pages/api/generated/"):
        html = _process_api_page_content(html, page)

    if page.file.src_path == "pages/api/index.md":
        _process_api_index_toc(page, config)
    elif (
        page.file.src_path.startswith("pages/api/")
        and page.file.src_path != "pages/api/index.md"
        and not page.file.src_path.startswith("pages/api/generated/")
        and page.meta.get("template") == "api-submodule.html"
    ):
        _process_api_submodule_toc(page, config)

    return html


def on_pre_build(config):
    """Export marimo notebooks and generate API pages before building docs."""
    project_root = Path(__file__).parent.parent
    docs_dir = Path(config["docs_dir"])

    # Generate per-class API pages and inject them into nav
    _generate_api_pages(docs_dir)
    _inject_generated_pages_into_nav(config, docs_dir)

    # Export marimo notebooks
    if os.environ.get("MKDOCS_SKIP_NOTEBOOKS"):
        print("[hooks] MKDOCS_SKIP_NOTEBOOKS set, skipping notebook export")
        return

    examples_dir = project_root / "examples"

    if not examples_dir.exists():
        return

    _skip_stems: set[str] = set()
    try:
        import yohou_nixtla  # noqa: F401
    except ModuleNotFoundError:
        _skip_stems |= {"nixtla_forecasters", "nixtla_panel"}

    notebooks = [
        p
        for p in examples_dir.rglob("*.py")
        if "__marimo__" not in p.parts
        and "bugs" not in p.parts
        and "__init__" not in p.name
        and p.stem not in _skip_stems
    ]
    if not notebooks:
        return

    docs_examples = project_root / "docs" / "examples"
    docs_examples.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []

    for notebook in notebooks:
        rel_path = notebook.relative_to(project_root)
        output_dir = docs_examples / notebook.stem

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

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
        except subprocess.CalledProcessError as e:
            failed.append(str(rel_path))
            print(f"[hooks] FAILED html {rel_path}: {e}", file=sys.stderr)
            if e.stderr:
                print(e.stderr, file=sys.stderr)
            continue
        except FileNotFoundError:
            print(
                "[hooks] marimo not found, skipping notebook export",
                file=sys.stderr,
            )
            break

    if failed:
        msg = f"[hooks] {len(failed)} notebook(s) had cell execution errors:\n"
        msg += "\n".join(f"  - {f}" for f in failed)
        raise RuntimeError(msg)


class _HtmlToMarkdown(HTMLParser):
    """HTML parser that converts mkdocs-material HTML to clean markdown."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._lines: list[str] = []
        self._line: list[str] = []
        self._list_stack: list[dict[str, int | str]] = []
        self._in_pre = False
        self._pre_buffer: list[str] = []
        self._pre_lang: str | None = None
        self._in_code_inline = False
        self._code_buffer: list[str] = []
        self._code_target: str = "line"
        self._skip_depth = 0
        self._in_table = False
        self._table_rows: list[list[str]] = []
        self._current_row: list[str] = []
        self._current_cell: list[str] = []
        self._row_has_th = False
        self._first_row_is_header = False
        self._in_highlight_table = False
        self._in_doc_section_title = False
        self._skip_next_table = False

    def get_markdown(self) -> str:
        """Return the accumulated markdown content."""
        self._flush_line()
        self._trim_trailing_blank_lines()
        return "\n".join(self._lines).strip() + "\n"

    def _trim_trailing_blank_lines(self) -> None:
        """Remove trailing blank lines from output."""
        while self._lines and not self._lines[-1].strip():
            self._lines.pop()

    def _flush_line(self) -> None:
        """Flush current line buffer to output."""
        if not self._line:
            return
        line = "".join(self._line).rstrip()
        self._lines.append(line)
        self._line = []

    def _ensure_blank_line(self) -> None:
        """Ensure there's a blank line before the next content."""
        if self._line:
            self._flush_line()
        if not self._lines or self._lines[-1].strip():
            self._lines.append("")

    def _start_block(self) -> None:
        """Start a new block-level element."""
        self._ensure_blank_line()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Handle HTML start tags and convert to markdown."""
        if self._skip_depth:
            self._skip_depth += 1
            return
        attr_map = {k: v or "" for k, v in attrs}
        if tag == "a" and "headerlink" in attr_map.get("class", ""):
            self._skip_depth = 1
            return
        if tag == "span" and "doc-section-title" in attr_map.get("class", ""):
            self._in_doc_section_title = True
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._flush_line()
            self._ensure_blank_line()
            level = int(tag[1])
            self._line.append("#" * level + " ")
        elif tag == "p":
            self._start_block()
        elif tag == "br":
            self._flush_line()
        elif tag == "ul":
            self._start_block()
            self._list_stack.append({"type": "ul", "count": 0})
        elif tag == "ol":
            self._start_block()
            self._list_stack.append({"type": "ol", "count": 1})
        elif tag == "li":
            self._flush_line()
            indent = "  " * max(len(self._list_stack) - 1, 0)
            if self._list_stack and self._list_stack[-1]["type"] == "ol":
                count = int(self._list_stack[-1]["count"])
                self._list_stack[-1]["count"] = count + 1
                bullet = f"{count}."
            else:
                bullet = "-"
            self._line.append(f"{indent}{bullet} ")
        elif tag == "pre":
            self._start_block()
            self._in_pre = True
            self._pre_buffer = []
            self._pre_lang = None
        elif tag == "code" and self._in_pre:
            class_name = attr_map.get("class", "")
            match = re.search(r"language-([a-zA-Z0-9_+-]+)", class_name)
            if match:
                self._pre_lang = match.group(1)
        elif tag == "code":
            self._in_code_inline = True
            self._code_buffer = []
            self._code_target = "cell" if self._in_table else "line"
        elif tag in {"strong", "b"}:
            self._line.append("**")
        elif tag in {"em", "i"}:
            self._line.append("*")
        elif tag == "table":
            if "highlighttable" in attr_map.get("class", ""):
                self._in_highlight_table = True
                return
            if self._skip_next_table:
                self._skip_next_table = False
                self._skip_depth = 1
                return
            self._start_block()
            self._in_table = True
            self._table_rows = []
            self._current_row = []
            self._current_cell = []
            self._row_has_th = False
            self._first_row_is_header = False
        elif tag == "td" and self._in_highlight_table and "linenos" in attr_map.get("class", ""):
            self._skip_depth = 1
        elif tag == "tr" and self._in_table:
            self._current_row = []
            self._row_has_th = False
        elif tag in {"th", "td"} and self._in_table:
            self._current_cell = []
            if tag == "th":
                self._row_has_th = True

    def handle_endtag(self, tag: str) -> None:
        """Handle HTML end tags and complete markdown conversion."""
        if self._skip_depth:
            self._skip_depth -= 1
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"} or tag == "p":
            self._flush_line()
            self._ensure_blank_line()
        elif tag in {"ul", "ol"}:
            if self._list_stack:
                self._list_stack.pop()
            self._flush_line()
            self._ensure_blank_line()
        elif tag == "li":
            self._flush_line()
        elif tag == "pre":
            self._in_pre = False
            self._flush_pre()
        elif tag == "code" and self._in_code_inline:
            code_text = "".join(self._code_buffer).strip()
            if code_text:
                wrapped = f"`{code_text}`"
                if self._code_target == "cell":
                    self._current_cell.append(wrapped)
                else:
                    self._line.append(wrapped)
            self._in_code_inline = False
        elif tag in {"strong", "b"}:
            self._line.append("**")
        elif tag in {"em", "i"}:
            self._line.append("*")
        elif tag in {"th", "td"} and self._in_table:
            cell_text = "".join(self._current_cell).strip()
            self._current_row.append(cell_text)
            self._current_cell = []
        elif tag == "tr" and self._in_table:
            if self._current_row:
                if not self._table_rows:
                    self._first_row_is_header = self._row_has_th
                self._table_rows.append(self._current_row)
            self._current_row = []
        elif tag == "table":
            if self._in_highlight_table:
                self._in_highlight_table = False
                return
            self._emit_table()
            self._in_table = False

    def handle_data(self, data: str) -> None:
        """Handle text data within HTML tags."""
        if self._skip_depth:
            return
        if self._in_doc_section_title:
            section_title = data.strip()
            if section_title == "Parameters:":
                self._skip_next_table = True
            self._in_doc_section_title = False
            return
        if self._in_pre:
            self._pre_buffer.append(data)
            return
        if self._in_code_inline:
            self._code_buffer.append(data)
            return
        text = data
        text = re.sub(r"\s+", " ", text)
        if not text:
            return
        if self._in_table and self._current_cell is not None:
            self._current_cell.append(text)
            return
        if self._line and self._line[-1].endswith(" "):
            text = text.lstrip()
        self._line.append(text)

    def _flush_pre(self) -> None:
        """Flush preformatted code block to markdown."""
        pre_text = "".join(self._pre_buffer)
        pre_text = pre_text.rstrip("\n")
        fence = f"```{self._pre_lang or ''}".rstrip()
        self._lines.append(fence)
        if pre_text:
            self._lines.extend(pre_text.splitlines())
        self._lines.append("```")
        self._lines.append("")
        self._pre_buffer = []
        self._pre_lang = None

    def _emit_table(self) -> None:
        """Emit accumulated table rows as markdown table."""
        if not self._table_rows:
            return
        column_count = max(len(row) for row in self._table_rows)
        rows = [row + [""] * (column_count - len(row)) for row in self._table_rows]
        if self._first_row_is_header:
            header = rows[0]
            body = rows[1:]
        else:
            header = [""] * column_count
            body = rows
        header_line = "| " + " | ".join(self._escape_cell(cell) for cell in header) + " |"
        separator = "| " + " | ".join("---" for _ in header) + " |"
        self._lines.append(header_line)
        self._lines.append(separator)
        for row in body:
            row_line = "| " + " | ".join(self._escape_cell(cell) for cell in row) + " |"
            self._lines.append(row_line)
        self._lines.append("")

    @staticmethod
    def _escape_cell(value: str) -> str:
        """Escape special characters in table cells."""
        return value.replace("|", r"\|").strip()


def _html_to_markdown(html: str) -> str:
    """Convert HTML to clean markdown using custom parser."""
    parser = _HtmlToMarkdown()
    parser.feed(html)
    return parser.get_markdown()


def _extract_article_html(html: str) -> str | None:
    """Extract the main article content from mkdocs HTML."""
    marker = '<article class="md-content__inner md-typeset">'
    start = html.find(marker)
    if start == -1:
        return None
    start += len(marker)
    end = html.find("</article>", start)
    if end == -1:
        return None
    return html[start:end]


def _html_path_for(relative: str, site_dir: Path) -> Path:
    """Convert markdown path to corresponding HTML path in site directory."""
    if relative == "index.md":
        return site_dir / "index.html"
    return site_dir / relative.removesuffix(".md") / "index.html"


def _is_excluded(relative_posix: str, patterns: list[str]) -> bool:
    """Check if a relative path matches any exclusion pattern."""
    return any(fnmatch.fnmatch(relative_posix, pattern) for pattern in patterns)


def _rewrite_api_links_in_html(html_file: Path) -> None:
    """Rewrite absolute /pages/api/ links to relative paths in exported HTML."""
    if not html_file.exists():
        return

    html_content = html_file.read_text(encoding="utf-8")

    # All exported notebooks live at site/examples/<slug>/index.html
    # so relative path to site root is ../../
    html_content = re.sub(
        r'href="/pages/api/',
        'href="../../pages/api/',
        html_content,
    )

    html_file.write_text(html_content, encoding="utf-8")


def _inject_rtd_css(html_file: Path) -> None:
    """Inject CSS to hide Read The Docs version menu flyout in marimo notebooks."""
    if not html_file.exists():
        return

    html_content = html_file.read_text(encoding="utf-8")

    rtd_css = """
  <style>
    readthedocs-flyout {
      display: none;
    }
  </style>
"""

    if "</head>" in html_content:
        html_content = html_content.replace("</head>", f"{rtd_css}</head>", 1)
        html_file.write_text(html_content, encoding="utf-8")


def on_post_build(config):
    """Copy markdown files for LLM consumption after build completes."""
    site_dir = Path(config["site_dir"])
    docs_dir = Path(config["docs_dir"])
    project_root = Path(__file__).parent.parent
    docs_examples = project_root / "docs" / "examples"

    if docs_examples.exists():
        for html_dir in docs_examples.iterdir():
            if not html_dir.is_dir() or html_dir.name.startswith("."):
                continue

            index_html = html_dir / "index.html"
            if not index_html.exists():
                continue

            target_dir = site_dir / "examples" / html_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)

            for file in html_dir.iterdir():
                if file.name == "CLAUDE.md" or file.is_dir():
                    continue
                shutil.copy2(file, target_dir / file.name)

            _inject_rtd_css(target_dir / "index.html")
            _rewrite_api_links_in_html(target_dir / "index.html")
            print(f"[hooks] copied examples/{html_dir.name}/ to site")

    exclude_patterns = ["examples/**/CLAUDE.md"]

    legacy_dir = site_dir / "llm"
    if legacy_dir.exists():
        shutil.rmtree(legacy_dir)

    llms_txt_source = docs_dir / "llms.txt"
    if llms_txt_source.exists():
        llms_txt_dest = site_dir / "llms.txt"
        shutil.copy2(llms_txt_source, llms_txt_dest)
        print("[hooks] copied llms.txt to site")

    copied_count = 0
    for md_file in sorted(docs_dir.rglob("*.md")):
        relative_posix = md_file.relative_to(docs_dir).as_posix()

        if _is_excluded(relative_posix, exclude_patterns):
            continue

        destination = site_dir / relative_posix
        destination.parent.mkdir(parents=True, exist_ok=True)

        html_path = _html_path_for(relative_posix, site_dir)
        if html_path.exists():
            html = html_path.read_text(encoding="utf-8")
            article_html = _extract_article_html(html)
            if article_html:
                markdown = _html_to_markdown(article_html)
                destination.write_text(markdown, encoding="utf-8")
                copied_count += 1
                continue

        destination.write_text(md_file.read_text(encoding="utf-8"), encoding="utf-8")
        copied_count += 1

    if copied_count > 0:
        print(f"[hooks] copied {copied_count} markdown files to site")
