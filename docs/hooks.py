"""MkDocs hooks for post-build processing."""

import fnmatch
import re
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path


def on_page_markdown(markdown, page, config, files):
    """Rewrite example links to work in both local and RTD environments.

    Converts absolute paths like /examples/ to relative paths based on page depth.
    This works on both local builds and Read the Docs.
    """
    # Calculate relative path based on page depth
    src_parts = page.file.src_path.split("/")

    # Calculate depth (pages/examples.md has depth 2, index.md has depth 0)
    depth = len(src_parts) if src_parts[-1] != "index.md" else len(src_parts) - 1

    # Build relative prefix: '../' repeated for each directory level
    prefix = "../" * depth

    # Replace absolute paths with relative paths
    markdown = re.sub(r"\]\(/examples/", f"]({prefix}examples/", markdown)
    return markdown


def on_pre_build(config):
    """Export marimo notebooks before building documentation.

    This ensures standalone HTML versions are available when mkdocs processes files.
    """
    project_root = Path(__file__).parent.parent
    examples_dir = project_root / "examples"

    if not examples_dir.exists():
        return

    # Find all marimo notebooks
    notebooks = list(examples_dir.glob("*.py"))
    if not notebooks:
        return

    docs_examples = project_root / "docs" / "examples"
    docs_examples.mkdir(parents=True, exist_ok=True)

    # Export each notebook
    for notebook in notebooks:
        output_dir = docs_examples / notebook.stem
        output_file = output_dir / "index.html"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "marimo",
                    "-y",
                    "-q",
                    "export",
                    "html-wasm",
                    str(notebook),
                    "-o",
                    str(output_file),
                    "--mode",
                    "edit",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"[hooks] exported {notebook.relative_to(project_root)} -> {output_file.relative_to(project_root)}")
        except subprocess.CalledProcessError as e:
            print(f"[hooks] Error exporting {notebook.name}: {e}", file=sys.stderr)
            if e.stderr:
                print(e.stderr, file=sys.stderr)
        except FileNotFoundError:
            print("[hooks] marimo not found, skipping notebook export", file=sys.stderr)
            break


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


def _fix_marimo_filename(html_file: Path, notebook_name: str) -> None:
    """Replace the default 'notebook.py' filename in marimo HTML exports.

    Marimo exports always use 'notebook.py' as the default filename. This function
    replaces it with the actual notebook name to show the correct title in browser tabs.
    """
    if not html_file.exists():
        return

    html_content = html_file.read_text(encoding="utf-8")

    # Replace both the marimo-filename tag and the config filename
    html_content = html_content.replace(
        "<marimo-filename hidden>notebook.py</marimo-filename>",
        f"<marimo-filename hidden>{notebook_name}.py</marimo-filename>",
    )
    html_content = html_content.replace(
        '"filename": "notebook.py"',
        f'"filename": "{notebook_name}.py"',
    )

    html_file.write_text(html_content, encoding="utf-8")


def _inject_rtd_css(html_file: Path) -> None:
    """Inject CSS to hide Read The Docs version menu flyout in marimo notebooks.

    This ensures marimo notebooks have the same clean appearance as other documentation
    pages by hiding the RTD version selector that appears in the bottom right corner.
    """
    if not html_file.exists():
        return

    html_content = html_file.read_text(encoding="utf-8")

    # CSS to hide the RTD flyout menu
    rtd_css = """
  <style>
    readthedocs-flyout {
      display: none;
    }
  </style>
"""

    # Inject the CSS before the closing </head> tag
    if "</head>" in html_content:
        html_content = html_content.replace("</head>", f"{rtd_css}</head>", 1)
        html_file.write_text(html_content, encoding="utf-8")


def on_post_build(config):
    """Copy markdown files for LLM consumption after build completes."""
    site_dir = Path(config["site_dir"])
    docs_dir = Path(config["docs_dir"])
    project_root = Path(__file__).parent.parent
    docs_examples = project_root / "docs" / "examples"

    # Copy standalone HTML example files
    if docs_examples.exists():
        for html_dir in docs_examples.iterdir():
            if not html_dir.is_dir() or html_dir.name.startswith("."):
                continue

            index_html = html_dir / "index.html"
            if not index_html.exists():
                continue

            # Create target directory in site
            target_dir = site_dir / "examples" / html_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)

            # Copy index.html and all other files in the directory
            for file in html_dir.iterdir():
                if file.name == "CLAUDE.md":
                    continue
                target_file = target_dir / file.name
                if file.is_file():
                    shutil.copy2(file, target_file)

            # Fix the marimo filename to show the actual notebook name in browser tabs
            _fix_marimo_filename(target_dir / "index.html", html_dir.name)

            # Inject CSS to hide RTD version menu in marimo notebooks
            _inject_rtd_css(target_dir / "index.html")

            print(f"[hooks] copied examples/{html_dir.name}/ to site")

    # Get exclude patterns from config
    # Note: mkdocs converts exclude_docs to a GitIgnoreSpec object, so we use an empty list
    exclude_patterns = []

    # Remove legacy llm/ directory if it exists
    legacy_dir = site_dir / "llm"
    if legacy_dir.exists():
        shutil.rmtree(legacy_dir)

    # Copy llms.txt if it exists
    llms_txt_source = docs_dir / "llms.txt"
    if llms_txt_source.exists():
        llms_txt_dest = site_dir / "llms.txt"
        shutil.copy2(llms_txt_source, llms_txt_dest)
        print("[hooks] copied llms.txt to site")

    # Process markdown files
    copied_count = 0
    for md_file in sorted(docs_dir.rglob("*.md")):
        relative_posix = md_file.relative_to(docs_dir).as_posix()

        # Skip excluded files
        if _is_excluded(relative_posix, exclude_patterns):
            continue

        destination = site_dir / relative_posix
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Try to convert from built HTML first
        html_path = _html_path_for(relative_posix, site_dir)
        if html_path.exists():
            html = html_path.read_text(encoding="utf-8")
            article_html = _extract_article_html(html)
            if article_html:
                markdown = _html_to_markdown(article_html)
                destination.write_text(markdown, encoding="utf-8")
                copied_count += 1
                continue

        # Fallback: copy original markdown
        destination.write_text(md_file.read_text(encoding="utf-8"), encoding="utf-8")
        copied_count += 1

    if copied_count > 0:
        print(f"[hooks] copied {copied_count} markdown files to site")
