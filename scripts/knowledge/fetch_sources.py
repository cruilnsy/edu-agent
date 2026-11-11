# scripts/knowledge/fetch_sources.py
# Download the knowledge base sources pinned in sources.json and clean them into plain Markdown.
# Output goes to data/knowledge/<topic>/<id>.md, which is git-ignored: the documents keep their
# own licenses (see sources.json) and are not redistributed with this repo.
#
# Usage: python -m scripts.knowledge.fetch_sources [--only ID ...]

import argparse
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("sources.json")
OUT_DIR = ROOT / "data" / "knowledge"

LICENSES = {"MIT", "Apache-2.0", "PSF-2.0", "CC-BY-4.0", "CC-BY-SA-2.5"}


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def fetch(repo: str, commit: str, path: str) -> str:
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{urllib.parse.quote(path)}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode("utf-8")


# ── Markdown / MDX ──

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
_FENCE = re.compile(r"^(```|~~~)(\w*).*?^\1[ \t]*$\n?", re.M | re.S)
_TAG_LINE = re.compile(r"^[ \t]*</?[A-Z]\w*(?:\s[^<>]*)?/?>[ \t]*$\n?", re.M)   # JSX tag alone on its line(s)
_ANCHOR_LINE = re.compile(r"^[ \t]*<a id=[^>]*/>[ \t]*$\n?", re.M)
_MDN_MACRO = re.compile(r"\{\{\s*(\w+)\s*(?:\((.*?)\))?\s*\}\}")
_QUOTED = re.compile(r"\"([^\"]*)\"|'([^']*)'")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_HTML_TAG = re.compile(r"</?(?:div|table|thead|tbody|tr|td|th|p|pre|code|img|span|br)\b[^>]*>")
_HEADING_ANCHOR = re.compile(r"^(#+ .*?)\s*(?:\[\[[\w-]+\]\]|\{\s*#[\w-]+\s*\})[ \t]*$", re.M)


def _mdn_macro(match: re.Match) -> str:
    """{{HTTPHeader("Origin")}} -> Origin; {{domxref("Window/fetch", "fetch()")}} -> fetch()."""
    args = [a or b for a, b in _QUOTED.findall(match.group(2) or "")]
    if not args:
        return ""
    label = args[1] if len(args) > 1 and args[1] else args[0]
    return label.rsplit("/", 1)[-1]


def _drop_blocks(text: str, start: str, end: str) -> str:
    """Remove every block from a line equal to `start` through the next line equal to `end`."""
    out, skipping = [], False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if not skipping and stripped == start:
            skipping = True
        elif skipping and stripped == end:
            skipping = False
        elif not skipping:
            out.append(line)
    return "".join(out)


def _inline_snippets(text: str, repo: str, commit: str, snippet_root: str) -> str:
    """Replace <ComponentPy /> with the Python snippet file it imports (LangGraph docs)."""
    imports = dict(re.findall(r"^import (\w+) from '(/snippets/[^']+)';$", text, re.M))

    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in imports or not name.endswith("Py"):
            return match.group(0)
        return fetch(repo, commit, snippet_root + imports[name]).strip() + "\n"

    return re.sub(r"^<(\w+) />$", replace, text, flags=re.M)


def clean_markdown(text: str, title: str) -> str:
    """Turn a Markdown/MDX doc page into plain Markdown, leaving code blocks untouched."""
    text = _FRONT_MATTER.sub("", text, count=1)
    text = _drop_blocks(text, "<Challenges>", "</Challenges>")    # react.dev exercises
    text = _drop_blocks(text, ":::js", ":::")                     # LangGraph JS variants
    text = re.sub(r"^:::\w*[ \t]*\n", "", text, flags=re.M)

    parts, last = [], 0
    for fence in _FENCE.finditer(text):
        parts.append(_clean_prose(text[last:fence.start()]))
        if fence.group(2) != "css":                                # Styling in react.dev sandboxes
            parts.append(fence.group(0))
        last = fence.end()
    parts.append(_clean_prose(text[last:]))
    prose = [p for p in parts if not _FENCE.match(p)]
    text = "".join(parts)

    if not any(re.search(r"^# ", p, re.M) for p in prose):        # Code comments are not H1s
        text = f"# {title}\n\n{text}"
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def _drop_open_components(text: str) -> str:
    """Drop multi-line components whose attributes hold JSX, e.g. <Card icon={<CodeIcon />} ... />."""
    out, skipping = [], False
    for line in text.splitlines(keepends=True):
        if not skipping and re.match(r"^\s*<[A-Z]\w*\s*$", line):
            skipping = True
        elif skipping:
            skipping = not line.rstrip().endswith(">")
        else:
            out.append(line)
    return "".join(out)


def _clean_prose(text: str) -> str:
    text = re.sub(r"^(import|export) .*$\n?", "", text, flags=re.M)
    text = _drop_open_components(text)
    text = _HEADING_ANCHOR.sub(r"\1", text)
    text = _HTML_TAG.sub("", text)
    text = re.sub(r"\{/\*.*?\*/\}", "", text, flags=re.S)
    text = _TAG_LINE.sub("", text)
    text = _ANCHOR_LINE.sub("", text)
    text = _MDN_MACRO.sub(_mdn_macro, text)
    return _IMAGE.sub("", text)


# ── reStructuredText (Python docs) ──

_RST_UNDERLINE = re.compile(r"^([=\-~^\"'*+#])\1{2,}$")
_RST_ROLE = re.compile(r":(\w+)(?::\w+)?:`([^`]+)`")
_CODE_ROLES = {"func", "class", "meth", "attr", "exc", "mod", "data", "const", "keyword", "envvar"}
_API_DIRECTIVES = {"function", "class", "method", "coroutinefunction", "coroutinemethod",
                   "awaitablefunction", "awaitablemethod", "attribute", "exception", "decorator", "data"}
_LABELS = {"note": "Note", "important": "Important", "warning": "Warning", "seealso": "See also",
           "versionadded": "Added in version", "versionchanged": "Changed in version",
           "deprecated": "Deprecated since version"}


def _rst_role(match: re.Match) -> str:
    """:func:`~asyncio.run` -> `run`; :term:`Coroutines <coroutine>` -> Coroutines."""
    role, target = match.group(1), match.group(2)
    if "<" in target:
        label = target.split("<")[0].strip()
    elif target.startswith("~"):
        label = target[1:].rsplit(".", 1)[-1]
    else:
        label = target.lstrip("!")
    return f"`{label}`" if role in _CODE_ROLES else label


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _literal_block(lines: list[str], i: int, marker_indent: int) -> tuple[list[str], int]:
    """Collect the indented block after a "::" marker or code directive as a fenced block."""
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines) or _indent(lines[i]) <= marker_indent:
        return [], i
    base, block = _indent(lines[i]), []
    while i < len(lines) and (not lines[i].strip() or _indent(lines[i]) >= base):
        block.append(lines[i][base:])
        i += 1
    while block and not block[-1].strip():
        block.pop()
    return ["", "```python", *block, "```", ""], i


def rst_to_markdown(text: str) -> str:
    """Good-enough RST -> Markdown for chunking: headings, API signatures, roles, admonitions,
    and literal blocks fenced so code comments are not read as headings."""
    lines, out, levels, i = text.splitlines(), [], [], 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        # Overlined heading: ===, title, ===
        if _RST_UNDERLINE.match(line) and i + 2 < len(lines) and lines[i + 2] == line and lines[i + 1].strip():
            key, title, i = "o" + line[0], lines[i + 1].strip(), i + 3
        elif line.strip() and _RST_UNDERLINE.match(nxt) and len(nxt) >= len(line.strip()):
            key, title, i = "u" + nxt[0], line.strip(), i + 2
        else:
            key = None
        if key:
            if key not in levels:
                levels.append(key)
            out.append("#" * min(levels.index(key) + 1, 6) + " " + title)
            continue

        directive = re.match(r"^\s*\.\. ([\w-]+)::\s*(.*)$", line)
        if directive:
            name, arg = directive.groups()
            i += 1
            while i < len(lines) and re.match(r"^\s+:[\w-]+:", lines[i]):   # Directive options
                i += 1
            if name in _API_DIRECTIVES:
                out.append(f"#### `{arg}`")
            elif name in _LABELS:
                out.append(f"*{_LABELS[name]}{' ' + arg if arg else ''}:*")
            elif name in ("code-block", "testcode", "doctest"):
                block, i = _literal_block(lines, i, _indent(line))
                out.extend(block)
            continue
        if re.match(r"^\s*\.\. _[\w-]+:$", line) or _RST_UNDERLINE.match(line):
            i += 1
            continue

        text_line = _RST_ROLE.sub(_rst_role, line).replace("``", "`").strip()
        if text_line.endswith("::"):
            text_line = text_line[:-3] if text_line.endswith(" ::") else text_line[:-1]
            out.append(text_line)
            block, i = _literal_block(lines, i + 1, _indent(line))
            out.extend(block)
            continue
        out.append(text_line)
        i += 1
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"


# ── Main ──

def build_document(source: dict) -> str:
    raw = fetch(source["repo"], source["commit"], source["path"])
    if source["path"].endswith(".rst"):
        body = rst_to_markdown(raw)
    else:
        if "snippet_root" in source:
            raw = _inline_snippets(raw, source["repo"], source["commit"], source["snippet_root"])
        body = clean_markdown(raw, source["title"])
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="Fetch only these source ids")
    args = parser.parse_args()

    manifest = load_manifest()
    for source in manifest["sources"]:
        if args.only and source["id"] not in args.only:
            continue
        out = OUT_DIR / source["topic"] / f"{source['id']}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(build_document(source), encoding="utf-8")
        print(f"{source['id']:<28} {out.stat().st_size:>7} bytes  ({source['license']})")


if __name__ == "__main__":
    main()
