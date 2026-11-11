# tests/unit/test_knowledge_sources.py
# The knowledge base manifest must pin every source to a commit and an allowed license, and the
# cleaners must strip doc-site markup without touching code blocks. No network access.

import re

from scripts.knowledge import fetch_sources as fs


def test_manifest_is_pinned_and_licensed():
    manifest = fs.load_manifest()
    ids = [s["id"] for s in manifest["sources"]]
    assert len(ids) == len(set(ids))
    for source in manifest["sources"]:
        assert source["topic"] in manifest["topics"], source["id"]
        assert re.fullmatch(r"[0-9a-f]{40}", source["commit"]), source["id"]
        assert source["license"] in fs.LICENSES, source["id"]
        assert source["url"].startswith("https://"), source["id"]


def test_markdown_cleaning_keeps_code_and_drops_markup():
    raw = """---
title: Demo
---
import Thing from './thing';

Use {{HTTPHeader("Origin")}} and {{domxref("Window/fetch", "fetch()")}}. Render `<SearchBar />`.

<Sandpack>

```js
import { useState } from 'react';
// <Note> in code stays
```

```css
body { color: red; }
```

</Sandpack>

<Challenges>
Exercise text
</Challenges>

## Section { #section }

<div class="termy">
Shown
</div>
"""
    out = fs.clean_markdown(raw, "Demo page")
    assert out.startswith("# Demo page\n")                   # Added when the doc has no H1
    assert "Use Origin and fetch()." in out
    assert "`<SearchBar />`" in out                           # Inline code is kept
    assert "import { useState } from 'react';" in out         # Code is kept, imports in prose are not
    assert "import Thing" not in out
    assert "// <Note> in code stays" in out
    assert "color: red" not in out and "Exercise text" not in out
    assert "<Sandpack>" not in out and "<div" not in out and "Shown" in out
    assert "## Section\n" in out


def test_comment_in_code_is_not_taken_as_h1():
    out = fs.clean_markdown("## Graphs\n\n```python\n# a comment\n```\n", "LangGraph")
    assert out.startswith("# LangGraph\n")


def test_langgraph_js_variants_are_dropped():
    raw = "# T\n\n:::python\nPython text\n:::\n\n:::js\nJS text\n:::\n"
    out = fs.clean_markdown(raw, "T")
    assert "Python text" in out and "JS text" not in out and ":::" not in out


def test_rst_conversion():
    raw = """====================
Coroutines and tasks
====================

Running
=======

.. function:: run(coro)

   Run :func:`~asyncio.run` with :term:`Coroutines <coroutine>`.
   Example::

       # a comment, not a heading
       asyncio.run(main())

   .. versionchanged:: 3.12
      Something changed.
"""
    out = fs.rst_to_markdown(raw)
    assert out.startswith("# Coroutines and tasks\n")
    assert "## Running" in out
    assert "#### `run(coro)`" in out
    assert "Run `run` with Coroutines." in out
    assert "```python\n# a comment, not a heading\nasyncio.run(main())\n```" in out
    assert "*Changed in version 3.12:*" in out
    assert ".. " not in out
