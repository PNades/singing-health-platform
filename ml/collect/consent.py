"""The consent text shown on the collection site (spec 2.5).

The wording has one home: the blockquote under "## Consent (shown on the site,
version `consent-vN`)" in docs/data_protocol.md. This module reads it at
startup and turns its paragraphs and **bold** spans into HTML. A missing or
malformed section is an error, so the site never starts without consent text.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

PROTOCOL = Path(__file__).resolve().parents[2] / "docs" / "data_protocol.md"
_HEADING = re.compile(r"^## Consent \(shown on the site, version `(consent-v\d+)`\)\s*$", re.M)


def load(path: Path = PROTOCOL) -> tuple[str, str]:
    """(version, html) of the consent text."""
    text = path.read_text(encoding="utf-8")
    m = _HEADING.search(text)
    if not m:
        raise RuntimeError(f"No consent section in {path}")
    lines = []
    for line in text[m.end() :].lstrip("\n").splitlines():
        if not line.startswith(">"):
            break
        lines.append(line[1:].strip())
    paras = [p.strip() for p in "\n".join(lines).split("\n\n") if p.strip()]
    if len(paras) < 3:
        raise RuntimeError(f"Consent text in {path} looks incomplete")
    out = []
    for p in paras:
        body = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(p.replace("\n", " ")))
        out.append(f"<p>{body}</p>")
    return m.group(1), "\n".join(out)
