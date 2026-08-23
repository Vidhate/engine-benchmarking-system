"""Loader for the checked-in support corpus.

Each document is a markdown file with a small frontmatter block and, after an
`<!-- ARCHIVED yyyy-mm-dd -->` marker, an outdated revision of the same page.
The archived half is never indexed; it exists so the `stale` retriever fault
can serve genuinely outdated — and genuinely contradictory — content instead
of a synthetic-looking placeholder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

CORPUS_DIR = Path(__file__).parent / "corpus"

_FRONTMATTER = re.compile(r"\A---\s*\n(?P<meta>.*?)\n---\s*\n(?P<body>.*)\Z", re.DOTALL)
_ARCHIVED = re.compile(r"^<!--\s*ARCHIVED\s+(?P<date>[\d-]+)\s*-->\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    tags: list[str] = field(default_factory=list)
    updated: str = ""
    text: str = ""
    stale_text: str = ""
    stale_updated: str = ""


def _parse_frontmatter(raw: str, path: Path) -> tuple[dict[str, str], str]:
    match = _FRONTMATTER.match(raw)
    if match is None:
        raise ValueError(f"{path}: missing frontmatter block")
    meta: dict[str, str] = {}
    for line in match.group("meta").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, match.group("body")


def parse_document(raw: str, path: Path) -> Document:
    meta, body = _parse_frontmatter(raw, path)
    missing = {"doc_id", "title", "tags", "updated"} - meta.keys()
    if missing:
        raise ValueError(f"{path}: frontmatter missing {sorted(missing)}")

    split = _ARCHIVED.search(body)
    if split is None:
        raise ValueError(f"{path}: missing '<!-- ARCHIVED ... -->' revision marker")
    current, archived = body[: split.start()], body[split.end() :]

    return Document(
        doc_id=meta["doc_id"],
        title=meta["title"],
        tags=[tag.strip() for tag in meta["tags"].split(",") if tag.strip()],
        updated=meta["updated"],
        text=current.strip(),
        stale_text=archived.strip(),
        stale_updated=split.group("date"),
    )


def load_corpus(directory: Path | str | None = None) -> list[Document]:
    """Load every `*.md` document, sorted by `doc_id` for determinism."""
    root = Path(directory) if directory is not None else CORPUS_DIR
    paths = sorted(root.glob("*.md"))
    docs = [parse_document(path.read_text(encoding="utf-8"), path) for path in paths]
    return sorted(docs, key=lambda d: d.doc_id)
