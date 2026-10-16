"""Load Markdown documents and split them into chunks for retrieval.

Chunking strategy — structure first, size second:
- split on `##` section headings, so each chunk is one coherent topic (an alarm code, a procedure);
- prefix every chunk with "Document title > Section" so the chunk keeps its context on its own;
- only if a section is long, cut it into overlapping word windows (overlap keeps sentences that
  straddle a boundary retrievable).
Each chunk carries the document's access groups (ACL) so permissions are enforced at search time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MAX_WORDS = 180
OVERLAP_WORDS = 40


@dataclass(frozen=True)
class Chunk:
    """One searchable piece of a document (one section) plus the metadata used for citations and access control."""
    chunk_id: str
    source: str           # file name, shown to the user as the citation
    title: str
    section: str
    text: str             # "Title > Section\n<body>" — what gets embedded and shown to the LLM
    access: frozenset[str]
    revision: str


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Split a `---`-delimited header of `key: value` lines from the body. Called by: chunk_document()."""
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S)
    if not match:
        return {}, raw
    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, match.group(2)


def _windows(words: list[str]) -> list[list[str]]:
    """Cut a long section into overlapping word windows; short sections stay whole. Called by: chunk_document()."""
    if len(words) <= MAX_WORDS:
        return [words]
    step = MAX_WORDS - OVERLAP_WORDS
    return [words[i:i + MAX_WORDS] for i in range(0, len(words) - OVERLAP_WORDS, step)]


def chunk_document(path: Path) -> list[Chunk]:
    """Read one Markdown file and return its chunks, one per `##` section (windows if long).
    Called by: load_corpus()."""
    meta, body = parse_front_matter(path.read_text(encoding="utf-8").replace("\r\n", "\n"))
    title = meta.get("title", path.stem)
    access = frozenset(g.strip() for g in meta.get("access", "").split(",") if g.strip())
    chunks: list[Chunk] = []
    # re.split with a capture group returns [preamble, heading1, body1, heading2, body2, ...]
    parts = re.split(r"^## (.+)$", body, flags=re.M)
    for heading, section_body in zip(parts[1::2], parts[2::2]):
        words = section_body.split()
        for n, window in enumerate(_windows(words)):
            chunks.append(Chunk(
                chunk_id=f"{path.stem}#{len(chunks)}",
                source=path.name,
                title=title,
                section=heading.strip() + (f" (part {n + 1})" if n else ""),
                text=f"{title} > {heading.strip()}\n{' '.join(window)}",
                access=access,
                revision=meta.get("revision", ""),
            ))
    return chunks


def load_corpus(docs_path: Path) -> list[Chunk]:
    """Chunk every .md file in the docs folder. Called by: ChatService.__init__() to build the index."""
    return [chunk for path in sorted(docs_path.glob("*.md")) for chunk in chunk_document(path)]
