"""PDF -> Markdown with Docling (keeps sections and tables, unlike plain text extraction).

Week 1 task: run this on 3 papers and READ the output. Note where tables, equations,
or supplementary info get mangled. That failure list is your first real engineering insight.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _converter():
    from docling.document_converter import DocumentConverter  # heavy import, load once

    return DocumentConverter()


def pdf_to_markdown(pdf_path: str, cache: bool = True) -> str:
    md_path = Path(pdf_path).with_suffix(".md")
    if cache and md_path.exists():
        return md_path.read_text()
    result = _converter().convert(pdf_path)
    md = result.document.export_to_markdown()
    md_path.write_text(md)
    return md
