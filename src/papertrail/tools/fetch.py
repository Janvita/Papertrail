"""Get a paper PDF from an arXiv id, a direct PDF URL (bioRxiv works), or a local path."""

from __future__ import annotations

import re
from pathlib import Path

import arxiv
import httpx

from papertrail.config import PDF_DIR

ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


def fetch_paper(source: str) -> dict:
    """Return {"paper_id", "pdf_path", "title", "year"} (title/year may be None)."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    # 1) local file
    p = Path(source)
    if p.exists():
        return {"paper_id": p.stem, "pdf_path": str(p), "title": None, "year": None}

    # 2) arXiv id
    if ARXIV_ID.match(source):
        result = next(arxiv.Client().results(arxiv.Search(id_list=[source])))
        pdf_path = PDF_DIR / f"{source}.pdf"
        if not pdf_path.exists():
            _download(f"https://arxiv.org/pdf/{source}", pdf_path)
        return {
            "paper_id": source,
            "pdf_path": str(pdf_path),
            "title": result.title,
            "year": result.published.year,
        }

    # 3) any direct PDF URL (e.g. bioRxiv ".../2024.01.01.123456v1.full.pdf")
    if source.startswith("http"):
        stem = re.sub(r"[^A-Za-z0-9._-]", "_", source.rstrip("/").split("/")[-1])[:80]
        stem = stem.removesuffix(".pdf")
        pdf_path = PDF_DIR / f"{stem}.pdf"
        if not pdf_path.exists():
            _download(source, pdf_path)
        return {"paper_id": stem, "pdf_path": str(pdf_path), "title": None, "year": None}

    raise ValueError(f"Don't know how to fetch: {source}")


def _download(url: str, dest: Path) -> None:
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        r = client.get(url, headers={"User-Agent": "PaperTrail/0.1 (research tool)"})
        r.raise_for_status()
        dest.write_bytes(r.content)
