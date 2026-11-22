"""Small helpers shared by the output modules."""
import re


def slug(title):
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60]


def paper_url(p):
    """Best link for a paper: arXiv, then DOI, then the OpenAlex page."""
    arxiv = (p.get("externalIds") or {}).get("ArXiv")
    if arxiv:
        return f"https://arxiv.org/abs/{arxiv}"
    return p.get("doi") or f"https://openalex.org/{p['paperId']}"


def link(p):
    """Markdown hyperlink: [title](url)."""
    return f"[{p['title']}]({paper_url(p)})"
