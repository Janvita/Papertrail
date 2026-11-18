"""Code Hunter agent: deterministic first, LLM only if needed.

Interview point: not every 'agent' step needs an LLM. Regex + GitHub API is cheaper,
faster, and more reliable for this job. Use the LLM only for the ambiguous cases.
"""

from __future__ import annotations

from papertrail.schemas import CodeInfo, PaperCard
from papertrail.tools.github_lookup import find_github_links, search_repo_by_title, verify_repo


def hunt_code(card: PaperCard, paper_md: str) -> CodeInfo:
    links = find_github_links(paper_md)
    info = CodeInfo(url_in_paper=links[0] if links else None)

    for link in links:
        facts = verify_repo(link)
        if facts:
            return info.model_copy(update={**facts, "verified": True})

    if links:
        info.notes = "Repo linked in paper but not reachable (deleted, private, or typo)."
        return info

    guess = search_repo_by_title(card.title)
    if guess and (facts := verify_repo(guess)):
        return info.model_copy(
            update={**facts, "verified": False, "notes": "No link in paper; found by title search. Check manually."}
        )

    info.notes = "No code found."
    return info
