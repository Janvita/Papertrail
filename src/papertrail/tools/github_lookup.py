"""Code Hunter's tools: find repo links in the paper text, then verify them on GitHub."""

from __future__ import annotations

import re
from typing import Optional

from github import Github, GithubException

from papertrail.config import GITHUB_TOKEN

GITHUB_URL = re.compile(r"https?://github\.com/[\w.-]+/[\w.-]+", re.IGNORECASE)


def find_github_links(text: str) -> list[str]:
    links = [m.rstrip(".,)") for m in GITHUB_URL.findall(text)]
    return list(dict.fromkeys(links))  # dedupe, keep order


def verify_repo(url: str) -> Optional[dict]:
    """Return repo facts, or None if it doesn't exist / isn't reachable."""
    full_name = "/".join(url.rstrip("/").split("/")[-2:]).removesuffix(".git")
    gh = Github(GITHUB_TOKEN) if GITHUB_TOKEN else Github()
    try:
        repo = gh.get_repo(full_name)
    except GithubException:
        return None
    try:
        repo.get_readme()
        has_readme = True
    except GithubException:
        has_readme = False
    return {
        "repo_url": repo.html_url,
        "stars": repo.stargazers_count,
        "last_push": repo.pushed_at.date().isoformat() if repo.pushed_at else None,
        "has_readme": has_readme,
    }


def search_repo_by_title(title: str) -> Optional[str]:
    """Fallback when the paper has no link. Week 2 stretch: make this smarter
    (check README mentions the paper title / arXiv id before trusting it)."""
    gh = Github(GITHUB_TOKEN) if GITHUB_TOKEN else Github()
    try:
        hits = gh.search_repositories(query=f'"{title[:80]}" in:readme')
        for repo in hits[:3]:
            return repo.html_url
    except GithubException:
        return None
    return None
