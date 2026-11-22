"""Steps 4-6: ranking, citation edges, reading order."""
import math
from . import config


def rank(papers, limit=config.MAX_PAPERS):
    """Blend of relevance, citation impact and recency (weights in config.py)."""
    if not papers:
        return []

    def impact(p):  # citations per year, log-scaled so one mega-cited paper does not dominate
        age = max(config.CURRENT_YEAR - p["year"], 1)
        return math.log1p((p["citationCount"] or 0) / age)

    max_imp = max(impact(p) for p in papers) or 1
    lo = min(p["year"] for p in papers)
    hi = max(p["year"] for p in papers)

    for p in papers:
        rel = config.MIN_RELEVANCE if p.get("relevance") is None else p["relevance"]
        recency = (p["year"] - lo) / max(hi - lo, 1)
        p["score"] = (config.W_RELEVANCE * rel / 3
                      + config.W_IMPACT * impact(p) / max_imp
                      + config.W_RECENCY * recency)
    return sorted(papers, key=lambda p: p["score"], reverse=True)[:limit]


def citation_edges(papers):
    """(A, B) means paper A cites paper B. Only pairs where both are in the set."""
    idset = {p["paperId"] for p in papers}
    return [(p["paperId"], r) for p in papers for r in p["refs"] if r in idset]


def reading_order(papers, edges):
    """Papers that others build on come first; year breaks ties."""
    by_id = {p["paperId"]: p for p in papers}
    cites = {pid: set() for pid in by_id}
    for citing, cited in edges:
        if by_id[cited]["year"] <= by_id[citing]["year"]:
            cites[citing].add(cited)
    order, done = [], set()
    while len(done) < len(by_id):
        ready = [pid for pid in by_id if pid not in done and cites[pid] <= done]
        if not ready:  # cycle or odd data: fall back to year
            ready = [min((x for x in by_id if x not in done), key=lambda x: by_id[x]["year"])]
        ready.sort(key=lambda x: by_id[x]["year"])
        order.append(ready[0])
        done.add(ready[0])
    return order
