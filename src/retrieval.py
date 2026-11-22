"""Step 2: search OpenAlex, keep real papers only, remove duplicates."""
import logging, re, time
from . import config, openalex

log = logging.getLogger("papertrail")


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def search(queries, per_query=config.PER_QUERY):
    by_id = {}
    for q in queries:
        data = openalex.get("/works", {
            "search": q,
            "filter": "has_abstract:true",
            "per_page": per_query,
            "select": config.OA_SELECT,
        })
        raw = data.get("results", [])
        usable = 0
        for w in raw:
            p = openalex.normalize(w)
            if not (p["title"] and p["abstract"] and p["year"]):
                continue
            if p["type"] not in config.ALLOWED_TYPES:
                continue
            usable += 1
            if p["paperId"] in by_id:
                if q not in by_id[p["paperId"]]["queries"]:
                    by_id[p["paperId"]]["queries"].append(q)
            else:
                p["queries"] = [q]
                by_id[p["paperId"]] = p
        log.info(f"  '{q}': {len(raw)} returned, {usable} usable")
        time.sleep(0.5)
    return dedupe_titles(list(by_id.values()))


def dedupe_titles(papers):
    """Preprint + published version of the same paper -> keep one, merge their info."""
    groups = {}
    for p in papers:
        groups.setdefault(norm_title(p["title"]), []).append(p)
    out = []
    for group in groups.values():
        group.sort(key=lambda p: p["citationCount"], reverse=True)
        best = group[0]
        for other in group[1:]:
            for q in other["queries"]:
                if q not in best["queries"]:
                    best["queries"].append(q)
            best["refs"] = list(dict.fromkeys(best["refs"] + other["refs"]))
            if not best["externalIds"] and other["externalIds"]:
                best["externalIds"] = other["externalIds"]
        out.append(best)
    removed = len(papers) - len(out)
    if removed:
        log.info(f"  removed {removed} duplicate titles")
    return out
