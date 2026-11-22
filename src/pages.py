"""Markdown output: index.md and one summary page per paper."""
from .utils import slug, link


def write_pages(topic, papers, info, edges, order, out, qstats=None):
    out.mkdir(parents=True, exist_ok=True)
    by_id = {p["paperId"]: p for p in papers}

    # Per-paper summary pages (only when extraction is on)
    if info is not None:
        for pid in order:
            p, e = by_id[pid], info.get(pid, {})
            builds_on = [link(by_id[c]) for a, c in edges if a == pid]
            rel = p.get("relevance")
            lines = [
                f"# {link(p)}",
                f"{p['year']} | {p['citationCount']} citations"
                + (f" | relevance {rel:g}/3" if rel is not None else ""), "",
                f"**Core idea:** {e.get('core_idea')}", "",
                f"**Architecture:** {e.get('architecture')}", "",
                f"**Datasets:** {', '.join(e.get('datasets') or []) or 'not stated'}",
                f"**Metrics:** {', '.join(e.get('metrics') or []) or 'not stated'}", "",
                f"**What is new:** {e.get('delta')}", "",
                f"**Read if:** {e.get('read_if')}", "",
                "**Builds on (within this set):** " + ("; ".join(builds_on) or "none found"),
            ]
            (out / f"{slug(p['title'])}.md").write_text("\n".join(lines))

    # Index page
    idx = [f"# Survey: {topic}", "",
           "[Open interactive citation graph](graph.html)", "",
           "## Reading order"]
    for i, pid in enumerate(order, 1):
        p = by_id[pid]
        summary = f" ([summary]({slug(p['title'])}.md))" if info is not None else ""
        idx.append(f"{i}. {link(p)} ({p['year']}){summary}")

    idx += ["", "## Timeline"]
    for p in sorted(papers, key=lambda x: x["year"]):
        idx.append(f"- {p['year']}: {link(p)}")

    if info is not None:
        idx += ["", "## Datasets across papers"]
        ds = {}
        for pid in order:
            for d in info.get(pid, {}).get("datasets") or []:
                ds.setdefault(d, []).append(by_id[pid]["title"])
        for d, ts in sorted(ds.items(), key=lambda kv: -len(kv[1])):
            idx.append(f"- {d}: {len(ts)} papers")

    if qstats:
        idx += ["", "## Search queries (which ones paid off)"]
        for r in qstats:
            avg = "n/a" if r["avg_relevance"] is None else r["avg_relevance"]
            idx.append(f"- `{r['query']}`: found {r['found']}, relevant {r['relevant']}, "
                       f"in final set {r['final']}, avg relevance {avg}")

    (out / "index.md").write_text("\n".join(idx))
