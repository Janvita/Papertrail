"""Writes a readable report of the recall check: every reference, every candidate, and the matches.

Usage: python -m evals.report <survey_dir>
Output: <survey_dir>/recall_report.md (survey_*/ is gitignored).
"""
import sys
from pathlib import Path

from src.retrieval import norm_title
from evals.common import load_candidates, load_eval_config, reference_titles


def _mark(flag):
    return "yes" if flag else "no"


def _cell(text):
    return str(text if text is not None else "").replace("|", "/").replace("\n", " ")


def build(survey_dir, cfg=None):
    cfg = cfg or load_eval_config()
    keywords = [k.lower() for k in cfg["topic_keywords"]]
    review_id = cfg["reference_review"]["openalex_id"]

    candidates = load_candidates(survey_dir)
    by_norm = {}
    for c in candidates:
        by_norm.setdefault(norm_title(c["title"]), c)
    final_norms = {norm_title(c["title"]) for c in candidates if c["in_final_set"]}

    refs = {}
    for t in reference_titles(review_id):
        refs.setdefault(norm_title(t), t)

    ref_rows = []
    for n, title in refs.items():
        on_topic = any(k in n for k in keywords)
        cand = by_norm.get(n)
        ref_rows.append({
            "title": title,
            "on_topic": on_topic,
            "in_pool": cand is not None,
            "in_final": n in final_norms,
            "relevance": cand.get("relevance") if cand else None,
        })
    # matched first, then on-topic, then the rest
    ref_rows.sort(key=lambda r: (not r["in_pool"], not r["on_topic"], r["title"].lower()))

    cand_rows = []
    for c in sorted(candidates, key=lambda c: (-(c.get("relevance") or 0), c["title"].lower())):
        cand_rows.append({
            "title": c["title"], "year": c["year"], "relevance": c.get("relevance"),
            "in_final": c["in_final_set"], "in_review": norm_title(c["title"]) in refs,
            "queries": ", ".join(c.get("queries") or []),
        })
    return {"review_id": review_id, "refs": ref_rows, "cands": cand_rows,
            "n_refs": len(refs), "n_on_topic": sum(r["on_topic"] for r in ref_rows)}


def render(data):
    lines = [
        f"# Recall report",
        "",
        f"Reference review: `{data['review_id']}` (\"Progress and challenges for the machine learning-based "
        "design of fit-for-purpose monoclonal antibodies\", 2022)",
        "",
        f"- References with a title: {data['n_refs']}",
        f"- Of those, look on-topic by keyword: {data['n_on_topic']}",
        f"- Candidates in this run: {len(data['cands'])}",
        f"- References found among candidates: {sum(r['in_pool'] for r in data['refs'])}",
        f"- Found in the final set: {sum(r['in_final'] for r in data['refs'])}",
        "",
        "## References found in the candidate pool",
        "",
        "| Reference title | In final set | Relevance score |",
        "|---|---|---|",
    ]
    matched = [r for r in data["refs"] if r["in_pool"]]
    for r in matched:
        lines.append(f"| {_cell(r['title'])} | {_mark(r['in_final'])} | {_cell(r['relevance'])} |")
    if not matched:
        lines.append("| (none) | | |")

    lines += [
        "",
        "## All candidates from the run (70)",
        "",
        "Sorted by relevance score. \"In review\" means the title matches a reference in the review.",
        "",
        "| # | Title | Year | Relevance | In final set | In review | Queries that found it |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, c in enumerate(data["cands"], 1):
        lines.append(f"| {i} | {_cell(c['title'])} | {_cell(c['year'])} | {_cell(c['relevance'])} | "
                     f"{_mark(c['in_final'])} | {_mark(c['in_review'])} | {_cell(c['queries'])} |")

    lines += [
        "",
        "## All references from the review",
        "",
        "Matched references first, then on-topic ones (keyword hit), then the rest. "
        "Only the title is used for matching, so a reference that appears in the pool under a different title will show as not found.",
        "",
        "| # | Reference title | On-topic (keyword) | Found in pool | Found in final set |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(data["refs"], 1):
        lines.append(f"| {i} | {_cell(r['title'])} | {_mark(r['on_topic'])} | {_mark(r['in_pool'])} | {_mark(r['in_final'])} |")
    return "\n".join(lines) + "\n"


def main(argv):
    if len(argv) != 1:
        raise SystemExit("usage: python -m evals.report <survey_dir>")
    data = build(argv[0])
    out = Path(argv[0]) / "recall_report.md"
    out.write_text(render(data))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
