"""Step 3: LLM relevance filter, so off-topic papers never reach the survey."""
from . import config


def score_relevance(papers, topic, llm):
    """Adds p['relevance'] (0-3) and p['relevance_reason'] to every paper.
    None means the call failed."""
    for i in range(0, len(papers), config.RELEVANCE_BATCH):
        batch = papers[i:i + config.RELEVANCE_BATCH]
        items = "\n\n".join(
            f"[{j}] {p['title']} ({p['year']})\n{p['abstract'][:700]}"
            for j, p in enumerate(batch)
        )
        try:
            data = llm.ask_json(
                f"A literature survey is being written on: {topic}\n\n"
                f"Rate how relevant each paper is to that topic, using ONLY its title and abstract.\n"
                "3 = directly about the topic\n"
                "2 = closely related (a method, dataset or background clearly used for the topic)\n"
                "1 = tangential (same broad field, different problem)\n"
                "0 = unrelated\n\n"
                f"{items}\n\n"
                "For each paper, also give a one-sentence reason for the score, pointing at what "
                "in the title/abstract drove it.\n"
                'Format: {"scores": [{"id": 0, "score": 3, "reason": "..."}, ...]} '
                "with one entry per paper."
            )
            results = {int(r["id"]): (float(r["score"]), r.get("reason"))
                       for r in data.get("scores", [])}
        except Exception as ex:
            print(f"  relevance batch at {i} failed ({ex}); keeping those papers unscored")
            results = {}
        for j, p in enumerate(batch):
            score, reason = results.get(j, (None, None))
            p["relevance"] = score
            p["relevance_reason"] = reason
        print(f"  scored {min(i + config.RELEVANCE_BATCH, len(papers))}/{len(papers)}")
    return papers


def filter_relevant(papers, min_relevance=config.MIN_RELEVANCE):
    """Keep papers at or above the threshold. Unscored papers are kept.
    If that leaves fewer than MIN_KEEP, keep the best MIN_KEEP instead."""
    def rel(p):
        return config.MIN_RELEVANCE if p.get("relevance") is None else p["relevance"]
    kept = [p for p in papers if rel(p) >= min_relevance]
    if len(kept) < config.MIN_KEEP:
        kept = sorted(papers, key=rel, reverse=True)[:config.MIN_KEEP]
        print(f"  filter was too strict, kept the best {len(kept)} instead")
    print(f"  relevance filter: kept {len(kept)} of {len(papers)}")
    return kept


def query_stats(queries, candidates, relevant, final):
    """Per query: how many papers it found, how many were relevant, how many made the final set."""
    rows = []
    for q in queries:
        found = [p for p in candidates if q in p["queries"]]
        scored = [p["relevance"] for p in found if p.get("relevance") is not None]
        rows.append({
            "query": q,
            "found": len(found),
            "relevant": sum(1 for p in found if p in relevant),
            "final": sum(1 for p in found if p in final),
            "avg_relevance": round(sum(scored) / len(scored), 2) if scored else None,
        })
    return rows
