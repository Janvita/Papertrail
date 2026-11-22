"""Step 7: structured summaries from abstracts (batched)."""
from . import config

SCHEMA_HINT = (
    '{"id": 0, "core_idea": "one sentence", "architecture": "...", '
    '"datasets": ["..."], "metrics": ["..."], '
    '"delta": "what is new versus earlier work", '
    '"read_if": "who should read this and why"}'
)


def extract_batch(batch, llm):
    items = "\n\n".join(
        f"[{i}] Title: {p['title']}\nYear: {p['year']}\nAbstract: {p['abstract']}"
        for i, p in enumerate(batch)
    )
    data = llm.ask_json(
        f"Below are {len(batch)} papers, numbered.\n\n{items}\n\n"
        "For EACH paper, fill this schema using ONLY its abstract. "
        "Use null when the abstract does not say. Keep the same id as the paper number.\n"
        f'Format: {{"results": [{SCHEMA_HINT}]}}'
    )
    out = {}
    for r in data.get("results", []):
        try:
            out[int(r["id"])] = r
        except (KeyError, ValueError, TypeError):
            continue
    return out


def extract_all(papers, llm):
    info = {}
    for i in range(0, len(papers), config.EXTRACT_BATCH):
        batch = papers[i:i + config.EXTRACT_BATCH]
        try:
            res = extract_batch(batch, llm)
        except Exception as ex:
            print(f"  batch starting at {i} failed ({ex}); trying papers one by one...")
            res = {}
            for j, p in enumerate(batch):
                try:
                    res[j] = extract_batch([p], llm)[0]
                except Exception as ex2:
                    print("  extract failed:", p["title"], ex2)
        for j, p in enumerate(batch):
            info[p["paperId"]] = res.get(j, {})
    return info
