"""Fake OpenAlex works, normalized papers and a scripted LLM. No network, no real model."""
from src import openalex


def inv_index(text):
    idx = {}
    for i, word in enumerate(text.split()):
        idx.setdefault(word, []).append(i)
    return idx


def make_work(wid, title, year=2022, type="article", cites=10, refs=(), doi=None,
              abstract="A study of antibody binding affinity prediction from sequence."):
    """A raw OpenAlex work, as openalex.get would return it."""
    return {
        "id": f"https://openalex.org/{wid}",
        "title": title,
        "publication_year": year,
        "abstract_inverted_index": inv_index(abstract) if abstract else None,
        "cited_by_count": cites,
        "doi": doi if doi is not None else f"https://doi.org/10.1000/{wid.lower()}",
        "referenced_works": [f"https://openalex.org/{r}" for r in refs],
        "type": type,
    }


def make_paper(wid, title=None, year=2022, cites=10, refs=(), relevance=3.0, queries=("q",)):
    """A normalized paper dict (the shape the pipeline passes around)."""
    p = openalex.normalize(make_work(wid, title or f"Paper {wid}", year=year, cites=cites, refs=refs))
    p["queries"] = list(queries)
    p["relevance"] = relevance
    return p


class FakeLLM:
    """Stands in for llm.LLM. `handler(prompt)` returns the parsed JSON for that prompt."""

    def __init__(self, handler, name="fake/model"):
        self.handler, self.name = handler, name
        self.prompts = []
        self.stats = {"api_calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def __str__(self):
        return self.name

    def ask_json(self, prompt):
        self.prompts.append(prompt)
        self.stats["api_calls"] += 1
        return self.handler(prompt)
