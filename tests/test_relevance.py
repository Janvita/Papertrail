import re

import pytest

from src import config, relevance
from tests.fakes import FakeLLM, make_paper


def score_handler(prompt):
    rows = re.findall(r"^\[(\d+)\] (.*) \(\d{4}\)$", prompt, flags=re.M)
    return {"scores": [{"id": int(i), "score": 0 if "offtopic" in title else 3,
                        "reason": f"reason for {title}"} for i, title in rows]}


def test_scores_and_reasons_are_attached_across_batches():
    papers = [make_paper(f"W{i}", title="offtopic paper" if i == 11 else f"On topic {i}", relevance=None)
              for i in range(12)]                      # 12 papers = batches of 10 + 2
    llm = FakeLLM(score_handler)
    relevance.score_relevance(papers, "antibodies", llm)
    assert len(llm.prompts) == 2
    assert papers[0]["relevance"] == 3.0
    assert papers[0]["relevance_reason"] == "reason for On topic 0"
    assert papers[11]["relevance"] == 0.0


def test_failed_batch_leaves_those_papers_unscored_and_keeps_the_rest():
    papers = [make_paper(f"W{i}", relevance=None) for i in range(12)]
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("model down")
        return score_handler(prompt)

    relevance.score_relevance(papers, "antibodies", FakeLLM(flaky))
    assert all(p["relevance"] == 3.0 for p in papers[:10])
    assert all(p["relevance"] is None and p["relevance_reason"] is None for p in papers[10:])


def test_filter_keeps_papers_at_or_above_threshold(monkeypatch):
    monkeypatch.setattr(config, "MIN_KEEP", 2)
    papers = [make_paper(f"W{i}", relevance=r) for i, r in enumerate([3, 3, 2, 1, 0])]
    kept = relevance.filter_relevant(papers, min_relevance=2)
    assert [p["paperId"] for p in kept] == ["W0", "W1", "W2"]


def test_filter_falls_back_to_best_min_keep_when_too_strict(monkeypatch):
    monkeypatch.setattr(config, "MIN_KEEP", 3)
    papers = [make_paper(f"W{i}", relevance=r) for i, r in enumerate([1, 3, 0, 1])]
    kept = relevance.filter_relevant(papers, min_relevance=2)   # only one paper >= 2
    assert len(kept) == 3
    assert kept[0]["paperId"] == "W1"                            # the 3.0 comes first
    assert "W2" not in {p["paperId"] for p in kept}              # the 0.0 is the one left out


def test_unscored_papers_are_kept_at_the_default_threshold(monkeypatch):
    monkeypatch.setattr(config, "MIN_KEEP", 1)
    papers = [make_paper("W0", relevance=None), make_paper("W1", relevance=0)]
    kept = relevance.filter_relevant(papers, min_relevance=config.MIN_RELEVANCE)
    assert [p["paperId"] for p in kept] == ["W0"]


def test_query_stats_counts_per_query():
    a = make_paper("A", relevance=3, queries=("q1", "q2"))
    b = make_paper("B", relevance=1, queries=("q1",))
    c = make_paper("C", relevance=None, queries=("q2",))
    rows = {r["query"]: r for r in relevance.query_stats(["q1", "q2"], [a, b, c], [a, c], [a])}
    assert rows["q1"] == {"query": "q1", "found": 2, "relevant": 1, "final": 1, "avg_relevance": 2.0}
    assert rows["q2"]["found"] == 2 and rows["q2"]["avg_relevance"] == 3.0   # unscored paper excluded
