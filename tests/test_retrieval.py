import pytest

from src import openalex, retrieval
from tests.fakes import make_work


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(retrieval.time, "sleep", lambda s: None)


def fake_get(results_by_query):
    def get(path, params=None):
        return {"results": results_by_query.get(params["search"], [])}
    return get


def test_type_filter_and_incomplete_records_are_dropped(monkeypatch):
    results = {"q": [
        make_work("W1", "Keeper article"),
        make_work("W2", "Keeper preprint", type="preprint"),
        make_work("W3", "Keeper review", type="review"),
        make_work("W4", "A book", type="book"),
        make_work("W5", "A dataset", type="dataset"),
        make_work("W6", "No abstract", abstract=None),
        make_work("W7", None),
        make_work("W8", "No year", year=None),
    ]}
    monkeypatch.setattr(openalex, "get", fake_get(results))
    papers = retrieval.search(["q"])
    assert {p["paperId"] for p in papers} == {"W1", "W2", "W3"}


def test_paper_found_by_two_queries_is_kept_once_with_both_queries(monkeypatch):
    results = {"q1": [make_work("W1", "Shared paper")],
               "q2": [make_work("W1", "Shared paper"), make_work("W2", "Only in q2")]}
    monkeypatch.setattr(openalex, "get", fake_get(results))
    papers = {p["paperId"]: p for p in retrieval.search(["q1", "q2"])}
    assert set(papers) == {"W1", "W2"}
    assert papers["W1"]["queries"] == ["q1", "q2"]
    assert papers["W2"]["queries"] == ["q2"]


def test_preprint_and_published_version_merge_into_the_more_cited_one(monkeypatch):
    results = {"q1": [make_work("W1", "Language Models for Antibody Affinity", cites=5,
                                refs=["R1"], doi="https://doi.org/10.48550/arxiv.2301.00001")],
               "q2": [make_work("W2", "language models for antibody affinity!", cites=50,
                                refs=["R2"])]}
    monkeypatch.setattr(openalex, "get", fake_get(results))
    papers = retrieval.search(["q1", "q2"])
    assert len(papers) == 1
    p = papers[0]
    assert p["paperId"] == "W2"                         # higher citation count wins
    assert p["queries"] == ["q2", "q1"]                 # both queries remembered
    assert p["refs"] == ["R2", "R1"]                    # references merged
    assert p["externalIds"] == {"ArXiv": "2301.00001"}  # arXiv id borrowed from the preprint


def test_norm_title_ignores_case_and_punctuation():
    assert retrieval.norm_title("Deep-Learning: for Antibodies!") == retrieval.norm_title("deep learning for antibodies")
