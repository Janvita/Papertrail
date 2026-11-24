import json
import re

import pytest

from src import config, openalex, pipeline, retrieval
from src.pages import write_pages
from tests.fakes import FakeLLM, make_paper, make_work

QUERIES = ["antibody affinity", "language model antibodies", "empty query"]

WORKS = {
    "antibody affinity": [
        make_work("W1", "Antibody affinity model one", year=2022, cites=100, refs=["W2"]),
        make_work("W2", "Foundations of antibody affinity", year=2020, cites=500),
        make_work("W3", "Offtopic crystallography of rice", year=2021, cites=900),
        make_work("W4", "An antibody affinity handbook", type="book"),
        make_work("W5", "Antibody affinity model five", year=2023, cites=20, refs=["W1", "W2"]),
    ],
    "language model antibodies": [
        make_work("W1", "Antibody affinity model one", year=2022, cites=100, refs=["W2"]),
        make_work("W6", "Language models for antibody affinity", year=2023, cites=5,
                  doi="https://doi.org/10.48550/arxiv.2301.00001"),
        make_work("W7", "Language Models for Antibody Affinity", year=2024, cites=40),
        make_work("W8", "Abstract is missing", abstract=None),
    ],
}


def handler(prompt):
    if prompt.startswith("I am starting a literature survey"):
        return {"queries": QUERIES}
    if prompt.startswith("A literature survey is being written"):
        rows = re.findall(r"^\[(\d+)\] (.*) \(\d{4}\)$", prompt, flags=re.M)
        return {"scores": [{"id": int(i), "score": 0 if "Offtopic" in t else 3,
                            "reason": f"fake reason for {t}"} for i, t in rows]}
    n = int(re.search(r"Below are (\d+) papers", prompt).group(1))
    return {"results": [{"id": i, "core_idea": "an idea", "architecture": "a net",
                         "datasets": ["SAbDab"], "metrics": ["Spearman"],
                         "delta": "something new", "read_if": "you like antibodies"} for i in range(n)]}


@pytest.fixture
def fake_world(monkeypatch):
    monkeypatch.setattr(retrieval.time, "sleep", lambda s: None)
    monkeypatch.setattr(openalex, "get", lambda path, params=None: {"results": WORKS.get(params["search"], [])})
    monkeypatch.setattr(config, "MIN_KEEP", 3)
    llms = {role: FakeLLM(handler, f"fake/{role}") for role in ("main", "filter", "topic")}
    monkeypatch.setattr(pipeline, "get_llm", lambda role="main": llms[role])
    return llms


def test_full_run_writes_every_output_file(fake_world, tmp_path):
    out = tmp_path / "survey"
    pipeline.run("antibody affinity", max_papers=6, out_dir=out)

    for name in ("index.md", "graph.html", "run.log", "run_manifest.json", "debug_candidates.json"):
        assert (out / name).exists(), name
    assert len(list(out.glob("*.md"))) == 1 + 4          # index + one page per final paper

    index = (out / "index.md").read_text()
    assert "## Reading order" in index and "## Timeline" in index and "Datasets across papers" in index
    assert "Offtopic" not in index                      # filtered out

    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["counts"] == {"candidates": 5, "unscored": 0, "relevant": 4, "final": 4,
                                  "citation_links": 3, "missing_summaries": 0}
    assert manifest["models"] == {"main": "fake/main", "filter": "fake/filter", "topic": "fake/topic"}
    assert manifest["queries"] == QUERIES

    debug = {d["title"]: d for d in json.loads((out / "debug_candidates.json").read_text())}
    assert debug["Offtopic crystallography of rice"]["in_final_set"] is False
    assert debug["Offtopic crystallography of rice"]["relevance"] == 0.0
    assert debug["Foundations of antibody affinity"]["relevance_reason"].startswith("fake reason")
    assert "An antibody affinity handbook" not in debug  # book dropped at retrieval


def test_run_log_contains_the_trace_a_reader_needs(fake_world, tmp_path):
    out = tmp_path / "survey"
    pipeline.run("antibody affinity", max_papers=6, out_dir=out)
    log = (out / "run.log").read_text()
    for expected in ("Queries used:", "'antibody affinity': 5 returned, 4 usable",
                     "removed 1 duplicate titles", "relevance filter: kept 4 of 5",
                     "fake reason for Offtopic", "Per-query ranking"):
        assert expected in log, expected


def test_no_extract_skips_summaries_and_the_main_model(fake_world, tmp_path):
    out = tmp_path / "survey"
    pipeline.run("antibody affinity", max_papers=6, do_extract=False, out_dir=out)
    assert len(list(out.glob("*.md"))) == 1              # only index.md
    assert fake_world["main"].prompts == []
    assert json.loads((out / "run_manifest.json").read_text())["extraction"] is False


def test_each_role_only_gets_its_own_kind_of_prompt(fake_world, tmp_path):
    pipeline.run("antibody affinity", max_papers=6, out_dir=tmp_path / "survey")
    assert all(p.startswith("I am starting") for p in fake_world["topic"].prompts)
    assert all(p.startswith("A literature survey") for p in fake_world["filter"].prompts)
    assert all(p.startswith("Below are") for p in fake_world["main"].prompts)


def test_unscored_paper_is_flagged_in_the_index(tmp_path):
    scored, unscored = make_paper("A", year=2021), make_paper("B", year=2022, relevance=None)
    write_pages("topic", [scored, unscored], None, [], ["A", "B"], tmp_path)
    index = (tmp_path / "index.md").read_text()
    assert index.count("[unscored]") == 1
    assert "[unscored] " not in index.split("\n")[3]     # the scored paper's line is clean
