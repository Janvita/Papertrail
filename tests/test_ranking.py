from src import ranking
from tests.fakes import make_paper


def ids(papers):
    return [p["paperId"] for p in papers]


def test_rank_puts_more_relevant_paper_first_when_all_else_equal():
    low, high = make_paper("LOW", relevance=1), make_paper("HIGH", relevance=3)
    assert ids(ranking.rank([low, high])) == ["HIGH", "LOW"]


def test_rank_prefers_higher_citation_rate_when_relevance_ties():
    quiet, cited = make_paper("QUIET", cites=5), make_paper("CITED", cites=1000)
    assert ids(ranking.rank([quiet, cited])) == ["CITED", "QUIET"]


def test_rank_respects_limit_and_handles_empty():
    papers = [make_paper(f"W{i}", cites=i) for i in range(5)]
    assert len(ranking.rank(papers, limit=3)) == 3
    assert ranking.rank([]) == []


def test_rank_treats_unscored_paper_like_the_threshold():
    unscored = make_paper("U", relevance=None)
    ranking.rank([unscored])
    assert unscored["score"] > 0


def test_citation_edges_only_inside_the_set():
    a = make_paper("A", refs=["B", "OUTSIDE"])
    b = make_paper("B")
    assert ranking.citation_edges([a, b]) == [("A", "B")]


def test_reading_order_puts_cited_papers_first_regardless_of_input_order():
    a = make_paper("A", year=2021)
    b = make_paper("B", year=2022, refs=["A"])
    c = make_paper("C", year=2023, refs=["B"])
    papers = [c, b, a]
    assert ranking.reading_order(papers, ranking.citation_edges(papers)) == ["A", "B", "C"]


def test_reading_order_breaks_ties_by_year():
    x, y = make_paper("X", year=2020), make_paper("Y", year=2019)
    assert ranking.reading_order([x, y], []) == ["Y", "X"]


def test_reading_order_survives_a_citation_cycle():
    a = make_paper("A", year=2020, refs=["B"])
    b = make_paper("B", year=2020, refs=["A"])
    order = ranking.reading_order([a, b], ranking.citation_edges([a, b]))
    assert sorted(order) == ["A", "B"]


def test_reading_order_ignores_a_citation_to_a_newer_paper():
    a = make_paper("A", year=2020, refs=["B"])   # odd data: 2020 paper "cites" a 2022 paper
    b = make_paper("B", year=2022)
    assert ranking.reading_order([a, b], ranking.citation_edges([a, b])) == ["A", "B"]
