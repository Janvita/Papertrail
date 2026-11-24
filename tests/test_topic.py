from src import topic
from tests.fakes import FakeLLM


def test_clean_query_strips_quotes_operators_and_parentheses():
    assert topic.clean_query('("antibody affinity" AND prediction) OR NOT sequence') == "antibody affinity prediction sequence"


def test_clean_query_does_not_touch_words_containing_operators():
    assert topic.clean_query("ORIGINAL ANDROID NOTEBOOK") == "ORIGINAL ANDROID NOTEBOOK"


def test_expand_topic_cleans_every_query_the_llm_returns():
    llm = FakeLLM(lambda prompt: {"queries": ['"antibody affinity" AND sequence', "deep   learning antibodies"]})
    assert topic.expand_topic("antibodies", llm) == ["antibody affinity sequence", "deep learning antibodies"]
    assert "antibodies" in llm.prompts[0]
