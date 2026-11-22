"""Step 1: turn a topic into search queries."""
import re


def clean_query(q):
    q = re.sub(r'\b(AND|OR|NOT)\b|["()]', " ", q)
    return re.sub(r"\s+", " ", q).strip()


def expand_topic(topic, llm):
    queries = llm.ask_json(
        f"I am starting a literature survey on: {topic}\n"
        "Give 4 distinct search queries that together cover foundational work, "
        "major method families, and recent advances. "
        "Every query must contain at least one term specific to this topic; never use "
        "a generic phrase on its own (for example 'graph neural network' or 'deep learning') "
        "because it returns papers from unrelated fields. "
        "Each query must be plain keywords, 3 to 8 words, with NO quotes and "
        "NO AND/OR operators. "
        'Format: {"queries": ["...", "..."]}'
    )["queries"]
    return [clean_query(q) for q in queries]
