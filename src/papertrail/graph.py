"""LangGraph pipeline for weeks 1-2:

    fetch -> parse -> read -> (validate ok?) --no--> repair (max N) --+
                                  |yes                                |
                                  v                                   |
                             code_hunter -> save  <-------------------+
"""

from __future__ import annotations

from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from papertrail.agents.code_hunter import hunt_code
from papertrail.agents.reader import read_paper, repair
from papertrail.config import CARD_DIR, MAX_EXTRACTION_RETRIES, READER_MODEL
from papertrail.schemas import PaperCard
from papertrail.tools.fetch import fetch_paper
from papertrail.tools.pdf_parse import pdf_to_markdown


class State(TypedDict, total=False):
    source: str
    model: str
    meta: dict
    paper_md: str
    card: Optional[PaperCard]
    errors: list[str]
    retries: int


def fetch_node(state: State) -> State:
    return {"meta": fetch_paper(state["source"]), "retries": 0}


def parse_node(state: State) -> State:
    return {"paper_md": pdf_to_markdown(state["meta"]["pdf_path"])}


def read_node(state: State) -> State:
    card, errors = read_paper(state["paper_md"], state["meta"], state.get("model", READER_MODEL))
    return {"card": card, "errors": errors}


def repair_node(state: State) -> State:
    raw = next((e[5:] for e in state["errors"] if e.startswith("RAW::")), "")
    msgs = [e for e in state["errors"] if not e.startswith("RAW::")]
    card, errors = repair(raw, msgs, state.get("model", READER_MODEL))
    return {"card": card, "errors": errors, "retries": state["retries"] + 1}


def route_after_read(state: State) -> str:
    if state.get("card") is not None:
        return "code_hunter"
    if state["retries"] < MAX_EXTRACTION_RETRIES:
        return "repair"
    return END  # give up; errors stay in state for logging


def code_node(state: State) -> State:
    card = state["card"]
    card.code = hunt_code(card, state["paper_md"])
    return {"card": card}


def save_node(state: State) -> State:
    CARD_DIR.mkdir(parents=True, exist_ok=True)
    card = state["card"]
    (CARD_DIR / f"{card.paper_id}.json").write_text(card.model_dump_json(indent=2))
    return {}


def build_graph():
    g = StateGraph(State)
    g.add_node("fetch", fetch_node)
    g.add_node("parse", parse_node)
    g.add_node("read", read_node)
    g.add_node("repair", repair_node)
    g.add_node("code_hunter", code_node)
    g.add_node("save", save_node)

    g.add_edge(START, "fetch")
    g.add_edge("fetch", "parse")
    g.add_edge("parse", "read")
    g.add_conditional_edges("read", route_after_read, ["code_hunter", "repair", END])
    g.add_conditional_edges("repair", route_after_read, ["code_hunter", "repair", END])
    g.add_edge("code_hunter", "save")
    g.add_edge("save", END)
    return g.compile()
