"""Field-level scoring of a predicted PaperCard against a hand-labeled golden card.

Week 1-2 metrics are deterministic (no LLM judge yet), so they're cheap and stable:
- datasets:  set F1 over normalized dataset names
- results:   recall of golden headline numbers (metric + value within tolerance)
- baselines: set F1 over normalized names
- code:      did we find the same repo as the golden card?
- evidence:  fraction of evidence quotes that actually appear in the paper text (hallucination check)
Week 7 adds an LLM-as-judge for free-text fields (problem, method, strengths, weaknesses).
"""

from __future__ import annotations

import re

from papertrail.schemas import PaperCard


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def set_f1(pred: list[str], gold: list[str]) -> float | None:
    p, g = {norm(x) for x in pred}, {norm(x) for x in gold}
    if not g and not p:
        return None  # nothing to score
    if not p or not g:
        return 0.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    precision, recall = tp / len(p), tp / len(g)
    return 2 * precision * recall / (precision + recall)


def results_recall(pred: PaperCard, gold: PaperCard, rel_tol: float = 0.01) -> float | None:
    golds = [r for r in gold.results if r.is_main_result] or gold.results
    if not golds:
        return None
    hits = 0
    for g in golds:
        for p in pred.results:
            if norm(p.metric) == norm(g.metric) and abs(p.value - g.value) <= rel_tol * max(abs(g.value), 1e-9):
                hits += 1
                break
    return hits / len(golds)


def code_match(pred: PaperCard, gold: PaperCard) -> float | None:
    g = gold.code.repo_url or gold.code.url_in_paper
    p = pred.code.repo_url or pred.code.url_in_paper
    if not g:
        return 1.0 if not p else 0.0  # correctly said "no code" vs. invented one
    return 1.0 if p and norm(p) == norm(g) else 0.0


def evidence_grounding(pred: PaperCard, paper_text: str) -> float | None:
    """Fraction of evidence quotes found (approximately) verbatim in the paper."""
    quotes = [pred.problem.evidence, pred.method_summary.evidence]
    quotes += [s.evidence for s in pred.strengths + pred.weaknesses]
    quotes = [q for q in quotes if q]
    if not quotes:
        return None
    hay = norm(paper_text)
    # use the middle 60 chars of the normalized quote to tolerate small edge differences
    found = sum(1 for q in quotes if (nq := norm(q)) and nq[len(nq) // 5 : len(nq) // 5 + 60] in hay)
    return found / len(quotes)


def score_card(pred: PaperCard, gold: PaperCard, paper_text: str | None = None) -> dict:
    scores = {
        "datasets_f1": set_f1([d.name for d in pred.datasets], [d.name for d in gold.datasets]),
        "results_recall": results_recall(pred, gold),
        "baselines_f1": set_f1(pred.baselines, gold.baselines),
        "code_match": code_match(pred, gold),
    }
    if paper_text is not None:
        scores["evidence_grounding"] = evidence_grounding(pred, paper_text)
    return scores
