"""Filter quality: re-score hand-labeled papers with each candidate model and compare.

Usage:
  python -m evals.filter_eval evals/labels/candidates.csv                 # prints the cost estimate only
  python -m evals.filter_eval evals/labels/candidates.csv --confirm-quota # calls Groq

Without --confirm-quota this makes no LLM call.
"""
import csv
import math
import sys

from src import config
from src.llm import LLM
from src import relevance
from evals import metrics
from evals.common import load_eval_config


def read_labels(path):
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("label") or "").strip() != ""]
    return rows


def to_papers(rows):
    return [{"paperId": r["id"], "title": r["title"], "year": r["year"],
             "abstract": r["abstract"] or "", "relevance": None, "relevance_reason": None}
            for r in rows]


def estimate(n_papers, n_models):
    batches = math.ceil(n_papers / config.RELEVANCE_BATCH)
    return batches, batches * n_models


def evaluate(rows, cfg):
    """Returns a list of result rows: metric, model, value, n. Calls Groq once per batch per model."""
    threshold = cfg["filter_eval"]["threshold"]
    labels = [int(float(r["label"])) for r in rows]
    results = []
    for model in cfg["filter_eval"]["models"]:
        papers = to_papers(rows)
        relevance.score_relevance(papers, cfg["topic"], LLM("groq", model))
        scored = [(p["relevance"], lab) for p, lab in zip(papers, labels) if p["relevance"] is not None]
        if not scored:
            results.append({"metric": "scored", "model": model, "value": 0, "n": len(rows)})
            continue
        s, l = [x for x, _ in scored], [y for _, y in scored]
        results += [
            {"metric": "spearman", "model": model, "value": metrics.spearman(s, l), "n": len(s)},
            {"metric": f"agreement@{threshold}", "model": model,
             "value": metrics.agreement(s, l, threshold), "n": len(s)},
            {"metric": "saturation", "model": model, "value": metrics.saturation(s), "n": len(s)},
            {"metric": "unscored", "model": model, "value": len(rows) - len(s), "n": len(rows)},
        ]
    return results


def main(argv):
    if not argv:
        raise SystemExit("usage: python -m evals.filter_eval <labels.csv> [--confirm-quota]")
    cfg = load_eval_config()
    rows = read_labels(argv[0])
    if not rows:
        raise SystemExit(f"no labeled rows in {argv[0]}; fill the 'label' column first")
    n_models = len(cfg["filter_eval"]["models"])
    batches, calls = estimate(len(rows), n_models)
    print(f"{len(rows)} labeled papers, {n_models} models, {batches} batches per model, {calls} Groq calls total")
    if "--confirm-quota" not in argv:
        print("Not calling Groq. Re-run with --confirm-quota to run the comparison.")
        return
    for r in evaluate(rows, cfg):
        print(f"{r['model']:<22} {r['metric']:<16} {r['value']!s:<22} n={r['n']}")


if __name__ == "__main__":
    main(sys.argv[1:])
