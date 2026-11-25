"""Retrieval recall: how much of the reference review's bibliography our search found.

Usage: python -m evals.recall <survey_dir>
Matching is by normalized title (debug_candidates.json has no OpenAlex IDs).
"""
import sys

from src.retrieval import norm_title
from evals import metrics
from evals.common import load_candidates, load_eval_config, reference_titles


def compute(survey_dir, cfg=None):
    cfg = cfg or load_eval_config()
    keywords = [k.lower() for k in cfg["topic_keywords"]]

    refs = {norm_title(t) for t in reference_titles(cfg["reference_review"]["openalex_id"])}
    on_topic = {r for r in refs if any(k in r for k in keywords)}

    candidates = load_candidates(survey_dir)
    pool = {norm_title(c["title"]) for c in candidates}
    final = {norm_title(c["title"]) for c in candidates if c["in_final_set"]}
    final_titles = sorted(c["title"] for c in candidates if c["in_final_set"] and norm_title(c["title"]) in refs)

    result = {
        "reference_count": len(refs),
        "on_topic_reference_count": len(on_topic),
        "candidate_count": len(candidates),
        "final_count": len(final),
        "pool_recall": metrics.recall(pool, refs),
        "final_recall": metrics.recall(final, refs),
        "pool_recall_on_topic": metrics.recall(pool, on_topic) if on_topic else None,
        "final_recall_on_topic": metrics.recall(final, on_topic) if on_topic else None,
        "matched_in_final": final_titles,
    }
    return result


def main(argv):
    if len(argv) != 1:
        raise SystemExit("usage: python -m evals.recall <survey_dir>")
    r = compute(argv[0])
    print(f"Reference review: {load_eval_config()['reference_review']['openalex_id']}")
    print(f"References with a title: {r['reference_count']} ({r['on_topic_reference_count']} look on-topic)")
    print(f"Candidates in run: {r['candidate_count']}, final set: {r['final_count']}")
    print(f"Pool recall:          {r['pool_recall']:.1%}")
    print(f"Final recall:         {r['final_recall']:.1%}")
    if r["pool_recall_on_topic"] is not None:
        print(f"Pool recall (on-topic refs):  {r['pool_recall_on_topic']:.1%}")
        print(f"Final recall (on-topic refs): {r['final_recall_on_topic']:.1%}")
    print("Final-set papers that are in the review's reference list:")
    for t in r["matched_in_final"] or ["(none)"]:
        print(f"  - {t}")
    print("Recall is a lower bound: the review cites subtopics this search never targeted.")


if __name__ == "__main__":
    main(sys.argv[1:])
