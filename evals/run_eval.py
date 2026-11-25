"""Entry point: run the checks and append rows to evals/results.csv (long format).

Usage:
  python -m evals.run_eval --tag baseline <survey_dir>                       # recall only, no LLM
  python -m evals.run_eval --tag baseline <survey_dir> --labels evals/labels/candidates.csv --confirm-quota
"""
import argparse
import csv
import datetime
from pathlib import Path

from evals import filter_eval, recall
from evals.common import ROOT, load_eval_config

RESULTS = ROOT / "evals" / "results.csv"
FIELDS = ["date", "tag", "metric", "model", "value", "n"]

RECALL_METRICS = ["pool_recall", "final_recall", "pool_recall_on_topic", "final_recall_on_topic",
                  "reference_count", "candidate_count", "final_count"]


def append_rows(tag, rows):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    new = not RESULTS.exists()
    date = datetime.date.today().isoformat()
    with RESULTS.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({"date": date, "tag": tag, **r})
    return len(rows)


def main():
    p = argparse.ArgumentParser(description="Run PaperTrail evaluations")
    p.add_argument("survey_dir")
    p.add_argument("--tag", required=True, help="short name for this run, e.g. baseline")
    p.add_argument("--labels", help="labeled CSV from make_label_sheet (enables the filter check)")
    p.add_argument("--confirm-quota", action="store_true", help="allow the Groq calls for the filter check")
    args = p.parse_args()

    cfg = load_eval_config()
    result = recall.compute(args.survey_dir, cfg)
    rows = []
    for name in RECALL_METRICS:
        if result[name] is None:
            continue
        is_rate = name.endswith("recall") or name.endswith("recall_on_topic")
        n = result["reference_count"] if is_rate else ""   # counts have no denominator
        rows.append({"metric": name, "model": "", "value": result[name], "n": n})
    print(f"Recall: pool {result['pool_recall']:.1%}, final {result['final_recall']:.1%}")

    if args.labels:
        if not args.confirm_quota:
            print("Filter check skipped: pass --confirm-quota to call Groq.")
        else:
            labeled = filter_eval.read_labels(args.labels)
            rows += filter_eval.evaluate(labeled, cfg)

    n = append_rows(args.tag, rows)
    print(f"Appended {n} rows to {RESULTS}")


if __name__ == "__main__":
    main()
