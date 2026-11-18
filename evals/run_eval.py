"""Run the pipeline on every golden paper and print a results table.

Usage:
    python evals/run_eval.py                         # default model
    python evals/run_eval.py --model anthropic/claude-haiku-5-5 --tag cheap
Results are appended to evals/results.csv so you can compare runs over time.
This CSV is your interview story: every change -> a new row -> a measured delta.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from metrics import score_card  # noqa: E402

from papertrail.config import GOLDEN_DIR, READER_MODEL  # noqa: E402
from papertrail.graph import build_graph  # noqa: E402
from papertrail.llm import USAGE_LOG  # noqa: E402
from papertrail.schemas import PaperCard  # noqa: E402

RESULTS_CSV = Path(__file__).parent / "results.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=READER_MODEL)
    ap.add_argument("--tag", default="baseline", help="Name this experiment, e.g. 'reranker' or 'cheap-model'")
    args = ap.parse_args()

    graph = build_graph()
    golden_files = sorted(p for p in GOLDEN_DIR.glob("*.json") if not p.name.startswith("_"))
    if not golden_files:
        sys.exit(f"No golden cards in {GOLDEN_DIR}. Copy _template.json and fill it in by hand first.")

    rows, failures = [], 0
    for path in golden_files:
        gold = PaperCard.model_validate_json(path.read_text())
        source = json.loads(path.read_text()).get("_source", gold.paper_id)
        t0 = time.time()
        final = graph.invoke({"source": source, "model": args.model})
        latency = time.time() - t0
        if not final.get("card"):
            failures += 1
            print(f"FAIL  {gold.paper_id}")
            continue
        s = score_card(final["card"], gold, final.get("paper_md"))
        s.update(paper_id=gold.paper_id, retries=final.get("retries", 0), latency_s=round(latency, 1))
        rows.append(s)
        print(f"ok    {gold.paper_id}  {s}")

    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(statistics.mean(vals), 3) if vals else None

    total_cost = sum(u["cost_usd"] or 0 for u in USAGE_LOG)
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag,
        "model": args.model,
        "n_papers": len(golden_files),
        "failures": failures,
        "datasets_f1": mean("datasets_f1"),
        "results_recall": mean("results_recall"),
        "baselines_f1": mean("baselines_f1"),
        "code_match": mean("code_match"),
        "evidence_grounding": mean("evidence_grounding"),
        "avg_retries": mean("retries"),
        "avg_latency_s": mean("latency_s"),
        "cost_per_paper_usd": round(total_cost / max(len(golden_files), 1), 4),
    }
    print("\nSUMMARY", json.dumps(summary, indent=2))

    new = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary))
        if new:
            w.writeheader()
        w.writerow(summary)


if __name__ == "__main__":
    main()
