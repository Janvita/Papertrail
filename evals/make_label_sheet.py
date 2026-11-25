"""Writes a blind labeling sheet: title, year and abstract only. No model scores.

Usage: python -m evals.make_label_sheet <survey_dir>
Then fill the 'label' column (0-3) and save as evals/labels/candidates.csv.
"""
import csv
import random
import sys
from pathlib import Path

from src.retrieval import norm_title
from evals.common import ROOT, abstract_index, load_candidates, load_eval_config

FIELDS = ["id", "title", "year", "abstract", "label"]


def build_rows(candidates, cfg):
    lab = cfg["labeling"]
    rng = random.Random(lab["seed"])
    sample = rng.sample(candidates, min(lab["sample_size"], len(candidates)))
    abstracts = abstract_index()
    rows = []
    for i, c in enumerate(sample):
        abstract = abstracts.get(norm_title(c["title"])) or ""
        rows.append({"id": i, "title": c["title"], "year": c["year"],
                     "abstract": abstract[:lab["abstract_chars"]], "label": ""})
    return rows


def main(argv):
    if len(argv) != 1:
        raise SystemExit("usage: python -m evals.make_label_sheet <survey_dir>")
    cfg = load_eval_config()
    rows = build_rows(load_candidates(argv[0]), cfg)
    out = ROOT / "evals" / "labels" / "candidates.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    missing = sum(1 for r in rows if not r["abstract"])
    print(f"Wrote {len(rows)} papers to {out}")
    print(f"{missing} have no abstract in the OpenAlex cache (title and year only)")
    print("Fill the 'label' column with 0-3 (0 unrelated, 1 tangential, 2 closely related, 3 directly on topic).")


if __name__ == "__main__":
    main(sys.argv[1:])
