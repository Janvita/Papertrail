"""Shared helpers for the eval scripts: config, survey candidates, reference titles, abstracts."""
import json
from pathlib import Path

import yaml

from src import config, openalex
from src.retrieval import norm_title

ROOT = Path(__file__).resolve().parents[1]


def load_eval_config():
    with (ROOT / "configs" / "evals.yaml").open() as f:
        return yaml.safe_load(f)


def load_candidates(survey_dir):
    """Candidates written by the pipeline's debug_candidates.json (title, year, relevance, in_final_set, ...)."""
    path = Path(survey_dir) / "debug_candidates.json"
    if not path.exists():
        raise SystemExit(f"no debug_candidates.json in {survey_dir}; run the pipeline first")
    return json.loads(path.read_text())


def reference_titles(review_id, batch=50):
    """Titles of every work the review cites. Two kinds of OpenAlex call, both cached."""
    work = openalex.get(f"/works/{review_id}", {"select": "id,referenced_works"})
    ids = [r.rsplit("/", 1)[-1] for r in work.get("referenced_works") or []]
    titles = []
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        data = openalex.get("/works", {"filter": "openalex:" + "|".join(chunk),
                                       "per-page": batch, "select": "id,title"})
        titles.extend(w.get("title") for w in data.get("results", []))
    return [t for t in titles if t]


def abstract_index():
    """normalized title -> abstract, read from the OpenAlex search cache (no network)."""
    index = {}
    for f in config.CACHE.glob("*.json"):
        try:
            data = json.loads(f.read_text())
        except (ValueError, OSError):
            continue
        for w in data.get("results", []) if isinstance(data, dict) else []:
            if w.get("title") and w.get("abstract_inverted_index"):
                index.setdefault(norm_title(w["title"]), openalex.abstract_from_index(w["abstract_inverted_index"]))
    return index
