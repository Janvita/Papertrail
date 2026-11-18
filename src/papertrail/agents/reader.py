"""Reader agent: paper markdown -> validated PaperCard.

Pattern: extract -> validate with Pydantic -> if invalid, feed the exact validation
errors back to the model and retry (a small Reflexion-style loop). Track how often
retries happen; that number is a metric you can improve and talk about.
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from papertrail.config import MAX_PAPER_CHARS
from papertrail.llm import complete, extract_json
from papertrail.schemas import PaperCard

SYSTEM = """You are a meticulous scientific reader. Extract a structured summary of the paper.

Rules:
- Use ONLY information stated in the paper. If something is not reported, leave it null or empty.
- For every `evidence` field, copy ONE sentence verbatim from the paper.
- `results`: numeric values only, include the metric name and dataset. Mark the headline number with is_main_result=true.
- `weaknesses`: include limitations the authors admit AND methodological issues you can see
  (small test set, random splits that may leak similar sequences, missing strong baselines).
- `builds_on`: titles of the 1-3 prior works the method most directly extends.
- Return ONLY a JSON object matching this JSON schema:
{schema}"""


def build_prompt(paper_md: str, meta: dict) -> str:
    header = f"paper_id: {meta['paper_id']}\n"
    if meta.get("title"):
        header += f"title (from metadata): {meta['title']}\n"
    if meta.get("year"):
        header += f"year (from metadata): {meta['year']}\n"
    return f"{header}\n--- PAPER START ---\n{paper_md[:MAX_PAPER_CHARS]}\n--- PAPER END ---"


def read_paper(paper_md: str, meta: dict, model: str) -> tuple[PaperCard | None, list[str]]:
    """Single extraction attempt. Returns (card or None, list of error messages)."""
    system = SYSTEM.format(schema=json.dumps(PaperCard.model_json_schema()))
    raw = complete(model, system, build_prompt(paper_md, meta))
    return _validate(raw)


def repair(previous_raw: str, errors: list[str], model: str) -> tuple[PaperCard | None, list[str]]:
    system = "Fix the JSON so it validates against the schema. Return ONLY the corrected JSON."
    user = f"Errors:\n{chr(10).join(errors)}\n\nJSON:\n{previous_raw}\n\nSchema:\n" + json.dumps(
        PaperCard.model_json_schema()
    )
    return _validate(complete(model, system, user))


def _validate(raw: str) -> tuple[PaperCard | None, list[str]]:
    try:
        data = extract_json(raw)
        return PaperCard.model_validate(data), []
    except json.JSONDecodeError as e:
        return None, [f"Invalid JSON: {e}", f"RAW::{raw}"]
    except ValidationError as e:
        return None, [f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()] + [
            f"RAW::{json.dumps(data)}"
        ]
