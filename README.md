# PaperTrail

Give it a research topic; it returns a literature survey: ranked key papers, a reading order, one summary page per paper, and an interactive citation graph.

## Quickstart

```bash
pdm install                   # needs PDM (brew install pdm); reads pyproject.toml / pdm.lock
cp .env.example .env          # add GROQ_API_KEY and OPENALEX_API_KEY
python run.py "antibody binding affinity prediction from sequence" --max-papers 10   # or: pdm run python run.py ...
```

Flags: `--max-papers N`, `--min-relevance N`, `--no-extract` (skip summaries, cheaper), `--out DIR`.

## How it works

```
topic -> 4 search queries (small LLM)
      -> OpenAlex search, dedupe
      -> relevance score 0-3 with a reason (small LLM), drop low scores
      -> rank (relevance + citations/year + recency), citation links, reading order
      -> per-paper summaries from abstracts (main LLM)
      -> index.md, one page per paper, graph.html
```

## Configuration

| File | What it controls |
|---|---|
| `.env` | API keys only |
| `configs/llm.yaml` | providers, and which model does topic expansion / relevance scoring / extraction |
| `configs/pipeline.yaml` | batch sizes, relevance threshold, ranking weights, paper counts |

## Output (`survey_<topic>/`)

- `index.md`, `<paper>.md`, `graph.html`: the survey
- `run.log`: timestamped trace of every step, including each paper's relevance score and reason
- `run_manifest.json`: models, config, counts per stage, timings and LLM usage for the run
- `debug_candidates.json`: every candidate with its score, reason and whether it made the final set

LLM and OpenAlex responses are cached on disk (`.llm_cache/`, `.oa_cache/`), so reruns are free. Changing a prompt invalidates its cache entries.

## Tests

```bash
pdm run pytest        # no network, no API calls
```
