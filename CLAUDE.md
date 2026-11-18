# CLAUDE.md: project context for PaperTrail

## What this project is
PaperTrail is a multi-agent **research copilot** for scientific papers. For a paper it extracts the problem, method, datasets, results, strengths, weaknesses, code location and the prior work it builds on. Later it compares papers fairly and traces how a research problem evolved over time.

- **Owner:** Janvita, AI Specialist (genomics ML + agentic systems).
- **Domain focus:** Antibody property prediction (public arXiv / bioRxiv / PubMed papers only).
- **Two goals, equally important:**
  1. A tool used daily to understand papers in the field.
  2. A portfolio project to discuss in ML/AI engineering interviews. Every design choice should be explainable, and every improvement should be **measured** (before/after numbers in `evals/results.csv`).

## Hard rules
- **Public papers only.** Never add code paths that send internal or confidential documents to external APIs.
- **Evals first.** Don't add a feature without a way to measure it. When changing prompts, models or retrieval, run `python evals/run_eval.py --tag <name>` and report the delta.
- **Don't invent facts in cards.** "Not reported" (null / empty) is better than a guess. Evidence fields must be verbatim sentences from the paper.
- **Use an LLM only where it's needed.** Prefer deterministic code (regex, APIs) when it is reliable (see `agents/code_hunter.py`).
- **The schema is the contract.** `src/papertrail/schemas.py` (`PaperCard`) is used both for agent output and for hand-labeled golden cards. If it changes, update `evals/golden/_template.json`, `evals/metrics.py` and the tests together.
- Never edit files in `evals/golden/` except `_template.json`. Those are hand labels.

## Architecture (current: weeks 1–2)
```
fetch -> parse (Docling) -> Reader -> valid? --no--> repair (<=2 retries)
                                        |yes
                                        v
                                  Code Hunter -> save card JSON (data/cards/)
```
- `src/papertrail/graph.py`: LangGraph `StateGraph` wiring the nodes above.
- `src/papertrail/agents/reader.py`: Extract → Pydantic validate → feed errors back and repair (Reflexion-style).
- `src/papertrail/agents/code_hunter.py`: Finds GitHub links in the text and verifies them via the GitHub API, with a title-search fallback.
- `src/papertrail/tools/`: `fetch.py` (arXiv id / PDF URL / local path), `pdf_parse.py` (Docling → Markdown, cached), `github_lookup.py`.
- `src/papertrail/llm.py`: The only place LLM calls happen (LiteLLM). It logs tokens and cost. Tracing (Langfuse) will plug in here.
- `src/papertrail/config.py`: Models and limits, overridable via `.env`.
- `evals/metrics.py`: Deterministic scores (datasets_f1, results_recall, baselines_f1, code_match, evidence_grounding).
- `evals/run_eval.py`: Runs all golden papers and appends a summary row to `evals/results.csv`.

## Tech stack
Python 3.11+, LangGraph, Pydantic v2, LiteLLM, Docling, arxiv, httpx, PyGithub, pytest.
Planned: sentence-transformers, Qdrant or pgvector, cross-encoder reranker, Semantic Scholar API, OpenAlex (`pyalex`), MCP Python SDK (FastMCP), NetworkX → Neo4j, Langfuse, DeepEval/Ragas, FastAPI, Streamlit, Docker.

## Commands
```bash
pip install -e ".[dev]"
pytest                                    # fast, no API calls
python -m papertrail.cli <arxiv_id|pdf_url|path> [--model ...]
python evals/run_eval.py --tag <experiment_name> [--model ...]
```

## Roadmap (about 12 weeks at 8–10 hrs/week)
| Weeks | Milestone |
|---|---|
| 1–2 | **(current)** Breakdown card, Code Hunter, 10 golden cards, baseline results |
| 3–4 | Card library + hybrid search + reranker; "ask anything" Q&A with citations |
| 5–6 | Section-aware context selection instead of naive truncation (measure it) |
| 7–8 | Critic agent (checks evidence against text), LLM-as-judge for free-text fields, comparison tables that flag unfair comparisons (different datasets/splits/metrics) |
| 9–10 | Citation tools (Semantic Scholar / OpenAlex) as MCP servers; evolution timeline / citation graph |
| 11–12 | Daily watch for new papers, model routing (cheap vs. strong), Langfuse tracing, Docker, Streamlit UI, **PaperQA2 baseline comparison** |

## What makes it different from existing tools (Elicit, Consensus, PaperQA2, Asta)
1. Code verification (does the repo exist, is it usable, does it match the paper).
2. Unfair-comparison detection across papers.
3. Personalized to one niche, with memory of what's been read.
4. Evolution timeline and critical comparison in one tool.

## How to help in this repo
- Explain *why* when making design choices. The owner wants to understand and discuss them in interviews, not just get working code.
- Keep changes small and focused. Suggest the matching eval/test with each change.
- After meaningful changes, suggest a one-line entry for the README "Results" or a learnings note in `notes/`.
- Update the "(current)" marker in the roadmap above when a phase is finished.
