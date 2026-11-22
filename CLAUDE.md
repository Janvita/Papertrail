Paper Trial

Python tool that takes a research topic and produces a literature survey: ranked key papers, a recommended reading order, a hyperlinked index, one summary page per paper, and an interactive citation graph.

It is also the owner's portfolio project: a research-paper agent tied to their daily work, meant to be extended with new techniques and discussed in interviews. The long-term goal is a genuinely agentic system (LangGraph, planner / retriever / reader / critic loop), not just a script. Prefer clear, well-structured, explainable code over clever code.

Run
pdm install                     # dependencies live in pyproject.toml (pinned in pdm.lock); add with `pdm add <pkg>`
cp .env.example .env            # needs OPENALEX_API_KEY and GROQ_API_KEY
python run.py "<topic name>" [--max-papers N] [--min-relevance N] [--no-extract] [--out DIR]
pdm run pytest                  # no network, no real LLM

Output goes to survey_<topic-slug>/: index.md, one .md per paper, graph.html, debug_candidates.json, run.log (full trace incl. relevance reasons), run_manifest.json (models, config, counts, timings, LLM usage).

Pipeline (src/pipeline.py wires these in order)
topic.py expand topic -> 4 search queries (LLM)
retrieval.py OpenAlex search per query, keep article/preprint/review, dedupe by normalized title
relevance.py LLM scores each candidate 0-3 from title + abstract; drop below MIN_RELEVANCE
ranking.py rank = 0.50 relevance + 0.35 log(citations/year) + 0.15 recency; then citation edges (A, B) = A cites B, only inside the set; then reading order (topological, year tiebreak)
extraction.py structured summary per paper from the abstract (batched LLM calls)
pages.py index.md + per-paper pages (titles hyperlinked: arXiv > DOI > OpenAlex)
graph.py pyvis graph; node number = reading order, color = year, size = citations, click = link

Other modules: config.py (loads configs/pipeline.yaml), llm_config.py (loads configs/llm.yaml), llm.py (provider-agnostic client), openalex.py (cached API), utils.py (slug, paper_url, link), __main__.py (CLI), run.py at the repo root (calls it, so `python run.py` works; `python -m src` is equivalent).

Core data shape

A paper is a plain dict (see openalex.normalize): paperId (e.g. "W2741809807"), title, year, abstract, citationCount, doi, type, externalIds ({"ArXiv": id} when known), refs (list of paperIds it cites), queries (queries that found it). Added later: relevance (0-3 or None), score. Keep this dict flat and JSON-serializable; add new fields rather than changing existing ones.

LLM setup
Everything goes through llm.get_llm("main" | "filter" | "topic"): main = extraction, filter = relevance scoring, topic = query expansion. Never create an OpenAI/Groq client anywhere else.
Providers and the model per role are set in configs/llm.yaml; .env holds API keys only. The owner only has a free Groq key: daily token limits are tight, so keep prompts short, batch calls, truncate abstracts, and never add per-paper LLM calls without telling them.
All LLM and OpenAlex responses are cached on disk (.llm_cache/, .oa_cache/), keyed by the request. Reruns are free, so: do not delete caches, do not bypass them, and note that changing a prompt string invalidates that prompt's cache entries (that costs quota).
ask_json must return parsed JSON; models sometimes wrap it in fences or <think> tags (already handled).
Conventions
Python 3.10+, standard library plus requests, openai, python-dotenv, pyvis. Ask before adding a dependency.
One responsibility per module; keep functions small and typed where it helps. New pipeline steps get their own module and one line in pipeline.py.
All tunables live in configs/*.yaml (read via config.py / llm_config.py), not hardcoded in functions. Secrets only in .env.
Never commit .env or the cache folders. Add a .gitignore for .env, .oa_cache/, .llm_cache/, survey_*/.
Log through logging.getLogger("papertrail") (never print): pipeline.py attaches the console and run.log handlers, so every message lands in run.log.
Do not reproduce long passages of papers in output. Summaries and short quotes only.
Testing

Only the LLM-role wiring is tested so far (tests/test_llm_roles.py). Next: add the rest of tests/ (pytest) with no network and no real LLM: monkeypatch openalex.get and LLM.ask_json with fakes (fake works include duplicates, off-topic titles and non-article types). Cover: dedupe, type filter, relevance filter fallback (MIN_KEEP), ranking order, citation_edges, reading_order with a cycle, and an end-to-end run() writing index.md and graph.html. Run pytest before reporting any task as done.

Roadmap (do one item at a time, in this order)
Repo hygiene: git init, .gitignore, tests as above.
Snowballing (snowball.py): take the top papers, add their most-cited references (from refs) and most-cited citing papers (OpenAlex filter=cites:<id>; fetch works by id with filter=openalex:W1|W2, check the OpenAlex docs for current limits), then re-score relevance and re-rank. Goal: a denser citation graph and a meaningful reading order. Cap the added candidates so Groq quota is not blown.
Method-family clustering: LLM labels each paper with a family (e.g. GNN, language model, diffusion); color the graph by family and add a family table to index.md.
Code links (code_links.py): regex GitHub URLs from abstract/full text, then Hugging Face papers API by arXiv id; resolve missing arXiv ids by title via the arXiv API (reject loose matches, rate limit ~1 request / 3 s, cache).
Full-text pros/cons: fetch arxiv.org/html/<id> (fallback ar5iv), keep only relevant sections, extract pros / limitations / datasets / key results with a supporting quote for each claim. Abstract-only papers are marked "abstract only".
Agentic refactor (LangGraph): planner -> retriever (with snowballing) -> reader -> critic -> synthesizer. The critic checks coverage (missing method families or year ranges) and extraction faithfulness, and can send the graph back to retrieval. Keep the existing modules as the nodes' building blocks.
Evaluation: compare found papers against a published survey on the same topic (recall); spot-check 10 extractions. Store the numbers for the README.
Polish: README with architecture diagram and example output, optional Streamlit UI.
How to work in this repo
Before changing anything non-trivial, read the relevant module and propose a short plan first.
Work on one roadmap item per session; keep diffs small and reviewable; commit after each working step.
Do not change the output format (index.md layout, graph behavior) unless the task says so.
After code changes: run the tests, then a real run on a small topic with --max-papers 10 if the keys are set.
If something would spend meaningful Groq quota or needs a new API key, say so before doing it.
When unsure about product or API details (OpenAlex, Groq, arXiv, Hugging Face), check the official docs instead of guessing; they change.