"""Wires all steps together."""
import json, logging, sys, time
from datetime import datetime
from pathlib import Path
from . import config, llm_config
from .llm import get_llm
from .topic import expand_topic
from .retrieval import search
from .relevance import score_relevance, filter_relevant, query_stats
from .ranking import rank, citation_edges, reading_order
from .extraction import extract_all
from .pages import write_pages
from .graph import write_graph
from .utils import slug


def _make_logger(out):
    """Every run gets its own run.log in the output dir, plus the usual console output."""
    log = logging.getLogger("papertrail")   # shared by every module
    log.setLevel(logging.INFO)
    for h in list(log.handlers):
        h.close()
    log.handlers.clear()
    log.propagate = False

    fh = logging.FileHandler(out / "run.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    log.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(sh)
    return log


def run(topic, max_papers=None, min_relevance=None, do_extract=True, out_dir=None):
    max_papers = max_papers or config.MAX_PAPERS
    min_relevance = config.MIN_RELEVANCE if min_relevance is None else min_relevance

    out = Path(out_dir) if out_dir else Path("survey_" + slug(topic))
    out.mkdir(parents=True, exist_ok=True)
    log = _make_logger(out)
    started, t0 = datetime.now(), time.perf_counter()
    counts, seconds = {}, {}

    def stage(name):
        seconds[name] = round(time.perf_counter() - t0 - sum(seconds.values()), 1)

    main_llm, filter_llm, topic_llm = get_llm("main"), get_llm("filter"), get_llm("topic")
    log.info(f"Topic: {topic}")
    log.info(f"Main LLM: {main_llm} | Filter LLM: {filter_llm} | Topic LLM: {topic_llm}")

    log.info("Expanding topic into search queries...")
    queries = expand_topic(topic, topic_llm)
    log.info("Queries used:")
    for i, q in enumerate(queries, 1):
        log.info(f"  {i}. {q}")
    stage("expand_topic")

    log.info("Searching OpenAlex...")
    candidates = search(queries)
    if not candidates:
        sys.exit("No papers found. Try a different topic.")
    counts["candidates"] = len(candidates)
    log.info(f"Found {len(candidates)} candidate papers.")
    stage("search")

    log.info(f"Scoring relevance of {len(candidates)} candidates with {filter_llm}...")
    score_relevance(candidates, topic, filter_llm)
    log.info("Relevance scores (highest first):")
    for p in sorted(candidates, key=lambda p: -(p.get("relevance") or 0)):
        reason = p.get("relevance_reason") or "no reason returned"
        log.info(f"  [{p.get('relevance')}] {p['title']} ({p['year']}) -- {reason}")
    counts["unscored"] = sum(1 for p in candidates if p.get("relevance") is None)
    stage("score_relevance")

    relevant = filter_relevant(candidates, min_relevance)
    counts["relevant"] = len(relevant)

    papers = rank(relevant, max_papers)
    counts["final"] = len(papers)
    log.info(f"Final set: {len(papers)} papers, ranked:")
    for i, p in enumerate(papers, 1):
        log.info(f"  {i}. [score {p['score']:.3f}, relevance {p.get('relevance')}, "
                 f"citations {p['citationCount']}] {p['title']} ({p['year']})")

    log.info("Building citation links...")
    edges = citation_edges(papers)
    counts["citation_links"] = len(edges)
    log.info(f"{len(edges)} links inside the set.")
    stage("filter_rank_edges")

    info = None
    if do_extract:
        log.info(f"Extracting summaries with {main_llm}...")
        info = extract_all(papers, main_llm)
        counts["missing_summaries"] = sum(1 for v in info.values() if not v)
        if counts["missing_summaries"]:
            log.warning(f"{counts['missing_summaries']} papers have no summary. "
                        "Rerun to retry (cached calls are free).")
        stage("extract")

    order = reading_order(papers, edges)
    qstats = query_stats(queries, candidates, relevant, papers)
    log.info("Per-query ranking (how each query contributed):")
    for r in qstats:
        log.info(f"  query '{r['query']}': found {r['found']}, relevant {r['relevant']}, "
                 f"final {r['final']}, avg relevance {r['avg_relevance']}")

    write_pages(topic, papers, info, edges, order, out, qstats)
    write_graph(papers, edges, out, order)
    stage("write_output")

    # Debug file: see exactly what the filter dropped and why
    final_ids = {p["paperId"] for p in papers}
    debug = [{"title": p["title"], "year": p["year"], "relevance": p.get("relevance"),
              "relevance_reason": p.get("relevance_reason"),
              "in_final_set": p["paperId"] in final_ids, "queries": p["queries"]}
             for p in sorted(candidates, key=lambda p: -(p.get("relevance") or 0))]
    (out / "debug_candidates.json").write_text(json.dumps(debug, indent=2))

    manifest = {
        "topic": topic,
        "started": started.isoformat(timespec="seconds"),
        "total_seconds": round(time.perf_counter() - t0, 1),
        "models": {role: str(get_llm(role)) for role in llm_config.ROLES},
        "queries": queries,
        "counts": counts,
        "stage_seconds": seconds,
        "llm_usage": {role: get_llm(role).stats for role in llm_config.ROLES},
        "pipeline_config": config.RAW,
        "min_relevance_used": min_relevance,
        "max_papers_used": max_papers,
        "extraction": do_extract,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    log.info(f"Done in {manifest['total_seconds']}s. Open {out}/index.md, {out}/graph.html, "
             f"{out}/run.log and {out}/run_manifest.json")
