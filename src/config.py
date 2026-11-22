"""Pipeline settings, loaded from configs/pipeline.yaml. Secrets are read from .env."""
from datetime import date
from pathlib import Path
import yaml
from dotenv import load_dotenv

load_dotenv()

with (Path(__file__).resolve().parents[1] / "configs" / "pipeline.yaml").open() as f:
    _cfg = yaml.safe_load(f)

RAW = _cfg   # full pipeline.yaml, saved in each run's manifest

CURRENT_YEAR = date.today().year

# Fields that openalex.normalize() reads. Change together with that function.
OA_SELECT = ("id,title,publication_year,abstract_inverted_index,"
             "cited_by_count,doi,referenced_works,type")

OA_BASE = _cfg["retrieval"]["openalex_base"]
PER_QUERY = _cfg["retrieval"]["per_query"]
ALLOWED_TYPES = set(_cfg["retrieval"]["allowed_types"])

RELEVANCE_BATCH = _cfg["relevance"]["batch_size"]
MIN_RELEVANCE = _cfg["relevance"]["min_relevance"]
MIN_KEEP = _cfg["relevance"]["min_keep"]

MAX_PAPERS = _cfg["ranking"]["max_papers"]
W_RELEVANCE = _cfg["ranking"]["weights"]["relevance"]
W_IMPACT = _cfg["ranking"]["weights"]["impact"]
W_RECENCY = _cfg["ranking"]["weights"]["recency"]

EXTRACT_BATCH = _cfg["extraction"]["batch_size"]

LLM_PAUSE = _cfg["llm"]["pause_seconds"]

CACHE = Path(".oa_cache")
LLM_CACHE = Path(".llm_cache")
CACHE.mkdir(exist_ok=True)
LLM_CACHE.mkdir(exist_ok=True)
