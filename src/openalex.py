"""OpenAlex access: cached GET, abstract rebuild, record normalization."""
import hashlib, json, os, random, re, sys, time
import requests
from . import config


def _headers():
    key = os.environ.get("OPENALEX_API_KEY")
    if not key:
        sys.exit("Set OPENALEX_API_KEY in your .env file")
    return {"Authorization": f"Bearer {key}"}


def get(path, params=None):
    """GET with disk cache, Retry-After handling and jittered backoff."""
    key = hashlib.md5(json.dumps([path, params], sort_keys=True).encode()).hexdigest()
    f = config.CACHE / f"{key}.json"
    if f.exists():  # reruns cost nothing
        return json.loads(f.read_text())

    for attempt in range(6):
        r = requests.get(f"{config.OA_BASE}{path}", params=params, headers=_headers(), timeout=30)
        if r.status_code == 429:
            try:
                wait = float(r.headers.get("Retry-After"))
            except (TypeError, ValueError):
                wait = 10 * (attempt + 1)
            wait += random.uniform(0, 2)
            print(f"  rate limited, waiting {wait:.0f}s...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        data = r.json()
        f.write_text(json.dumps(data))
        return data
    raise RuntimeError(f"OpenAlex kept rate limiting: {path}")


def abstract_from_index(inv):
    """OpenAlex stores abstracts as {word: [positions]}. Rebuild the text."""
    if not inv:
        return None
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def normalize(w):
    """Convert an OpenAlex work into the dict shape the rest of the code uses."""
    doi = w.get("doi") or ""
    m = re.search(r"10\.48550/arxiv\.(.+)$", doi, re.I)
    return {
        "paperId": w["id"].rsplit("/", 1)[-1],
        "title": w.get("title"),
        "year": w.get("publication_year"),
        "abstract": abstract_from_index(w.get("abstract_inverted_index")),
        "citationCount": w.get("cited_by_count") or 0,
        "doi": doi,
        "type": w.get("type"),
        "externalIds": {"ArXiv": m.group(1)} if m else {},
        "refs": [r.rsplit("/", 1)[-1] for r in (w.get("referenced_works") or [])],
        "queries": [],   # which search queries found this paper
    }
