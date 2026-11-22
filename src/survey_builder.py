import os, sys, json, time, re, hashlib, random, textwrap
from urllib.parse import quote
from pathlib import Path
import requests
from openai import OpenAI
from dotenv import load_dotenv
from pyvis.network import Network

load_dotenv()

# ---------- Config ----------
MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
GROQ_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_KEY:
    sys.exit("Set GROQ_API_KEY in your .env file")
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_KEY)

OA = "https://api.openalex.org"
OA_KEY = os.environ.get("OPENALEX_API_KEY")
if not OA_KEY:
    sys.exit("Set OPENALEX_API_KEY in your .env file")
OA_HEADERS = {"Authorization": f"Bearer {OA_KEY}"}
OA_SELECT = ("id,title,publication_year,abstract_inverted_index,"
             "cited_by_count,doi,referenced_works")

MAX_PAPERS = 25
PER_QUERY = 30
BATCH = 5             # papers summarized per LLM call; lower to 3 if you see JSON errors
USE_JSON_MODE = True  # asks Groq to return valid JSON; set False if you get a 400 error
EXTRACT = True        # True: per-paper summary pages + datasets list (uses LLM calls)
                      # False: only index.md (hyperlinked titles) + graph.html, no extraction

CACHE = Path(".oa_cache")
CACHE.mkdir(exist_ok=True)
LLM_CACHE = Path(".llm_cache")
LLM_CACHE.mkdir(exist_ok=True)


# ---------- LLM helpers (Groq) ----------
def ask(prompt):
    key = hashlib.md5((MODEL + prompt).encode()).hexdigest()
    f = LLM_CACHE / f"{key}.txt"
    if f.exists():  # reruns cost nothing
        return f.read_text()

    kwargs = {}
    if USE_JSON_MODE:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(5):
        try:
            time.sleep(2)
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                **kwargs,
            )
            text = resp.choices[0].message.content
            f.write_text(text)
            return text
        except Exception as ex:
            msg = str(ex)
            low = msg.lower()
            if "per day" in low or "rpd" in low or "tpd" in low:
                sys.exit("Daily quota used up. Rerun later (finished calls are cached).")
            if "429" in msg or "503" in msg or "rate" in low:
                wait = 15 * (attempt + 1)
                print(f"  LLM busy or rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("LLM unavailable after retries")


def ask_json(prompt):
    text = ask(prompt + "\n\nReply with a single JSON object only, no prose, no code fences.")
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    return json.loads(text)


# ---------- OpenAlex helpers ----------
def oa_get(path, params=None):
    """GET with disk cache, Retry-After handling and jittered backoff."""
    key = hashlib.md5(json.dumps([path, params], sort_keys=True).encode()).hexdigest()
    f = CACHE / f"{key}.json"
    if f.exists():  # reruns cost nothing
        return json.loads(f.read_text())

    for attempt in range(6):
        r = requests.get(f"{OA}{path}", params=params, headers=OA_HEADERS, timeout=30)
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
    """Convert an OpenAlex work into the shape the rest of the script uses."""
    doi = w.get("doi") or ""
    m = re.search(r"10\.48550/arxiv\.(.+)$", doi, re.I)
    return {
        "paperId": w["id"].rsplit("/", 1)[-1],  # e.g. W2741809807
        "title": w.get("title"),
        "year": w.get("publication_year"),
        "abstract": abstract_from_index(w.get("abstract_inverted_index")),
        "citationCount": w.get("cited_by_count") or 0,
        "doi": doi,
        "externalIds": {"ArXiv": m.group(1)} if m else {},
        "refs": [r.rsplit("/", 1)[-1] for r in (w.get("referenced_works") or [])],
    }


# ---------- 1. Topic expansion ----------
def clean_query(q):
    q = re.sub(r'\b(AND|OR|NOT)\b|["()]', " ", q)
    return re.sub(r"\s+", " ", q).strip()


def expand_topic(topic):
    queries = ask_json(
        f"I am starting a literature survey on: {topic}\n"
        "Give 4 distinct search queries that together cover foundational work, "
        "major method families, and recent advances. "
        "Each query must be plain keywords, 3 to 8 words, with NO quotes and "
        "NO AND/OR operators. "
        'Format: {"queries": ["...", "..."]}'
    )["queries"]
    return [clean_query(q) for q in queries]


# ---------- 2. Retrieval ----------
def search(queries, per_query=PER_QUERY):
    papers = {}
    for q in queries:
        data = oa_get("/works", {
            "search": q,
            "filter": "has_abstract:true",
            "per_page": per_query,
            "select": OA_SELECT,
        })
        for w in data.get("results", []):
            p = normalize(w)
            if p["title"] and p["abstract"] and p["year"]:
                papers[p["paperId"]] = p
        time.sleep(0.5)
    return list(papers.values())


# ---------- 3. Ranking: citations per year, so recent work is not buried ----------
def rank(papers, now=2026):
    def score(p):
        age = max(now - p["year"], 1)
        return (p["citationCount"] or 0) / age
    return sorted(papers, key=score, reverse=True)[:MAX_PAPERS]


# ---------- 4. Citation links, kept only between papers in the set ----------
def citation_edges(papers):
    idset = {p["paperId"] for p in papers}
    return [(p["paperId"], r) for p in papers for r in p["refs"] if r in idset]


# ---------- 5. Extraction from abstracts (batched) ----------
SCHEMA_HINT = (
    '{"id": 0, "core_idea": "one sentence", "architecture": "...", '
    '"datasets": ["..."], "metrics": ["..."], '
    '"delta": "what is new versus earlier work", '
    '"read_if": "who should read this and why"}'
)


def extract_batch(batch):
    items = "\n\n".join(
        f"[{i}] Title: {p['title']}\nYear: {p['year']}\nAbstract: {p['abstract']}"
        for i, p in enumerate(batch)
    )
    data = ask_json(
        f"Below are {len(batch)} papers, numbered.\n\n{items}\n\n"
        "For EACH paper, fill this schema using ONLY its abstract. "
        "Use null when the abstract does not say. Keep the same id as the paper number.\n"
        f'Format: {{"results": [{SCHEMA_HINT}]}}'
    )
    out = {}
    for r in data.get("results", []):
        try:
            out[int(r["id"])] = r
        except (KeyError, ValueError, TypeError):
            continue
    return out


def extract_all(papers):
    info = {}
    for i in range(0, len(papers), BATCH):
        batch = papers[i:i + BATCH]
        try:
            res = extract_batch(batch)
        except Exception as ex:
            print(f"  batch starting at {i} failed ({ex}); trying papers one by one...")
            res = {}
            for j, p in enumerate(batch):
                try:
                    res[j] = extract_batch([p])[0]
                except Exception as ex2:
                    print("  extract failed:", p["title"], ex2)
        for j, p in enumerate(batch):
            info[p["paperId"]] = res.get(j, {})
    return info


# ---------- 6. Reading order: papers cited by others come first, year breaks ties ----------
def reading_order(papers, edges):
    by_id = {p["paperId"]: p for p in papers}
    cites = {pid: set() for pid in by_id}  # pid -> papers it builds on
    for citing, cited in edges:
        if by_id[cited]["year"] <= by_id[citing]["year"]:
            cites[citing].add(cited)
    order, done = [], set()
    while len(done) < len(by_id):
        ready = [pid for pid in by_id if pid not in done and cites[pid] <= done]
        if not ready:  # cycle or odd data: fall back to year
            ready = [min((x for x in by_id if x not in done), key=lambda x: by_id[x]["year"])]
        ready.sort(key=lambda x: by_id[x]["year"])
        order.append(ready[0])
        done.add(ready[0])
    return order


# ---------- Output helpers ----------
def slug(title):
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60]


def paper_url(p):
    """Best link for a paper: arXiv, then DOI, then the OpenAlex page."""
    arxiv = (p.get("externalIds") or {}).get("ArXiv")
    if arxiv:
        return f"https://arxiv.org/abs/{arxiv}"
    return p.get("doi") or f"https://openalex.org/{p['paperId']}"


def link(p):
    """Markdown hyperlink: [title](url)."""
    return f"[{p['title']}]({paper_url(p)})"


def write_pages(topic, papers, info, edges, order, out):
    out.mkdir(parents=True, exist_ok=True)
    by_id = {p["paperId"]: p for p in papers}

    # Per-paper summary pages (only when extraction is on)
    if info is not None:
        for pid in order:
            p, e = by_id[pid], info.get(pid, {})
            builds_on = [link(by_id[c]) for a, c in edges if a == pid]
            lines = [
                f"# {link(p)}",
                f"{p['year']} | {p['citationCount']} citations", "",
                f"**Core idea:** {e.get('core_idea')}", "",
                f"**Architecture:** {e.get('architecture')}", "",
                f"**Datasets:** {', '.join(e.get('datasets') or []) or 'not stated'}",
                f"**Metrics:** {', '.join(e.get('metrics') or []) or 'not stated'}", "",
                f"**What is new:** {e.get('delta')}", "",
                f"**Read if:** {e.get('read_if')}", "",
                "**Builds on (within this set):** " + ("; ".join(builds_on) or "none found"),
            ]
            (out / f"{slug(p['title'])}.md").write_text("\n".join(lines))

    # Index page
    idx = [f"# Survey: {topic}", "",
           "[Open interactive citation graph](graph.html)", "",
           "## Reading order"]
    for i, pid in enumerate(order, 1):
        p = by_id[pid]
        summary = f" ([summary]({slug(p['title'])}.md))" if info is not None else ""
        idx.append(f"{i}. {link(p)} ({p['year']}){summary}")

    idx += ["", "## Timeline"]
    for p in sorted(papers, key=lambda x: x["year"]):
        idx.append(f"- {p['year']}: {link(p)}")

    if info is not None:
        idx += ["", "## Datasets across papers"]
        ds = {}
        for pid in order:
            for d in info.get(pid, {}).get("datasets") or []:
                ds.setdefault(d, []).append(by_id[pid]["title"])
        for d, ts in sorted(ds.items(), key=lambda kv: -len(kv[1])):
            idx.append(f"- {d}: {len(ts)} papers")

    (out / "index.md").write_text("\n".join(idx))


def number_badge(n, color):
    """SVG circle with the reading-order number inside, as a data URI for vis-network."""
    size = 54 if n < 10 else 44 if n < 100 else 34
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100">'
        f'<circle cx="50" cy="50" r="46" fill="{color}" stroke="#222" stroke-width="3"/>'
        f'<text x="50" y="50" text-anchor="middle" dominant-baseline="central" '
        f'font-family="Arial" font-weight="bold" font-size="{size}" fill="#fff" '
        f'stroke="#222" stroke-width="2" paint-order="stroke">{n}</text></svg>'
    )
    return "data:image/svg+xml;charset=utf-8," + quote(svg)


def write_graph(papers, edges, out, order):
    """Interactive citation graph. Arrow A -> B means A cites B.
    Number inside each node = reading order (1 = read first).
    Color: blue (older) to red (newer). Size: citation count.
    Click a node: a panel shows the full title as a hyperlink to the paper.
    Double-click a node: opens the paper in a new tab."""
    out.mkdir(parents=True, exist_ok=True)
    years = [p["year"] for p in papers]
    lo, hi = min(years), max(years)

    net = Network(height="850px", width="100%", directed=True, cdn_resources="in_line")
    net.set_options("""var options = {
      "nodes": {"borderWidth": 1,
                "font": {"size": 13, "face": "arial", "background": "rgba(255,255,255,0.85)"}},
      "edges": {"arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
                "smooth": {"type": "continuous"}, "color": {"opacity": 0.5}},
      "physics": {"solver": "forceAtlas2Based",
                  "forceAtlas2Based": {"gravitationalConstant": -150, "centralGravity": 0.004,
                                       "springLength": 260, "springConstant": 0.04,
                                       "avoidOverlap": 1},
                  "stabilization": {"iterations": 300}},
      "interaction": {"hover": true, "tooltipDelay": 100}
    }""")

    rank_of = {pid: i for i, pid in enumerate(order, 1)}
    info = {}
    for p in papers:
        t = (p["year"] - lo) / max(hi - lo, 1)          # 0 = old, 1 = new
        color = f"hsl({int(220 - 200 * t)}, 70%, 50%)"   # blue -> red
        short = p["title"] if len(p["title"]) <= 45 else p["title"][:42] + "..."
        label = textwrap.fill(short, 20) + f"\n({p['year']})"   # wrapped = narrower = less overlap
        n = rank_of[p["paperId"]]
        net.add_node(
            p["paperId"],
            label=label,
            title=f"Read #{n}: {p['title']} ({p['year']}, {p['citationCount']} citations)",
            shape="image",
            image=number_badge(n, color),
            size=16 + min(p["citationCount"], 5000) ** 0.5 / 4,
        )
        info[p["paperId"]] = {"title": p["title"], "year": p["year"], "order": n,
                              "cites": p["citationCount"], "url": paper_url(p)}
    for citing, cited in edges:
        net.add_edge(citing, cited)

    path = out / "graph.html"
    net.write_html(str(path))

    # Inject a click panel with a real hyperlink
    data = json.dumps(info).replace("</", "<\\/")
    extra = """
<div id="paper-panel" style="position:fixed;top:12px;right:12px;max-width:340px;background:#fff;
  border:1px solid #ccc;border-radius:8px;padding:12px;font-family:Arial,sans-serif;font-size:14px;
  display:none;box-shadow:0 2px 8px rgba(0,0,0,.15);z-index:10"></div>
<div style="position:fixed;bottom:12px;left:12px;background:#fff;border:1px solid #ccc;border-radius:8px;
  padding:8px 12px;font-family:Arial,sans-serif;font-size:13px;color:#333;z-index:10">
  Number in node = suggested reading order (1 = read first).<br>
  Blue = older, red = newer. Bigger = more citations.<br>
  Click a node for its link.</div>
<script>
window.addEventListener("load", function () {
  var papers = __DATA__;
  var panel = document.getElementById("paper-panel");
  network.on("click", function (params) {
    if (!params.nodes.length) { panel.style.display = "none"; return; }
    var p = papers[params.nodes[0]];
    panel.textContent = "";
    var a = document.createElement("a");
    a.href = p.url; a.target = "_blank"; a.rel = "noopener";
    a.textContent = p.title; a.style.fontWeight = "bold";
    var meta = document.createElement("div");
    meta.textContent = "Read #" + p.order + " | " + p.year + " | " + p.cites + " citations";
    meta.style.marginTop = "6px"; meta.style.color = "#555";
    panel.appendChild(a); panel.appendChild(meta);
    panel.style.display = "block";
  });
  network.on("doubleClick", function (params) {
    if (params.nodes.length) window.open(papers[params.nodes[0]].url, "_blank");
  });
});
</script>
""".replace("__DATA__", data)
    html = path.read_text(encoding="utf-8")
    html = html.replace("</body>", extra + "</body>", 1) if "</body>" in html else html + extra
    path.write_text(html, encoding="utf-8")


# ---------- Main ----------
def main(topic):
    print("Expanding topic...")
    queries = expand_topic(topic)
    print("Queries:", queries)
    papers = rank(search(queries))
    if not papers:
        sys.exit("No papers found. Try a different topic.")
    print(f"Kept {len(papers)} papers. Building citation links...")
    edges = citation_edges(papers)
    print(f"{len(edges)} links inside the set.")

    info = None
    if EXTRACT:
        print("Extracting...")
        info = extract_all(papers)
        missing = sum(1 for v in info.values() if not v)
        if missing:
            print(f"Warning: {missing} papers have no summary. Rerun to retry (cached calls are free).")

    order = reading_order(papers, edges)
    out = Path("survey_" + slug(topic))
    write_pages(topic, papers, info, edges, order, out)
    write_graph(papers, edges, out, order)
    print(f"Done. Open {out}/index.md and {out}/graph.html")


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]) or "antibody property prediction")