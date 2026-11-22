"""Interactive citation graph (pyvis).

Arrow A -> B means A cites B. Number inside a node = reading order (1 = read first).
Color: blue (older) to red (newer). Size: citation count.
Click a node: panel with the hyperlinked title. Double-click: open the paper.
"""
import json, textwrap
from urllib.parse import quote
from pyvis.network import Network
from .utils import paper_url


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


OPTIONS = """var options = {
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
}"""

EXTRA_HTML = """
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
"""


def write_graph(papers, edges, out, order):
    out.mkdir(parents=True, exist_ok=True)
    years = [p["year"] for p in papers]
    lo, hi = min(years), max(years)
    rank_of = {pid: i for i, pid in enumerate(order, 1)}

    net = Network(height="850px", width="100%", directed=True, cdn_resources="in_line")
    net.set_options(OPTIONS)

    info = {}
    for p in papers:
        t = (p["year"] - lo) / max(hi - lo, 1)          # 0 = old, 1 = new
        color = f"hsl({int(220 - 200 * t)}, 70%, 50%)"   # blue -> red
        short = p["title"] if len(p["title"]) <= 45 else p["title"][:42] + "..."
        label = textwrap.fill(short, 20) + f"\n({p['year']})"
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

    data = json.dumps(info).replace("</", "<\\/")
    extra = EXTRA_HTML.replace("__DATA__", data)
    html = path.read_text(encoding="utf-8")
    html = html.replace("</body>", extra + "</body>", 1) if "</body>" in html else html + extra
    path.write_text(html, encoding="utf-8")
