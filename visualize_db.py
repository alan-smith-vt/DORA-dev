"""
explore_db.py
-------------
Flask app that serves an interactive drill-down treemap of Baytown_chunks.db.
Only one folder level is sent to the browser at a time — no memory issues.
Single-child folders are automatically skipped so you land straight at the
first level that actually branches.

    pip install flask
    python explore_db.py
    open http://localhost:5000
"""

import sqlite3
from collections import defaultdict
from flask import Flask, jsonify, request, render_template_string

DB_PATH = "BAYT_PROD_chunks.db"

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load all file stats once at startup
# ---------------------------------------------------------------------------

def load_file_stats():
    print("Loading file stats from DB (one-time)...")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT filename, token_count FROM chunks").fetchall()
    conn.close()

    def normalize(path):
        """Strip _<N> chunk suffix and Windows drive letter, normalize slashes."""
        base, sep, suffix = path.rpartition("_")
        src = base if (sep and suffix.isdigit()) else path
        src = src.replace("\\", "/")
        if len(src) >= 2 and src[1] == ":":
            src = src[2:]
        return src.lstrip("/")

    file_tokens = defaultdict(int)
    file_chunks = defaultdict(int)
    for row in rows:
        src    = normalize(row["filename"])
        tokens = row["token_count"] or 0
        file_tokens[src] += tokens
        file_chunks[src] += 1

    print(f"  {len(file_tokens):,} source files loaded")
    return file_tokens, file_chunks


FILE_TOKENS, FILE_CHUNKS = load_file_stats()


def path_parts(src_file):
    # Paths are pre-normalized at load time — just split
    return [p for p in src_file.split("/") if p]


def children_of(prefix_parts):
    depth    = len(prefix_parts)
    children = defaultdict(lambda: {"file_count": 0, "chunk_count": 0, "token_count": 0, "is_folder": False})

    for src_file, tokens in FILE_TOKENS.items():
        parts = path_parts(src_file)
        if len(parts) <= depth:
            continue
        if parts[:depth] != prefix_parts:
            continue

        child_name          = parts[depth]
        is_folder           = len(parts) > depth + 1
        c                   = children[child_name]
        c["is_folder"]      = c["is_folder"] or is_folder
        c["file_count"]    += 1
        c["chunk_count"]   += FILE_CHUNKS[src_file]
        c["token_count"]   += tokens

    result = [
        {"label": name, "is_folder": s["is_folder"],
         "file_count": s["file_count"], "chunk_count": s["chunk_count"],
         "token_count": s["token_count"]}
        for name, s in children.items()
    ]
    result.sort(key=lambda x: -x["token_count"])
    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/children")
def api_children():
    path         = request.args.get("path", "")
    prefix_parts = [p for p in path.split("/") if p]

    # Auto-skip single-child folder chains so the user lands at a real branch
    skipped = []
    while True:
        kids = children_of(prefix_parts)
        if len(kids) != 1 or not kids[0]["is_folder"]:
            break
        skipped.append(kids[0]["label"])
        prefix_parts.append(kids[0]["label"])

    return jsonify({
        "path":     "/".join(prefix_parts),
        "skipped":  skipped,
        "children": kids,
        "total":    len(kids),
    })


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Baytown Document Explorer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0f0f1a; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; height: 100vh; display: flex; flex-direction: column; }
  #header { padding: 12px 20px; background: #1a1a2e; border-bottom: 1px solid #333; display: flex; align-items: center; gap: 16px; flex-shrink: 0; flex-wrap: wrap; }
  #header h1 { font-size: 16px; font-weight: 600; color: #a0c4ff; white-space: nowrap; }
  #breadcrumb { display: flex; align-items: center; gap: 4px; font-size: 13px; flex-wrap: wrap; flex: 1; }
  .crumb { color: #a0c4ff; cursor: pointer; padding: 2px 6px; border-radius: 4px; }
  .crumb:hover { background: #2a2a4e; }
  .crumb-sep { color: #444; }
  .crumb-skipped { color: #555; font-style: italic; cursor: default; }
  #stats { font-size: 12px; color: #888; white-space: nowrap; }
  #chart { flex: 1; min-height: 0; }
  #loading { position: absolute; inset: 0; background: rgba(15,15,26,0.85); display: flex; align-items: center; justify-content: center; font-size: 18px; color: #a0c4ff; pointer-events: none; opacity: 0; transition: opacity 0.15s; }
  #loading.visible { opacity: 1; pointer-events: all; }
</style>
</head>
<body>

<div id="header">
  <h1>Baytown Explorer</h1>
  <div id="breadcrumb"></div>
  <div id="stats"></div>
</div>
<div id="chart"></div>
<div id="loading">Loading…</div>

<script>
// pathStack entries: { path, label, skippedPrefix }
// skippedPrefix: the collapsed single-child segments above this entry
let currentPath = "";
let pathStack   = [];

async function loadPath(path, stackOverride) {
  document.getElementById("loading").classList.add("visible");

  const resp = await fetch("/api/children?path=" + encodeURIComponent(path));
  const data = await resp.json();

  // Server may have skipped ahead — update current path to wherever it landed
  currentPath = data.path;

  // If stackOverride provided, use it; otherwise we already pushed before calling
  if (stackOverride !== undefined) pathStack = stackOverride;

  // If server skipped segments on this navigation, record them on the top entry
  if (data.skipped && data.skipped.length && pathStack.length) {
    pathStack[pathStack.length - 1].skippedAbove = data.skipped;
  }

  renderBreadcrumb(data.skipped || []);
  renderChart(data);

  const fileTotal = data.children.reduce((s, c) => s + c.file_count, 0);
  document.getElementById("stats").textContent =
    data.total.toLocaleString() + " items · " + fileTotal.toLocaleString() + " files";

  document.getElementById("loading").classList.remove("visible");
}

function renderBreadcrumb(rootSkipped) {
  const bc = document.getElementById("breadcrumb");
  bc.innerHTML = "";

  function addSep() {
    const s = document.createElement("span");
    s.className = "crumb-sep";
    s.textContent = "/";
    bc.appendChild(s);
  }

  function addCrumb(label, onclick) {
    const c = document.createElement("span");
    c.className = "crumb";
    c.textContent = label;
    c.onclick = onclick;
    bc.appendChild(c);
  }

  function addSkipped(segs) {
    if (!segs || !segs.length) return;
    addSep();
    const s = document.createElement("span");
    s.className = "crumb-skipped";
    s.title = segs.join("/");
    s.textContent = "…";
    bc.appendChild(s);
  }

  // Root crumb
  addCrumb("⌂ root", () => loadPath("", []));

  // Show any segments that were auto-skipped at root level
  addSkipped(rootSkipped.length && pathStack.length === 0 ? rootSkipped : []);

  pathStack.forEach((entry, i) => {
    addSep();
    addSkipped(entry.skippedAbove || []);
    if (entry.skippedAbove && entry.skippedAbove.length) addSep();
    addCrumb(entry.label, () => {
      loadPath(entry.path, pathStack.slice(0, i + 1));
    });
  });
}

function renderChart(data) {
  const kids = data.children;
  if (!kids.length) {
    document.getElementById("chart").innerHTML =
      '<div style="padding:40px;color:#666;text-align:center">No children found</div>';
    return;
  }

  const trace = {
    type: "treemap",
    labels:    kids.map(c => c.label),
    parents:   kids.map(() => ""),
    values:    kids.map(c => c.token_count || 1),
    customdata: kids.map(c => [c.file_count, c.chunk_count, c.token_count, c.is_folder ? 1 : 0]),
    hovertemplate:
      "<b>%{label}</b><br>" +
      "Files:  %{customdata[0]:,}<br>" +
      "Chunks: %{customdata[1]:,}<br>" +
      "Tokens: %{customdata[2]:,}<br>" +
      "<extra></extra>",
    texttemplate: "%{label}<br>%{customdata[0]:,} files",
    marker: {
      colors:     kids.map(c => c.chunk_count > 0 ? c.token_count / c.chunk_count : 0),
      colorscale: "Viridis",
      showscale:  true,
      colorbar: { title: "Tokens/Chunk", tickfont: { color: "#aaa" }, titlefont: { color: "#aaa" } },
    },
    textfont: { color: "#fff" },
  };

  const el = document.getElementById("chart");
  Plotly.react(el, [trace],
    { margin: { t: 10, l: 10, r: 10, b: 10 }, paper_bgcolor: "#0f0f1a", font: { color: "#e0e0e0" } },
    { responsive: true, displayModeBar: false }
  );

  el.removeAllListeners && el.removeAllListeners("plotly_click");
  el.on("plotly_click", function(evt) {
    const child = kids[evt.points[0].pointNumber];
    if (!child || !child.is_folder) return;

    // Use server-returned path as base, never build paths client-side
    const newPath = data.path ? data.path + "/" + child.label : child.label;
    pathStack.push({ path: newPath, label: child.label });
    loadPath(newPath);
  });
}

loadPath("");
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Starting server — open http://localhost:5000")
    app.run(debug=False, port=5000)