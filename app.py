from flask import Flask, render_template, request, jsonify, send_from_directory
from main import SearchEngine, IndexBuilder, load_index, save_index
import os
import json
from statistics import mean

app = Flask(__name__, template_folder="templates")

# -----------------------------
# Config
# -----------------------------
DOCS_FOLDER = "docs"
DEFAULT_MODE = "tfidf"   # default is TF-IDF

# Accept a few aliases; always work with lowercase internally
ALLOWED_MODES = {
    "tfidf": "tfidf",
    "tf-idf": "tfidf",
    "lsi": "lsi",
    "doc2vec": "doc2vec",
    "d2v": "doc2vec",
}
def normalize_mode(m: str) -> str:
    if not m:
        return DEFAULT_MODE
    m = m.strip().lower()
    return ALLOWED_MODES.get(m, DEFAULT_MODE)

def index_filename(mode: str) -> str:
    """Return consistent filename per model; keep a legacy fallback."""
    mode = normalize_mode(mode)
    legacy = "index_data.pkl"
    modern = f"index_data_{mode}.pkl"
    # Prefer model-specific; if only legacy exists, use that to avoid breaking old installs.
    return modern if os.path.exists(modern) or not os.path.exists(legacy) else legacy

# -----------------------------
# Engine init / switching
# -----------------------------
def build_or_load(mode: str):
    """
    Only LOAD here. If the index file doesn't exist, raise so callers
    can decide (e.g., /search should return an error instead of building).
    """
    mode = normalize_mode(mode)
    idx_file = index_filename(mode)
    if os.path.exists(idx_file):
        print(f"🔹 Loading existing {mode} index: {idx_file}")
        data = load_index(idx_file)
        return data
    else:
        # Do NOT build here; building in a request handler can hang on Windows.
        raise FileNotFoundError(f"Index for mode '{mode}' not found. Build it first.")

def initialize_search_engine(init_mode: str):
    init_mode = normalize_mode(init_mode)
    try:
        data = build_or_load(init_mode)
    except FileNotFoundError:
        # Ensure server comes up: auto-build TF-IDF only if missing at startup.
        print(f"⚠️ {init_mode.upper()} index missing at startup. Building default TF-IDF...")
        data = IndexBuilder("tfidf").build(DOCS_FOLDER)
        save_index(index_filename("tfidf"), data)
        init_mode = "tfidf"
    return SearchEngine(data, init_mode)

def switch_to_mode(new_mode: str):
    """Hot-swap global search engine to desired mode (normalized)."""
    global search_engine
    new_mode = normalize_mode(new_mode)
    if new_mode == search_engine.mode:
        return
    data = build_or_load(new_mode)  # may raise FileNotFoundError
    search_engine = SearchEngine(data, new_mode)
    print(f"✅ Switched to {new_mode.upper()}")

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html", current_mode=search_engine.mode)

@app.route("/pdf/<path:filename>")
def serve_pdf(filename):
    # Serve from docs/ for clickable links
    return send_from_directory(DOCS_FOLDER, filename)

@app.route("/search")
def search():
    global search_engine

    query = request.args.get("query", "").strip()
    mode = normalize_mode(request.args.get("mode", search_engine.mode))
    top_k = int(request.args.get("k", 5))

    if not query:
        return jsonify({"results": [], "mode": search_engine.mode})

    # Switch model if needed, but don't build here
    if mode != search_engine.mode:
        try:
            print(f"🔄 Switching model to {mode} for search...")
            switch_to_mode(mode)
        except FileNotFoundError as e:
            return jsonify({
                "error": str(e),
                "need_build": True,
                "mode": mode
            }), 409  # inform UI to build first

    try:
        paths, scores, _docs = search_engine.query(query, top_k=top_k)
    except Exception as e:
        # Avoid infinite spinner on the frontend
        return jsonify({"error": f"search failed: {e}", "mode": search_engine.mode}), 500

    results = []
    for (abs_path, page_idx, sentiment), score in zip(paths, scores):
        name = os.path.basename(abs_path)
        page = int(page_idx) + 1
        url = f"/pdf/{name}#page={page}"  # open at page
        results.append({
            "name": name,
            "path": abs_path,
            "page": page,
            "score": float(score),
            "sentiment": float(sentiment),
            "url": url
        })

    return jsonify({"results": results, "mode": search_engine.mode})

@app.route("/build_index", methods=["POST"])
def build_index():
    global search_engine
    mode = normalize_mode(request.args.get("mode", search_engine.mode))

    print(f"🛠️ Rebuilding {mode.upper()} index (requested from UI)...")
    try:
        data = IndexBuilder(mode).build(DOCS_FOLDER)
        save_index(index_filename(mode), data)
        # Refresh engine only if active mode matches
        if mode == search_engine.mode:
            search_engine = SearchEngine(data, mode)
    except Exception as e:
        return jsonify({"error": f"build failed: {e}", "mode": mode}), 500

    return jsonify({"message": f"{mode.upper()} index built successfully!", "mode": mode})

@app.route("/metrics")
def metrics():
    """
    Compute macro-averaged Precision@k, Recall@k, F1@k
    for pages and docs using eval_set.json.
    """
    global search_engine
    mode = normalize_mode(request.args.get("mode", search_engine.mode))
    k = int(request.args.get("k", 5))

    # Switch model if needed (do not build here)
    if mode != search_engine.mode:
        try:
            print(f"🔄 Switching model to {mode} for metrics...")
            switch_to_mode(mode)
        except FileNotFoundError as e:
            return jsonify({
                "error": str(e),
                "need_build": True,
                "mode": mode
            }), 409

    eval_path = "eval_set.json"
    if not os.path.exists(eval_path):
        return jsonify({"error": "eval_set.json not found"}), 404

    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            ev = json.load(f)
    except Exception as e:
        return jsonify({"error": f"could not read eval_set.json: {e}"}), 500

    k = int(ev.get("k", k))
    queries = ev.get("queries", [])

    def prf(pred_set, true_set, denom):
        tp = len(pred_set & true_set)
        precision = tp / max(denom, 1)
        recall = tp / max(len(true_set), 1)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return precision, recall, f1

    page_P, page_R, page_F = [], [], []
    doc_P, doc_R, doc_F = [], [], []

    try:
        for q in queries:
            text = q["text"]
            paths, scores, _agg = search_engine.query(text, top_k=k)

            # predicted pages (filename, 1-based page)
            pred_pages = []
            seen_pages = set()
            for (abs_path, page_idx, _sent), _s in zip(paths, scores):
                name = os.path.basename(abs_path)
                key = (name, int(page_idx) + 1)
                if key not in seen_pages:
                    seen_pages.add(key)
                    pred_pages.append(key)
                    if len(pred_pages) >= k:
                        break

            # predicted docs (unique order, up to k)
            pred_docs = []
            seen_docs = set()
            for (abs_path, _page_idx, _sent), _s in zip(paths, scores):
                name = os.path.basename(abs_path)
                if name not in seen_docs:
                    seen_docs.add(name)
                    pred_docs.append(name)
                    if len(pred_docs) >= k:
                        break

            true_pages = {(fn, int(p)) for fn, p in q.get("relevant_pages", [])}
            true_docs = set(q.get("relevant_docs", []))

            pP, rP, fP = prf(set(pred_pages), true_pages, len(pred_pages) or k)
            pD, rD, fD = prf(set(pred_docs),  true_docs,  len(pred_docs)  or k)

            page_P.append(pP); page_R.append(rP); page_F.append(fP)
            doc_P.append(pD);  doc_R.append(rD);  doc_F.append(fD)
    except Exception as e:
        return jsonify({"error": f"metrics failed: {e}", "mode": search_engine.mode}), 500

    result = {
        "mode": search_engine.mode,
        "k": k,
        "pages": {
            "precision": round(mean(page_P), 4) if page_P else 0.0,
            "recall":    round(mean(page_R), 4) if page_R else 0.0,
            "f1":        round(mean(page_F), 4) if page_F else 0.0,
        },
        "docs": {
            "precision": round(mean(doc_P), 4) if doc_P else 0.0,
            "recall":    round(mean(doc_R), 4) if doc_R else 0.0,
            "f1":        round(mean(doc_F), 4) if doc_F else 0.0,
        }
    }
    return jsonify(result)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # important for Windows + multiprocessing in IndexBuilder
    search_engine = initialize_search_engine(DEFAULT_MODE)
    print(f"🚀 Flask running with model: {search_engine.mode.upper()}")
    app.run(debug=True)
