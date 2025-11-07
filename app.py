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
DEFAULT_MODE = "tfidf"  # "tfidf" | "lsi" | "doc2vec"

def index_filename(mode: str) -> str:
    """Consistent filename per model; keep legacy fallback."""
    legacy = "index_data.pkl"
    modern = f"index_data_{mode}.pkl"
    # Prefer model-specific; if only legacy exists, use that.
    return modern if os.path.exists(modern) or not os.path.exists(legacy) else legacy

# -----------------------------
# Engine init / switching
# -----------------------------
def build_or_load(mode: str):
    idx_file = index_filename(mode)
    if os.path.exists(idx_file):
        print(f"🔹 Loading existing {mode} index: {idx_file}")
        data = load_index(idx_file)
    else:
        print(f"⚙️ Building new {mode} index from PDFs...")
        data = IndexBuilder(mode).build(DOCS_FOLDER)
        save_index(idx_file, data)
        print(f"✅ {mode.upper()} index built and saved at {idx_file}")
    return data

def initialize_search_engine(init_mode: str):
    data = build_or_load(init_mode)
    return SearchEngine(data, init_mode)

def switch_to_mode(new_mode: str):
    """Hot-swap global search engine to desired mode."""
    global search_engine
    if new_mode == search_engine.mode:
        return
    data = build_or_load(new_mode)
    search_engine = SearchEngine(data, new_mode)

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html", current_mode=search_engine.mode)

@app.route("/pdf/<path:filename>")
def serve_pdf(filename):
    # serve from docs/ for clickable links
    return send_from_directory(DOCS_FOLDER, filename)

@app.route("/search")
def search():
    global search_engine

    query = request.args.get("query", "").strip()
    mode = request.args.get("mode", search_engine.mode)
    top_k = int(request.args.get("k", 5))

    if not query:
        return jsonify({"results": [], "mode": search_engine.mode})

    # Switch model if needed
    if mode != search_engine.mode:
        print(f"🔄 Switching model to {mode} for search...")
        switch_to_mode(mode)

    paths, scores, _docs = search_engine.query(query, top_k=top_k)

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
    mode = request.args.get("mode", search_engine.mode)

    print(f"🛠️ Rebuilding {mode.upper()} index (requested from UI)...")
    data = IndexBuilder(mode).build(DOCS_FOLDER)
    save_index(index_filename(mode), data)
    # Refresh engine only if active mode matches
    if mode == search_engine.mode:
        search_engine = SearchEngine(data, mode)
    return jsonify({"message": f"{mode.upper()} index built successfully!", "mode": mode})

@app.route("/metrics")
def metrics():
    """
    Compute macro-averaged Precision@k, Recall@k, F1@k
    for pages and docs using eval_set.json.
    """
    global search_engine
    mode = request.args.get("mode", search_engine.mode)
    k = int(request.args.get("k", 5))

    # Switch model if needed
    if mode != search_engine.mode:
        print(f"🔄 Switching model to {mode} for metrics...")
        switch_to_mode(mode)

    eval_path = "eval_set.json"
    if not os.path.exists(eval_path):
        return jsonify({"error": "eval_set.json not found"}), 404

    with open(eval_path, "r", encoding="utf-8") as f:
        ev = json.load(f)

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

    result = {
        "mode": search_engine.mode,
        "k": k,
        "pages": {
            "precision": round(mean(page_P), 4),
            "recall":    round(mean(page_R), 4),
            "f1":        round(mean(page_F), 4),
        },
        "docs": {
            "precision": round(mean(doc_P), 4),
            "recall":    round(mean(doc_R), 4),
            "f1":        round(mean(doc_F), 4),
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
