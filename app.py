from flask import Flask, render_template, request, jsonify, send_from_directory
from main import SearchEngine, IndexBuilder, load_index, save_index
import os
import json
from statistics import mean

# -----------------------------
# 1️⃣ Initialize Flask
# -----------------------------
app = Flask(__name__, template_folder="templates")

# -----------------------------
# 2️⃣ Config
# -----------------------------
DOCS_FOLDER = "docs"
DEFAULT_MODE = "tfidf"   # default model is TF-IDF

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
    """Return consistent filename per model; keep legacy fallback."""
    mode = normalize_mode(mode)
    legacy = "index_data.pkl"
    modern = f"index_data_{mode}.pkl"
    return modern if os.path.exists(modern) or not os.path.exists(legacy) else legacy

# -----------------------------
# 3️⃣ Engine Initialization & Helpers
# -----------------------------
def build_or_load(mode: str):
    """Load an index; raise FileNotFoundError if missing."""
    mode = normalize_mode(mode)
    idx_file = index_filename(mode)
    if os.path.exists(idx_file):
        print(f"🔹 Loading existing {mode} index: {idx_file}")
        data = load_index(idx_file)
        return data
    else:
        raise FileNotFoundError(f"Index for mode '{mode}' not found. Build it first.")


def initialize_search_engine(init_mode: str):
    init_mode = normalize_mode(init_mode)
    try:
        data = build_or_load(init_mode)
    except FileNotFoundError:
        # Auto-build TF-IDF on first startup
        print(f"⚠️ {init_mode.upper()} index missing. Building TF-IDF by default...")
        data = IndexBuilder("tfidf").build(DOCS_FOLDER)
        save_index(index_filename("tfidf"), data)
        init_mode = "tfidf"
    return SearchEngine(data, init_mode)


def switch_to_mode(new_mode: str):
    global search_engine
    new_mode = normalize_mode(new_mode)
    if new_mode == search_engine.mode:
        return
    data = build_or_load(new_mode)
    search_engine = SearchEngine(data, new_mode)
    print(f"✅ Switched to {new_mode.upper()} mode")

# -----------------------------
# 4️⃣ Routes
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html", current_mode=search_engine.mode)

@app.route("/pdf/<path:filename>")
def serve_pdf(filename):
    return send_from_directory(DOCS_FOLDER, filename)

@app.route("/search")
def search():
    global search_engine
    query = request.args.get("query", "").strip()
    mode = normalize_mode(request.args.get("mode", search_engine.mode))
    top_k = int(request.args.get("k", 5))
    result_type = request.args.get("result_type", "page").strip().lower()  # 'page' or 'document'

    if not query:
        return jsonify({"results": [], "mode": search_engine.mode})

    if mode != search_engine.mode:
        try:
            print(f"🔄 Switching model to {mode} for search...")
            switch_to_mode(mode)
        except FileNotFoundError as e:
            return jsonify({
                "error": str(e),
                "need_build": True,
                "mode": mode
            }), 409

    try:
        # search_engine.query returns: paths (list of (abs_path, page_idx, sentiment)), scores (list), _docs (optional)
        paths, scores, _docs = search_engine.query(query, top_k=top_k)
    except Exception as e:
        return jsonify({"error": f"Search failed: {e}", "mode": search_engine.mode}), 500

    # Build page-level results first (unchanged behavior) -- includes sentiment and raw similarity score
    page_results = []
    for (abs_path, page_idx, sentiment), score in zip(paths, scores):
        name = os.path.basename(abs_path)
        page = int(page_idx) + 1
        url = f"/pdf/{name}#page={page}"
        page_results.append({
            "name": name,
            "path": abs_path,
            "page": page,
            "score": float(score),            # raw similarity score
            "sentiment": float(sentiment),    # per-page sentiment
            "url": url
        })

    # If the UI asks for page-wise results, return them directly
    if result_type == "page":
        return jsonify({"results": page_results, "mode": search_engine.mode, "result_type": "page"})

    # Otherwise, produce document-wise aggregated results with normalization
    # Group pages by document name
    docs = {}
    for item in page_results:
        name = item["name"]
        docs.setdefault(name, []).append(item)

    doc_aggregates = []
    for name, pages in docs.items():
        scores_list = [p["score"] for p in pages]
        sents_list = [p["sentiment"] for p in pages]
        # Aggregate: use mean of page scores and sentiments
        agg_score = mean(scores_list) if scores_list else 0.0
        agg_sentiment = mean(sents_list) if sents_list else 0.0
        doc_aggregates.append({
            "name": name,
            "path": pages[0]["path"],
            "pages": pages,                       # include page-level details if UI wants to expand
            "raw_score": float(agg_score),        # un-normalized aggregated score
            "raw_sentiment": float(agg_sentiment)
        })

    # Normalize aggregated scores to 0-1 range by dividing by max (simple, stable normalization)
    max_score = max((d["raw_score"] for d in doc_aggregates), default=0.0)
    if max_score > 0:
        for d in doc_aggregates:
            d["normalized_score"] = float(d["raw_score"] / max_score)
    else:
        for d in doc_aggregates:
            d["normalized_score"] = 0.0

    # Sort by normalized_score desc and return top_k documents
    doc_aggregates.sort(key=lambda x: x["normalized_score"], reverse=True)
    doc_results = doc_aggregates[:top_k]

    # For convenience include both normalized score and raw score in response
    return jsonify({
        "results": doc_results,
        "mode": search_engine.mode,
        "result_type": "document",
        "note": "document-wise results are aggregated (mean) across pages and normalized to [0,1] by dividing by the max aggregated score"
    })

@app.route("/build_index", methods=["POST"])
def build_index():
    global search_engine
    mode = normalize_mode(request.args.get("mode", search_engine.mode))
    print(f"🛠️ Rebuilding {mode.upper()} index...")

    try:
        data = IndexBuilder(mode).build(DOCS_FOLDER)
        save_index(index_filename(mode), data)
        if mode == search_engine.mode:
            search_engine = SearchEngine(data, mode)
    except Exception as e:
        return jsonify({"error": f"Build failed: {e}", "mode": mode}), 500

    return jsonify({"message": f"{mode.upper()} index built successfully!", "mode": mode})

# -----------------------------
# 5️⃣ Metrics (single-mode)
# -----------------------------
@app.route("/metrics")
def metrics():
    global search_engine
    mode = normalize_mode(request.args.get("mode", search_engine.mode))
    k_req = int(request.args.get("k", 5))

    if mode != search_engine.mode:
        try:
            switch_to_mode(mode)
        except FileNotFoundError as e:
            return jsonify({"error": str(e), "need_build": True, "mode": mode}), 409

    eval_path = "eval_set.json"
    if not os.path.exists(eval_path):
        return jsonify({"error": "eval_set.json not found"}), 404

    with open(eval_path, "r", encoding="utf-8") as f:
        ev = json.load(f)

    k = int(ev.get("k", k_req))
    queries = ev.get("queries", [])

    def prf(pred_set, true_set, denom):
        tp = len(pred_set & true_set)
        precision = tp / max(denom, 1)
        recall = tp / max(len(true_set), 1)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return precision, recall, f1

    page_P, page_R, page_F = [], [], []
    doc_P, doc_R, doc_F = [], [], []
    evaluated, skipped = 0, 0

    for q in queries:
        text = q.get("text", "")
        if not text:
            skipped += 1
            continue

        try:
            paths, scores, _agg = search_engine.query(text, top_k=k)
        except Exception as e:
            skipped += 1
            continue

        pred_pages, seen_pages = [], set()
        for (abs_path, page_idx, _sent), _s in zip(paths, scores):
            name = os.path.basename(abs_path)
            key = (name, int(page_idx) + 1)
            if key not in seen_pages:
                seen_pages.add(key)
                pred_pages.append(key)
                if len(pred_pages) >= k:
                    break

        pred_docs, seen_docs = [], set()
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
        evaluated += 1

    result = {
        "mode": search_engine.mode,
        "k": k,
        "evaluated": evaluated,
        "skipped": skipped,
        "pages": {
            "precision": round(mean(page_P), 4) if page_P else 0.0,
            "recall": round(mean(page_R), 4) if page_R else 0.0,
            "f1": round(mean(page_F), 4) if page_F else 0.0,
        },
        "docs": {
            "precision": round(mean(doc_P), 4) if doc_P else 0.0,
            "recall": round(mean(doc_R), 4) if doc_R else 0.0,
            "f1": round(mean(doc_F), 4) if doc_F else 0.0,
        }
    }
    return jsonify(result)

# -----------------------------
# 6️⃣ Metrics (all modes)
# -----------------------------
@app.route("/metrics_all")
def metrics_all():
    global search_engine
    modes = ["tfidf", "lsi", "doc2vec"]
    k_req = int(request.args.get("k", 5))

    eval_path = "eval_set.json"
    if not os.path.exists(eval_path):
        return jsonify({"error": "eval_set.json not found"}), 404

    with open(eval_path, "r", encoding="utf-8") as f:
        ev = json.load(f)

    k = int(ev.get("k", k_req))
    queries = ev.get("queries", [])

    def prf(pred_set, true_set, denom):
        tp = len(pred_set & true_set)
        precision = tp / max(denom, 1)
        recall = tp / max(len(true_set), 1)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return precision, recall, f1

    def evaluate_mode(mode):
        try:
            data = build_or_load(mode)
        except FileNotFoundError as e:
            return {"mode": mode, "need_build": True, "error": str(e)}

        try:
            switch_to_mode(mode)
        except FileNotFoundError as e:
            return {"mode": mode, "need_build": True, "error": str(e)}

        page_P, page_R, page_F = [], [], []
        doc_P, doc_R, doc_F = [], [], []
        evaluated, skipped = 0, 0

        for q in queries:
            text = q.get("text", "")
            if not text:
                skipped += 1
                continue
            try:
                paths, scores, _agg = search_engine.query(text, top_k=k)
            except Exception:
                skipped += 1
                continue

            pred_pages, seen_pages = [], set()
            for (abs_path, page_idx, _sent), _s in zip(paths, scores):
                name = os.path.basename(abs_path)
                key = (name, int(page_idx) + 1)
                if key not in seen_pages:
                    seen_pages.add(key)
                    pred_pages.append(key)
                    if len(pred_pages) >= k:
                        break

            pred_docs, seen_docs = [], set()
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
            evaluated += 1

        return {
            "mode": mode,
            "k": k,
            "evaluated": evaluated,
            "skipped": skipped,
            "pages": {
                "precision": round(mean(page_P), 4) if page_P else 0.0,
                "recall": round(mean(page_R), 4) if page_R else 0.0,
                "f1": round(mean(page_F), 4) if page_F else 0.0,
            },
            "docs": {
                "precision": round(mean(doc_P), 4) if doc_P else 0.0,
                "recall": round(mean(doc_R), 4) if doc_R else 0.0,
                "f1": round(mean(doc_F), 4) if doc_F else 0.0,
            }
        }

    results = {m: evaluate_mode(m) for m in modes}
    return jsonify(results)

# -----------------------------
# 7️⃣ Run Server
# -----------------------------
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    search_engine = initialize_search_engine(DEFAULT_MODE)
    print(f"🚀 Flask running with model: {search_engine.mode.upper()}")
    app.run(debug=True)
