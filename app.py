from flask import Flask, render_template, request, jsonify
from main import SearchEngine, IndexBuilder, load_index, save_index
import os

app = Flask(__name__, template_folder="templates")

# ------------------------------------------------------------
# ⚙️ Configuration
# ------------------------------------------------------------
INDEX_FILE = "index_data.pkl"
DOCS_FOLDER = "docs"
MODE = "tfidf"  # default model: "tfidf" | "lsi" | "doc2vec"


# ------------------------------------------------------------
# 🧠 Initialize or Load Search Engine
# ------------------------------------------------------------
def initialize_search_engine():
    """Build or load the PDF index and return a SearchEngine instance."""
    if os.path.exists(INDEX_FILE):
        print("🔹 Loading existing index...")
        index_data = load_index(INDEX_FILE)
    else:
        print("⚙️ Building new index from PDFs...")
        indexer = IndexBuilder(MODE)
        index_data = indexer.build(DOCS_FOLDER)
        save_index(INDEX_FILE, index_data)
        print("✅ Index built and saved.")
    return SearchEngine(index_data, MODE)


# ------------------------------------------------------------
# 🏠 Home Page Route
# ------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ------------------------------------------------------------
# 🔍 Search Route
# ------------------------------------------------------------
@app.route("/search")
def search():
    global search_engine  # ✅ declare global before using it

    query = request.args.get("query", "")
    mode = request.args.get("mode", MODE)  # use URL param or default MODE

    if not query.strip():
        return jsonify({"results": []})

    # 🔄 Switch model dynamically if user changes it
    if mode != search_engine.mode:
        print(f"🔄 Switching model to {mode}...")
        index_file = f"index_data_{mode}.pkl"

        if os.path.exists(index_file):
            index_data = load_index(index_file)
        else:
            print(f"⚙️ Building new {mode} index...")
            indexer = IndexBuilder(mode)
            index_data = indexer.build(DOCS_FOLDER)
            save_index(index_file, index_data)
            print(f"✅ {mode.upper()} index built and saved.")

        search_engine = SearchEngine(index_data, mode)

    # Perform the search
    paths, scores, docs = search_engine.query(query, top_k=5)

    results = [
        {
            "path": p[0],
            "name": os.path.basename(p[0]),
            "score": float(s),
            "sentiment": float(p[2])
        }
        for p, s in zip(paths, scores)
    ]

    return jsonify({"results": results})


# ------------------------------------------------------------
# 🚀 App Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # ✅ Required for Windows multiprocessing

    search_engine = initialize_search_engine()
    print(f"🚀 Flask app running with model: {MODE.upper()}")
    print("📂 Docs folder:", DOCS_FOLDER)
    app.run(debug=True)
