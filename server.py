from main import IndexBuilder, SearchEngine
from models import IndexBuildTask, SearchTask
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
import uuid
import pickle
from typing import List, Optional
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import time
from uuid import uuid4
import nltk

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/docs", StaticFiles(directory="docs"), name="pdfs")


class PDFFile(BaseModel):
    name: str
    path: str


@app.get("/list-pdfs", response_model=List[PDFFile])
def list_pdfs(docs_path: str = "docs"):
    pdf_files = []
    for root, _, files in os.walk(docs_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                full_path = os.path.join(root, file)
                pdf_files.append(PDFFile(name=os.path.basename(
                    full_path).split('.')[0], path=full_path))
    return pdf_files


def save_index(index_file, data):
    with open(index_file, 'wb') as f:
        pickle.dump(data, f)


def load_index(index_file):
    with open(index_file, 'rb') as f:
        return pickle.load(f)


def convert_sentiment_to_label(sentiment_score):
    if sentiment_score > 0.2:
        return "positive"
    elif sentiment_score < -0.2:
        return "negative"
    else:
        return "neutral"


# Dictionary to keep track of tasks
tasks = {}


class IndexTask:
    def __init__(self):
        self._cancel = False

    async def run(self, taskConfig):
        task = taskConfig["taskData"]
        task_id = taskConfig["task_id"]
        index_file = task.index_file.split(".")[0]
        index_file = f"{index_file}.{task.mode}.pkl"

        try:
            if os.path.exists(index_file) and not task.update_index:
                pass
            else:
                print("Building new index...")
                indexer = IndexBuilder(task.mode)
                index_data = indexer.build(task.docs_path)
                save_index(index_file, index_data)
                # Fix: Update task status to "success"
                tasks[task_id]["status"] = "success"
                print(f"Index saved to {task.index_file}")
        finally:
            self._cancel = False


@app.post("/build-index")
def build_index(taskData: IndexBuildTask, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    task = IndexTask()
    taskConfig = {
        "taskData": taskData,
        "task_id": task_id,
    }
    tasks[task_id] = {"task": task, "status": "in_progress"}
    background_tasks.add_task(task.run, taskConfig)

    return {"task_id": task_id, "status": "in_progress"}


@app.get("/index-status/{task_id}")
def get_index_status(task_id: str):
    task_info = tasks.get(task_id)
    if task_info:
        return {"status": task_info["status"]}
    raise HTTPException(status_code=404, detail="Task not found")


@app.post("/stop-index/{task_id}")
def stop_index(task_id: str):
    task_info = tasks.get(task_id)
    if task_info and task_info["status"] == "in_progress":
        task_info["task"].cancel()
        task_info["status"] = "cancelled"
        return {"status": "cancelled"}
    raise HTTPException(
        status_code=404, detail="Task not found or not in progress")


class Page(BaseModel):
    path: str
    document_name: str
    page_number: int
    score: float
    sentiment: float


class Document(BaseModel):
    path: str
    document_name: str
    sentiment: float
    cumulative_score: float


class SearchResults(BaseModel):
    pages: List[Page]
    docs: List[Document]
    query_time: float
    query_id: str
    query: str
    mode: str
    index: str


def search_index(query: str, index: str, mode: str) -> SearchResults:
    start = time.time()
    index_file = f"{index}.{mode}.pkl"
    index_data = load_index(index_file)
    search_engine = SearchEngine(index_data, mode)
    paths, scores, sorted_docs = search_engine.query(query)

    pages = [
        Page(
            path=path,
            page_number=page + 1,
            score=scores[i],
            sentiment=sentiment,
            document_name=os.path.basename(path).split(
                '.')[0]  # Extract the document name
        )
        for i, (path, page, sentiment) in enumerate(paths)
    ]

    docs = [
        Document(
            document_name=os.path.basename(doc).split('.')[0],
            cumulative_score=detail['score'],
            sentiment=detail['average_sentiment'],
            path=doc
        )
        for doc, detail in sorted_docs
    ]
    end = time.time()

    return SearchResults(pages=pages, docs=docs, query_time=(end-start) * 10**3, query_id=str(uuid4()), query=query, mode=mode, index=index)


@app.post("/query", response_model=SearchResults)
def query_index(task: SearchTask):
    try:
        results = search_index(task.query, task.index_file, task.mode)
        return results
    except Exception as e:
        print("Error: " + str(e))
        raise HTTPException(status_code=500, detail=str(e))


valid_modes = {"tfidf", "doc2vec", "lsi"}


def parse_filename(pkl_file):
    parts = pkl_file.split('.')

    if len(parts) != 3:
        return None

    name, mode, _ = parts

    if mode not in valid_modes:
        return None

    return {"name": name, "mode": mode}


def get_valid_index_files():
    valid_files = []
    root_dir = os.path.dirname(os.path.realpath(__file__))

    files = os.listdir(root_dir)
    (root_dir, files)
    for filename in files:
        file_path = os.path.join(root_dir, filename)
        if os.path.isfile(file_path) and file_path.lower().endswith('.pkl'):
            result = parse_filename(file_path.split('/')[-1])
            if result:
                valid_files.append(result)

    return valid_files


@app.get("/get-index-list")
def get_indexes():
    return get_valid_index_files()
