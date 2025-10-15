from pydantic import BaseModel
from typing import Optional


class IndexBuildTask(BaseModel):
    docs_path: Optional[str] = "docs"
    index_file: Optional[str] = "index_data.pkl"
    update_index: Optional[bool] = False
    mode: Optional[str] = "tfidf"


class SearchTask(BaseModel):
    index_file: Optional[str] = "index_data.pkl"
    mode: Optional[str] = "tfidf"
    query: str
