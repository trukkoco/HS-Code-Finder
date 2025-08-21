from __future__ import annotations
import os, csv, time
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
from .search_core import HybridSearch, load_master, get_meta

DATA_PATH = os.getenv("MASTER_PATH", "data/master.csv")
SYN_PATH = os.getenv("SYNONYMS_PATH", "data/synonyms.csv")
FEEDBACK_PATH = os.getenv("FEEDBACK_PATH", "data/feedback.csv")

app = FastAPI(title="HS Code Search API", version="0.6.1")

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: int = 5
    chapter: Optional[str] = None
    heading_prefix: Optional[str] = None
    use_embeddings: bool = False

class SuggestRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: int = 10
    chapter: Optional[str] = None
    heading_prefix: Optional[str] = None

class FeedbackRequest(BaseModel):
    query: str
    selected_hs6: Optional[str] = None
    selected_hs_code_new: Optional[str] = None
    selected_hs_code_old: Optional[str] = None
    corrected_hs6: Optional[str] = None
    corrected_hs_code_new: Optional[str] = None
    corrected_hs_code_old: Optional[str] = None
    notes: Optional[str] = None
    source: Optional[str] = "ui"

_search: Optional[HybridSearch] = None

def _ensure_loaded():
    global _search
    if _search is None:
        df = load_master(DATA_PATH)
        _search = HybridSearch(df, synonyms_path=SYN_PATH)

@app.on_event("startup")
def startup():
    _ensure_loaded()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/meta")
def meta():
    _ensure_loaded()
    return get_meta(_search.df)

@app.get("/synonyms")
def get_synonyms():
    if not os.path.exists(SYN_PATH):
        return {"csv": "term,expansion\n"}
    with open(SYN_PATH, "r", encoding="utf-8") as f:
        return {"csv": f.read()}

@app.post("/update_synonyms")
def update_synonyms(csv_text: str):
    with open(SYN_PATH, "w", encoding="utf-8") as f:
        f.write(csv_text if csv_text.strip() else "term,expansion\n")
    _ensure_loaded()
    _search.reload_synonyms(SYN_PATH)
    return {"ok": True}

@app.post("/search")
def search(req: SearchRequest):
    _ensure_loaded()
    results = _search.search(
        query=req.query,
        top_k=req.top_k,
        chapter=req.chapter,
        heading_prefix=req.heading_prefix,
        use_embeddings=req.use_embeddings,
    )
    return {"query": req.query, "results": results}

@app.post("/suggest")
def suggest(req: SuggestRequest):
    _ensure_loaded()
    suggestions = _search.suggest(
        query=req.query,
        top_k=req.top_k,
        chapter=req.chapter,
        heading_prefix=req.heading_prefix,
    )
    return {"query": req.query, "suggestions": suggestions}

@app.get("/code_lookup")
def code_lookup(code: str):
    _ensure_loaded()
    res = _search.search_by_code(code)
    return {"code": code, "results": res}

@app.get("/related")
def related(heading: str, top_k: int = 10):
    _ensure_loaded()
    res = _search.related_by_heading(heading, top_k=top_k)
    return {"heading": heading, "results": res}

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
    headers = ["timestamp","source","query","selected_hs6","selected_hs_code_new","selected_hs_code_old","corrected_hs6","corrected_hs_code_new","corrected_hs_code_old","notes"]
    write_header = not os.path.exists(FEEDBACK_PATH)
    with open(FEEDBACK_PATH, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(headers)
        w.writerow([
            int(time.time()), req.source or "ui", req.query or "", req.selected_hs6 or "", req.selected_hs_code_new or "",
            req.selected_hs_code_old or "", req.corrected_hs6 or "", req.corrected_hs_code_new or "", req.corrected_hs_code_old or "", req.notes or ""
        ])
    return {"ok": True}
