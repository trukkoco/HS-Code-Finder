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
# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import re, unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack

HS_XLSX_PATH = "HSCodeMaster-v3.3customers.xlsx"  # or a URL / mounted path

def _zero_pad(code, width):
    if pd.isna(code): return ""
    s = str(code).strip()
    return s.zfill(width) if s.isdigit() else s

def _clean(t: str) -> str:
    if not t: return ""
    t = unicodedata.normalize("NFKC", str(t).lower())
    t = re.sub(r"[-_/.,;:(){}\[\]<>+]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

@st.cache_resource(show_spinner=True)
def load_index(master_path: str):
    df = pd.read_excel(master_path, sheet_name="HSCodeMaster")
    df["OldHSCode"] = df["OldHSCode"].apply(lambda x: _zero_pad(x, 8))
    df["NewHSCode"] = df["NewHSCode"].apply(lambda x: _zero_pad(x, 12))
    df["text_en"] = df["LongDescEn"].fillna("")
    df["text_ar"] = df["LongDescAr"].fillna("")
    df["text_combined"] = (df["text_en"] + " " + df["text_ar"]).astype(str).str.strip()
    df = df[df["NewHSCode"].str.len() >= 11].reset_index(drop=True)

    corpus = [_clean(x) for x in df["text_combined"].tolist()]
    wv = TfidfVectorizer(ngram_range=(1,2), analyzer="word", min_df=1, max_features=200000)
    Xw = wv.fit_transform(corpus)
    cv = TfidfVectorizer(ngram_range=(3,5), analyzer="char", min_df=1)
    Xc = cv.fit_transform(corpus)
    X = hstack([Xw, Xc]).tocsr()
    nn = NearestNeighbors(n_neighbors=25, metric="cosine").fit(X)
    return df, wv, cv, nn

def _embed(texts, wv, cv):
    prep = [_clean(x) for x in texts]
    return hstack([wv.transform(prep), cv.transform(prep)]).tocsr()

def _scale_conf(dists):
    sim = 1.0 - dists
    s = (sim - 0.1) / (1.0 - 0.1 + 1e-9)  # anchored 10%..100%
    return np.clip(s, 0, 1) * 100.0

def classify(df, wv, cv, nn, query: str, top_k: int = 5):
    Q = _embed([query], wv, cv)
    dists, idxs = nn.kneighbors(Q, n_neighbors=top_k, return_distance=True)
    dists, idxs = dists[0], idxs[0]
    conf = _scale_conf(dists)
    rows = []
    for rank, (i, c) in enumerate(zip(idxs, conf), start=1):
        row = df.iloc[int(i)]
        rows.append({
            "Rank": rank,
            "HS12 Code": row["NewHSCode"],
            "HS8 Parent": row["OldHSCode"],
            "Description (EN)": row["LongDescEn"],
            "Unit": row["StatisticalQtyUnit"],
            "Duty %": float(row["DutyPercentage"] or 0),
            "Confidence %": float(np.round(c, 2)),
        })
    return pd.DataFrame(rows)

st.title("UAE 12-digit HS Classifier")
st.caption("Maps free-form product descriptions to 12-digit HS codes (UAE).")

with st.sidebar:
    hs_path = st.text_input("Path to HS master Excel", value=HS_XLSX_PATH)
    top_k = st.slider("Top K", min_value=1, max_value=15, value=5)
    st.write("Make sure your Excel includes the 'HSCodeMaster' sheet.")

df, wv, cv, nn = load_index(hs_path)

query = st.text_input("Enter product description", value="night vision binoculars")
btn = st.button("Classify")

if btn and query.strip():
    out = classify(df, wv, cv, nn, query.strip(), top_k)
    st.dataframe(out, use_container_width=True)
    st.download_button("Download CSV", data=out.to_csv(index=False), file_name="hs_results.csv", mime="text/csv")
