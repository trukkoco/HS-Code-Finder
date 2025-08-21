from __future__ import annotations
import re, os
from typing import List, Dict, Any, Optional
import pandas as pd
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz

# Optional embeddings
USE_ST = False
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    USE_ST = True
except Exception:
    USE_ST = False

def _normalize_keep_unicode(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[\t\r\n]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

STOPWORDS = set("a an the and or for of in on to with without from by as other others into over under between within per each every any your our his her their its is are was were be been being have has had do does did can could should would will shall may might must not no".split())

def _tokenize(text: str) -> List[str]:
    return [w for w in _normalize_keep_unicode(text).split() if w not in STOPWORDS]

def _confidence_from_score(s: float) -> str:
    if s >= 0.75: return "high"
    if s >= 0.50: return "medium"
    return "low"

def load_master(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    for c in ["hs_code_new","hs_code_old","hs6","chapter","heading","description","description_ar","description_en"]:
        if c not in df.columns: df[c] = ""
    return df

def _load_synonyms(path: str) -> Dict[str,str]:
    if not os.path.exists(path): return {}
    import csv
    syn = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = (row.get("term") or "").strip().lower()
            e = (row.get("expansion") or "").strip().lower()
            if t and e:
                syn[t] = e
    return syn

WEIGHTS = dict(bm25=0.55, fuzzy=0.35, emb=0.10)

def get_meta(df: pd.DataFrame) -> Dict[str, Any]:
    chapters = sorted({str(x).strip() for x in df.get("chapter", []) if str(x).strip()})
    headings = sorted({str(x).strip() for x in df.get("heading", []) if str(x).strip()})
    prefixes = sorted({h[:4] for h in headings if len(h) >= 4})
    return {"chapters": chapters, "heading_prefixes": prefixes, "count": int(len(df))}

class HybridSearch:
    def __init__(self, df: pd.DataFrame, synonyms_path: Optional[str] = None):
        self.df = df.copy()
        self.synonyms_path = synonyms_path
        self.synonyms = _load_synonyms(synonyms_path) if synonyms_path else {}
        self._build_index()

    def _build_index(self):
        comp = (self.df["description_en"].fillna("") + " " +
                self.df["description_en"].fillna("") + " " +
                self.df["description_ar"].fillna("")).map(_normalize_keep_unicode)
        self.df["_combined"] = comp.map(self._expand_with_synonyms)
        docs = [_tokenize(t) for t in self.df["_combined"].tolist()]
        self.bm25 = BM25Okapi(docs)
        self.df["_norm_en"] = self.df["description_en"].map(_normalize_keep_unicode)
        self.df["_norm_ar"] = self.df["description_ar"].map(_normalize_keep_unicode)
        # embeddings
        self.emb_model = None
        self.doc_emb = None
        if USE_ST:
            try:
                self.emb_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
                self.doc_emb = self.emb_model.encode(self.df["_combined"].tolist(), convert_to_numpy=True, show_progress_bar=False)
            except Exception:
                self.emb_model = None
                self.doc_emb = None

    def reload_synonyms(self, path: Optional[str] = None):
        if path: self.synonyms_path = path
        if self.synonyms_path:
            self.synonyms = _load_synonyms(self.synonyms_path)
        self._build_index()

    def _expand_with_synonyms(self, text: str) -> str:
        words = text.split()
        extras = [self.synonyms[w] for w in words if w in self.synonyms]
        return text + (" " + " ".join(extras) if extras else "")

    def _filter_idx(self, chapter: Optional[str], heading_prefix: Optional[str]) -> List[int]:
        idx = list(range(len(self.df)))
        if chapter:
            idx = [i for i in idx if str(self.df.iloc[i]["chapter"]).startswith(str(chapter))]
        if heading_prefix:
            idx = [i for i in idx if str(self.df.iloc[i]["heading"]).startswith(str(heading_prefix))]
        return idx

    def _bm25_scores(self, q: str, idxs: List[int]) -> List[float]:
        tokens = _tokenize(_normalize_keep_unicode(q))
        scores = self.bm25.get_scores(tokens).tolist()
        return [scores[i] for i in idxs]

    def _fuzzy_scores(self, q: str, idxs: List[int]) -> List[float]:
        qn = _normalize_keep_unicode(q)
        scores_all = []
        for en, ar in zip(self.df["_norm_en"].tolist(), self.df["_norm_ar"].tolist()):
            base = en if en.strip() else ar
            scores_all.append(fuzz.token_set_ratio(qn, base)/100.0)
        return [scores_all[i] for i in idxs]

    def _emb_scores(self, q: str, idxs: List[int]):
        if not self.emb_model or self.doc_emb is None: return None
        qv = self.emb_model.encode([_normalize_keep_unicode(q)], convert_to_numpy=True, show_progress_bar=False)[0]
        import numpy as np
        qn = qv / (np.linalg.norm(qv)+1e-8)
        dn = self.doc_emb / (np.linalg.norm(self.doc_emb, axis=1, keepdims=True)+1e-8)
        sim = (dn @ qn).tolist()
        sim01 = [(s+1)/2 for s in sim]
        return [sim01[i] for i in idxs]

    def _rank(self, query: str, chapter: Optional[str], heading_prefix: Optional[str], use_embeddings: bool):
        idxs = self._filter_idx(chapter, heading_prefix)
        if not idxs:
            return [], [], [], [], [], None
        bm_raw = self._bm25_scores(query, idxs)
        bmax = max(bm_raw) if bm_raw else 1.0
        bm_n = [s/(bmax+1e-8) for s in bm_raw]
        fz = self._fuzzy_scores(query, idxs)
        emb = self._emb_scores(query, idxs) if use_embeddings else None
        w_b, w_f, w_e = 0.55, 0.35, (0.10 if emb is not None else 0.0)
        final = []
        for i in range(len(idxs)):
            s = w_b*bm_n[i] + w_f*fz[i] + (w_e*emb[i] if emb is not None else 0.0)
            final.append(s)
        ord_local = sorted(range(len(final)), key=lambda k: final[k], reverse=True)
        ordered_idxs = [idxs[k] for k in ord_local]
        return ordered_idxs, final, bm_raw, bm_n, fz, emb

    def search(self, query: str, top_k: int = 5, chapter: Optional[str]=None, heading_prefix: Optional[str]=None, use_embeddings: bool=False) -> List[Dict[str, Any]]:
        order, final, bm, bm_n, fz, emb = self._rank(query, chapter, heading_prefix, use_embeddings)
        if not order: return []
        take = order[:top_k]
        results = []
        q_tokens = set(_tokenize(query))
        for j, idx in enumerate(take):
            row = self.df.iloc[idx]
            matched = [w for w in q_tokens if w and (w in row["_norm_en"] or w in row["_norm_ar"])]
            logic = dict(
                bm25=bm[j], bm25_norm=bm_n[j],
                fuzzy=fz[j]*100.0, fuzzy_norm=fz[j],
                embedding=(emb[j] if emb is not None else None),
                weights=dict(bm25=0.55, fuzzy=0.35, emb=(0.10 if emb is not None else 0.0)),
                matched_terms=matched,
                filters=dict(chapter=chapter, heading_prefix=heading_prefix)
            )
            results.append(dict(
                hs_code_new=row.get("hs_code_new",""),
                hs_code_old=row.get("hs_code_old",""),
                hs6=row.get("hs6",""),
                chapter=row.get("chapter",""),
                heading=row.get("heading",""),
                description_en=row.get("description_en",""),
                description_ar=row.get("description_ar",""),
                confidence=_confidence_from_score(logic["bm25_norm"] if logic["bm25_norm"] is not None else 0.0),
                score=round(float(logic["bm25_norm"] if logic["bm25_norm"] is not None else 0.0), 4),
                logic=logic
            ))
        return results

    def suggest(self, query: str, top_k: int = 10, chapter: Optional[str]=None, heading_prefix: Optional[str]=None) -> List[Dict[str, Any]]:
        order, final, bm, bm_n, fz, _ = self._rank(query, chapter, heading_prefix, use_embeddings=False)
        take = order[:top_k]
        out = []
        seen = set()
        for k, idx in enumerate(take):
            row = self.df.iloc[idx]
            text = row["description_en"] if row["description_en"].strip() else row["description_ar"]
            label = text[:120]
            tail = " | ".join([c for c in [row['hs_code_new'], row['hs6'], row['hs_code_old']] if c])
            if tail: label = f"{label} — {tail}"
            if text in seen: continue
            seen.add(text)
            out.append(dict(
                text=text, label=label,
                hs_code_new=row["hs_code_new"], hs_code_old=row["hs_code_old"], hs6=row["hs6"],
                chapter=row["chapter"], heading=row["heading"], score=round(final[k],4)
            ))
        return out

    def search_by_code(self, code: str) -> List[Dict[str, Any]]:
        c = (code or "").strip()
        if not c: return []
        c_digits = re.sub(r"[^0-9]", "", c)
        res = self.df[ (self.df["hs_code_new"].str.contains(c, na=False)) | (self.df["hs_code_old"].str.contains(c, na=False)) | (self.df["hs6"].str.contains(c_digits, na=False)) ]
        out = []
        for _, row in res.head(20).iterrows():
            out.append(dict(
                hs_code_new=row.get("hs_code_new",""), hs_code_old=row.get("hs_code_old",""), hs6=row.get("hs6",""),
                chapter=row.get("chapter",""), heading=row.get("heading",""),
                description_en=row.get("description_en",""), description_ar=row.get("description_ar",""),
                confidence="high", score=1.0, logic=dict(reason="code lookup")
            ))
        return out

    def related_by_heading(self, heading: str, top_k: int = 10) -> List[Dict[str, Any]]:
        h = (heading or "").strip()
        res = self.df[self.df["heading"].astype(str).str.startswith(h)].head(top_k)
        out = []
        for _, row in res.iterrows():
            out.append(dict(
                hs_code_new=row.get("hs_code_new",""), hs_code_old=row.get("hs_code_old",""), hs6=row.get("hs6",""),
                chapter=row.get("chapter",""), heading=row.get("heading",""),
                description_en=row.get("description_en",""), description_ar=row.get("description_ar",""),
                confidence="medium", score=0.5, logic=dict(reason="same heading")
            ))
        return out
