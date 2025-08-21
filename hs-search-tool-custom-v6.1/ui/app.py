import os, requests, streamlit as st, pandas as pd
from io import BytesIO
import re, json

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="HS Code Finder — Full", layout="wide")
st.title("🔎 HS Code Finder — Full Build (Typeahead + Filters + Batch + Document + Feedback + Admin)")

def fetch_meta():
    try:
        r = requests.get(f"{API_BASE}/meta", timeout=10)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return {"chapters": [], "heading_prefixes": [], "count": 0}

meta = fetch_meta()
chapters = ["(All)"] + meta.get("chapters", [])
prefixes = ["(All)"] + meta.get("heading_prefixes", [])

STOPWORDS = set("a an the and or for of in on to with without from by as other others into over under between within per each every any your our his her their its is are was were be been being have has had do does did can could should would will shall may might must not no".split())

def _normalize_keep_unicode(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[\t\r\n]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def extract_text_from_file(file) -> str:
    name = (file.name or "").lower()
    data = file.read()
    if name.endswith(".txt"):
        for enc in ("utf-8","utf-16","cp1252","latin-1"):
            try: return data.decode(enc)
            except Exception: continue
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(BytesIO(data)) as pdf:
                for page in pdf.pages[:50]:
                    text_parts.append(page.extract_text() or "")
            text = "\n".join(text_parts).strip()
        except Exception:
            text = ""
        return text
    if name.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return ""
    return ""

def extract_keywords(text: str, top_k: int = 12):
    t = _normalize_keep_unicode(text)
    # FIXED regex: hyphen is last inside class; Arabic range uses real \u escapes via Python string
    tokens = [w for w in re.findall("[A-Za-z\u0600-\u06FF0-9][A-Za-z\u0600-\u06FF0-9'/-]{2,}", t)]
    tokens = [w for w in tokens if w not in STOPWORDS]
    freq = {}
    for w in tokens:
        freq[w] = freq.get(w, 0) + 1
    boosts = {"cotton":2, "woven":2, "knitted":2, "men":1.5, "women":1.5, "horse":2, "equine":2,
              "engine":2, "electronic":1.5, "furniture":1.5, "wood":1.2, "toy":1.2, "beverage":1.2}
    scored = [(w, freq[w]*boosts.get(w,1.0)) for w in freq]
    scored.sort(key=lambda x: x[1], reverse=True)
    out = []
    seen = set()
    for w, _ in scored:
        if w in seen: continue
        seen.add(w); out.append(w)
        if len(out) >= top_k: break
    return out

# Preferences
PREF_PATH = "data/user_prefs.json"
def load_prefs():
    try:
        with open(PREF_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
def save_prefs(p):
    try:
        with open(PREF_PATH, "w", encoding="utf-8") as f:
            json.dump(p, f, indent=2)
        return True
    except Exception:
        return False

prefs = load_prefs()

tab1, tab2, tab3, tab4 = st.tabs(["Search", "Batch classify", "Document analyze", "Admin"])

# ---------- Search tab ----------
with tab1:
    st.subheader("Natural language search")
    cols = st.columns(5)
    with cols[0]:
        q = st.text_input("Describe your product (EN/AR):", value=prefs.get("last_query",""), key="q")
    with cols[1]:
        chapter_sel = st.selectbox("Chapter", options=chapters, index=prefs.get("chapter_idx",0))
    with cols[2]:
        heading_sel = st.selectbox("Heading prefix (4-digit)", options=prefixes, index=prefs.get("heading_idx",0))
    with cols[3]:
        use_emb = st.checkbox("Use embeddings (if installed)", value=prefs.get("use_emb", False))
    with cols[4]:
        auto_search = st.checkbox("Auto-search on suggestion pick", value=prefs.get("auto_search", False))

    topk = st.number_input("Top K", 1, 20, prefs.get("topk",5), 1)

    chosen_text = None
    if len(q.strip()) >= 2:
        try:
            payload = {"query": q, "top_k": 12}
            if chapter_sel != "(All)": payload["chapter"] = chapter_sel
            if heading_sel != "(All)": payload["heading_prefix"] = heading_sel
            rs = requests.post(f"{API_BASE}/suggest", json=payload, timeout=10)
            if rs.ok:
                sugs = rs.json().get("suggestions", [])
                options = [s.get("label","") for s in sugs]
                sel = st.selectbox("Suggestions (optional):", options=["(none)"]+options, index=0, key="sugg_sel")
                if sel != "(none)":
                    idx = options.index(sel)
                    chosen_text = sugs[idx].get("text","")
        except Exception:
            pass

    final_q = chosen_text if chosen_text else q
    filters = {}
    if chapter_sel != "(All)": filters["chapter"] = chapter_sel
    if heading_sel != "(All)": filters["heading_prefix"] = heading_sel

    do_search = st.button("Search", type="primary") or (auto_search and chosen_text)

    if do_search:
        if not final_q.strip():
            st.warning("Please enter a description (suggestions are optional).")
        else:
            prefs.update({"last_query": q, "chapter_idx": chapters.index(chapter_sel) if chapter_sel in chapters else 0,
                          "heading_idx": prefixes.index(heading_sel) if heading_sel in prefixes else 0,
                          "use_emb": use_emb, "auto_search": auto_search, "topk": int(topk)})
            save_prefs(prefs)
            try:
                payload = {"query": final_q, "top_k": int(topk), "use_embeddings": bool(use_emb), **filters}
                r = requests.post(f"{API_BASE}/search", json=payload, timeout=60)
                r.raise_for_status()
                data = r.json().get("results", [])
                if not data:
                    st.info("No results. Try refining terms or adjust filters.")
                else:
                    rows = []
                    for i, res in enumerate(data, 1):
                        with st.container(border=True):
                            st.text(f"#{i}  HS New: {res.get('hs_code_new','')}   HS Old: {res.get('hs_code_old','')}   HS6: {res.get('hs6','')}")
                            st.write("**Chapter:**", res.get("chapter",""), "  **Heading:**", res.get("heading",""))
                            if res.get("description_en"):
                                st.write("**EN:**", res.get("description_en",""))
                            if res.get("description_ar"):
                                st.write("**AR:**", res.get("description_ar",""))
                            st.write("**Confidence:**", res.get("confidence",""), "  **Score:**", res.get("score",0))
                            with st.expander("Logic (why this matched)"):
                                st.json(res.get("logic", {}))
                        rows.append({
                            "hs_code_new": res.get("hs_code_new",""),
                            "hs_code_old": res.get("hs_code_old",""),
                            "hs6": res.get("hs6",""),
                            "chapter": res.get("chapter",""),
                            "heading": res.get("heading",""),
                            "description_en": res.get("description_en",""),
                            "description_ar": res.get("description_ar",""),
                            "confidence": res.get("confidence",""),
                            "score": res.get("score",""),
                        })
                    st.download_button("Download results CSV", data=pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"), file_name="search_results.csv", mime="text/csv")
            except requests.RequestException as e:
                st.error(f"Search error: {e}")

    st.markdown("---")
    st.subheader("Quick code lookup")
    code = st.text_input("Enter HS code (any length, new/old/HS6):", value="")
    if st.button("Lookup code"):
        rr = requests.get(f"{API_BASE}/code_lookup", params={"code": code}, timeout=10)
        if rr.ok:
            data = rr.json().get("results", [])
            if not data:
                st.info("No exact/partial matches for that code.")
            else:
                for r1 in data:
                    st.write(f"- {r1.get('hs_code_new','')} | {r1.get('hs6','')} — {r1.get('description_en') or r1.get('description_ar','')}")

# ---------- Batch tab ----------
with tab2:
    st.subheader("Batch classify (CSV/Excel)")
    f = st.file_uploader("Upload file", type=["csv","xlsx","xls"])
    desc_col = st.text_input("Description column name (leave blank to auto-detect)")
    topk_b = st.number_input("Top K per row", 1, 10, 3, 1)
    include_logic = st.checkbox("Include simple rationale columns", value=True)

    if st.button("Run batch"):
        if not f:
            st.warning("Please upload a file.")
        else:
            try:
                if f.name.lower().endswith((".xlsx",".xls")):
                    df = pd.read_excel(f, dtype=str).fillna("")
                else:
                    df = pd.read_csv(f, dtype=str).fillna("")
                chosen = desc_col.strip() if desc_col.strip() else None
                if not chosen:
                    for c in df.columns:
                        if c.strip().lower() in ["description","product","item","goods","desc","details","longdescen","longdescar"]:
                            chosen = c; break
                if not chosen:
                    st.error(f"Could not detect description column. Columns: {list(df.columns)}")
                else:
                    rows = []
                    for _, row in df.iterrows():
                        qrow = str(row.get(chosen,"")).strip()
                        if not qrow:
                            rows.append({**row.to_dict(), "hs_code_new":"","hs_code_old":"","hs6":"","confidence":"","score":"","rationale":"Empty description"})
                            continue
                        payload = {"query": qrow, "top_k": int(topk_b)}
                        r = requests.post(f"{API_BASE}/search", json=payload, timeout=60)
                        best = {}
                        if r.ok:
                            res = r.json().get("results", [])
                            best = res[0] if res else {}
                        out = {**row.to_dict()}
                        out.update({
                            "hs_code_new": best.get("hs_code_new",""),
                            "hs_code_old": best.get("hs_code_old",""),
                            "hs6": best.get("hs6",""),
                            "confidence": best.get("confidence",""),
                            "score": best.get("score",""),
                        })
                        if include_logic:
                            out["matched_terms"] = "; ".join(best.get("logic",{}).get("matched_terms", [])) if best else ""
                            out["bm25_norm"] = best.get("logic",{}).get("bm25_norm","") if best else ""
                            out["fuzzy_norm"] = best.get("logic",{}).get("fuzzy_norm","") if best else ""
                        rows.append(out)
                    out_df = pd.DataFrame(rows)
                    st.success("Batch complete. Download CSV below.")
                    st.download_button("Download CSV", data=out_df.to_csv(index=False).encode("utf-8-sig"), file_name="classified_results.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Batch error: {e}")

# ---------- Document tab ----------
with tab3:
    st.subheader("Upload a document → Extract → Search")
    doc = st.file_uploader("Upload a product document (PDF, DOCX, or TXT)", type=["pdf","docx","txt"], key="docuploader")
    topk_d = st.number_input("Top K results", 1, 20, 5, 1, key="topkdoc")
    if doc is not None:
        text = extract_text_from_file(doc)
        if not text.strip():
            st.error("Could not extract text. If this is a scanned PDF (image), OCR isn't enabled in this build.")
        else:
            st.text_area("Extracted text (editable):", value=text[:5000], height=200, key="doc_text")
            kws = extract_keywords(text)
            st.write("**Detected keywords:**", ", ".join(kws) if kws else "(none)")
            query_seed = " ".join(kws[:10]) if kws else text[:200]
            q_edit = st.text_input("Search query (you can edit):", value=query_seed, key="doc_query")
            if st.button("Analyze & Search", type="primary"):
                try:
                    payload = {"query": q_edit, "top_k": int(topk_d)}
                    r = requests.post(f"{API_BASE}/search", json=payload, timeout=60)
                    r.raise_for_status()
                    data = r.json().get("results", [])
                    if not data:
                        st.info("No results. Try adding specific product attributes in the query.")
                    else:
                        for i, res in enumerate(data, 1):
                            with st.container(border=True):
                                st.text(f"#{i}  HS New: {res.get('hs_code_new','')}   HS Old: {res.get('hs_code_old','')}   HS6: {res.get('hs6','')}")
                                st.write("**Chapter:**", res.get("chapter",""), "  **Heading:**", res.get("heading",""))
                                if res.get("description_en"):
                                    st.write("**EN:**", res.get("description_en",""))
                                if res.get("description_ar"):
                                    st.write("**AR:**", res.get("description_ar",""))
                                st.write("**Confidence:**", res.get("confidence",""), "  **Score:**", res.get("score",0))
                                with st.expander("Logic (why this matched)"):
                                    st.json(res.get("logic", {}))
                except requests.RequestException as e:
                    st.error(f"Search error: {e}")

# ---------- Admin tab ----------
with tab4:
    st.subheader("Admin — synonyms & index")
    try:
        sr = requests.get(f"{API_BASE}/synonyms", timeout=10)
        csv_text = sr.json().get("csv","") if sr.ok else "term,expansion\n"
    except Exception:
        csv_text = "term,expansion\n"
    csv_edit = st.text_area("Edit synonyms.csv (term,expansion)", value=csv_text, height=180)
    cols = st.columns(3)
    with cols[0]:
        if st.button("Save synonyms"):
            ur = requests.post(f"{API_BASE}/update_synonyms", json=csv_edit, timeout=10)
            if ur.ok:
                st.success("Synonyms saved & index reloaded.")
            else:
                st.error(f"Save failed: {ur.status_code}")
    with cols[1]:
        if st.button("Reload index"):
            ur = requests.post(f"{API_BASE}/update_synonyms", json=csv_edit, timeout=10)
            if ur.ok: st.success("Index reloaded.")
    with cols[2]:
        st.info("Embeddings are optional: `pip install -r requirements-embeddings.txt` then tick the checkbox in Search.")
