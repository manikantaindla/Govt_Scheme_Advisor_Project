import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
import google.generativeai as genai

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except Exception:
    DDGS_AVAILABLE = False


# =========================================================
# CONFIG
# =========================================================
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
PDF_DIR = DATA_DIR / "sources" / "pdfs_raw"
LINKS_JSON = DATA_DIR / "sources" / "scheme_links.json"
INDEX_DIR = DATA_DIR / "index"
META_PARQUET = INDEX_DIR / "meta.parquet"

TOP_K = 6
MIN_SCORE = 0.08

INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan",
    "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal", "Delhi", "Jammu and Kashmir", "Ladakh",
    "Puducherry", "Chandigarh", "Andaman and Nicobar Islands",
    "Dadra and Nagar Haveli and Daman and Diu", "Lakshadweep",
]

SCHEME_OPTIONS = [
    "Education / Scholarship",
    "Pension",
    "Housing",
    "Home Loan",
    "Car Loan",
    "Marriage Assistance",
    "Farmer / Agriculture",
    "Women Welfare",
    "Disability Support",
    "Health / Medical",
    "Business / Self Employment",
    "Unemployment / Job Support",
    "Skill Development / Training",
    "Minority Welfare",
    "Student Hostel / Fee Reimbursement",
    "Old Age Support",
    "Widow Support",
    "Child Welfare",
    "Insurance",
    "General Welfare",
]

SCHEME_KEYWORDS = {
    "Education / Scholarship": "education scholarship fee reimbursement student school college hostel study support",
    "Pension": "pension widow old age disability social security pension monthly support",
    "Housing": "housing house home plot construction shelter scheme",
    "Home Loan": "home loan housing finance subsidy interest subsidy loan",
    "Car Loan": "vehicle loan car loan transport subsidy self employment vehicle assistance",
    "Marriage Assistance": "marriage assistance shaadi kalyana lakshmi wedding support",
    "Farmer / Agriculture": "farmer agriculture crop loan rythu kisan subsidy irrigation seeds",
    "Women Welfare": "women welfare girl women assistance financial support",
    "Disability Support": "disability divyang handicap assistance pension aid appliances",
    "Health / Medical": "health medical treatment hospital insurance aarogyasri healthcare support",
    "Business / Self Employment": "business self employment entrepreneurship subsidy loan startup support",
    "Unemployment / Job Support": "unemployment allowance job support livelihood support",
    "Skill Development / Training": "skill development training coaching employment courses free training",
    "Minority Welfare": "minority welfare scholarship support subsidy",
    "Student Hostel / Fee Reimbursement": "student hostel fee reimbursement pre matric post matric scholarship",
    "Old Age Support": "old age senior citizen pension support",
    "Widow Support": "widow pension widow assistance support",
    "Child Welfare": "child welfare child support girl child support",
    "Insurance": "insurance accident insurance life cover health cover",
    "General Welfare": "government welfare scheme assistance subsidy benefit",
}

load_dotenv()


# =========================================================
# HELPERS
# =========================================================
def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def chunk_text(text: str, max_chars: int = 1400, overlap: int = 200) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def parse_pdf(pdf_path: Path) -> List[Tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        txt = clean_text(txt)
        if txt:
            pages.append((i, txt))
    return pages


def expand_user_query(user_query: str) -> str:
    q = user_query.lower().strip()

    if any(word in q for word in ["educ", "study", "student", "scholarship", "college", "school", "fee"]):
        return user_query + " scholarship fee reimbursement student education support"

    if "pension" in q:
        return user_query + " widow old age disability pension"

    if any(word in q for word in ["marriage", "wedding", "shaadi", "kalyana"]):
        return user_query + " marriage assistance kalyana shaadi"

    if any(word in q for word in ["farmer", "agriculture", "crop", "rythu", "kisan"]):
        return user_query + " farmer agriculture crop support"

    if any(word in q for word in ["house", "housing", "home"]):
        return user_query + " housing house home scheme"

    if "loan" in q:
        return user_query + " loan subsidy finance assistance scheme"

    if any(word in q for word in ["health", "medical", "hospital"]):
        return user_query + " health medical treatment support insurance"

    return user_query


def is_state_matching(evidence: List[Dict], selected_state: str) -> bool:
    selected_state_l = (selected_state or "").lower()
    if not selected_state_l:
        return False

    alias_map = {
        "andhra pradesh": ["andhra pradesh", "ap"],
        "telangana": ["telangana", "tg"],
        "karnataka": ["karnataka"],
        "bihar": ["bihar"],
        "tamil nadu": ["tamil nadu", "tn"],
        "kerala": ["kerala"],
        "maharashtra": ["maharashtra"],
    }
    aliases = alias_map.get(selected_state_l, [selected_state_l])

    for e in evidence:
        text = (e.get("text", "") or "").lower()
        file_name = (e.get("file_name", "") or "").lower()
        blob = f"{text} {file_name}"
        if any(alias in blob for alias in aliases):
            return True

    return False


def build_user_query(
    scheme_type: str,
    state: str,
    category: str,
    age: int,
    income: int,
    extra_text: str,
) -> str:
    base_keywords = SCHEME_KEYWORDS.get(scheme_type, scheme_type)
    extra_text = (extra_text or "").strip()

    raw = f"{scheme_type} {base_keywords} {extra_text}".strip()
    expanded = expand_user_query(raw)

    return (
        f"{expanded} for {category} category in {state}, "
        f"age {age}, annual income {income}"
    )


def format_output(text: str) -> str:
    if not text:
        return ""

    # Normalize spaces
    text = text.replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)

    # Define keys
    keys = [
        "Scheme Name:",
        "Description:",
        "Who can apply:",
        "How to apply:",
        "Official Notice Link:",
        "Official Apply Link:",
        "Possible Scheme:",
    ]

    # Force proper line breaks BEFORE each key
    for key in keys:
        text = re.sub(
            rf"\s*{re.escape(key)}\s*",
            f"\n\n{key}\n",
            text,
            flags=re.IGNORECASE
        )

    # Clean extra spaces/newlines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text

def make_links_clickable(text: str) -> str:
    if not text:
        return ""

    url_pattern = r"(https?://[^\s\]\)]+)"

    def repl(match):
        url = match.group(1).rstrip(".,;)")
        return f"[{url}]({url})"

    return re.sub(url_pattern, repl, text)


# =========================================================
# GEMINI
# =========================================================
@st.cache_resource
def get_gemini_model():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


# =========================================================
# LOCAL PDF STORE
# =========================================================
def build_local_store():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No PDFs found in: {PDF_DIR}")

    rows = []
    for pdf in pdfs:
        doc_id = pdf.stem
        pages = parse_pdf(pdf)

        for page_no, page_text in pages:
            chunks = chunk_text(page_text)
            for chunk_no, ck in enumerate(chunks, start=1):
                rows.append(
                    {
                        "doc_id": doc_id,
                        "file_name": pdf.name,
                        "page_no": int(page_no),
                        "chunk_no": int(chunk_no),
                        "text": ck,
                    }
                )

    if not rows:
        raise RuntimeError("No extractable text found in PDFs.")

    df = pd.DataFrame(rows)
    df.to_parquet(META_PARQUET, index=False)
    return df


@st.cache_resource
def load_local_store():
    if META_PARQUET.exists():
        return pd.read_parquet(META_PARQUET)
    return None


def retrieve(query: str, top_k: int = TOP_K) -> List[Dict]:
    meta = load_local_store()
    if meta is None or meta.empty:
        return []

    texts = meta["text"].fillna("").astype(str).tolist()
    docs = texts + [query]

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(docs)

    doc_matrix = matrix[:-1]
    query_vec = matrix[-1]
    scores = cosine_similarity(query_vec, doc_matrix).flatten()
    top_ids = np.argsort(scores)[::-1][:top_k]

    out = []
    for idx in top_ids:
        row = meta.iloc[int(idx)].to_dict()
        out.append(
            {
                "score": float(scores[idx]),
                "doc_id": str(row.get("doc_id", "")),
                "file_name": str(row.get("file_name", "")),
                "page_no": int(row.get("page_no", 0)),
                "chunk_no": int(row.get("chunk_no", 0)),
                "text": str(row.get("text", "")),
            }
        )
    return out


# =========================================================
# OFFICIAL LINKS
# =========================================================
def load_scheme_links() -> List[Dict]:
    if not LINKS_JSON.exists():
        return []
    try:
        return json.loads(LINKS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return []


def match_links_from_evidence(evidence: List[Dict]) -> List[Dict]:
    links_db = load_scheme_links()
    if not links_db:
        return []

    ev_doc_ids = {e.get("doc_id", "").lower() for e in evidence if e.get("doc_id")}
    ev_files = {e.get("file_name", "").lower() for e in evidence if e.get("file_name")}

    matched = []
    for scheme in links_db:
        doc_ids = [d.lower() for d in scheme.get("doc_ids", [])]
        file_names = [f.lower() for f in scheme.get("file_names", [])]
        scheme_name = scheme.get("scheme_name", "").lower()

        found = False
        if doc_ids and any(d in ev_doc_ids for d in doc_ids):
            found = True
        elif file_names and any(f in ev_files for f in file_names):
            found = True
        elif scheme_name and any(scheme_name in f for f in ev_files):
            found = True

        if found:
            matched.append(scheme)

    return matched


# =========================================================
# FALLBACK SEARCH
# =========================================================
def fallback_search(query: str, state: str, category: str = "", max_results: int = 10) -> List[Dict]:
    if not DDGS_AVAILABLE:
        return []

    query = (query or "").strip()
    state = (state or "").strip()
    category = (category or "").strip()

    state_portal_map = {
        "Karnataka": ["karnataka.gov.in", "sevasindhu.karnataka.gov.in"],
        "Telangana": ["telangana.gov.in", "cgg.gov.in", "telanganaepass.cgg.gov.in"],
        "Andhra Pradesh": ["ap.gov.in", "gsws-nbm.ap.gov.in"],
        "Tamil Nadu": ["tn.gov.in"],
        "Kerala": ["kerala.gov.in"],
        "Maharashtra": ["maharashtra.gov.in"],
        "Odisha": ["odisha.gov.in"],
        "West Bengal": ["wb.gov.in"],
        "Delhi": ["delhi.gov.in"],
        "Bihar": ["bihar.gov.in", "state.bihar.gov.in"],
        "Uttar Pradesh": ["up.gov.in"],
        "Rajasthan": ["rajasthan.gov.in"],
        "Gujarat": ["gujarat.gov.in"],
        "Madhya Pradesh": ["mp.gov.in"],
    }

    extra_domains = state_portal_map.get(state, [])

    search_queries = [
        f"{query} {state} government scheme official",
        f"{query} {state} official portal",
        f"{query} {state} application form official",
        f"{query} {state} {category} site:gov.in",
        f"{query} {state} site:nic.in",
        f"{query} {state} site:services.india.gov.in",
        f"{query} {state} site:myscheme.gov.in",
    ]

    for domain in extra_domains:
        search_queries.append(f"{query} site:{domain}")

    results = []
    seen = set()

    try:
        with DDGS() as ddgs:
            for sq in search_queries:
                try:
                    found = ddgs.text(sq, max_results=max_results)
                except Exception:
                    continue

                for r in found:
                    url = r.get("href") or r.get("url") or ""
                    title = r.get("title", "") or ""
                    body = r.get("body", "") or r.get("snippet", "") or ""

                    if not url or url in seen:
                        continue
                    seen.add(url)

                    url_l = url.lower()
                    text_blob = f"{title} {body}".lower()

                    allowed_domains = [
                        ".gov.in",
                        ".nic.in",
                        "services.india.gov.in",
                        "myscheme.gov.in",
                        "india.gov.in",
                    ] + [d.lower() for d in extra_domains]

                    if not any(x in url_l for x in allowed_domains):
                        continue

                    if state.lower() not in text_blob and state.lower() not in url_l:
                        continue

                    results.append(
                        {
                            "scheme_name": title if title else "Possible Scheme",
                            "source_url": url,
                            "pdf_url": url if ".pdf" in url_l else "",
                            "apply_link": url if any(k in url_l for k in ["apply", "application", "login", "register", "form"]) else "",
                            "snippet": body,
                            "confidence": "medium",
                            "office_note": "Please verify details on the official government website.",
                        }
                    )

                if len(results) >= 3:
                    break

    except Exception:
        return []

    unique = []
    used = set()
    for r in results:
        if r["source_url"] not in used:
            unique.append(r)
            used.add(r["source_url"])

    return unique[:8]


# =========================================================
# LLM ANSWERS
# =========================================================
def llm_answer(profile: Dict, query: str, evidence: List[Dict], verified_links: List[Dict]) -> str:
    model = get_gemini_model()
    if model is None:
        return "❌ GEMINI_API_KEY missing in .env"

    ev_text = "\n\n".join(
        [f"[{e['file_name']} | page {e['page_no']}]\n{e['text']}" for e in evidence]
    )

    links_text = "\n".join(
        [
            f"Scheme: {x.get('scheme_name', '')}\n"
            f"Notice Links: {', '.join(x.get('source_links', []))}\n"
            f"Apply Link: {x.get('apply_link', '')}"
            for x in verified_links
        ]
    ) or "No verified links available."

    prompt = f"""
You are an Indian Government Scheme Advisor.

STRICT RULES:
- Use ONLY the given evidence and links.
- Do NOT guess or invent.
- Keep answer SHORT, CLEAR, and EASY.
- Use simple English.
- Each label must be on a separate line.
- Each value must start on the next line.
- Description must include amount if it is available in the evidence.
- If something is missing, write: Not clearly available.
-Give in news lines. Each sub heading should be in new line Ex- in 1st line Scheme name , in 2nd line Description etc.

Return exactly in this structure:

Scheme Name:
<answer>
next line
Description:
<what the scheme is and amount if available>
next line
Who can apply:
<answer>
next lines
How to apply:
<answer>
next line
Official Notice Link:
<link or Not clearly available>
next line
Official Apply Link:
<link or Not clearly available>

User:
{profile}

Query:
{query}

Evidence:
{ev_text}

Links:
{links_text}
"""

    try:
        resp = model.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception as e:
        return f"❌ Gemini error: {e}"


def llm_answer_from_fallback(profile: Dict, query: str, fallback_results: List[Dict]) -> str:
    model = get_gemini_model()
    if model is None:
        return "❌ GEMINI_API_KEY missing in .env"

    src_text = "\n\n".join(
        [
            f"Title: {x.get('scheme_name', '')}\n"
            f"Source: {x.get('source_url', '')}\n"
            f"PDF: {x.get('pdf_url', '')}\n"
            f"Apply: {x.get('apply_link', '')}\n"
            f"Snippet: {x.get('snippet', '')}"
            for x in fallback_results[:5]
        ]
    )

    prompt = f"""
You are an Indian Government Scheme Advisor.

STRICT RULES:
- Use ONLY the given search results.
- Do NOT guess beyond the given data.
- Keep answer SHORT, CLEAR, and EASY.
- Use simple English.
- Each label must be on a separate line.
- Each value must start on the next line.
- Description must include amount if it is visible in the search results.
- If unsure, write: Not clearly available
-Give in news lines. Each sub heading should be in new line Ex- in 1st line Scheme name , in 2nd line Description etc.

Return exactly in this structure:

Scheme Name:
<best matching scheme name>
next line
Description:
<what the scheme is and amount if available>
next line
Who can apply:
<answer>
next line
How to apply:
<answer>
next line
Official Notice Link:
<link or Not clearly available>
next line
Official Apply Link:
<link or Not clearly available>

User:
{profile}

Query:
{query}

Search Results:
{src_text}
"""

    try:
        resp = model.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception as e:
        return f"❌ Gemini error: {e}"


# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="Govt Scheme Advisor", layout="wide")
st.title("🇮🇳 Govt Scheme Advisor")
st.caption("Uses local PDFs first. If the selected state is not found in local data, it checks official government sources.")

st.subheader("Profile")
c1, c2, c3, c4 = st.columns(4)

default_state_index = INDIAN_STATES.index("Telangana") if "Telangana" in INDIAN_STATES else 0

with c1:
    state = st.selectbox("State", INDIAN_STATES, index=default_state_index)
with c2:
    age = st.number_input("Age", min_value=0, max_value=120, value=25)
with c3:
    income = st.number_input("Annual income (₹)", min_value=0, value=150000, step=1000)
with c4:
    category = st.selectbox("Category", ["General", "EWS", "OBC/BC", "SC", "ST", "Minority"])

d1, d2 = st.columns([1, 1])

with d1:
    scheme_type = st.selectbox("Scheme Type", SCHEME_OPTIONS, index=0)

with d2:
    language = st.selectbox("Language", ["English", "Telugu"])

extra_text = st.text_input(
    "Additional details (optional)",
    placeholder="Example: pre matric, girls, disabled, widow, hostel, loan subsidy",
)

col1, col2 = st.columns([1, 1])

with col1:
    search_btn = st.button("Search Schemes", type="primary")
with col2:
    build_btn = st.button("Build / Rebuild Local PDF Store")

if build_btn:
    try:
        with st.spinner("Reading PDFs and building local store..."):
            build_local_store()
            st.cache_resource.clear()
            st.success("Local PDF store built successfully.")
    except Exception as e:
        st.error(f"Build failed: {e}")

if search_btn:
    profile = {
        "state": state.strip(),
        "age": int(age),
        "annual_income": int(income),
        "category": category,
        "language": language,
        "scheme_type": scheme_type,
    }

    query = build_user_query(
        scheme_type=scheme_type,
        state=state,
        category=category,
        age=int(age),
        income=int(income),
        extra_text=extra_text,
    )

    with st.spinner("Searching local PDFs..."):
        evidence = retrieve(query, top_k=TOP_K)

    best_score = evidence[0]["score"] if evidence else 0.0
    state_match = is_state_matching(evidence, state)

    if evidence and best_score >= MIN_SCORE and state_match:
        matched_links = match_links_from_evidence(evidence)

        with st.spinner("Preparing clear answer..."):
            answer = llm_answer(
                profile=profile,
                query=query,
                evidence=evidence,
                verified_links=matched_links,
            )

        st.subheader("Answer")
        formatted = format_output(answer)
        formatted = make_links_clickable(formatted)
        st.markdown(formatted)

        if matched_links:
            st.subheader("Official Links")
            for item in matched_links:
                st.markdown(f"**{item.get('scheme_name', 'Scheme')}**")
                if item.get("apply_link"):
                    st.markdown(f"[Apply Here]({item['apply_link']})")
                for src in item.get("source_links", []):
                    st.markdown(f"[Official Source]({src})")
                if item.get("office_note"):
                    st.info(item["office_note"])

        with st.expander("Evidence used"):
            for e in evidence:
                st.markdown(
                    f"**{e['file_name']} | page {e['page_no']} | score {e['score']:.3f}**"
                )
                st.write(e["text"])

    else:
        if evidence and not state_match:
            st.info("Local PDF data does not match the selected state. Searching official sources instead.")
        else:
            st.warning("No strong local match found. Trying official web search...")

        with st.spinner("Searching official government sources..."):
            fallback_results = fallback_search(query, state, category)

        if not fallback_results:
            st.error("No useful official search results found.")
        else:
            with st.spinner("Preparing clear answer..."):
                answer = llm_answer_from_fallback(
                    profile=profile,
                    query=query,
                    fallback_results=fallback_results,
                )

            st.subheader("Answer")
            formatted = format_output(answer)
            formatted = make_links_clickable(formatted)
            st.markdown(formatted)

            st.subheader("Useful Official Links")
            for item in fallback_results:
                st.markdown(f"**{item.get('scheme_name', 'Possible Scheme')}**")
                if item.get("source_url"):
                    st.markdown(f"[Official Source]({item['source_url']})")
                if item.get("pdf_url"):
                    st.markdown(f"[PDF Link]({item['pdf_url']})")
                if item.get("apply_link"):
                    st.markdown(f"[Apply Here]({item['apply_link']})")
                if item.get("office_note"):
                    st.info(item["office_note"])
                if item.get("snippet"):
                    st.caption(item["snippet"])