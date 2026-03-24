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

    text = text.replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)

    keys = [
        "Scheme Name:",
        "Description:",
        "Who can apply:",
        "How to apply:",
        "Official Notice Link:",
        "Official Apply Link:",
        "Possible Scheme:",
    ]

    for key in keys:
        text = re.sub(
            rf"\s*{re.escape(key)}\s*",
            f"\n\n{key}\n",
            text,
            flags=re.IGNORECASE
        )

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def make_links_clickable(text: str) -> str:
    if not text:
        return ""

    url_pattern = r"(https?://[^\s<>\]\)]+)"

    def repl(match):
        url = match.group(1).rstrip(".,;)")
        return f'<a href="{url}" target="_blank">{url}</a>'

    return re.sub(url_pattern, repl, text)


def format_html_text(text: str) -> str:
    if not text:
        return "Not clearly available"
    text = text.replace("\n", "<br>")
    text = make_links_clickable(text)
    return text


def parse_answer_sections(text: str) -> Dict[str, str]:
    keys = [
        "Scheme Name",
        "Description",
        "Who can apply",
        "How to apply",
        "Official Notice Link",
        "Official Apply Link",
        "Possible Scheme",
    ]

    sections = {key: "Not clearly available" for key in keys}
    if not text:
        return sections

    normalized = format_output(text)

    pattern = r"(Scheme Name|Description|Who can apply|How to apply|Official Notice Link|Official Apply Link|Possible Scheme):\s*\n(.*?)(?=\n(?:Scheme Name|Description|Who can apply|How to apply|Official Notice Link|Official Apply Link|Possible Scheme):|\Z)"
    matches = re.findall(pattern, normalized, flags=re.DOTALL | re.IGNORECASE)

    for key, value in matches:
        proper_key = next((k for k in keys if k.lower() == key.lower()), key)
        clean_value = value.strip()
        if clean_value:
            sections[proper_key] = clean_value

    return sections


def render_answer_card(answer_text: str):
    sections = parse_answer_sections(answer_text)

    scheme_name = sections.get("Scheme Name", "Not clearly available")
    description = format_html_text(sections.get("Description", "Not clearly available"))
    who_can_apply = format_html_text(sections.get("Who can apply", "Not clearly available"))
    how_to_apply = format_html_text(sections.get("How to apply", "Not clearly available"))
    notice_link = format_html_text(sections.get("Official Notice Link", "Not clearly available"))
    apply_link = format_html_text(sections.get("Official Apply Link", "Not clearly available"))

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-badge">Scheme Result</div>
            <div class="result-title">{scheme_name}</div>
            <div class="result-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-card-title">Who can apply</div>
                <div class="info-card-value">{who_can_apply}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-card-title">How to apply</div>
                <div class="info-card-value">{how_to_apply}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    col3, col4 = st.columns(2)

    with col3:
        st.markdown(
            f"""
            <div class="link-card">
                <div class="info-card-title">Official Notice Link</div>
                <div class="info-card-value">{notice_link}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="link-card">
                <div class="info-card-title">Official Apply Link</div>
                <div class="info-card-value">{apply_link}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# GEMINI
# =========================================================
@st.cache_resource
def get_gemini_model():
    api_key = st.secrets.get("GEMINI_API_KEY", "").strip()

    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        st.error("No Gemini API key found.")
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
- VERY IMPORTANT:
  - Each label MUST be on a new line
  - Each value MUST be on the next line
  - Do NOT write everything in one paragraph
  - Follow exact line-by-line format strictly
- Description must include amount if it is available in the evidence.
- If something is missing, write: Not clearly available.

Return exactly in this structure:

Scheme Name:
<answer>

Description:
<what the scheme is and amount if available>

Who can apply:
<answer>

How to apply:
<answer>

Official Notice Link:
<link or Not clearly available>

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
- VERY IMPORTANT:
  - Each label MUST be on a new line
  - Each value MUST be on the next line
  - Do NOT write everything in one paragraph
  - Follow exact line-by-line format strictly
- Description must include amount if it is visible in the search results.
- If unsure, write: Not clearly available

Return exactly in this structure:

Scheme Name:
<best matching scheme name>

Description:
<what the scheme is and amount if available>

Who can apply:
<answer>

How to apply:
<answer>

Official Notice Link:
<link or Not clearly available>

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
st.set_page_config(
    page_title="Govt Scheme Advisor",
    page_icon="🇮🇳",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
        color: #e5e7eb;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .hero-section {
        background: linear-gradient(135deg, #111827, #1d4ed8);
        color: white;
        padding: 30px 34px;
        border-radius: 22px;
        margin-bottom: 22px;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.35);
        border: 1px solid rgba(255,255,255,0.08);
    }

    .hero-title {
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 15px;
        line-height: 1.7;
        color: rgba(255,255,255,0.92);
    }

    .section-heading {
        font-size: 13px;
        font-weight: 800;
        color: #60a5fa;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 8px;
        margin-bottom: 10px;
    }

    .panel-box {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
    }

    .tip-text {
        font-size: 13px;
        color: #9ca3af;
        margin-top: 8px;
        line-height: 1.6;
    }

    .result-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 20px;
        padding: 22px;
        margin-bottom: 16px;
        box-shadow: 0 10px 26px rgba(0, 0, 0, 0.28);
    }

    .result-badge {
        display: inline-block;
        background: rgba(37, 99, 235, 0.16);
        color: #93c5fd;
        font-size: 12px;
        font-weight: 700;
        padding: 6px 10px;
        border-radius: 999px;
        margin-bottom: 12px;
        border: 1px solid rgba(96, 165, 250, 0.18);
    }

    .result-title {
        font-size: 27px;
        font-weight: 800;
        color: #f9fafb;
        margin-bottom: 12px;
        line-height: 1.3;
    }

    .result-description {
        font-size: 15px;
        color: #d1d5db;
        line-height: 1.8;
        white-space: normal;
        word-break: break-word;
    }

    .info-card, .link-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 14px;
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.22);
        min-height: 180px;
    }

    .info-card-title {
        font-size: 13px;
        font-weight: 800;
        color: #60a5fa;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 10px;
    }

    .info-card-value {
        font-size: 15px;
        color: #e5e7eb;
        line-height: 1.8;
        white-space: normal;
        word-break: break-word;
    }

    .info-card-value a,
    .result-description a {
        color: #93c5fd !important;
        text-decoration: none;
        font-weight: 600;
    }

    .info-card-value a:hover,
    .result-description a:hover {
        color: #bfdbfe !important;
        text-decoration: underline;
    }

    .stButton > button {
        height: 48px;
        border-radius: 14px;
        font-size: 15px;
        font-weight: 700;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        box-shadow: 0 10px 22px rgba(37, 99, 235, 0.28);
    }

    .stButton > button:not([kind="primary"]) {
        background: #1f2937;
        color: #e5e7eb;
        border: 1px solid #374151;
    }

    .stTextInput label,
    .stNumberInput label,
    .stSelectbox label {
        color: #d1d5db !important;
        font-weight: 600;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {
        border-radius: 12px !important;
        background-color: #0f172a !important;
        color: #f9fafb !important;
        border: 1px solid #374151 !important;
    }

    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        border-radius: 12px !important;
        background-color: #0f172a !important;
        color: #f9fafb !important;
        border: 1px solid #374151 !important;
    }

    div[data-testid="stExpander"] {
        background: #111827 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 16px !important;
        overflow: hidden;
    }

    div[data-testid="stExpander"] summary {
        color: #e5e7eb !important;
        font-weight: 700;
    }

    [data-testid="stAlert"] {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-section">
        <div class="hero-title">Govt Scheme Advisor</div>
        <div class="hero-subtitle">
            Discover relevant government schemes using local PDF evidence first,
            then official government sources when required.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="section-heading">Applicant Profile</div>', unsafe_allow_html=True)
st.markdown('<div class="panel-box">', unsafe_allow_html=True)

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

d1, d2 = st.columns(2)

with d1:
    scheme_type = st.selectbox("Scheme Type", SCHEME_OPTIONS, index=0)

with d2:
    language = st.selectbox("Language", ["English", "Telugu"])

extra_text = st.text_input(
    "Additional details",
    placeholder="Example: widow, hostel, pre matric, disability, girls, minority",
)

st.markdown(
    '<div class="tip-text">Add a few specific details to improve scheme matching quality.</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-heading">Actions</div>', unsafe_allow_html=True)
st.markdown('<div class="panel-box">', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    search_btn = st.button("Search Schemes", type="primary", use_container_width=True)

with col2:
    build_btn = st.button("Build / Refresh Local PDF Store", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

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

        st.markdown('<div class="section-heading">Search Result</div>', unsafe_allow_html=True)
        render_answer_card(answer)

        if matched_links:
            st.markdown('<div class="section-heading">Official Links</div>', unsafe_allow_html=True)
            for item in matched_links:
                st.markdown(f"**{item.get('scheme_name', 'Scheme')}**")
                if item.get("apply_link"):
                    st.markdown(f"[Apply Here]({item['apply_link']})")
                for src in item.get("source_links", []):
                    st.markdown(f"[Official Source]({src})")
                if item.get("office_note"):
                    st.info(item["office_note"])

        with st.expander("Evidence Used"):
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

            st.markdown('<div class="section-heading">Search Result</div>', unsafe_allow_html=True)
            render_answer_card(answer)

            st.markdown('<div class="section-heading">Useful Official Links</div>', unsafe_allow_html=True)
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
