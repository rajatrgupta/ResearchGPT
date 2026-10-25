import streamlit as st
import requests
import time
import os
import re
import markdown
import json
try:
    from fpdf import FPDF
except ImportError:
    FPDF = None
from datetime import datetime, timedelta

# ==========================================
# PAGE CONFIGURATION & METADATA
# ==========================================
st.set_page_config(
    page_title="ResearchGPT | Autonomous Multi-Agent Intelligence Engine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API Configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "https://researchgpt-backend-ygh8.onrender.com/api/v1/research")
BACKEND_HEALTH_URL = API_BASE_URL.replace("/api/v1/research", "/health")

def clean_html(raw_html: str) -> str:
    """Strips leading/trailing whitespace on every line to prevent Markdown code block triggers."""
    return "\n".join(line.strip() for line in raw_html.splitlines() if line.strip())

# ==========================================
# DYNAMIC TRENDING TOPICS ENGINE
# ==========================================
# Four research domains — we always pick exactly ONE topic per domain
# so cards are always diverse regardless of what's trending.
_DOMAIN_SLOTS = [
    {
        "key": "ai_tech",
        "tag": "🤖 AI & TECHNOLOGY",
        "query": "most searched AI machine learning research topics 2025 2026",
    },
    {
        "key": "biomedical",
        "tag": "🧬 BIOMEDICAL & HEALTH",
        "query": "most searched biomedical health medical research topics 2025 2026",
    },
    {
        "key": "energy_climate",
        "tag": "⚡ ENERGY & CLIMATE",
        "query": "most searched clean energy climate research topics 2025 2026",
    },
    {
        "key": "geopolitics_economy",
        "tag": "🌐 GEOPOLITICS & ECONOMY",
        "query": "most searched geopolitics economics global policy research topics 2025 2026",
    },
]

# Hardcoded evergreen fallbacks — one per domain slot (shown if Tavily fails or no key)
_FALLBACK_TOPICS = [
    {
        "tag": "🤖 AI & TECHNOLOGY",
        "title": "Reasoning vs. Retrieval: The Next Frontier of LLM Architecture",
        "text": "Analyze the architectural tradeoffs, benchmark performance, and real-world deployment outcomes of reasoning-native LLMs (o3, Gemini 2.5) versus retrieval-augmented generation pipelines in enterprise AI through 2026.",
        "directive": "Analyze the architectural tradeoffs, benchmark performance, and real-world deployment outcomes of reasoning-native LLMs (o3, Gemini 2.5) versus retrieval-augmented generation pipelines in enterprise AI through 2026.",
    },
    {
        "tag": "🧬 BIOMEDICAL & HEALTH",
        "title": "GLP-1 Receptor Agonists: Obesity, Cardiovascular Risk & the Metabolic Revolution",
        "text": "Synthesize clinical trial evidence, long-term cardiovascular outcomes, and global health economics of GLP-1 agonists (semaglutide, tirzepatide) in obesity and metabolic disease management.",
        "directive": "Synthesize clinical trial evidence, long-term cardiovascular outcomes, and global health economics of GLP-1 agonists (semaglutide, tirzepatide) in obesity and metabolic disease management.",
    },
    {
        "tag": "⚡ ENERGY & CLIMATE",
        "title": "Small Modular Reactors: The Nuclear Renaissance in the Energy Transition",
        "text": "Evaluate the economic viability, regulatory status, safety benchmarks, and grid-scale deployment timelines of Small Modular Reactors (SMRs) as a decarbonization pillar in OECD nations through 2035.",
        "directive": "Evaluate the economic viability, regulatory status, safety benchmarks, and grid-scale deployment timelines of Small Modular Reactors (SMRs) as a decarbonization pillar in OECD nations through 2035.",
    },
    {
        "tag": "🌐 GEOPOLITICS & ECONOMY",
        "title": "De-Dollarization: BRICS Currency Initiatives and the Shifting Global Reserve Order",
        "text": "Assess the structural drivers, geopolitical motivations, macroeconomic feasibility, and projected timeline of BRICS-led de-dollarization efforts and their impact on USD reserve currency dominance.",
        "directive": "Assess the structural drivers, geopolitical motivations, macroeconomic feasibility, and projected timeline of BRICS-led de-dollarization efforts and their impact on USD reserve currency dominance.",
    },
]

_TRENDING_CACHE_KEY = "_trending_topics_cache"
# 72-hour TTL = max ~40 Tavily calls/month even with frequent cold starts.
# Tavily free tier = 1,000 searches/month. Main usage is actual research runs
# (~5 searches per run). This keeps trending-topic overhead at <5% of quota.
_TRENDING_CACHE_TTL_HOURS = 72


def _parse_topic_from_tavily(result: dict, domain_tag: str) -> dict | None:
    """Extract a clean research topic card from a single Tavily search result."""
    title = (result.get("title") or "").strip()
    content = (result.get("content") or "").strip()
    if not title or len(title) < 10:
        return None
    # Build a concise but rich directive from title + snippet
    snippet = content[:220].rstrip() if content else title
    if not snippet.endswith((".","?","!")):
        snippet = snippet.rsplit(" ", 1)[0]  # trim to last full word
    directive = f"{title}. {snippet}" if snippet and snippet != title else title
    # Cap directive length
    if len(directive) > 400:
        directive = directive[:397] + "..."
    return {
        "tag": domain_tag,
        "title": title[:90],
        "text": snippet[:220],
        "directive": directive,
    }


@st.cache_data(ttl=_TRENDING_CACHE_TTL_HOURS * 3600, show_spinner=False)
def fetch_trending_topics() -> list[dict]:
    """
    Fetches 4 trending research topics — one per domain — via Tavily Search API.
    Results are cached for 72 hours (process-level, shared across all users).
    Falls back to hardcoded evergreen topics on any error (missing key,
    network failure, quota exhaustion, or HTTP 429 rate-limit).

    Monthly Tavily quota impact:
      - 4 calls per cold start, cached 72 hrs → ~40 calls/month worst-case.
      - Free tier = 1,000 searches/month; main usage is research runs (~5/run).
      - This feature consumes <5% of the free quota.
    """
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    if not tavily_key or tavily_key.startswith("your_"):
        # No key configured — silently show evergreen fallback, no API call
        return _FALLBACK_TOPICS

    topics: list[dict] = []
    for slot in _DOMAIN_SLOTS:
        fallback_idx = _DOMAIN_SLOTS.index(slot)
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": tavily_key,
                    "query": slot["query"],
                    "search_depth": "basic",   # cheapest tier — 1 credit each
                    "max_results": 3,           # reduced from 5 → fewer tokens processed
                    "include_answer": False,    # no LLM answer → no extra cost
                },
                timeout=8,
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                found = False
                for r in results:
                    parsed = _parse_topic_from_tavily(r, slot["tag"])
                    if parsed:
                        topics.append(parsed)
                        found = True
                        break  # one topic per domain — stop immediately
                if not found:
                    topics.append(_FALLBACK_TOPICS[fallback_idx])
            elif resp.status_code == 429:
                # Rate-limited — use fallback for ALL remaining slots to avoid
                # burning more quota on a session that's already throttled
                remaining = _FALLBACK_TOPICS[fallback_idx:]
                topics.extend(remaining)
                break
            else:
                topics.append(_FALLBACK_TOPICS[fallback_idx])
        except Exception:
            topics.append(_FALLBACK_TOPICS[fallback_idx])

    # Safety: if something went very wrong, return full fallback list
    return topics if len(topics) == 4 else _FALLBACK_TOPICS

# ==========================================
# ULTRA-PREMIUM TIER-1 DESIGN SYSTEM (CSS)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-main: #07090E;
        --bg-sidebar: #0C1019;
        --bg-card: #111726;
        --bg-card-hover: #162035;
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-accent: rgba(59, 130, 246, 0.35);
        --primary: #3B82F6;
        --primary-gradient: linear-gradient(135deg, #2563EB 0%, #3B82F6 50%, #60A5FA 100%);
        --accent-purple: #8B5CF6;
        --accent-emerald: #10B981;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
    }

    /* Force consistent global theme across light & dark browser modes */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    [data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }

    [data-testid="stSidebar"] * {
        color: var(--text-primary);
    }

    /* Custom Header / Badges */
    .brand-title {
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
        margin-bottom: 0.2rem;
    }

    .brand-subtitle {
        font-size: 0.78rem;
        color: var(--text-muted);
        font-weight: 500;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.1);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }

    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 8px #10B981;
        animation: pulse-green 2s infinite;
    }

    @keyframes pulse-green {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.9); }
    }

    /* Hero Canvas Elements */
    .hero-container {
        padding: 2.5rem 2rem 2rem 2rem;
        border-radius: 18px;
        background: radial-gradient(circle at 50% 0%, rgba(37, 99, 235, 0.15) 0%, rgba(17, 23, 38, 0.6) 70%, transparent 100%), var(--bg-card);
        border: 1px solid var(--border-subtle);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        background: rgba(59, 130, 246, 0.12);
        color: #60A5FA;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1rem;
        border: 1px solid rgba(59, 130, 246, 0.25);
    }

    .hero-heading {
        font-size: 2.3rem;
        font-weight: 800;
        line-height: 1.15;
        letter-spacing: -0.03em;
        margin-bottom: 0.8rem;
        color: #FFFFFF;
    }

    .hero-gradient-text {
        background: linear-gradient(135deg, #60A5FA 20%, #C084FC 80%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-desc {
        font-size: 1rem;
        line-height: 1.6;
        color: var(--text-secondary);
        max-width: 850px;
        margin-bottom: 1.5rem;
    }

    /* Directive & Feature Cards */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--text-primary);
        margin: 1.5rem 0 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .card-box {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 1.25rem;
        height: 100%;
        transition: all 0.25s ease;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .card-box:hover {
        border-color: var(--border-accent);
        transform: translateY(-2px);
        background: var(--bg-card-hover);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 0 15px rgba(59, 130, 246, 0.1);
    }

    .card-tag {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #60A5FA;
        margin-bottom: 0.5rem;
    }

    .card-title {
        font-size: 0.98rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.4rem;
        line-height: 1.35;
    }

    .card-text {
        font-size: 0.82rem;
        color: var(--text-secondary);
        line-height: 1.5;
        margin-bottom: 0.8rem;
    }

    /* Stage Execution Stepper */
    .stepper-container {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 12px;
        margin: 1.2rem 0;
    }

    @media (max-width: 900px) {
        .stepper-container {
            grid-template-columns: 1fr;
        }
    }

    .step-node {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1rem 0.85rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .step-node.completed {
        border-color: rgba(16, 185, 129, 0.4);
        background: rgba(16, 185, 129, 0.06);
    }

    .step-node.active {
        border-color: #3B82F6;
        background: rgba(59, 130, 246, 0.15);
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.35);
        transform: scale(1.02);
    }

    .step-icon {
        font-size: 1.3rem;
        margin-bottom: 0.35rem;
    }

    .step-label {
        font-size: 0.84rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.2rem;
    }

    .step-desc {
        font-size: 0.7rem;
        color: var(--text-muted);
        line-height: 1.3;
    }

    /* Live Telemetry Box */
    .telemetry-card {
        background: linear-gradient(180deg, #131A2B 0%, #0D1321 100%);
        border: 1px solid var(--border-accent);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
    }

    /* Streamlit Input / Widget Overrides */
    div[data-baseweb="textarea"] {
        background-color: #111827 !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="textarea"]:focus-within {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 1px #3B82F6 !important;
    }

    textarea {
        color: #F8FAFC !important;
        font-family: inherit !important;
    }

    /* Primary Launch Button */
    button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        padding: 0.6rem 1rem !important;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4) !important;
        transition: all 0.2s ease !important;
    }

    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6) !important;
    }

    /* Secondary / Preset Buttons */
    button[kind="secondary"] {
        background-color: #162035 !important;
        color: #E2E8F0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }

    button[kind="secondary"]:hover {
        background-color: #1E2D4A !important;
        border-color: rgba(59, 130, 246, 0.5) !important;
        color: #FFFFFF !important;
    }

    /* Metric Badges in Report */
    .metrics-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-bottom: 1.5rem;
    }

    .metric-badge {
        background: #111827;
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 8px 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .metric-badge-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        color: var(--text-muted);
        font-weight: 600;
    }

    .metric-badge-value {
        font-size: 0.88rem;
        font-weight: 700;
        color: #F8FAFC;
    }

    /* Report Markdown Viewer */
    .report-viewport {
        background-color: #0E1424;
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 2.5rem;
        font-size: 1rem;
        line-height: 1.7;
        color: #E2E8F0;
    }

    .report-viewport h1, .report-viewport h2, .report-viewport h3 {
        color: #FFFFFF;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-top: 1.8rem;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 0.4rem;
    }

    .report-viewport blockquote {
        border-left: 3px solid #3B82F6;
        padding-left: 1rem;
        color: #94A3B8;
        font-style: italic;
        background: rgba(59, 130, 246, 0.05);
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid var(--border-subtle);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        color: var(--text-secondary);
        font-weight: 600;
        font-size: 0.88rem;
        padding: 10px 18px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #111827 !important;
        color: #60A5FA !important;
        border-bottom: 2px solid #3B82F6 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# ROBUST EXECUTIVE PDF GENERATOR
# ==========================================
if FPDF:
    class ExecutiveReportPDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(120, 130, 145)
            self.cell(0, 8, "RESEARCHGPT INTEL ENGINE | AUTONOMOUS EXECUTIVE SYNTHESIS", border=0, align="L")
            self.cell(0, 8, datetime.now().strftime("%B %d, %Y"), border=0, align="R")
            self.ln(10)
            self.set_draw_color(220, 225, 235)
            self.set_line_width(0.3)
            self.line(10, 18, 200, 18)
            self.ln(4)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", size=8)
            self.set_text_color(150, 160, 175)
            self.cell(0, 10, f"Page {self.page_no()}/{{nb}}  |  Generated autonomously via LangGraph & Gemini 3.8", align="C")

def generate_pdf(md_text: str, query_title: str = "Executive Research Synthesis") -> bytes:
    """Generates an executive, highly polished PDF with sanitized unicode and styling."""
    if FPDF is None:
        raise RuntimeError("fpdf2 library not installed.")
        
    pdf = ExecutiveReportPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    
    # Clean typography sanitizer for fpdf2 standard latin-1 fonts
    clean_text = md_text
    replacements = {
        "‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
        "•": "-", "…": "...", "→": "->", "←": "<-", "≥": ">=", "≤": "<=",
        "≈": "~", "±": "+/-", "✔": "[x]", "✅": "[OK]", "🔬": "", "🧠": "",
        "🚀": "", "📄": "", "⚙️": "", "🛡️": "", "📊": "", "⚡": "", "🎯": "",
        "🧬": "", "🌐": "", "⏱️": "", "🆔": "", "✨": "", "🚨": "", "🛠️": "",
        "•": "-"
    }
    for orig, repl in replacements.items():
        clean_text = clean_text.replace(orig, repl)

    safe_clean_text = clean_text.encode('latin-1', 'replace').decode('latin-1')
        
    try:
        html = markdown.markdown(safe_clean_text, extensions=['tables', 'fenced_code'])
        html = re.sub(r'<h1>(.*?)</h1>', r'<h1 style="color:#0f172a; font-size:18pt;">\1</h1>', html)
        html = re.sub(r'<h2>(.*?)</h2>', r'<h2 style="color:#1e3a8a; font-size:14pt;">\1</h2>', html)
        html = re.sub(r'<h3>(.*?)</h3>', r'<h3 style="color:#2563eb; font-size:12pt;">\1</h3>', html)
        pdf.set_font("Helvetica", size=10)
        pdf.write_html(html)
    except Exception:
        pdf = ExecutiveReportPDF()
        pdf.alias_nb_pages()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        safe_title = query_title[:80].encode('latin-1', 'replace').decode('latin-1')
        pdf.cell(0, 10, safe_title, ln=True)
        pdf.set_font("Courier", size=9)
        pdf.multi_cell(0, 4.5, safe_clean_text)
        
    return bytes(pdf.output())

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "run_id" not in st.session_state:
    st.session_state.run_id = None
if "status" not in st.session_state:
    st.session_state.status = None
if "final_report" not in st.session_state:
    st.session_state.final_report = None
if "current_stage" not in st.session_state:
    st.session_state.current_stage = "initialized"
if "main_query_textarea" not in st.session_state:
    st.session_state.main_query_textarea = ""
if "active_query_text" not in st.session_state:
    st.session_state.active_query_text = ""
if "_pending_directive" not in st.session_state:
    st.session_state._pending_directive = None

# ==========================================
# CENTRALIZED RESEARCH EXECUTION ENGINE
# ==========================================
def launch_research(directive_text: str):
    """Centralized launcher: validates directive, dispatches to backend, and handles state transitions."""
    clean_text = directive_text.strip()
    if not clean_text:
        st.error("⚠️ Please provide a research directive.")
        return

    # NOTE: Do NOT set st.session_state.main_query_textarea here — it's bound
    # to a widget and will raise StreamlitWidgetAlreadyInstantiatedError.
    # Instead, store the active query separately so the UI can display it.
    st.session_state.active_query_text = clean_text
    st.session_state.run_id = None
    st.session_state.status = "pending"
    st.session_state.final_report = None
    st.session_state.current_stage = "initialized"

    with st.spinner("Dispatching DAG to ResearchGPT Orchestrator..."):
        try:
            response = requests.post(API_BASE_URL, json={"query": clean_text}, timeout=25)
            if response.status_code == 202:
                data = response.json()
                st.session_state.run_id = data["run_id"]
                st.session_state.status = data.get("status", "pending")
                st.session_state.current_stage = "initialized"
                st.rerun()
            else:
                st.error(f"Engine Dispatch Error ({response.status_code}): {response.text}")
                st.session_state.status = None
        except Exception as e:
            st.error(f"Backend Handshake Failed: {str(e)}")
            st.session_state.status = None

def load_directive_to_input(directive_text: str):
    """Loads a directive into the sidebar textarea for user customization."""
    # Use the pending directive mechanism so the widget key is set before the
    # widget is instantiated (avoids StreamlitWidgetAlreadyInstantiatedError).
    st.session_state._pending_directive = directive_text
    st.rerun()

# ==========================================
# STEPPER & TELEMETRY HTML BUILDERS (NO INDENTATION)
# ==========================================
def build_stepper_html(current_idx: int, stages_map: dict) -> str:
    agent_stages = [
        ("planner", stages_map["planner"]),
        ("search", stages_map["search"]),
        ("retriever", stages_map["retriever"]),
        ("critic", stages_map["critic"]),
        ("writer", stages_map["writer"])
    ]
    nodes = []
    for key, val in agent_stages:
        if current_idx > val["index"]:
            node_class = "step-node completed"
            icon_display = "✅"
        elif current_idx == val["index"]:
            node_class = "step-node active"
            icon_display = val["icon"]
        else:
            node_class = "step-node"
            icon_display = val["icon"]

        node_html = (
            f'<div class="{node_class}">'
            f'<div class="step-icon">{icon_display}</div>'
            f'<div class="step-label">{val["label"]}</div>'
            f'<div class="step-desc">{val["desc"]}</div>'
            f'</div>'
        )
        nodes.append(node_html)

    return f'<div class="stepper-container">{"".join(nodes)}</div>'

def build_telemetry_html(current_stage: str, stage_desc: str, elapsed_seconds: int, current_status: str) -> str:
    return (
        '<div class="telemetry-card">'
        '<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">'
        '<div>'
        '<span style="font-size: 0.72rem; text-transform: uppercase; color: #94A3B8; font-weight: 700; letter-spacing: 0.05em;">Active Orchestration Node</span>'
        '<div style="font-size: 1.1rem; font-weight: 800; color: #60A5FA; margin-top: 2px;">'
        f'<code>{current_stage.upper()} NODE</code> &nbsp;&bull;&nbsp; <span style="font-size: 0.88rem; color: #E2E8F0; font-weight: 500;">{stage_desc}</span>'
        '</div>'
        '</div>'
        '<div style="display: flex; gap: 14px;">'
        '<div style="background: rgba(255,255,255,0.05); padding: 6px 12px; border-radius: 8px; text-align: center;">'
        '<div style="font-size: 0.68rem; color: #64748B; text-transform: uppercase;">Elapsed</div>'
        f'<div style="font-size: 0.95rem; font-weight: 700; color: #F8FAFC;">{elapsed_seconds}s</div>'
        '</div>'
        '<div style="background: rgba(255,255,255,0.05); padding: 6px 12px; border-radius: 8px; text-align: center;">'
        '<div style="font-size: 0.68rem; color: #64748B; text-transform: uppercase;">Engine State</div>'
        f'<div style="font-size: 0.95rem; font-weight: 700; color: #34D399;">{current_status.upper()}</div>'
        '</div>'
        '</div>'
        '</div>'
        '</div>'
    )

# ==========================================
# SIDEBAR: ENTERPRISE COMMAND COCKPIT
# ==========================================
with st.sidebar:
    st.markdown(clean_html("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.5rem;">
            <div style="background: linear-gradient(135deg, #2563EB, #7C3AED); width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 20px; box-shadow: 0 4px 12px rgba(37,99,235,0.4);">🔬</div>
            <div>
                <div class="brand-title">ResearchGPT</div>
                <div class="brand-subtitle">Autonomous Multi-Agent Engine</div>
            </div>
        </div>
        <div style="margin-bottom: 1rem;">
            <span class="status-pill"><span class="status-dot"></span> LangGraph DAG Online</span>
        </div>
    """), unsafe_allow_html=True)
    
    st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 0.8rem 0 1.2rem 0;'>", unsafe_allow_html=True)
    
    st.markdown("**🎯 Research Directive**")

    # Apply any pending directive (from card click) before widget is instantiated
    if st.session_state._pending_directive is not None:
        st.session_state.main_query_textarea = st.session_state._pending_directive
        st.session_state._pending_directive = None

    st.text_area(
        label="Directive Input",
        placeholder="Enter your research topic, hypothesis, or analytical question...",
        height=130,
        label_visibility="collapsed",
        key="main_query_textarea"
    )

    with st.expander("⚙️ Synthesis Parameters", expanded=False):
        st.selectbox(
            "Synthesis Depth",
            ["Deep Academic Synthesis (5 Sub-Queries)", "Targeted Executive Briefing (3 Sub-Queries)"],
            index=0,
            key="param_depth"
        )
        st.selectbox(
            "Domain Filtering Tier",
            ["Tier-1 High-Trust (Academic, .edu, .gov, arXiv, Reuters)", "Global Verified Web"],
            index=0,
            key="param_domain"
        )
        st.caption("Active Model: Google Gemini 3.8 Flash with Key Rotation")
        st.caption("Vector Store: Qdrant Isolated Collection")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    
    is_processing = st.session_state.status in ["pending", "running"]
    
    col_run, col_reset = st.columns([3, 1])
    with col_run:
        launch_clicked = st.button(
            "🚀 Launch Synthesis",
            type="primary",
            use_container_width=True,
            disabled=is_processing
        )
    with col_reset:
        reset_clicked = st.button("🔄", help="Reset workspace and state", use_container_width=True, disabled=is_processing)

    if reset_clicked:
        st.session_state.run_id = None
        st.session_state.status = None
        st.session_state.final_report = None
        st.session_state.current_stage = "initialized"
        st.session_state.main_query_textarea = ""
        st.session_state.active_query_text = ""
        st.rerun()

    if launch_clicked:
        launch_research(st.session_state.main_query_textarea)

    st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 1.5rem 0 1rem 0;'>", unsafe_allow_html=True)
    st.markdown(clean_html("""
        <div style="font-size: 0.72rem; color: #64748B; line-height: 1.5;">
            <b>Architecture:</b> Cyclic LangGraph DAG<br>
            <b>Validation:</b> RAGAS Context Critique<br>
            <b>Vectors:</b> Qdrant Cosine Reranking<br>
            <b>Persistence:</b> PostgreSQL Engine
        </div>
    """), unsafe_allow_html=True)

# ==========================================
# MAIN WORKSPACE CANVAS
# ==========================================

# ----------------------------------------------------
# 1. EMPTY STATE: TIER-1 EXECUTIVE RESEARCH CANVAS
# ----------------------------------------------------
if not st.session_state.run_id:
    # Hero Showcase
    st.markdown(clean_html("""
        <div class="hero-container">
            <div class="hero-badge">Autonomous Multi-Agent Intelligence Engine</div>
            <div class="hero-heading">Deep Academic & Strategic Research <br><span class="hero-gradient-text">Engineered for Rigorous Synthesis.</span></div>
            <div class="hero-desc">
                ResearchGPT bypasses shallow single-shot LLM queries using a cyclic 5-agent LangGraph state machine. It decomposes hypotheses into targeted sub-queries, scrapes high-trust academic literature, vectorizes chunks in Qdrant, dynamically critiques factual relevance, and synthesizes publication-grade reports with inline citations.
            </div>
        </div>
    """), unsafe_allow_html=True)

    # Interactive Quick-Start Directives — dynamically fetched & cached 24hrs
    st.markdown("<div class='section-title'>🔥 Trending Research Directives</div>", unsafe_allow_html=True)
    st.caption("Refreshed daily via live search · 4 different domains · Click to instantly launch autonomous synthesis.")

    # Fetch topics (cached 24h; falls back to evergreen list on any error)
    trending_topics = fetch_trending_topics()

    # Render in a 2×2 grid: [0,1] on left column, [2,3] on right column
    p1, p2 = st.columns(2)
    col_map = {0: p1, 1: p1, 2: p2, 3: p2}

    for idx, topic in enumerate(trending_topics[:4]):
        with col_map[idx]:
            if idx in (1, 3):
                st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
            tag   = topic.get("tag", "📌 RESEARCH")
            title = topic.get("title", "Trending Research Topic")
            text  = topic.get("text", "")
            directive = topic.get("directive", title)

            st.markdown(clean_html(f"""
                <div class="card-box">
                    <div>
                        <div class="card-tag">{tag}</div>
                        <div class="card-title">{title}</div>
                        <div class="card-text">{text}</div>
                    </div>
                </div>
            """), unsafe_allow_html=True)
            if st.button("🚀 Launch Research", key=f"launch_preset_{idx+1}", type="primary", use_container_width=True):
                launch_research(directive)

    # Core Capabilities Grid
    st.markdown("<div class='section-title'>🛡️ Production Architecture & Quality Guardrails</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(clean_html("""
            <div class="card-box">
                <div style="font-size: 24px; margin-bottom: 8px;">🧭</div>
                <div class="card-title">LangGraph Cyclic DAG</div>
                <div class="card-text">Iterative state machine orchestrating 5 specialized agent nodes with conditional retry loops.</div>
            </div>
        """), unsafe_allow_html=True)
    with c2:
        st.markdown(clean_html("""
            <div class="card-box">
                <div style="font-size: 24px; margin-bottom: 8px;">🌐</div>
                <div class="card-title">Trust-Tier Web Filtering</div>
                <div class="card-text">Blocks spam and homework platforms; extracts exclusively from peer-reviewed, academic, and verified sources.</div>
            </div>
        """), unsafe_allow_html=True)
    with c3:
        st.markdown(clean_html("""
            <div class="card-box">
                <div style="font-size: 24px; margin-bottom: 8px;">🧬</div>
                <div class="card-title">Qdrant Vector Memory</div>
                <div class="card-text">High-dimensional embeddings with cosine similarity reranking for isolated, run-specific semantic memory.</div>
            </div>
        """), unsafe_allow_html=True)
    with c4:
        st.markdown(clean_html("""
            <div class="card-box">
                <div style="font-size: 24px; margin-bottom: 8px;">⚖️</div>
                <div class="card-title">RAGAS Critic Verification</div>
                <div class="card-text">Automated quality gatekeeper scoring retrieved evidence and triggering query rewrites if relevancy is below threshold.</div>
            </div>
        """), unsafe_allow_html=True)

# ----------------------------------------------------
# 2. RUNNING / POLLING STATE: LIVE MULTI-AGENT TELEMETRY
# ----------------------------------------------------
else:
    if st.session_state.status not in ["completed", "failed"]:
        active_title = st.session_state.active_query_text or "Active Research Directive"
        run_preview = st.session_state.run_id[:8] if st.session_state.run_id else "..."
        
        st.markdown(clean_html(f"""
            <div style="padding: 1.2rem 1.5rem; background: rgba(59, 130, 246, 0.08); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 14px; margin-bottom: 1.5rem;">
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <span class="status-pill" style="margin-bottom: 6px;"><span class="status-dot"></span> Autonomous Synthesis in Progress</span>
                        <div style="font-size: 1.3rem; font-weight: 700; color: #FFFFFF; margin-top: 4px;">{active_title}</div>
                    </div>
                    <div style="text-align: right;">
                        <code style="background: #111827; padding: 4px 8px; border-radius: 6px; font-size: 0.78rem; color: #94A3B8;">Run ID: {run_preview}...</code>
                    </div>
                </div>
            </div>
        """), unsafe_allow_html=True)

        stages_map = {
            "initialized": {"index": 0, "label": "Pipeline Init", "icon": "⚡", "desc": "Allocating LangGraph state & session memory", "pct": 10},
            "planner": {"index": 1, "label": "Planner Agent", "icon": "🧭", "desc": "Decomposing query into 5 sub-hypotheses", "pct": 30},
            "search": {"index": 2, "label": "Search Agent", "icon": "🌐", "desc": "Scraping verified sources & academic docs", "pct": 55},
            "retriever": {"index": 3, "label": "Retriever (Qdrant)", "icon": "🧬", "desc": "Embedding & cosine similarity reranking", "pct": 75},
            "critic": {"index": 4, "label": "Critic Gatekeeper", "icon": "⚖️", "desc": "Grading relevancy & triggering optimization", "pct": 88},
            "writer": {"index": 5, "label": "Writer Agent", "icon": "📑", "desc": "Synthesizing executive report with citations", "pct": 96},
            "completed": {"index": 6, "label": "Completed", "icon": "✅", "desc": "Final report generated & stored", "pct": 100}
        }

        progress_placeholder = st.empty()
        stepper_placeholder = st.empty()
        telemetry_placeholder = st.empty()

        start_poll_time = time.time()
        consecutive_errors = 0

        while st.session_state.status not in ["completed", "failed"]:
            try:
                poll_res = requests.get(f"{API_BASE_URL}/{st.session_state.run_id}", timeout=25)
                if poll_res.status_code == 200:
                    consecutive_errors = 0
                    poll_data = poll_res.json()
                    current_status = poll_data.get("status")
                    current_stage = poll_data.get("current_stage", "initialized").lower()
                    st.session_state.current_stage = current_stage
                    elapsed_seconds = int(time.time() - start_poll_time)

                    stage_info = stages_map.get(current_stage, stages_map["initialized"])
                    current_idx = stage_info["index"]

                    progress_placeholder.progress(stage_info["pct"], text=f"Stage: {stage_info['label']} ({stage_info['pct']}%)")
                    stepper_placeholder.markdown(build_stepper_html(current_idx, stages_map), unsafe_allow_html=True)
                    telemetry_placeholder.markdown(build_telemetry_html(current_stage, stage_info['desc'], elapsed_seconds, current_status), unsafe_allow_html=True)

                    if current_status in ["completed", "failed"]:
                        st.session_state.status = current_status
                        if current_status == "completed":
                            st.session_state.final_report = poll_data.get("final_report")
                        else:
                            st.session_state.error_message = poll_data.get("error_message")
                        st.rerun()
                        break
                elif poll_res.status_code == 429:
                    time.sleep(5)
                    continue
                time.sleep(3)
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors > 8:
                    st.error(f"Telemetry stream interrupted: {str(e)}")
                    break
                time.sleep(3.5)

    # ----------------------------------------------------
    # 3. COMPLETED STATE: PUBLICATION-GRADE DELIVERABLE VIEW
    # ----------------------------------------------------
    elif st.session_state.status == "completed" and st.session_state.final_report:
        report_text = st.session_state.final_report
        word_count = len(report_text.split())
        reading_time = max(1, round(word_count / 220))
        active_title = st.session_state.active_query_text or "Autonomous Research Report"
        run_preview = st.session_state.run_id[:8] if st.session_state.run_id else "..."

        st.markdown(clean_html(f"""
            <div style="padding: 1.5rem 1.8rem; background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(17, 24, 39, 0.8) 100%); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 16px; margin-bottom: 1.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                    <div>
                        <span class="status-pill" style="background: rgba(16, 185, 129, 0.2); color: #34D399; border-color: rgba(16, 185, 129, 0.4);">
                            <span class="status-dot"></span> Synthesis Complete & Grounded
                        </span>
                        <div style="font-size: 1.6rem; font-weight: 800; color: #FFFFFF; margin-top: 6px;">
                            {active_title}
                        </div>
                    </div>
                </div>
            </div>
        """), unsafe_allow_html=True)

        st.markdown(clean_html(f"""
            <div class="metrics-bar">
                <div class="metric-badge">
                    <span style="font-size: 16px;">⏱️</span>
                    <div>
                        <div class="metric-badge-label">Read Time</div>
                        <div class="metric-badge-value">~{reading_time} min</div>
                    </div>
                </div>
                <div class="metric-badge">
                    <span style="font-size: 16px;">📑</span>
                    <div>
                        <div class="metric-badge-label">Volume</div>
                        <div class="metric-badge-value">{word_count:,} words</div>
                    </div>
                </div>
                <div class="metric-badge">
                    <span style="font-size: 16px;">🛡️</span>
                    <div>
                        <div class="metric-badge-label">Critic Verification</div>
                        <div class="metric-badge-value" style="color: #34D399;">Passed (≥0.75)</div>
                    </div>
                </div>
                <div class="metric-badge">
                    <span style="font-size: 16px;">🧬</span>
                    <div>
                        <div class="metric-badge-label">Vector Reranking</div>
                        <div class="metric-badge-value" style="color: #60A5FA;">Qdrant Isolated</div>
                    </div>
                </div>
                <div class="metric-badge">
                    <span style="font-size: 16px;">🆔</span>
                    <div>
                        <div class="metric-badge-label">Run Trace ID</div>
                        <div class="metric-badge-value" style="font-family: monospace;">{run_preview}</div>
                    </div>
                </div>
            </div>
        """), unsafe_allow_html=True)

        action_col1, action_col2, action_col3 = st.columns([1.5, 1.5, 1.2])
        
        with action_col1:
            st.download_button(
                label="💾 Download Markdown (.md)",
                data=report_text,
                file_name=f"ResearchGPT_Report_{run_preview}.md",
                mime="text/markdown",
                use_container_width=True
            )
            
        with action_col2:
            try:
                pdf_bytes = generate_pdf(report_text, query_title=active_title)
                st.download_button(
                    label="📄 Download Executive PDF (.pdf)",
                    data=pdf_bytes,
                    file_name=f"ResearchGPT_Report_{run_preview}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"PDF compilation note: {str(e)}")

        with action_col3:
            if st.button("✨ New Research Directive", use_container_width=True):
                st.session_state.run_id = None
                st.session_state.status = None
                st.session_state.final_report = None
                st.session_state.current_stage = "initialized"
                st.session_state.main_query_textarea = ""
                st.session_state.active_query_text = ""
                st.rerun()

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        tab_report, tab_architecture, tab_telemetry = st.tabs([
            "📑 Executive Synthesis Deliverable",
            "🧠 Agent DAG Architecture & Traces",
            "⚙️ System Audit & Diagnostic Logs"
        ])

        with tab_report:
            st.markdown('<div class="report-viewport">', unsafe_allow_html=True)
            st.markdown(report_text)
            st.markdown("</div>", unsafe_allow_html=True)

        with tab_architecture:
            st.markdown("### Multi-Agent LangGraph Orchestration Pipeline")
            st.caption("Visual representation of the executed state machine that generated this report.")
            
            st.markdown("""
```mermaid
flowchart LR
    Start(["User Directive"]):::node --> Planner["1. Planner Node<br/><i>5 Sub-Queries</i>"]:::node
    Planner --> Search["2. Search Node<br/><i>Tavily / Academic</i>"]:::node
    Search --> Retriever["3. Retriever Node<br/><i>Qdrant Embeddings</i>"]:::node
    Retriever --> Critic{"4. Critic Node<br/><i>Ragas Evaluation</i>"}:::node
    Critic -->|"Score < 0.75"| Optimizer["Optimizer Node<br/><i>Query Rewrite</i>"]:::node
    Optimizer --> Search
    Critic -->|"Score >= 0.75"| Writer["5. Writer Node<br/><i>Synthesis & Citations</i>"]:::node
    Writer --> Deliverable(["Final Publication Report"]):::node

    classDef node fill:#111827,stroke:#3B82F6,stroke-width:1.5px,color:#F8FAFC;
```
            """)
            
            st.markdown("""
            #### Executed Agent Responsibilities:
            - **Planner Node:** Converted the unstructured directive into 5 orthogonal, academically rigorous research angles.
            - **Search Node:** Dispatched concurrent search requests filtered against low-trust homework/spam domains.
            - **Retriever Node (Qdrant):** Chunked scraped text, generated dense vector embeddings, and performed cosine similarity search.
            - **Critic Node:** Evaluated contextual grounding against hallucination thresholds.
            - **Writer Node:** Synthesized the verified context into a structured, highly cited executive deliverable.
            """)

        with tab_telemetry:
            st.markdown("### Execution Telemetry & Database Audit")
            st.code(f"""
RUN_IDENTIFIER: {st.session_state.run_id}
EXECUTION_STATUS: {st.session_state.status.upper()}
ORCHESTRATION_ENGINE: LangGraph Cyclic State Machine
PRIMARY_LLM: Google Gemini 3.8 Flash (Key Auto-Rotation Enabled)
VECTOR_DATABASE: Qdrant Engine (Isolated Namespace: {st.session_state.run_id})
RELATIONAL_STORE: PostgreSQL via SQLAlchemy & Alembic
TIMESTAMP: {datetime.utcnow().isoformat()}Z
            """, language="yaml")

    # ----------------------------------------------------
    # 4. FAILED STATE: COMPREHENSIVE DIAGNOSTICS & RETRY
    # ----------------------------------------------------
    elif st.session_state.status == "failed":
        st.markdown(clean_html(f"""
            <div style="padding: 1.5rem; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.35); border-radius: 14px; margin-bottom: 1.5rem;">
                <div style="font-size: 1.3rem; font-weight: 800; color: #EF4444; margin-bottom: 0.5rem;">
                    🚨 Autonomous Synthesis Pipeline Interrupted
                </div>
                <div style="color: #E2E8F0; font-size: 0.95rem; margin-bottom: 1rem;">
                    {st.session_state.get("error_message", "An unexpected exception occurred during agent graph execution.")}
                </div>
            </div>
        """), unsafe_allow_html=True)

        col_f1, col_f2 = st.columns([1, 3])
        with col_f1:
            if st.button("🔄 Restart Research Session", type="primary", use_container_width=True):
                st.session_state.run_id = None
                st.session_state.status = None
                st.session_state.final_report = None
                st.session_state.current_stage = "initialized"
                st.rerun()

        with st.expander("🛠️ System Diagnostics & Traceback", expanded=True):
            st.code(f"""
RUN_ID: {st.session_state.run_id}
STAGE: {st.session_state.get('current_stage')}
STATUS: FAILED
ERROR_DETAILS: {st.session_state.get('error_message')}
RECOMMENDATION: Verify that Google Gemini or Tavily search quotas are active and check server logs.
            """, language="yaml")
