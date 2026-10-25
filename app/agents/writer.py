"""
Writer Agent for DeepTrace
"""

import time
import re
import urllib.parse
from typing import Dict, Any, List, Tuple
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from app.graph.state import ResearchState
from app.core.llm import get_llm

KNOWN_PUBLISHERS = {
    "nature.com": "Nature",
    "ncbi.nlm.nih.gov": "NIH / PubMed",
    "pubmed.ncbi.nlm.nih.gov": "PubMed Central",
    "ieee.org": "IEEE Xplore",
    "arxiv.org": "arXiv",
    "who.int": "WHO",
    "reuters.com": "Reuters",
    "bloomberg.com": "Bloomberg",
    "wsj.com": "Wall Street Journal",
    "ft.com": "Financial Times",
    "sciencedirect.com": "ScienceDirect",
    "thelancet.com": "The Lancet",
    "nejm.org": "NEJM",
    "jamanetwork.com": "JAMA",
    "cell.com": "Cell",
    "pnas.org": "PNAS",
    "fda.gov": "US FDA",
    "cdc.gov": "CDC",
    "whitehouse.gov": "White House",
    "brookings.edu": "Brookings",
    "technologyreview.com": "MIT Tech Review",
    "economist.com": "The Economist",
    "acm.org": "ACM Digital Library",
    "springer.com": "Springer",
    "wiley.com": "Wiley Online Library",
    "biorxiv.org": "bioRxiv",
    "medrxiv.org": "medRxiv"
}

def extract_publisher_name(url: str, title: str = "") -> str:
    """Extract a clean, recognizable publisher or journal name from a URL."""
    if not url:
        return "Academic Source"
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        for known_domain, pub_name in KNOWN_PUBLISHERS.items():
            if domain == known_domain or domain.endswith("." + known_domain):
                return pub_name
        # Fallback: clean domain name like 'healthline.com' -> 'Healthline'
        parts = domain.split(".")
        if len(parts) >= 2:
            base = parts[-2].capitalize()
            if len(base) > 2:
                return base
        return domain
    except Exception:
        return "Verified Source"

def format_retrieved_documents(documents: list[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Formats retrieved vector chunks into an enriched, numbered research catalog for the LLM."""
    formatted_context = ""
    sources_meta = []
    
    for i, doc in enumerate(documents):
        idx = i + 1
        question = doc.get("question", "General Topic")
        title = (doc.get("title") or "Untitled Document").strip()
        snippet = (doc.get("snippet") or "").strip()
        source = doc.get("source") or doc.get("link") or ""
        publisher = extract_publisher_name(source, title)
        
        sources_meta.append({
            "idx": idx,
            "title": title,
            "url": source,
            "publisher": publisher
        })
        
        formatted_context += f"--- Source [{idx}]: {publisher} ---\n"
        formatted_context += f"Target Dimension: {question}\n"
        formatted_context += f"Title: {title}\n"
        formatted_context += f"Direct URL: {source}\n"
        formatted_context += f"Evidence Excerpt: {snippet}\n\n"

    formatted_context += "=== MANDATORY CITATION KEYS (USE THESE EXACT LINKS) ===\n"
    formatted_context += "Every metric, finding, or bullet point MUST cite one of these keys with its exact URL:\n"
    for s in sources_meta:
        if s["url"] and s["url"].startswith("http"):
            formatted_context += f"- Source [{s['idx']}]: [[{s['publisher']}]({s['url']})]  -> Full Title: \"{s['title']}\"\n"
    formatted_context += "========================================================\n\n"
    
    return formatted_context, sources_meta

def post_process_citations(report_text: str, sources_meta: List[Dict[str, Any]]) -> str:
    """
    Airtight post-processor:
    1. Converts any unlinked bracketed titles e.g. [Title] or [Research Report]
       into full clickable Markdown links [[Publisher](URL)] using sources_meta.
    2. Ensures the 'Key Sources & References' section has fully clickable links.
    """
    if not sources_meta:
        return report_text

    processed = report_text

    # Build title and publisher lookup maps
    url_by_idx = {s["idx"]: s for s in sources_meta if s.get("url")}
    
    # 1. Fix unlinked bracket references: [Some Title] -> [[Publisher](URL)]
    def replace_bracket(match):
        raw = match.group(0)
        inner = match.group(1).strip()
        
        # Don't touch if it's already a markdown link [text](url) or starts with http or is bold
        if inner.startswith("http") or inner.startswith("**"):
            return raw

        # Check if inner matches any Source index like "1" or "Source 1" or "[1]"
        idx_match = re.search(r'\b(?:Source\s*)?(\d{1,2})\b', inner, re.IGNORECASE)
        if idx_match:
            idx = int(idx_match.group(1))
            if idx in url_by_idx:
                s = url_by_idx[idx]
                return f"[[{s['publisher']}]({s['url']})]"

        if len(inner) < 3:
            return raw
                
        # Check against titles or publishers
        inner_clean = inner.lower()
        best_match = None
        for s in sources_meta:
            title_clean = s["title"].lower()
            pub_clean = s["publisher"].lower()
            if not s["url"]:
                continue
            if inner_clean in title_clean or title_clean in inner_clean or inner_clean == pub_clean:
                best_match = s
                break
            # Word overlap match
            inner_words = set(re.findall(r'\w{4,}', inner_clean))
            title_words = set(re.findall(r'\w{4,}', title_clean))
            if inner_words and len(inner_words & title_words) >= max(2, len(inner_words) * 0.5):
                best_match = s
                break
                
        if best_match:
            return f"[[{best_match['publisher']}]({best_match['url']})]"
            
        return raw

    # Match square brackets that are NOT followed by '('
    processed = re.sub(r'(?<!\!)\[([^\[\]]+)\](?!\()', replace_bracket, processed)

    # 2. Fix 'Key Sources & References' section if titles were listed without hyperlinks
    ref_header = "## Key Sources & References"
    if ref_header in processed:
        parts = processed.split(ref_header)
        body = parts[0]
        refs_section = parts[1]
        
        # If references lines lack hyperlinks, reconstruct them cleanly
        if "(" not in refs_section or "http" not in refs_section:
            clean_refs = f"\n\n"
            for s in sources_meta:
                if s["url"] and s["url"].startswith("http"):
                    clean_refs += f"{s['idx']}. [[{s['publisher']}] {s['title']}]({s['url']})\n"
            processed = body + ref_header + clean_refs

    return processed

def writer_node(state: ResearchState) -> Dict[str, Any]:
    query = state.get("query", "")
    retrieved_documents = state.get("retrieved_documents", [])
    degradation_mode = state.get("degradation_mode", False)
    
    if degradation_mode:
        fallback_report = f"# Research Report: {query}\n\n"
        fallback_report += "## Insufficient Data / Exhausted Research Limits\n"
        fallback_report += "The autonomous research system exhausted its maximum retry loops without finding enough high-quality context to meet the minimum verification threshold. The generated report has been aborted to prevent hallucinations."
        
        return {
            "report": fallback_report,
            "status": "writing_degraded"
        }

    if not retrieved_documents:
        fallback_report = f"# Research Report: {query}\n\n"
        fallback_report += "## Error\n"
        fallback_report += "Unfortunately, the retrieval system was unable to find highly relevant documents for this query. The report generation was aborted to prevent hallucination."
        
        return {
            "report": fallback_report,
            "status": "writing_failed",
            "errors": ["Writer received empty retrieved_documents list."]
        }

    context_str, sources_meta = format_retrieved_documents(retrieved_documents)

    prompt = PromptTemplate(
        template="""You are a Principal AI Scientist and Senior Technical Consultant (combining deep academic rigor with executive-level strategic clarity).
Current Reference Year: 2026. (Treat 2024-2025 as recent historical baseline, 2026 as current state, and 2027-2030 as projection horizons).

Your task is to synthesize the retrieved research findings into an exhaustive, highly dense, analytical, and professional Markdown research report.

CRITICAL RULES - READ CAREFULLY:
1. MANDATORY INLINE HYPERLINK CITATIONS:
   Every factual claim, metric, bullet point, and comparative finding MUST end with an inline clickable Markdown hyperlink citation.
   STRICT SYNTAX: `[[Publisher Name](Exact_URL)]` or `[[Source #[idx]: Publisher](Exact_URL)]`.
   Example format:
   - **Diagnostic Sensitivity & Triaging:** In a multi-center trial (N=1,420), AI-assisted diagnostic screening improved sensitivity for non-small cell lung cancer from 74.2% to 88.6% [[NIH / PubMed](https://www.ncbi.nlm.nih.gov/...)] while reducing turnaround time by 32% [[The Lancet](https://www.thelancet.com/...)].
   CRITICAL: Never output raw brackets like `[Title]` or `[Research Report]` without the `(URL)` markdown link. Every citation must be a clickable hyperlink.

2. MAXIMUM EMPIRICAL SPECIFICITY (NO FLUFF OR SURFACE BULLETS):
   - Zero tolerance for generic filler sentences (e.g., "AI is transforming healthcare", "There are many opportunities and challenges"). Start directly with high-impact data.
   - EVERY bullet point MUST contain concrete empirical data, specific percentages, named institutions/hospitals, sample sizes (N=...), dollar amounts ($ USD), sensitivity/specificity benchmarks, error rates, or specific regulatory frameworks (e.g., FDA 510(k), EU AI Act Tier-2, ISO 42001, HIPAA BAA).
   - If comparing approaches, use dense structured contrast bullets with bold inline headers.

3. THEMATIC COVERAGE & ISOLATION:
   - Organize into 3 to 4 distinct H2 sections covering the deepest technical, empirical, economic, and regulatory dimensions.
   - Never repeat concepts across sections. Group related points so each section provides strictly net-new depth.
   - Use bolded inline prefixes for every bullet point (e.g., `- **Algorithmic Latency & Throughput:** ...`, `- **Capital Expenditure vs. Net ROI:** ...`).

4. EXECUTIVE SYNTHESIS (MANDATORY HIGH-CONVICTION TAKEAWAY):
   - You must ALWAYS conclude with a section titled "## Executive Synthesis".
   - Do NOT write generic boilerplate. Write 2 sharp, analytical paragraphs:
     Paragraph 1: The core architectural, clinical, or empirical bottleneck defining this domain right now in 2026.
     Paragraph 2: The high-stakes strategic roadmap, investment priorities, and operational imperatives for decision-makers over the next 24-36 months.

5. KEY SOURCES & REFERENCES (MANDATORY NUMBERED LIST):
   - You must ALWAYS append a final "## Key Sources & References" section.
   - Every entry MUST be a full clickable Markdown hyperlink in this format:
     1. [[Publisher/Organization] Full Article Title](Exact_URL) - Brief 1-sentence note of what empirical data this source provided.

Output the final Markdown report directly. Do NOT wrap it in JSON.

User Query: {query}

--- RESEARCH FINDINGS & SOURCES ---
{context}
--- END FINDINGS ---

Write the comprehensive research report now:
""",
        input_variables=["query", "context"]
    )

    llm_config = get_llm()
    writer_llm = llm_config["llm"]
    
    chain = prompt | writer_llm

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            start_time = time.time()
            if attempt == 1:
                print("[Writer] START LLM report synthesis...")
            else:
                print(f"[Writer] RETRY {attempt}/{max_retries} LLM report synthesis...")
            
            result = chain.invoke({
                "query": query,
                "context": context_str
            })
            
            raw_content = result.content
            if isinstance(raw_content, list):
                final_content = "\n".join([chunk.get("text", "") if isinstance(chunk, dict) else str(chunk) for chunk in raw_content])
            elif not isinstance(raw_content, str):
                final_content = str(raw_content)
            else:
                final_content = raw_content

            # Apply post-processing to guarantee 100% hyperlink citation integrity
            final_content = post_process_citations(final_content, sources_meta)

            elapsed = time.time() - start_time
            print(f"[Writer] COMPLETE LLM report synthesis in {elapsed:.2f}s")
            
            return {
                "report": final_content,
                "status": "writing_complete"
            }
            
        except Exception as e:
            error_str = str(e).lower()
            is_retryable = "503" in error_str or "unavailable" in error_str or "connection" in error_str or "timeout" in error_str or "429" in error_str or "exhausted" in error_str
            
            if is_retryable and attempt < max_retries:
                wait_time = 10 * attempt # progressive backoff for rate limits
                print(f"[Writer] Warning: LLM call failed with retryable error (attempt {attempt}/{max_retries}). Waiting {wait_time}s... Error: {str(e)}")
                time.sleep(wait_time)
                continue
                
            elapsed = time.time() - start_time if 'start_time' in locals() else 0.0
            error_msg = f"Writer failed to generate report in {elapsed:.2f}s: {str(e)}"
            print(f"[Writer] FAILED LLM report synthesis after {attempt} attempts: {error_msg}")
            return {
                "errors": [error_msg],
                "status": "writing_failed",
                "report": "An error occurred during report generation."
            }
