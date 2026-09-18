"""
Retriever Agent for DeepTrace

This module defines the Retriever Node. Its responsibility is to bridge the gap
between the raw data gathered by the Search Agent and the synthesis performed
by the Writer Agent. It commits the raw search results to the Qdrant Vector DB
for long-term memory, and then immediately performs a semantic similarity search
to extract only the top most relevant chunks to answer the user's overarching query.

Phase 4A Update:
Implements Trust-Tier Reranking Pipeline. Retrieves an over-fetched pool of
documents, applies Domain Authority multipliers, deduplicates exact URLs, and
enforces domain diversity.

Phase 4B Update:
Reads run_id from ResearchState and passes it to store_documents() and
search_similar() to enable run-level Qdrant isolation. Validates that run_id
is present before retrieval rather than silently querying the full collection.
"""

import time
import urllib.parse
from typing import Dict, Any
from app.graph.state import ResearchState
from app.vectorstore.qdrant_store import store_documents, search_similar

# ==========================================
# PHASE 4A: TRUST-TIER RERANKING CONSTANTS
# ==========================================
RERANK_FETCH_LIMIT = 20
FINAL_RETRIEVAL_LIMIT = 5
MAX_CHUNKS_PER_DOMAIN = 2

# TLDs get a direct endswith check. Specific domains get an exact or subdomain check.
HIGH_TRUST_TLDS = [".gov", ".edu", ".mil"]
HIGH_TRUST_DOMAINS = ["who.int", "nature.com", "ncbi.nlm.nih.gov", "ieee.org"]
MEDIUM_TRUST_DOMAINS = ["reuters.com", "bbc.co.uk", "apnews.com", "wsj.com", "bloomberg.com", "nytimes.com"]
LOW_TRUST_DOMAINS = ["reddit.com", "quora.com", "medium.com", "twitter.com", "yahoo.com"]


def get_trust_multiplier(url: str) -> tuple[float, str]:
    """
    Parses the domain from a URL and returns the appropriate
    Trust-Tier multiplier and tier name.
    Strictly checks domain or subdomain matches to prevent spoofing.
    """
    if not url:
        return 1.0, "DEFAULT_TRUST"

    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]

        # 1. High Trust TLD Check
        for tld in HIGH_TRUST_TLDS:
            if domain.endswith(tld):
                return 1.5, "HIGH_TRUST"

        # 2. High Trust Specific Domains
        for d in HIGH_TRUST_DOMAINS:
            if domain == d or domain.endswith("." + d):
                return 1.5, "HIGH_TRUST"

        # 3. Medium Trust Specific Domains
        for d in MEDIUM_TRUST_DOMAINS:
            if domain == d or domain.endswith("." + d):
                return 1.2, "MEDIUM_TRUST"

        # 4. Low Trust Specific Domains
        for d in LOW_TRUST_DOMAINS:
            if domain == d or domain.endswith("." + d):
                return 0.5, "LOW_TRUST"

        return 1.0, "DEFAULT_TRUST"

    except Exception:
        return 1.0, "DEFAULT_TRUST"

def retriever_node(state: ResearchState) -> Dict[str, Any]:
    """
    Executes the vector storage, retrieval, and reranking phase.

    Phase 4B: Reads run_id from state to stamp stored documents and filter
    retrieval. Fails with a clear error if run_id is absent rather than
    falling back to unfiltered global collection search.

    Args:
        state: The current ResearchState containing 'query', 'search_results',
               'run_id', and 'iteration_count'.

    Returns:
        A dictionary containing the semantically filtered and reranked
        'retrieved_documents', a 'status' indicator, and optionally 'errors'.
    """

    query = state.get("query", "")
    search_results = state.get("search_results", [])

    # Phase 4B: Read and validate run_id. The production pipeline must always
    # provide a run_id. If it is missing, fail fast with a clear error rather
    # than silently querying the entire unfiltered collection.
    run_id = state.get("run_id", "")
    if not run_id:
        return {
            "retrieved_documents": [],
            "errors": [
                "Retriever Node: 'run_id' is missing from ResearchState. "
                "Cannot perform run-isolated retrieval. Ensure main() generates "
                "a run_id and includes it in the initial state."
            ],
            "status": "retrieval_failed"
        }

    # cycle_number is the current iteration_count at time of storage.
    # This is informational metadata only — retrieval is NOT filtered by cycle.
    # All cycles within the same run remain eligible for retrieval.
    cycle_number = state.get("iteration_count", 0)

    # Defensive check
    if not search_results:
        return {
            "retrieved_documents": [],
            "errors": ["Retriever Node received empty search_results list."],
            "status": "retrieval_failed"
        }

    if not query:
        return {
            "retrieved_documents": [],
            "errors": ["Retriever Node requires a user query to perform similarity search."],
            "status": "retrieval_failed"
        }

    accumulated_errors = []

    try:
        # STEP 1: Storage (Long-Term Memory)
        # Documents are tagged with run_id and cycle_number in the payload.
        start_store = time.time()
        print("[Retriever] START Qdrant storage...")
        
        store_documents(
            documents=search_results,
            collection_name="deeptrace_research",
            run_id=run_id,
            cycle_number=cycle_number,
        )
        
        elapsed_store = time.time() - start_store
        print(f"[Retriever] COMPLETE Qdrant storage in {elapsed_store:.2f}s")

    except Exception as e:
        elapsed_store = time.time() - start_store if 'start_store' in locals() else 0.0
        error_msg = f"Retriever Node failed during storage phase in {elapsed_store:.2f}s: {str(e)}"
        accumulated_errors.append(error_msg)
        print(f"[Retriever] FAILED Qdrant storage: {error_msg}")
        return {
            "retrieved_documents": [],
            "errors": accumulated_errors,
            "status": "retrieval_failed"
        }

    try:
        # STEP 2: Retrieval (Semantic Over-fetch, run-isolated)
        # Only documents belonging to this run_id are candidates.
        # Documents from previous application runs are excluded by the payload filter.
        start_search = time.time()
        print("[Retriever] START Qdrant semantic search...")
        
        raw_docs = search_similar(
            query=query,
            collection_name="deeptrace_research",
            limit=RERANK_FETCH_LIMIT,
            run_id=run_id,
        )
        
        elapsed_search = time.time() - start_search
        print(f"[Retriever] COMPLETE Qdrant semantic search in {elapsed_search:.2f}s")

        # STEP 3: Domain Authority Reranking
        reranked_docs = []
        for raw_doc in raw_docs:
            # Create a shallow copy to prevent mutating the original Qdrant payload
            doc = raw_doc.copy()

            url = doc.get("source") or doc.get("link") or ""
            original_score = doc.get("_qdrant_score", 0.0)

            multiplier, tier = get_trust_multiplier(url)
            final_score = original_score * multiplier

            # Enrich document with observability metrics
            doc["trust_tier"] = tier
            doc["original_score"] = round(original_score, 4)
            doc["reranked_score"] = round(final_score, 4)

            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            doc["_domain"] = domain if domain else "unknown"

            reranked_docs.append(doc)

        # Sort descending by the new reranked_score
        reranked_docs.sort(key=lambda x: x["reranked_score"], reverse=True)

        # STEP 4: Deduplicate and enforce Source Diversity
        final_docs = []
        seen_urls = set()
        domain_counts = {}

        discarded_by_dedup = 0
        discarded_by_diversity = 0

        for doc in reranked_docs:
            url = doc.get("source") or doc.get("link") or ""
            domain = doc.get("_domain", "unknown")

            # Exact URL Deduplication
            if url and url in seen_urls:
                discarded_by_dedup += 1
                continue

            # Source Diversity limit
            count = domain_counts.get(domain, 0)
            if count >= MAX_CHUNKS_PER_DOMAIN:
                discarded_by_diversity += 1
                continue

            # Document accepted
            if url:
                seen_urls.add(url)
            domain_counts[domain] = count + 1

            # Cleanup internal helper keys before returning to state
            doc.pop("_domain", None)
            doc.pop("_qdrant_score", None)

            final_docs.append(doc)

            if len(final_docs) >= FINAL_RETRIEVAL_LIMIT:
                break

        # Calculate rank cutoffs (the unevaluated tail)
        discarded_by_rank = len(reranked_docs) - len(final_docs) - discarded_by_dedup - discarded_by_diversity

        print("\n[Retriever Internal Stats]")
        print(f"  - Run ID (first 8)       : {run_id[:8]}...")
        print(f"  - Cycle stored           : {cycle_number}")
        print(f"  - Fetched from Qdrant    : {len(raw_docs)}")
        print(f"  - Discarded by Dedup     : {discarded_by_dedup}")
        print(f"  - Discarded by Diversity : {discarded_by_diversity}")
        print(f"  - Discarded by Rank Limit: {discarded_by_rank}")
        print(f"  - Final Documents        : {len(final_docs)}\n")

    except Exception as e:
        elapsed_search = time.time() - start_search if 'start_search' in locals() else 0.0
        error_msg = f"Retriever Node failed during reranking phase in {elapsed_search:.2f}s: {str(e)}"
        accumulated_errors.append(error_msg)
        print(f"[Retriever] FAILED Qdrant semantic search/reranking: {error_msg}")
        return {
            "retrieved_documents": [],
            "errors": accumulated_errors,
            "status": "retrieval_failed"
        }

    return {
        "retrieved_documents": final_docs,
        "errors": accumulated_errors,
        "status": "retrieval_complete"
    }
