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
"""

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
    
    Args:
        state: The current ResearchState containing 'query' and 'search_results'.
        
    Returns:
        A dictionary containing the semantically filtered and reranked 
        'retrieved_documents', a 'status' indicator, and optionally 'errors'.
    """
    
    query = state.get("query", "")
    search_results = state.get("search_results", [])
    
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
        store_documents(documents=search_results, collection_name="deeptrace_research")
        
    except Exception as e:
        error_msg = f"Retriever Node failed during storage phase: {str(e)}"
        accumulated_errors.append(error_msg)
        print(error_msg)
        return {
            "retrieved_documents": [],
            "errors": accumulated_errors,
            "status": "retrieval_failed"
        }

    try:
        # STEP 2: Retrieval (Semantic Over-fetch)
        raw_docs = search_similar(
            query=query, 
            collection_name="deeptrace_research", 
            limit=RERANK_FETCH_LIMIT
        )
        
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
        print(f"  - Fetched from Qdrant    : {len(raw_docs)}")
        print(f"  - Discarded by Dedup     : {discarded_by_dedup}")
        print(f"  - Discarded by Diversity : {discarded_by_diversity}")
        print(f"  - Discarded by Rank Limit: {discarded_by_rank}")
        print(f"  - Final Documents        : {len(final_docs)}\n")
                
    except Exception as e:
        error_msg = f"Retriever Node failed during reranking phase: {str(e)}"
        accumulated_errors.append(error_msg)
        print(error_msg)
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
