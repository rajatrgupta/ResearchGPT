"""
DeepTrace Search Provider Abstraction

This module defines the provider abstraction layer that sits between the
Search Agent and the underlying search APIs. The Search Agent depends on
the SearchProvider interface, not on any specific implementation.

Architecture:
    SearchProvider (Abstract Base)
    +--> TavilySearchProvider     (Primary: high-quality, trust-tier-friendly)
    +--> DuckDuckGoSearchProvider  (Fallback: free, no API key required)
    +--> ChainedSearchProvider    (Orchestrates: Tavily -> DDG on failure)

All providers return results in the same normalized format expected by
the Search Agent and the downstream Retriever / Trust-Tier pipeline:
    {
        "question": str,   # The query that produced this result
        "title":    str,   # Page title
        "snippet":  str,   # Relevant text excerpt
        "source":   str    # URL
    }

Phase 4B Domain Filtering Strategy:
    Tavily supports two domain-control parameters:
      - include_domains: SOFT preference. Encourages results from these
        domains but does NOT hard-restrict recall. If the specified domains
        have no matching content, Tavily still returns broad results.
      - exclude_domains: HARD blocklist. Guarantees these domains never
        appear in results. Used for known LOW_TRUST sources.

    Our strategy uses query-intent classification to select appropriate
    include_domains per query rather than a single fixed news-wire list:
      - scientific: Prefer academic/research domains (nature.com, arxiv.org, ieee.org)
      - policy:     Prefer official/government domains (who.int, congress.gov, etc.)
      - news:       Prefer quality journalism (reuters.com, bbc.co.uk, bloomberg.com)
      - general:    No include_domains - let Tavily rank freely

    exclude_domains is always applied regardless of category (LOW_TRUST blocklist).

    The Phase 4A Trust-Tier reranker in retriever.py remains the FINAL
    authority on source ranking. Tavily domain filtering is only an
    upstream pre-filter that improves the quality of inputs to the reranker.
"""

import os
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================
# Centralized search configuration. Changing these values here affects
# all providers uniformly without touching agent code.
MAX_RESULTS_PER_QUERY = 3  # Matches historical DDG behavior (max_results=3)
                            # With 5 sub-questions: 15 raw results per cycle.
                            # With 3 retry queries: 9 results per retry cycle.
                            # Over-fetching (20->5) happens at the Qdrant layer.

# Environment variable name for the Tavily API key.
TAVILY_API_KEY_ENV = "TAVILY_API_KEY"

# Timeout for individual search API calls
SEARCH_TIMEOUT = 15.0

# ==========================================
# QUERY INTENT CLASSIFICATION
# ==========================================
# Deterministic keyword-based classifier. No LLM call. No new agent.
# Priority order: scientific > policy > news > general
# Multi-word phrases use substring match on the full lowercased query.
# Single words use exact word-token match to avoid false positives.

SCIENTIFIC_KEYWORDS = [
    # Multi-word phrases (substring match)
    "artificial intelligence", "machine learning", "deep learning",
    "neural network", "natural language processing", "computer vision",
    "data science", "large language model", "generative ai",
    "climate change", "climate model", "drug discovery", "gene therapy",
    # Single words (exact token match)
    "research", "study", "clinical", "biology", "chemistry", "physics",
    "algorithm", "genome", "quantum", "engineering", "laboratory",
    "medical", "scientific", "technology", "computing", "robotics",
    "semiconductor", "nanotechnology", "neuroscience", "astronomy",
    "epidemiology", "vaccine", "protein", "simulation",
]

POLICY_KEYWORDS = [
    # Multi-word phrases
    "national security", "foreign policy", "public policy", "court ruling",
    "executive order", "international law", "trade agreement",
    # Single words (exact token match)
    "regulation", "law", "policy", "government", "legislation",
    "compliance", "federal", "agency", "treaty", "military",
    "defense", "congress", "parliament", "senate", "governance",
    "oversight", "regulatory", "sanctions", "diplomacy", "geopolitical",
]

NEWS_KEYWORDS = [
    # Multi-word phrases
    "breaking news", "current events", "stock market", "economic outlook",
    # Single words (exact token match)
    "latest", "today", "recent", "breaking", "crisis",
    "war", "conflict", "economy", "news", "announced",
    "2024", "2025", "2026",
]

# ==========================================
# PER-CATEGORY DOMAIN PREFERENCE LISTS
# ==========================================
# These are passed as include_domains to Tavily (SOFT preference only).
# Tavily will prioritize these domains when they have relevant content,
# but will NOT sacrifice recall when they don't.

SCIENTIFIC_PREFERRED_DOMAINS: List[str] = [
    "nature.com",
    "ncbi.nlm.nih.gov",
    "ieee.org",
    "arxiv.org",
    "scholar.google.com",
]

POLICY_PREFERRED_DOMAINS: List[str] = [
    "who.int",
    "un.org",
    "ec.europa.eu",
    "congress.gov",
    "whitehouse.gov",
]

NEWS_PREFERRED_DOMAINS: List[str] = [
    "reuters.com",
    "bbc.co.uk",
    "apnews.com",
    "bloomberg.com",
    "nytimes.com",
    "wsj.com",
]

# LOW_TRUST domains: hard blocklist applied to ALL categories.
# Tavily guarantees these URLs never appear in results.
TAVILY_BLOCKED_DOMAINS: List[str] = [
    "reddit.com",
    "quora.com",
    "medium.com",
    "twitter.com",
    "yahoo.com",
    "brainly.in",
    "vedantu.com",
    "gradesfixer.com",
    "scribd.com",
    "coursehero.com",
    "studocu.com"
]


def classify_query_intent(query: str) -> str:
    """
    Deterministically classifies a search query into one of four intent categories.

    Classification is keyword-based with no LLM call. Priority order:
      scientific > policy > news > general

    This priority ensures that a query like "AI regulation policy" resolves to
    "scientific" rather than "policy", since AI research context is more relevant
    for Tavily domain hints than regulatory sources.

    Args:
        query: The search query string (sub-question or retry query).

    Returns:
        One of: "scientific", "policy", "news", "general"
    """
    query_lower = query.lower()
    query_words = set(query_lower.split())

    def matches(keywords: List[str]) -> bool:
        for kw in keywords:
            if " " in kw:
                # Multi-word phrase: substring match on full query string
                if kw in query_lower:
                    return True
            else:
                # Single word: exact token match to avoid false positives
                if kw in query_words:
                    return True
        return False

    if matches(SCIENTIFIC_KEYWORDS):
        return "scientific"
    if matches(POLICY_KEYWORDS):
        return "policy"
    if matches(NEWS_KEYWORDS):
        return "news"
    return "general"


def get_preferred_domains(intent: str) -> List[str]:
    """
    Returns the appropriate include_domains list for the given intent category.

    For 'general', returns an empty list so Tavily ranks freely without
    any inappropriate domain preference.
    """
    if intent == "scientific":
        return SCIENTIFIC_PREFERRED_DOMAINS
    elif intent == "policy":
        return POLICY_PREFERRED_DOMAINS
    elif intent == "news":
        return NEWS_PREFERRED_DOMAINS
    else:
        # general: no soft preference — let Tavily rank by relevance alone
        return []


# ==========================================
# ABSTRACT BASE: SearchProvider
# ==========================================

class SearchProvider(ABC):
    """
    Abstract interface for all search providers in DeepTrace.

    Every concrete provider must implement search() with this exact
    contract so the Search Agent can call any provider interchangeably.
    """

    @abstractmethod
    def search(self, query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> List[Dict[str, Any]]:
        """
        Execute a search for the given query and return normalized results.

        Args:
            query:       The search string to execute.
            max_results: Maximum number of results to return per query.

        Returns:
            A list of result dictionaries. Each dictionary contains:
              - "question" (str): The query that produced this result.
              - "title"    (str): Page or article title.
              - "snippet"  (str): Relevant text excerpt from the page.
              - "source"   (str): Full URL of the source.

            Returns an empty list on failure rather than raising, so the
            caller can apply fallback logic cleanly.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name for logging and observability."""
        pass


# ==========================================
# TAVILY SEARCH PROVIDER (PRIMARY)
# ==========================================

class TavilySearchProvider(SearchProvider):
    """
    Primary search provider powered by the Tavily API.

    Tavily returns high-quality, curated web results that are significantly
    more likely to include journalistic and academic sources (reuters.com,
    bbc.co.uk, .gov, .edu, etc.) compared to DuckDuckGo. This directly
    feeds better material into the Phase 4A Trust-Tier reranker.

    Domain Filtering (Phase 4B enhancement):
        include_domains: Per-query soft preference selected by classify_query_intent().
            - scientific queries: academic/research domains
            - policy queries:     official/government domains
            - news queries:       quality journalism domains
            - general queries:    no preference (Tavily ranks freely)
        exclude_domains: Hard block of LOW_TRUST sources (reddit, quora,
            medium, twitter, yahoo). These are guaranteed to never appear.

    Requires:
        TAVILY_API_KEY environment variable to be set.

    Graceful Degradation:
        If the API key is missing, is_available() returns False and
        search() returns [] immediately, allowing ChainedSearchProvider
        to fall through to DuckDuckGo.
    """

    def __init__(self):
        """
        Lazy-instantiates the Tavily client only if the API key is present.
        Never raises during __init__ -- failures surface via is_available().
        """
        self._client = None
        self._api_key = os.environ.get(TAVILY_API_KEY_ENV, "")

        if self._api_key:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=self._api_key)
            except ImportError:
                print("[TavilySearchProvider] Warning: 'tavily-python' package not installed. "
                      "Install it with: pip install tavily-python")
            except Exception as e:
                # Never expose the API key -- log only the exception type
                print(f"[TavilySearchProvider] Warning: Failed to initialize Tavily client: "
                      f"{type(e).__name__}")
        else:
            print(f"[TavilySearchProvider] Warning: '{TAVILY_API_KEY_ENV}' environment variable "
                  "is not set. Tavily provider unavailable. DuckDuckGo fallback will be used.")

    @property
    def name(self) -> str:
        return "Tavily"

    def is_available(self) -> bool:
        """Returns True if the Tavily client was successfully initialized."""
        return self._client is not None

    def search(self, query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> List[Dict[str, Any]]:
        """
        Execute a Tavily search with intent-aware domain filtering.

        Domain filtering applied per query:
          1. classify_query_intent(query) determines the intent category.
          2. get_preferred_domains(intent) selects the include_domains list.
          3. TAVILY_BLOCKED_DOMAINS is always applied as exclude_domains.

        This ensures academic queries get academic domain preferences,
        news queries get journalism preferences, and general queries get
        Tavily's unbiased free-ranking behavior.

        Tavily response structure:
          response["results"] -> list of dicts with keys:
            title, url, content, score, raw_content

        Returns [] on any failure so ChainedSearchProvider can fall back.
        """
        if not self.is_available():
            return []
            
        start_time = time.time()
        print(f"[TavilySearchProvider] START search for: '{query[:60]}...'")

        try:
            # Determine query intent and select appropriate domain preference
            intent = classify_query_intent(query)
            preferred_domains = get_preferred_domains(intent)

            # Build search call — only include include_domains when non-empty
            search_kwargs = dict(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_answer=False,
                include_raw_content=False,
                exclude_domains=TAVILY_BLOCKED_DOMAINS,
                timeout=SEARCH_TIMEOUT, # Native SDK timeout integration
            )
            if preferred_domains:
                search_kwargs["include_domains"] = preferred_domains

            response = self._client.search(**search_kwargs)

            results = response.get("results", [])

            if not results:
                elapsed = time.time() - start_time
                print(f"[TavilySearchProvider] COMPLETE in {elapsed:.2f}s (0 results)")
                return []

            normalized = []
            for item in results:
                if not isinstance(item, dict):
                    continue

                title = item.get("title", "")
                url = item.get("url", "")
                # Tavily uses "content" for the extracted text snippet
                snippet = item.get("content", "")

                # Skip entries with no usable content
                if not snippet and not title:
                    continue

                normalized.append({
                    "question": query,
                    "title": title,
                    "snippet": snippet,
                    "source": url,
                })

            elapsed = time.time() - start_time
            print(f"[TavilySearchProvider] COMPLETE in {elapsed:.2f}s ({len(normalized)} results)")
            return normalized

        except Exception as e:
            elapsed = time.time() - start_time
            # Log error type and truncated message -- never log the API key
            print(f"[TavilySearchProvider] FAILED in {elapsed:.2f}s for query '{query[:60]}...': "
                  f"{type(e).__name__}: {str(e)[:120]}")
            return []


# ==========================================
# DUCKDUCKGO SEARCH PROVIDER (FALLBACK)
# ==========================================

class DuckDuckGoSearchProvider(SearchProvider):
    """
    Fallback search provider using DuckDuckGo.

    This wraps the existing DuckDuckGoSearchAPIWrapper behavior that was
    previously embedded directly in app/agents/search.py. No API key required.
    Returns the same normalized result structure as TavilySearchProvider.
    """

    @property
    def name(self) -> str:
        return "DuckDuckGo"

    def search(self, query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> List[Dict[str, Any]]:
        """
        Execute a DuckDuckGo search and return normalized results.

        Uses DuckDuckGoSearchAPIWrapper.results() which returns a list of
        dicts with keys: title, snippet, link. These map directly to our
        normalized format.

        Returns [] on any failure.
        """
        start_time = time.time()
        print(f"[DuckDuckGoSearchProvider] START fallback search for: '{query[:60]}...'")
        
        try:
            from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
            search_tool = DuckDuckGoSearchAPIWrapper()
            raw_results = search_tool.results(query, max_results=max_results)

            if not isinstance(raw_results, list) or not raw_results:
                elapsed = time.time() - start_time
                print(f"[DuckDuckGoSearchProvider] COMPLETE in {elapsed:.2f}s (0 results)")
                return []

            normalized = []
            for item in raw_results:
                if not isinstance(item, dict):
                    continue

                normalized.append({
                    "question": query,
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "source": item.get("link", ""),
                })

            elapsed = time.time() - start_time
            print(f"[DuckDuckGoSearchProvider] COMPLETE in {elapsed:.2f}s ({len(normalized)} results)")
            return normalized

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[DuckDuckGoSearchProvider] FAILED in {elapsed:.2f}s for query '{query[:60]}...': "
                  f"{type(e).__name__}: {str(e)[:120]}")
            return []


# ==========================================
# CHAINED SEARCH PROVIDER (ORCHESTRATOR)
# ==========================================

class ChainedSearchProvider(SearchProvider):
    """
    Orchestrates a Tavily-first, DuckDuckGo-fallback search strategy.

    Execution logic per query:
      1. Attempt the primary provider (Tavily).
      2. If primary returns non-empty results -> return them immediately.
      3. If primary returns [] for any reason (unavailable, error, or no
         results found) -> log the fallback event and try the fallback
         provider (DuckDuckGo).
      4. Return fallback results, or [] if both fail.

    This ensures the pipeline never hard-fails due to a missing Tavily
    API key or a temporary Tavily outage.
    """

    def __init__(self, primary: SearchProvider, fallback: SearchProvider):
        self._primary = primary
        self._fallback = fallback

    @property
    def name(self) -> str:
        # ASCII-only: avoids Windows cp1252 encoding crash when printed to console
        return f"Chained({self._primary.name} -> {self._fallback.name})"

    def search(self, query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> List[Dict[str, Any]]:
        """
        Try the primary provider first; fall back to secondary on failure
        or empty result.

        The fallback is triggered when the primary returns an empty list
        for any reason:
          - Tavily API key is missing
          - Tavily raised an exception (network, rate limit, etc.)
          - Tavily returned zero results for this query
        """
        # Attempt primary provider
        primary_results = self._primary.search(query=query, max_results=max_results)

        if primary_results:
            return primary_results

        # Primary returned empty -- fall through to fallback
        print(f"[ChainedSearchProvider] '{self._primary.name}' returned no results for "
              f"query: '{query[:60]}...'. Activating fallback: '{self._fallback.name}'.")

        fallback_results = self._fallback.search(query=query, max_results=max_results)
        return fallback_results


# ==========================================
# FACTORY FUNCTION
# ==========================================

def get_search_provider() -> SearchProvider:
    """
    Factory function that builds and returns the configured search provider chain.

    Returns a ChainedSearchProvider configured with:
      - TavilySearchProvider as primary (with intent-aware domain filtering)
      - DuckDuckGoSearchProvider as fallback

    The Search Agent calls this function and uses the returned provider
    without knowing which concrete provider is active. All provider
    selection and fallback orchestration happens inside this module.

    Returns:
        SearchProvider: A ready-to-use provider instance.
    """
    primary = TavilySearchProvider()
    fallback = DuckDuckGoSearchProvider()
    return ChainedSearchProvider(primary=primary, fallback=fallback)
