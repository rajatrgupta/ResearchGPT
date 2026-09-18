"""
DeepTrace Search Provider Package

Exposes the search provider abstraction and factory function.
Import `get_search_provider` to get the active provider chain.
"""

from app.search.providers import (
    SearchProvider,
    TavilySearchProvider,
    DuckDuckGoSearchProvider,
    ChainedSearchProvider,
    get_search_provider,
)

__all__ = [
    "SearchProvider",
    "TavilySearchProvider",
    "DuckDuckGoSearchProvider",
    "ChainedSearchProvider",
    "get_search_provider",
]
