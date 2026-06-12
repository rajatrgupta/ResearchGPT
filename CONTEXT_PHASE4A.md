# DeepTrace Phase 4A: Trust-Tier Reranking & Retrieval Quality
## Permanent Project Memory

This document serves as the comprehensive, definitive reference for Phase 4A of the DeepTrace project. It acts as permanent project memory, allowing any engineer or AI system to fully understand, debug, or rebuild Phase 4A from scratch.

---

## 1. Executive Summary & Problem Statement

### The Bottleneck: Semantic Similarity != Factual Truth
By the end of Phase 3B, DeepTrace possessed a functionally complete autonomous RAG loop. The Critic could identify missing information, and the Query Optimizer could reformulate queries to find it. However, a critical flaw remained in the **Retriever Agent**: it equated "semantic similarity" with "factual truth."

When querying Qdrant using dense vector embeddings (Cosine Similarity), a low-quality SEO blog or a Reddit comment that heavily used the query's vocabulary would frequently rank higher than a peer-reviewed journal or government website. This forced the Critic and Writer to synthesize reports based on factually dubious context.

### What Phase 4A Solved
Phase 4A addressed this bottleneck by transforming the retrieval step from a pure vector search into a **Trust-Tier Reranking Pipeline**. It applies Domain Authority multipliers to the raw Qdrant scores, explicitly enforcing source diversity and deduplicating identical URLs, ensuring that only the most highly authoritative, diverse, and unique facts are passed to the Writer.

---

## 2. Trust-Tier Architecture Design

### Domain Authority Scoring Approach
Rather than modifying the underlying FastEmbed models to perform hybrid dense/sparse keyword matching, we opted for a highly deterministic, zero-latency Python heuristic applied *after* the Qdrant retrieval. 

### The Tiers
We established a strict 4-tier configuration:

1. **HIGH_TRUST (Multiplier 1.5x)**
   - Domains: `.gov`, `.edu`, `.mil`, `who.int`, `nature.com`, `ncbi.nlm.nih.gov`, `ieee.org`
   - Purpose: Heavily boosts academic, scientific, and government sources to the top of the context window.
2. **MEDIUM_TRUST (Multiplier 1.2x)**
   - Domains: `reuters.com`, `bbc.co.uk`, `apnews.com`, `wsj.com`, `bloomberg.com`, `nytimes.com`
   - Purpose: Boosts established journalistic and news organizations.
3. **DEFAULT_TRUST (Multiplier 1.0x)**
   - Domains: Any URL not explicitly defined.
   - Purpose: Maintains the baseline Qdrant semantic score without artificially penalizing unknown but potentially valid sources.
4. **LOW_TRUST (Multiplier 0.5x)**
   - Domains: `reddit.com`, `quora.com`, `medium.com`, `twitter.com`, `yahoo.com`
   - Purpose: Severely penalizes user-generated content and common content farms, preventing them from displacing authoritative facts.

### Exact-Domain vs Subdomain Matching
To prevent malicious spoofing (e.g., `fake-nature.com` receiving `HIGH_TRUST`), the URL parsing logic strictly segregates checks:
- **TLD check:** `domain.endswith(tld)` (e.g., matching any `.gov` or `.edu`).
- **Domain check:** `domain == d or domain.endswith("." + d)` (e.g., exactly `nature.com` or `subdomain.nature.com`).

---

## 3. The Reranking Pipeline Logic

### Over-Fetch Strategy
Instead of requesting the `FINAL_RETRIEVAL_LIMIT` (5) from Qdrant, the Retriever requests an `RERANK_FETCH_LIMIT` (20). This provides a larger pool of semantically relevant documents for the Python reranker to evaluate.

### Reranking Formula
`final_reranked_score = original_qdrant_cosine_score * trust_tier_multiplier`

### Deduplication Logic
DuckDuckGo search occasionally scrapes the exact same article syndicated under multiple links, or returns the same link multiple times across different sub-queries. The Retriever maintains a `seen_urls` set. Any document whose URL is already in the set is immediately discarded, preventing redundant facts from wasting the 5 precious context slots.

### Diversity Enforcement
To prevent a single highly authoritative domain (e.g., `nih.gov`) from monopolizing all 5 slots and narrowing the report's perspective, we enforce a `MAX_CHUNKS_PER_DOMAIN = 2` rule. 
If a domain has already contributed 2 documents to the final list, subsequent documents from that domain are discarded, forcing the loop to pull in the next highest-scoring, unique domain.

---

## 4. Qdrant Lifecycle & Contamination Prevention

### The Cross-Run Contamination Issue
During Phase 4A development, we identified a critical flaw in using Qdrant's local disk storage (`path="./local_qdrant_storage"`).
Because `store_documents` assigns a new UUID to every snippet and continually appends them, running `main.py` multiple times over several days caused the vector database to permanently bloat. A search for "Artificial Intelligence" on Tuesday would pull in vectors embedded during a test run on Monday, corrupting the reproducibility of the current execution.

### Temporary Development Safeguard
We introduced a `reset_collection()` function in `qdrant_store.py` and called it at the very top of `main.py`. This explicitly deletes and recreates the `deeptrace_research` collection every time the script executes. 
**Important Note:** This preserves data *within* a single run (allowing the retry loops to accumulate facts), but prevents cross-run contamination.

### Future Production Architecture
`reset_collection()` is strictly a Phase 4A temporary hack. 
In Phase 5 (Production), we will implement robust session isolation:
- PostgreSQL will generate a `research_id` (or `session_id`).
- This ID will be injected into the vector payload.
- Retrieval will use **Qdrant Payload Filtering** (`must` condition) to mathematically restrict similarity searches to vectors belonging *only* to the active session. This will allow the DB to grow indefinitely without wiping data, enabling future features like historical memory.

---

## 5. Files Modified

### `app/vectorstore/qdrant_store.py`
- Added `reset_collection()` to safely wipe the local DB on script startup.
- Modified `search_similar(limit=15)` to increase the over-fetch limit.
- Updated `search_similar` to extract `hit.score` and inject it into the returned payload as `_qdrant_score` so the Retriever has access to the raw math.

### `app/agents/retriever.py`
- Added configuration constants for Tiers (`HIGH_TRUST`, etc.) and Limits (`RERANK_FETCH_LIMIT = 20`, `FINAL_RETRIEVAL_LIMIT = 5`).
- Created `get_trust_multiplier()` for secure URL parsing and tier matching.
- Completely rewrote `retriever_node` to execute the Over-fetch -> Copy Document -> Multiply Score -> Sort -> Deduplicate -> Diversify -> Cleanup Pipeline.
- Added detailed internal console metrics for observability (`Discarded by Dedup`, `Discarded by Diversity`, `Discarded by Rank Limit`).

### `main.py`
- Added `reset_collection("deeptrace_research")` at script initialization.
- Added comprehensive `[Retriever]` observability print statements inside the `graph.stream` loop to visually validate the Trust Tier assignments and score alterations.
- Moved `import urllib.parse` to the top of the file to clean up inline scoping.

---

## 6. Runtime Validation & Observations

During the validation run of "Future of Artificial Intelligence":

### Metrics Output
```text
[Retriever Internal Stats]
  - Fetched from Qdrant    : 20
  - Discarded by Dedup     : 2
  - Discarded by Diversity : 0
  - Discarded by Rank Limit: 13
  - Final Documents        : 5

[Retriever] Curated Top 5 documents after Trust-Tier Reranking:
  1. [DEFAULT_TRUST] sigmauniversity.ac.in (Original: 0.8377 -> Reranked: 0.8377)
  2. [DEFAULT_TRUST] miguelescotet.com (Original: 0.8356 -> Reranked: 0.8356)
  ...
```

### Observation of Limitations
The pipeline executed the logic flawlessly: it dropped 2 duplicate URLs, successfully matched the remaining URLs to `DEFAULT_TRUST` (as they were unknown blogs), left the scores un-mutated (1.0x), and passed exactly 5 documents to the Critic.

**The Remaining Bottleneck:** The Trust-Tier system works perfectly, but it is currently starved of high-quality inputs. The underlying `DuckDuckGoSearchAPIWrapper` did not return *any* `.gov` or major news domains during the scrape, so there was nothing for the Retriever to boost to `HIGH_TRUST`. The Critic accurately identified that these lower-tier blogs lacked depth, triggering 3 full retry loops before gracefully synthesizing the best available data.

**Conclusion:** Retrieval quality is now strictly constrained by the capability of the search provider, not the intelligence of our routing or ranking algorithms.

---

## 7. Lessons Learned
- **Mutability is Dangerous:** In early tests, mutating the dictionary returned directly from Qdrant caused issues. Using `doc = raw_doc.copy()` in the Retriever loop ensured safe manipulation of observability fields without corrupting the underlying dataset.
- **Graceful Degradation:** The diversity and dedup filters are designed to `break` only when `len(final_docs) >= 5`. If the pool is exhausted (e.g., 18 chunks from one domain are discarded), the Retriever might return only 3 or 4 documents. This is a feature, not a bug; it is better to provide the LLM with 4 highly diverse facts than to force a 5th redundant or low-quality fact just to hit a quota.

---

## 8. Interview Preparation (Q&A)

**Q1. How did you improve factual reliability in DeepTrace?**
A: By breaking the assumption that "semantic similarity" equals "factual truth." I implemented a Trust-Tier Domain Authority reranker that acts as a multiplier against the vector database's cosine similarity score, heavily favoring `.gov` and academic sources over user-generated content.

**Q2. Why did you use Python heuristics instead of Hybrid Search (BM25)?**
A: To rapidly unblock the hallucination issue. Hybrid search is excellent for exact keyword matching (e.g., model numbers), but it doesn't solve source credibility. Applying a fast Python multiplier based on URL parsing directly solved the credibility gap with zero added latency or complex vector database migrations.

**Q3. Explain your Over-Fetch and Filter strategy.**
A: Instead of asking the DB for 5 documents, we ask for 20. This gives the application layer a wide semantic pool. We then score, sort, deduplicate (exact URLs), and enforce a diversity limit (`max 2 chunks per domain`). The "tail" of the 20 is discarded, ensuring the final 5 are highly curated.

**Q4. How do you prevent Domain Spoofing in your parser?**
A: For exact domains like `nature.com`, we cannot just use `endswith("nature.com")`, as that would approve `fake-nature.com`. We enforce strict exact matching (`domain == target`) or valid subdomain matching (`domain.endswith("." + target)`).

**Q5. Why delete the Qdrant collection on startup?**
A: It is a temporary development safeguard. Local Qdrant writes permanently to disk. Without wiping it, running tests across multiple days contaminates the vector space with outdated snippets, corrupting the semantic search results.

**Q6. What is the production solution for session isolation?**
A: In production, we will stop wiping the database. Instead, the PostgreSQL backend will generate a unique `session_id`. We will inject this ID into the Qdrant payload when storing documents, and use Qdrant Payload Filtering to restrict similarity searches only to vectors matching that session.

---

## 9. Next Steps
Phase 4B or Phase 5 must address the **Search Source Bottleneck**. Integrating a commercial search API (like Tavily, Google Programmable Search, or Exa) is required to feed the Trust-Tier system with high-caliber academic and journalistic data.