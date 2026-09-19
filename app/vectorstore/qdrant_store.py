"""
Qdrant Vector Store Integration for DeepTrace

=========================================
CORE CONCEPTS & EXPLANATIONS
=========================================
1. What is RAG?
   Retrieval-Augmented Generation (RAG) injects factual data into LLM prompts.
   Instead of guessing, the LLM reads external search results and synthesizes them.

2. Why do LLMs hallucinate?
   LLMs are probabilistic (guessing the next word). Without a real-time factual
   reference, they will confidently generate statistically probable, but factually
   incorrect, information.

3. What is a Vector Database?
   A database optimized to store, index, and query high-dimensional arrays (vectors)
   based on their semantic similarity rather than exact keyword matches.

4. Why Qdrant instead of PostgreSQL?
   Qdrant is natively built in Rust for highly concurrent vector search workloads.
   It utilizes HNSW graphs out-of-the-box, offering vastly superior speed and scaling
   for pure vector similarity search compared to a bolted-on PostgreSQL extension.

5. What are embeddings?
   Numerical representations of text. An embedding model translates sentences into
   coordinates in a high-dimensional space where similar concepts are grouped together.

6. How retrieval works internally?
   - Query text -> converted to an embedding vector.
   - Database calculates mathematical distance (e.g., Cosine Similarity) between the
     query vector and stored document vectors.
   - Database returns the top-K vectors with the shortest distance.

=========================================
PRODUCTION PATTERNS
=========================================
Why Lazy Loading is Better Than Global Initialization:
If we initialize `QdrantClient` and `TextEmbedding` at the module level (globally),
they execute the moment this file is imported. This can cause the app to crash at startup
if the Qdrant server isn't ready or env vars aren't loaded yet. It also consumes memory
for the ML models immediately. Lazy loading (using getter functions) defers instantiation
until the exact moment the tool is called, making the system resilient, memory-efficient,
and easier to test/mock.

Phase 4B: Run-Level Isolation
store_documents() now stamps run_id and cycle onto every payload. search_similar()
accepts an optional run_id and applies a Qdrant payload filter so each run only
retrieves its own documents. This eliminates cross-run contamination without
removing the reset_collection() local-development safeguard.
"""

import os
import uuid
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from langchain_huggingface import HuggingFaceEndpointEmbeddings
# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384


_client = None
_embedding_model = None

def get_qdrant_client() -> QdrantClient:
    """
    Lazy loads the Qdrant client. Uses local disk storage for testing,
    bypassing the need for a Docker container while persisting data across calls.
    """
    global _client
    if _client is None:
        _client = QdrantClient(path="./local_qdrant_storage")
    return _client


def get_embedding_model() -> HuggingFaceEndpointEmbeddings:
    """
    Lazy loads the HuggingFaceEndpointEmbeddings model using a Singleton pattern.
    Uses the API instead of downloading local models, preventing OOM crashes on Render.
    """
    global _embedding_model
    if _embedding_model is None:
        hf_token = os.environ.get("HF_TOKEN") or "hf_your_fallback_token_if_needed"
        _embedding_model = HuggingFaceEndpointEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            task="feature-extraction",
            huggingfacehub_api_token=hf_token,
        )
    return _embedding_model


def create_collection(collection_name: str = "deeptrace_research") -> None:
    """
    Creates a new collection in Qdrant if it does not already exist.
    """
    try:
        client = get_qdrant_client()
        if not client.collection_exists(collection_name):
            print(f"Creating new Qdrant collection: '{collection_name}'...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                ),
            )
            print("Collection created successfully.")
        else:
            print(f"Collection '{collection_name}' already exists. Skipping creation.")
    except Exception as e:
        print(f"Error creating collection '{collection_name}': {e}")
        raise

def reset_collection(collection_name: str = "deeptrace_research") -> None:
    """
    Deletes and recreates the collection to ensure a clean state for a new run,
    preventing cross-run contamination while allowing retries to accumulate data safely.
    """
    try:
        client = get_qdrant_client()
        if client.collection_exists(collection_name):
            print(f"Deleting existing Qdrant collection: '{collection_name}' to prevent cross-run contamination...")
            client.delete_collection(collection_name)
        create_collection(collection_name)
    except Exception as e:
        print(f"Error resetting collection: {e}")


def store_documents(
    documents: List[Dict[str, Any]],
    collection_name: str = "deeptrace_research",
    run_id: str = "",
    cycle_number: int = 0,
) -> None:
    """
    Converts text documents into embeddings and stores them in Qdrant.

    Phase 4B: Each stored point's payload is stamped with:
      - run_id:  The UUID4 of the current main() execution. Used by search_similar()
                 to filter retrieval to only this run's documents, preventing stale
                 documents from previous runs competing in the similarity search.
      - cycle:   The iteration_count at the time of storage. Informational metadata
                 for future freshness scoring or debugging. NOT used for filtering.

    Args:
        documents:        List of normalized search result dicts to store.
        collection_name:  Target Qdrant collection.
        run_id:           Active run's UUID4. Empty string if called outside pipeline.
        cycle_number:     Active iteration_count from ResearchState.
    """
    if not documents:
        print("No documents provided to store.")
        return

    try:
        client = get_qdrant_client()

        # Ensure collection exists before attempting to store vectors
        if not client.collection_exists(collection_name):
            print(f"Collection '{collection_name}' does not exist. Creating it now before storing...")
            create_collection(collection_name)

        embedding_model = get_embedding_model()

        # Extract the raw text (snippets) that we want to embed for semantic search
        texts = [doc.get("snippet", "") for doc in documents]

        # Generate embeddings using Google GenAI API.
        embeddings = embedding_model.embed_documents(texts)

        points = []
        # Loop over our documents and their corresponding newly-generated vectors
        for doc, vector in zip(documents, embeddings):
            point_id = str(uuid.uuid4())

            # Phase 4B: Stamp run_id and cycle onto the payload without mutating
            # the original document dict (which is shared with ResearchState).
            payload = {
                **doc,
                "run_id": run_id,
                "cycle": cycle_number,
            }

            point = PointStruct(
                id=point_id,
                vector=vector,           # It is already a standard Python list from LangChain
                payload=payload          # Store all metadata including run/cycle tags
            )
            points.append(point)

        # Upsert securely writes our points to the database.
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"Successfully stored {len(points)} documents in Qdrant "
              f"(run={run_id[:8]}..., cycle={cycle_number}).")

    except Exception as e:
        print(f"Error storing documents in Qdrant collection '{collection_name}': {e}")
        raise


def search_similar(
    query: str,
    collection_name: str = "deeptrace_research",
    limit: int = 15,
    run_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Searches the vector database for documents most similar to the provided query.

    Phase 4B: When run_id is provided, applies a Qdrant payload filter that restricts
    results to only documents that were stored in the current run. This prevents
    documents from previous application runs from polluting the retrieval results.

    Within a single run, all cycles' documents remain eligible (isolation is at the
    run boundary, not the cycle boundary). This preserves cumulative research behavior
    while eliminating cross-run contamination.

    Args:
        query:            The search string to find similar documents for.
        collection_name:  Target Qdrant collection.
        limit:            Maximum number of similar documents to return.
        run_id:           If provided, restricts results to documents from this run only.
                          None = no filter (backward compatible, for testing only).
    """
    try:
        client = get_qdrant_client()
        embedding_model = get_embedding_model()

        # 1. Convert the query text into its vector representation
        query_vector = embedding_model.embed_query(query)

        # 2. Build the optional run-level payload filter
        query_filter = None
        if run_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="run_id",
                        match=MatchValue(value=run_id)
                    )
                ]
            )

        # 3. Perform the semantic similarity search in Qdrant
        search_result = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True
        )

        # 4. Extract and return just the original payload (text and metadata)
        retrieved_docs = []
        for hit in search_result.points:
            if hit.payload:
                payload_copy = hit.payload.copy()
                payload_copy["_qdrant_score"] = hit.score
                retrieved_docs.append(payload_copy)

        return retrieved_docs

    except Exception as e:
        print(f"Error searching similar documents in Qdrant: {e}")
        # Return an empty list so upstream nodes don't crash, allowing graceful degradation
        return []