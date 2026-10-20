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
from dotenv import load_dotenv
load_dotenv()

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
class ResilientEmbeddingModel:
    """
    Resilient Embedding adapter:
    1. Primary: Google Generative AI Cloud Embeddings with multi-key failover (0 MB RAM, fast, perfect for 512MB Render free tier).
    2. Fallback: FastEmbed (Local, ONNX-accelerated, for local CLI development only).
    3. Fallback: HuggingFace Endpoint (if HF_TOKEN is configured).
    """
    def __init__(self):
        self._mode = None
        self._google_instances = []
        self._google = None
        self._fastembed = None
        self._hf = None
        self._hf_local = None
        self._vector_size = 3072
        self._init_model()

    @property
    def vector_size(self) -> int:
        return self._vector_size

    def _init_model(self):
        # 1. Primary: Google Generative AI Cloud Embeddings with multi-key rotation
        api_keys_str = (
            os.environ.get("GOOGLE_API_KEYS")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GEMINI_API_KEYS")
        )
        api_keys = [k.strip() for k in api_keys_str.split(",") if k.strip()] if api_keys_str else []

        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        for idx, key in enumerate(api_keys):
            try:
                emb_instance = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",
                    google_api_key=key
                )
                test_vec = emb_instance.embed_query("warmup")
                self._google_instances.append(emb_instance)
                if not self._google:
                    self._google = emb_instance
                    self._vector_size = len(test_vec)
                    self._mode = "google_genai"
                print(f"[Embeddings] Google key #{idx+1} verified successfully (Vector size: {len(test_vec)}).")
            except Exception as e:
                print(f"[Embeddings] Google key #{idx+1} failed verification: {e}")

        if self._google_instances:
            print(f"[Embeddings] Initialized {len(self._google_instances)} active Google Generative AI embedding instance(s).")
            return

        # On Render Free tier (512MB RAM), FastEmbed will cause an instant OOM kernel crash.
        # Guard against loading heavy ONNX runtimes in constrained cloud environments.
        is_render = os.environ.get("RENDER") == "true" or os.environ.get("IS_RENDER") == "true"
        if is_render:
            raise RuntimeError(
                "Google Generative AI embeddings failed to initialize on Render. "
                "FastEmbed local ONNX runtime is blocked on Render to prevent 512MB OOM crash. "
                "Please verify your GOOGLE_API_KEY / GEMINI_API_KEY in Render environment settings."
            )

        # 2. Secondary Fallback: FastEmbed (Local ONNX for local CLI development)
        try:
            from fastembed import TextEmbedding
            print("[Embeddings] Initializing FastEmbed (BAAI/bge-small-en-v1.5) for local execution...")
            self._fastembed = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            self._mode = "fastembed"
            self._vector_size = 384
            print("[Embeddings] FastEmbed initialized successfully.")
            return
        except Exception as e:
            print(f"[Embeddings] FastEmbed initialization unavailable: {e}. Trying HuggingFace fallback...")

        # 3. Tertiary Fallback: HuggingFace Endpoint (only if real token is provided)
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token and not hf_token.startswith("hf_your_fallback"):
            try:
                from langchain_huggingface import HuggingFaceEndpointEmbeddings
                print("[Embeddings] Initializing HuggingFaceEndpointEmbeddings...")
                self._hf = HuggingFaceEndpointEmbeddings(
                    model="sentence-transformers/all-MiniLM-L6-v2",
                    task="feature-extraction",
                    huggingfacehub_api_token=hf_token,
                )
                self._mode = "huggingface"
                self._vector_size = 384
                print("[Embeddings] HuggingFaceEndpointEmbeddings initialized successfully.")
                return
            except Exception as e:
                print(f"[Embeddings] HuggingFaceEndpointEmbeddings failed: {e}.")

        raise RuntimeError("No embedding provider could be initialized.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self._mode == "google_genai":
            last_err = None
            for instance in self._google_instances:
                try:
                    return instance.embed_documents(texts)
                except Exception as e:
                    last_err = e
                    continue
            raise RuntimeError(f"All Google embedding keys failed for embed_documents: {last_err}")
        elif self._mode == "fastembed":
            return [v.tolist() for v in self._fastembed.embed(texts)]
        elif self._mode == "huggingface":
            return self._hf.embed_documents(texts)
        elif self._mode == "sentence_transformers":
            return self._hf_local.embed_documents(texts)
        raise RuntimeError("Embedding model is not initialized.")

    def embed_query(self, query: str) -> List[float]:
        if self._mode == "google_genai":
            last_err = None
            for instance in self._google_instances:
                try:
                    return instance.embed_query(query)
                except Exception as e:
                    last_err = e
                    continue
            raise RuntimeError(f"All Google embedding keys failed for embed_query: {last_err}")
        elif self._mode == "fastembed":
            return list(self._fastembed.embed([query]))[0].tolist()
        elif self._mode == "huggingface":
            return self._hf.embed_query(query)
        elif self._mode == "sentence_transformers":
            return self._hf_local.embed_query(query)
        raise RuntimeError("Embedding model is not initialized.")


# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384


_client = None
_embedding_model = None

def get_qdrant_client() -> QdrantClient:
    """
    Lazy loads the Qdrant client. Uses in-memory storage for high speed,
    zero disk lock contention across concurrent background threads, and minimal resource footprint.
    """
    global _client
    if _client is None:
        _client = QdrantClient(location=":memory:")
    return _client


def get_embedding_model() -> ResilientEmbeddingModel:
    """
    Lazy loads the ResilientEmbeddingModel using a Singleton pattern.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = ResilientEmbeddingModel()
    return _embedding_model


def create_collection(collection_name: str = "deeptrace_research", vector_size: Optional[int] = None) -> None:
    """
    Creates a new collection in Qdrant if it does not already exist.
    Dynamically uses the active embedding model's vector size.
    """
    try:
        client = get_qdrant_client()
        embedding_model = get_embedding_model()
        v_size = vector_size or embedding_model.vector_size
        if not client.collection_exists(collection_name):
            print(f"Creating new Qdrant collection: '{collection_name}' (vector size {v_size})...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=v_size,
                    distance=Distance.COSINE
                ),
            )
            print("Collection created successfully.")
        else:
            print(f"Collection '{collection_name}' already exists. Skipping creation.")
    except Exception as e:
        error_msg = str(e).lower()
        if "already exists" in error_msg or "bad parameter" in error_msg or "api misuse" in error_msg:
            print(f"Collection '{collection_name}' likely created by another concurrent thread. Proceeding.")
        else:
            print(f"Error creating collection '{collection_name}': {e}")
            raise

def reset_collection(collection_name: str = "deeptrace_research", vector_size: Optional[int] = None) -> None:
    """
    Deletes and recreates the collection to ensure a clean state for a new run,
    preventing cross-run contamination while allowing retries to accumulate data safely.
    """
    try:
        client = get_qdrant_client()
        if client.collection_exists(collection_name):
            print(f"Deleting existing Qdrant collection: '{collection_name}' to prevent cross-run contamination...")
            client.delete_collection(collection_name)
        create_collection(collection_name, vector_size=vector_size)
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
        embedding_model = get_embedding_model()

        # Extract the raw text (snippets) that we want to embed for semantic search
        texts = [doc.get("snippet", "") for doc in documents]

        # Generate embeddings using active embedding model (Cloud Google or FastEmbed)
        embeddings = embedding_model.embed_documents(texts)
        if not embeddings:
            print("No embeddings generated.")
            return

        detected_size = len(embeddings[0])

        # Ensure collection exists before attempting to store vectors
        if not client.collection_exists(collection_name):
            print(f"Collection '{collection_name}' does not exist. Creating it now before storing...")
            create_collection(collection_name, vector_size=detected_size)

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