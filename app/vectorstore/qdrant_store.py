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
"""

import os
import uuid
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding

# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384


_client = None

def get_qdrant_client() -> QdrantClient:
    """
    Lazy loads the Qdrant client. Uses local disk storage for testing,
    bypassing the need for a Docker container while persisting data across calls.
    """
    global _client
    if _client is None:
        _client = QdrantClient(path="./local_qdrant_storage")
    return _client


def get_embedding_model() -> TextEmbedding:
    """
    Lazy loads the TextEmbedding model. 
    This prevents the heavy model from loading into memory upon module import.
    """
    return TextEmbedding(model_name=EMBEDDING_MODEL_NAME)


def create_collection(collection_name: str = "deeptrace_research") -> None:
    """
    Creates a new collection in Qdrant if it does not already exist.
    
    A "collection" in Qdrant is equivalent to a "table" in a relational database.
    We must define the size of the vectors (384 for our chosen model) and the 
    distance metric used to compare them (Cosine similarity is standard for text).
    
    Args:
        collection_name (str): The name of the collection to create.
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
        # In a robust production app, we would raise this or log to a telemetry service
        raise


def store_documents(documents: List[Dict[str, Any]], collection_name: str = "deeptrace_research") -> None:
    """
    Converts text documents into embeddings and stores them in Qdrant.
    
    Args:
        documents: A list of dictionaries representing the search results. 
                   Expected format: {"snippet": "...", "title": "...", "source": "...", "question": "..."}
        collection_name: The Qdrant collection to store the data in.
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
        
        # Generate embeddings locally using fastembed.
        embeddings_generator = embedding_model.embed(texts)
        
        points = []
        # Loop over our documents and their corresponding newly-generated vectors
        for doc, vector in zip(documents, embeddings_generator):
            point_id = str(uuid.uuid4())
            
            point = PointStruct(
                id=point_id,
                vector=vector.tolist(), # Convert numpy array to standard Python list
                payload=doc             # Store all metadata (title, snippet, link)
            )
            points.append(point)
            
        # Upsert securely writes our points to the database.
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"Successfully stored {len(points)} documents in Qdrant.")
        
    except Exception as e:
        print(f"Error storing documents in Qdrant collection '{collection_name}': {e}")
        raise


def search_similar(query: str, collection_name: str = "deeptrace_research", limit: int = 3) -> List[Dict[str, Any]]:
    """
    Searches the vector database for documents most similar to the provided query.
    
    Args:
        query (str): The user's search string (e.g., a specific sub-question).
        collection_name (str): The collection to search within.
        limit (int): How many top results to return.
        
    Returns:
        A list of dictionaries containing the metadata (payload) of the closest matches.
    """
    try:
        client = get_qdrant_client()
        embedding_model = get_embedding_model()
        
        # 1. Convert the query text into its vector representation
        query_vector = list(embedding_model.embed([query]))[0]
        
        # 2. Perform the semantic similarity search in Qdrant
        search_result = client.query_points(
            collection_name=collection_name,
            query=query_vector.tolist(),
            limit=limit
        )
        
        # 3. Extract and return just the original payload (text and metadata)
        retrieved_docs = []
        for hit in search_result.points:
            if hit.payload:
                retrieved_docs.append(hit.payload)
            
        return retrieved_docs
        
    except Exception as e:
        print(f"Error searching similar documents in Qdrant: {e}")
        # Return an empty list so upstream nodes don't crash, allowing graceful degradation
        return []