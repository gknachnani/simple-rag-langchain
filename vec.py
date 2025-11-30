"""
Vector Store Examples: Qdrant & Weaviate (Local Setup)
Demonstrates creating, adding documents, and searching with local vector databases

Updated with correct imports for LangChain 1.0+
"""

# ============================================================================
# INSTALLATION INSTRUCTIONS
# ============================================================================
# For Qdrant:
#   pip install langchain-qdrant qdrant-client
#
# For Weaviate:
#   pip install langchain-weaviate weaviate-client
#
# For Ollama (for real embeddings):
#   pip install langchain-ollama
#
# For local Weaviate, you need Docker:
#   docker run -d -p 8080:8080 -p 50051:50051 \
#     --name weaviate \
#     cr.weaviate.io/semitechnologies/weaviate:latest
# ============================================================================

import os
import sys

# ============================================================================
# CORRECT IMPORTS FOR LANGCHAIN 1.0+
# ============================================================================
# ✓ Correct imports:
from langchain_core.documents import Document

# ❌ INCORRECT imports that cause errors:
# from langchain_core.memory import ...        # ERROR: memory not in langchain_core
# from langchain.schema import Document        # DEPRECATED: use langchain_core.documents

# ✓ If you need memory/chat history, use these instead:
# from langchain.memory import ConversationBufferMemory
# from langchain_community.chat_message_histories import ChatMessageHistory

print("✓ All imports loaded correctly!")
print("✓ Using langchain_core.documents.Document (correct)\n")

# ============================================================================
# FILTER SYNTAX COMPARISON: ChromaDB vs Qdrant
# ============================================================================
# ChromaDB (simple dict):
#   filter={"topic": "rag"}
#
# Qdrant (Filter object):
#   from qdrant_client.models import Filter, FieldCondition, MatchValue
#   filter=Filter(must=[FieldCondition(key="metadata.topic", match=MatchValue(value="rag"))])
#
# ============================================================================

# ============================================================================
# EXAMPLE 1: QDRANT (LOCAL)
# ============================================================================
print("=" * 80)
print("QDRANT LOCAL VECTOR STORE EXAMPLE")
print("=" * 80)
print("\nIMPORTANT: Qdrant filter syntax")
print("- Use Filter() with FieldCondition() for metadata filtering")
print("- Not like ChromaDB (simple dict filters don't work)")
print("=" * 80)

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from langchain_ollama import OllamaEmbeddings

print("\nInitializing Ollama embeddings (nomic-embed-text)...")
print("Make sure Ollama is running: ollama serve")

# Initialize Ollama embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
print("✓ Ollama embeddings initialized")

# ============================================================================
# OPTION 1: Qdrant In-Memory (No persistence)
# ============================================================================
print("\n--- Option 1: Qdrant In-Memory ---")

# Create in-memory Qdrant client
qdrant_client_memory = QdrantClient(location=":memory:")

# Create Qdrant vector store
qdrant_client_memory.recreate_collection(
    collection_name="my_collection_memory",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)
qdrant_store_memory = QdrantVectorStore(
    client=qdrant_client_memory,
    collection_name="my_collection_memory",
    embedding=embeddings
)

# Add documents
sample_docs = [
    Document(
        page_content="RAG combines retrieval and generation",
        metadata={"topic": "rag", "difficulty": "intermediate"}
    ),
    Document(
        page_content="LangChain simplifies LLM applications",
        metadata={"topic": "langchain", "difficulty": "beginner"}
    ),
    Document(
        page_content="Vector databases enable semantic search",
        metadata={"topic": "vectordb", "difficulty": "intermediate"}
    )
]

qdrant_store_memory.add_documents(sample_docs)
print("✓ Added documents to Qdrant (in-memory)")

# Search without filter
print("\n--- Basic Search ---")
results = qdrant_store_memory.similarity_search(
    "Tell me about RAG",
    k=2
)

print("Search results:")
for i, doc in enumerate(results, 1):
    print(f"  {i}. {doc.page_content}")
    print(f"     Metadata: {doc.metadata}")

# Search with metadata filter
print("\n--- Search with Metadata Filter ---")
# Qdrant requires Filter object with FieldCondition
qdrant_filter = Filter(
    must=[
        FieldCondition(
            key="metadata.topic",
            match=MatchValue(value="rag")
        )
    ]
)

results_filtered = qdrant_store_memory.similarity_search(
    "Tell me about RAG",
    k=2,
    filter=qdrant_filter
)

print("Filtered search results (topic='rag'):")
for i, doc in enumerate(results_filtered, 1):
    print(f"  {i}. {doc.page_content}")
    print(f"     Metadata: {doc.metadata}")

# Example: Multiple filter conditions (AND logic)
print("\n--- Multiple Filters Example ---")
multi_filter = Filter(
    must=[
        FieldCondition(key="metadata.topic", match=MatchValue(value="rag")),
        FieldCondition(key="metadata.difficulty", match=MatchValue(value="intermediate"))
    ]
)
# results_multi = qdrant_store_memory.similarity_search("RAG", k=2, filter=multi_filter)
print("ℹ️  For multiple conditions, use Filter(must=[condition1, condition2, ...])")

# ============================================================================
# OPTION 2: Qdrant with Local Persistence
# ============================================================================
print("\n" + "=" * 80)
print("--- Option 2: Qdrant with Local Persistence ---")
print("=" * 80)

# Create persistent Qdrant client
qdrant_path = "./qdrant_data"
qdrant_client_persistent = QdrantClient(path=qdrant_path)

qdrant_client_persistent.recreate_collection(
    collection_name="my_collection_persistent",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

# Create Qdrant vector store with persistence
qdrant_store_persistent = QdrantVectorStore(
    client=qdrant_client_persistent,
    collection_name="my_collection_persistent",
    embedding=embeddings
)

# Add documents
qdrant_store_persistent.add_documents(sample_docs)
print(f"✓ Added documents to Qdrant (persistent at {qdrant_path})")

# Search
results = qdrant_store_persistent.similarity_search(
    "Tell me about LangChain",
    k=2
)

print("\nSearch results:")
for i, doc in enumerate(results, 1):
    print(f"  {i}. {doc.page_content}")
    print(f"     Metadata: {doc.metadata}")

# ============================================================================
# OPTION 3: Qdrant from_documents (Recommended)
# ============================================================================
print("\n" + "=" * 80)
print("--- Option 3: Qdrant from_documents (Recommended) ---")
print("=" * 80)

# Create directly from documents (easier)
qdrant_store_easy = QdrantVectorStore.from_documents(
    documents=sample_docs,
    embedding=embeddings,
    path="./qdrant_easy",  # Local persistence
    collection_name="rag_collection"
)

print("✓ Created Qdrant store from documents")

# Search with score
results_with_scores = qdrant_store_easy.similarity_search_with_score(
    "Vector databases",
    k=3
)

print("\nSearch results with scores:")
for doc, score in results_with_scores:
    print(f"  Score: {score:.4f}")
    print(f"  Content: {doc.page_content}")
    print(f"  Metadata: {doc.metadata}")
    print()

# ============================================================================
# EXAMPLE 2: WEAVIATE (LOCAL)
# ============================================================================
# 
# Note: This example requires Docker to run a local Weaviate instance
# Run: docker run -d -p 8080:8080 -p 50051:50051 --name weaviate cr.weaviate.io/semitechnologies/weaviate:latest
print("\n" + "=" * 80)
print("WEAVIATE LOCAL VECTOR STORE EXAMPLE")
print("=" * 80)
print("\nNote: Make sure Weaviate is running locally on port 8080")
print("Docker command: docker run -d -p 8080:8080 -p 50051:50051 \\")
print("  --name weaviate cr.weaviate.io/semitechnologies/weaviate:latest\n")

try:
    import weaviate
    from langchain_weaviate import WeaviateVectorStore

    # ============================================================================
    # Connect to local Weaviate instance
    # ============================================================================
    print("--- Connecting to Local Weaviate ---")

    # Create Weaviate client (local)
    weaviate_client = weaviate.connect_to_local(
        host="localhost",
        port=8080,
        grpc_port=50051
    )

    print("✓ Connected to local Weaviate")

    # ============================================================================
    # Create Weaviate vector store
    # ============================================================================
    print("\n--- Creating Weaviate Vector Store ---")

    # Method 1: Create from existing client
    weaviate_store = WeaviateVectorStore(
        client=weaviate_client,
        index_name="MyDocuments",
        text_key="text",
        embedding=embeddings
    )

    # Add documents
    weaviate_store.add_documents(sample_docs)
    print("✓ Added documents to Weaviate")

    # ============================================================================
    # Search
    # ============================================================================
    print("\n--- Basic Search ---")
    results = weaviate_store.similarity_search(
        "Tell me about RAG",
        k=2
    )

    print("Search results:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc.page_content}")
        print(f"     Metadata: {doc.metadata}")

    # ============================================================================
    # Search with metadata filter
    # ============================================================================
    print("\n--- Search with Metadata Filter ---")

    # Weaviate uses where filter syntax
    results_filtered = weaviate_store.similarity_search(
        "Tell me about databases",
        k=2,
        where_filter={
            "path": ["difficulty"],
            "operator": "Equal",
            "valueText": "intermediate"
        }
    )

    print("Filtered search results (difficulty=intermediate):")
    for i, doc in enumerate(results_filtered, 1):
        print(f"  {i}. {doc.page_content}")
        print(f"     Metadata: {doc.metadata}")

    # ============================================================================
    # Search with score
    # ============================================================================
    print("\n--- Search with Scores ---")
    results_with_scores = weaviate_store.similarity_search_with_score(
        "Vector databases",
        k=3
    )

    print("Search results with scores:")
    for doc, score in results_with_scores:
        print(f"  Score: {score:.4f}")
        print(f"  Content: {doc.page_content}")
        print(f"  Metadata: {doc.metadata}")
        print()

    # ============================================================================
    # Method 2: Create from documents (easier)
    # ============================================================================
    print("\n--- Creating Weaviate from Documents ---")

    weaviate_store_easy = WeaviateVectorStore.from_documents(
        documents=sample_docs,
        embedding=embeddings,
        client=weaviate_client,
        index_name="EasyDocuments"
    )

    print("✓ Created Weaviate store from documents")

    # Search
    results = weaviate_store_easy.similarity_search("LangChain", k=2)
    print("\nSearch results:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc.page_content}")

    # Close connection
    weaviate_client.close()
    print("\n✓ Closed Weaviate connection")

except Exception as e:
    print(f"✗ Weaviate error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure Weaviate is running: docker ps")
    print("2. Start Weaviate: docker run -d -p 8080:8080 -p 50051:50051 \\")
    print("     --name weaviate cr.weaviate.io/semitechnologies/weaviate:latest")
    print("3. Check if port 8080 is available")

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("VECTOR STORE COMPARISON SUMMARY")
print("=" * 80)

comparison = """
┌─────────────┬──────────────────────┬──────────────────────┐
│  Feature    │  Qdrant              │  Weaviate            │
├─────────────┼──────────────────────┼──────────────────────┤
│ Setup       │ Easy (no Docker)     │ Requires Docker      │
│ Persistence │ Local file/in-memory │ Docker volume        │
│ Performance │ Fast                 │ Very fast            │
│ Scalability │ Good                 │ Excellent            │
│ Filters     │ Simple dict          │ Where filter syntax  │
│ Memory      │ Low                  │ Medium               │
│ Best for    │ Local development    │ Production-ready     │
└─────────────┴──────────────────────┴──────────────────────┘

RECOMMENDATION:
- For local development & testing: Use Qdrant (no Docker needed)
- For production or advanced features: Use Weaviate (better scalability)
- For quick prototypes: ChromaDB (easiest setup)
"""

print(comparison)

# ============================================================================
# COMPLETE EXAMPLE WITH REAL EMBEDDINGS (Ollama)
# ============================================================================
print("\n" + "=" * 80)
print("BONUS: REAL OLLAMA EMBEDDINGS EXAMPLE")
print("=" * 80)
print("\nThis example uses real Ollama embeddings (if available)")

try:
    from langchain_ollama import OllamaEmbeddings
    from langchain_qdrant import QdrantVectorStore

    print("\n--- Testing Ollama Connection ---")

    # Initialize Ollama embeddings
    ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Test embeddings
    test_text = "This is a test with real Ollama embeddings"
    test_embedding = ollama_embeddings.embed_query(test_text)

    print(f"✓ Ollama embeddings working!")
    print(f"  Model: nomic-embed-text")
    print(f"  Embedding dimension: {len(test_embedding)}")
    print(f"  Sample values: {test_embedding[:5]}")

    # Create Qdrant store with real embeddings
    print("\n--- Creating Qdrant with Ollama Embeddings ---")

    ollama_qdrant_store = QdrantVectorStore.from_documents(
        documents=sample_docs,
        embedding=ollama_embeddings,
        path="./qdrant_ollama",
        collection_name="ollama_collection"
    )

    print("✓ Created Qdrant store with Ollama embeddings")

    # Search with real embeddings
    query = "Tell me about RAG"
    results = ollama_qdrant_store.similarity_search(query, k=2)

    print(f"\nSearch query: '{query}'")
    print("Results:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc.page_content}")
        print(f"     Topic: {doc.metadata.get('topic')}")

    print("\n✓ Ollama integration successful!")
    print("\nTo use in RAG:")
    print("  retriever = ollama_qdrant_store.as_retriever(search_kwargs={'k': 4})")

except ImportError:
    print("✗ langchain-ollama not installed")
    print("\nInstall with: pip install langchain-ollama")
    print("\nExample code to use when installed:")
    print("""
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

# Initialize embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Create vector store
qdrant_store = QdrantVectorStore.from_documents(
    documents=your_documents,
    embedding=embeddings,
    path="./qdrant_local",
    collection_name="my_collection"
)

# Use as retriever in RAG
retriever = qdrant_store.as_retriever(search_kwargs={"k": 4})
    """)

except Exception as e:
    print(f"✗ Error with Ollama: {e}")
    print("\nMake sure:")
    print("1. Ollama is running: ollama serve")
    print("2. Model is downloaded: ollama pull nomic-embed-text")
    print("3. langchain-ollama is installed: pip install langchain-ollama")

# ============================================================================
# SUMMARY AND NEXT STEPS
# ============================================================================
print("\n" + "=" * 80)
print("✓ VECTOR STORE EXAMPLES COMPLETED!")
print("=" * 80)

summary = """
What we covered:
1. ✓ Qdrant (in-memory) - No persistence, fast testing
2. ✓ Qdrant (persistent) - Local file storage
3. ✓ Qdrant (from_documents) - Easiest method
4. ✓ Weaviate (local Docker) - Production-ready
5. ✓ Ollama embeddings - Real embeddings integration

Key Takeaways:
- Use Qdrant for local development (no Docker needed)
- Use Weaviate for production (requires Docker)
- Always use correct imports: from langchain_core.documents import Document
- Avoid deprecated imports: from langchain.schema import Document

Next Steps:
1. Choose your vector store (Qdrant recommended for local)
2. Install required packages
3. Replace DummyEmbeddings with OllamaEmbeddings
4. Integrate into your RAG pipeline
5. Test with your own documents

For help:
- Qdrant docs: https://qdrant.tech/documentation/
- Weaviate docs: https://weaviate.io/developers/weaviate
- LangChain docs: https://python.langchain.com/docs/
"""

print(summary)
print("=" * 80)
