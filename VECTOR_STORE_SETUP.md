# Local Vector Store Setup Guide

## Quick Start

### 1. Install Packages

```bash
# For Qdrant
pip install langchain-qdrant qdrant-client

# For Weaviate
pip install langchain-weaviate weaviate-client
```

### 2. Run the Examples

```bash
# Run all examples
python vec.py
```

---

## Qdrant Setup (Easiest - No Docker Required)

### Installation
```bash
pip install langchain-qdrant qdrant-client
```

### Usage Options

#### Option 1: In-Memory (No persistence)
```python
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# In-memory client
client = QdrantClient(location=":memory:")

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="my_collection",
    embedding=embeddings
)
```

#### Option 2: Local Persistence (Recommended)
```python
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Persistent client
client = QdrantClient(path="./qdrant_data")

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="my_collection",
    embedding=embeddings
)
```

#### Option 3: from_documents (Easiest)
```python
vectorstore = QdrantVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    path="./qdrant_local",
    collection_name="rag_collection"
)
```

### Features
- ✅ No Docker required
- ✅ Fast setup and execution
- ✅ Low memory usage
- ✅ Great for local development
- ✅ Simple metadata filtering

---

## Weaviate Setup (Production-Ready)

### Prerequisites
**Docker is required** for local Weaviate

### Installation

#### Step 1: Install Python Package
```bash
pip install langchain-weaviate weaviate-client
```

#### Step 2: Start Local Weaviate Server
```bash
# Start Weaviate container
docker run -d \
  -p 8080:8080 \
  -p 50051:50051 \
  --name weaviate \
  cr.weaviate.io/semitechnologies/weaviate:latest

# Check if running
docker ps
```

#### Step 3: Verify Weaviate is Running
```bash
# Should return {"links":[...]}
curl http://localhost:8080/v1/meta
```

### Usage

```python
import weaviate
from langchain_weaviate import WeaviateVectorStore
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Connect to local Weaviate
client = weaviate.connect_to_local(
    host="localhost",
    port=8080,
    grpc_port=50051
)

# Create vector store
vectorstore = WeaviateVectorStore(
    client=client,
    index_name="MyDocuments",
    text_key="text",
    embedding=embeddings
)

# Or create from documents
vectorstore = WeaviateVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    client=client,
    index_name="MyDocuments"
)

# Don't forget to close
client.close()
```

### Features
- ✅ Production-ready
- ✅ Excellent scalability
- ✅ Advanced filtering
- ✅ GraphQL API
- ✅ Multi-tenancy support

### Troubleshooting Weaviate

#### Container not starting?
```bash
# Check logs
docker logs weaviate

# Restart container
docker restart weaviate

# Remove and recreate
docker rm -f weaviate
docker run -d -p 8080:8080 -p 50051:50051 --name weaviate \
  cr.weaviate.io/semitechnologies/weaviate:latest
```

#### Port already in use?
```bash
# Find process using port 8080
lsof -i :8080

# Use different port
docker run -d -p 8081:8080 -p 50052:50051 --name weaviate \
  cr.weaviate.io/semitechnologies/weaviate:latest
```

---

## Comparison Table

| Feature | ChromaDB | Qdrant | Weaviate |
|---------|----------|--------|----------|
| **Setup Difficulty** | Easy | Easy | Medium (Docker) |
| **Persistence** | Local DB | File/Memory | Docker volume |
| **Performance** | Good | Fast | Very Fast |
| **Memory Usage** | Medium | Low | Medium |
| **Scalability** | Good | Good | Excellent |
| **Filtering** | Simple | Simple | Advanced |
| **Best For** | Prototyping | Local Dev | Production |
| **Python 3.13** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Docker Required** | ❌ No | ❌ No | ✅ Yes |

---

## Recommendations

### For Local Development (No Cloud)
1. **Quick prototypes**: ChromaDB (easiest)
2. **Local development**: Qdrant (fast, no Docker)
3. **Production testing**: Weaviate (Docker required)

### For RAG Applications
- **Small projects (<1M docs)**: ChromaDB or Qdrant
- **Medium projects**: Qdrant
- **Large projects (>1M docs)**: Weaviate

### For Python 3.13
- ✅ **ChromaDB**: Fully compatible
- ✅ **Qdrant**: Fully compatible
- ✅ **Weaviate**: Fully compatible

---

## Complete RAG Example (Qdrant + Ollama)

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load documents
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# 2. Split
splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
chunks = splitter.split_documents(docs)

# 3. Create embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 4. Create Qdrant vector store
vectorstore = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    path="./qdrant_local",
    collection_name="rag_collection"
)

# 5. Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 6. Configure LLM
llm = ChatOllama(model="gemma3:1b", temperature=0)

# 7. Create RAG chain
prompt = ChatPromptTemplate.from_template(
    "Answer based on context:\n\n{context}\n\nQuestion: {question}"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 8. Ask questions
answer = rag_chain.invoke("What is this document about?")
print(answer)
```

---

## Switching Between Vector Stores

All three vector stores work with the same LangChain interface:

```python
# Just change the vectorstore creation, everything else stays the same!

# ChromaDB
from langchain_chroma import Chroma
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma")

# Qdrant
from langchain_qdrant import QdrantVectorStore
vectorstore = QdrantVectorStore.from_documents(docs, embeddings, path="./qdrant")

# Weaviate
from langchain_weaviate import WeaviateVectorStore
vectorstore = WeaviateVectorStore.from_documents(docs, embeddings, client=client)

# Then use the same retriever interface
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```

---

## Next Steps

1. Run `python vec.py` to see all examples
2. Choose the vector store that fits your needs
3. Integrate into your RAG pipeline
4. Experiment with different embedding models
5. Test with your own documents

**Happy building!** 🚀
