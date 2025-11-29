# LangChain 1.0+ Import Reference

## Common Import Errors and Fixes

### ❌ ERROR: `ModuleNotFoundError: No module named 'langchain_core.memory'`

**Cause**: `memory` is not in `langchain_core`

**Fix**: Use the correct import location:
```python
# ✓ CORRECT - Memory is in langchain (not langchain_core)
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationSummaryMemory

# ✓ CORRECT - Chat history is in langchain_community
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory
```

---

## Import Cheat Sheet

### Documents
```python
# ✓ CORRECT (LangChain 1.0+)
from langchain_core.documents import Document

# ❌ DEPRECATED (Old LangChain)
from langchain.schema import Document
from langchain.docstore.document import Document
```

### Embeddings
```python
# ✓ CORRECT - OpenAI
from langchain_openai import OpenAIEmbeddings

# ✓ CORRECT - Ollama (Local)
from langchain_ollama import OllamaEmbeddings

# ✓ CORRECT - HuggingFace
from langchain_huggingface import HuggingFaceEmbeddings

# ❌ WRONG
from langchain_core.embeddings import ...  # Embeddings are NOT in core
```

### Vector Stores
```python
# ✓ CORRECT - ChromaDB
from langchain_chroma import Chroma

# ✓ CORRECT - Qdrant
from langchain_qdrant import QdrantVectorStore

# ✓ CORRECT - Weaviate
from langchain_weaviate import WeaviateVectorStore

# ✓ CORRECT - FAISS
from langchain_community.vectorstores import FAISS

# ❌ WRONG
from langchain_core.vectorstores import ...  # Only InMemoryVectorStore is in core
```

### LLMs
```python
# ✓ CORRECT - OpenAI
from langchain_openai import ChatOpenAI, OpenAI

# ✓ CORRECT - Ollama (Local)
from langchain_ollama import ChatOllama, OllamaLLM

# ✓ CORRECT - Google
from langchain_google_genai import ChatGoogleGenerativeAI

# ✓ CORRECT - Anthropic
from langchain_anthropic import ChatAnthropic
```

### Text Splitters
```python
# ✓ CORRECT
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import TokenTextSplitter

# ❌ DEPRECATED (still works but old)
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

### Document Loaders
```python
# ✓ CORRECT
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import WebBaseLoader
```

### Prompts
```python
# ✓ CORRECT
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
```

### Runnables (LCEL)
```python
# ✓ CORRECT
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableSequence
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
```

### Memory & Chat History
```python
# ✓ CORRECT - Memory classes
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationSummaryMemory
from langchain.memory import ConversationSummaryBufferMemory

# ✓ CORRECT - Chat message histories
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

# ❌ WRONG - These don't exist!
from langchain_core.memory import ...  # ERROR!
```

---

## Package Organization

LangChain 1.0+ is organized into several packages:

### `langchain-core` (Base abstractions)
- ✅ Documents: `Document`
- ✅ Prompts: `ChatPromptTemplate`, `PromptTemplate`
- ✅ Runnables: `RunnablePassthrough`, etc.
- ✅ Messages: `HumanMessage`, `AIMessage`
- ✅ Output Parsers: `StrOutputParser`
- ❌ NOT embeddings, LLMs, vector stores, memory

### `langchain-community` (Community integrations)
- ✅ Document Loaders: `PyPDFLoader`, etc.
- ✅ Vector Stores: `FAISS` (legacy)
- ✅ Chat Message Histories
- ✅ Many other community tools

### `langchain-openai`
- ✅ `ChatOpenAI`, `OpenAI`
- ✅ `OpenAIEmbeddings`

### `langchain-ollama`
- ✅ `ChatOllama`, `OllamaLLM`
- ✅ `OllamaEmbeddings`

### `langchain-chroma`
- ✅ `Chroma` vector store

### `langchain-qdrant`
- ✅ `QdrantVectorStore`

### `langchain-weaviate`
- ✅ `WeaviateVectorStore`

### `langchain-text-splitters`
- ✅ `RecursiveCharacterTextSplitter`
- ✅ `CharacterTextSplitter`

### `langchain` (Main package - has memory)
- ✅ `ConversationBufferMemory`
- ✅ Other memory classes
- ⚠️ Many deprecated imports (prefer specific packages)

---

## Quick Migration Guide

### If you see: `ModuleNotFoundError: No module named 'X'`

1. **Check the package name** - It's probably in a separate package
2. **Install the package**:
   ```bash
   # For OpenAI
   pip install langchain-openai

   # For Ollama
   pip install langchain-ollama

   # For ChromaDB
   pip install langchain-chroma

   # For Qdrant
   pip install langchain-qdrant

   # For text splitters
   pip install langchain-text-splitters
   ```

3. **Use the correct import**:
   ```python
   # Old way (deprecated)
   from langchain.embeddings import OpenAIEmbeddings

   # New way (correct)
   from langchain_openai import OpenAIEmbeddings
   ```

---

## Installation Quick Reference

```bash
# Core (required)
pip install langchain langchain-core langchain-community

# Text processing
pip install langchain-text-splitters pypdf tiktoken

# Local LLM & Embeddings (Ollama)
pip install langchain-ollama

# OpenAI (if using cloud)
pip install langchain-openai openai

# Vector stores (choose one or more)
pip install langchain-chroma chromadb          # ChromaDB
pip install langchain-qdrant qdrant-client     # Qdrant
pip install langchain-weaviate weaviate-client # Weaviate
pip install faiss-cpu                          # FAISS

# Optional utilities
pip install python-dotenv  # For .env files
pip install jupyter notebook ipykernel  # For notebooks
```

---

## Complete Working Example

```python
# ============================================================================
# CORRECT IMPORTS (LangChain 1.0+)
# ============================================================================
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore

# ============================================================================
# BUILD RAG PIPELINE
# ============================================================================

# 1. Load documents
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# 2. Split
splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
chunks = splitter.split_documents(docs)

# 3. Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 4. Vector store
vectorstore = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    path="./qdrant_local",
    collection_name="rag"
)

# 5. Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 6. LLM
llm = ChatOllama(model="gemma3:1b", temperature=0)

# 7. Prompt
prompt = ChatPromptTemplate.from_template(
    "Answer based on context:\n\n{context}\n\nQuestion: {question}"
)

# 8. RAG Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 9. Ask questions
answer = rag_chain.invoke("What is this document about?")
print(answer)
```

---

## Need Help?

- **LangChain Docs**: https://python.langchain.com/docs/
- **Migration Guide**: https://python.langchain.com/docs/versions/migrating_chains/
- **API Reference**: https://api.python.langchain.com/

---

**Last Updated**: 2025
**LangChain Version**: 1.0+
