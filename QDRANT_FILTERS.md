# Qdrant Filter Syntax Guide

## The Error You Encountered

```
AttributeError: 'dict' object has no attribute 'must'
```

**Cause**: Qdrant doesn't accept simple dictionary filters like ChromaDB. You must use `Filter` objects.

## ✅ Correct Qdrant Filter Syntax

### Single Condition Filter

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Create filter
qdrant_filter = Filter(
    must=[
        FieldCondition(
            key="metadata.topic",  # Note: use "metadata.field_name"
            match=MatchValue(value="rag")
        )
    ]
)

# Use in search
results = vectorstore.similarity_search(
    "your query",
    k=4,
    filter=qdrant_filter
)
```

### Multiple Conditions (AND logic)

```python
# All conditions must match
multi_filter = Filter(
    must=[
        FieldCondition(key="metadata.topic", match=MatchValue(value="rag")),
        FieldCondition(key="metadata.difficulty", match=MatchValue(value="intermediate"))
    ]
)
```

### OR Logic

```python
# Any condition can match
or_filter = Filter(
    should=[
        FieldCondition(key="metadata.topic", match=MatchValue(value="rag")),
        FieldCondition(key="metadata.topic", match=MatchValue(value="langchain"))
    ]
)
```

### NOT Logic

```python
# Exclude certain values
not_filter = Filter(
    must_not=[
        FieldCondition(key="metadata.difficulty", match=MatchValue(value="beginner"))
    ]
)
```

### Combined Logic (AND + OR + NOT)

```python
complex_filter = Filter(
    must=[
        FieldCondition(key="metadata.topic", match=MatchValue(value="rag"))
    ],
    should=[
        FieldCondition(key="metadata.difficulty", match=MatchValue(value="intermediate")),
        FieldCondition(key="metadata.difficulty", match=MatchValue(value="advanced"))
    ],
    must_not=[
        FieldCondition(key="metadata.deprecated", match=MatchValue(value=True))
    ]
)
```

## ❌ Common Mistakes

### Mistake 1: Using Simple Dictionary (ChromaDB style)
```python
# ❌ WRONG - This works in ChromaDB but NOT in Qdrant
filter={"topic": "rag"}  # AttributeError!
```

### Mistake 2: Wrong Key Name
```python
# ❌ WRONG - Missing "metadata." prefix
FieldCondition(key="topic", match=MatchValue(value="rag"))

# ✅ CORRECT - Use "metadata.field_name"
FieldCondition(key="metadata.topic", match=MatchValue(value="rag"))
```

### Mistake 3: Forgetting MatchValue
```python
# ❌ WRONG
FieldCondition(key="metadata.topic", match="rag")

# ✅ CORRECT
FieldCondition(key="metadata.topic", match=MatchValue(value="rag"))
```

## 📊 ChromaDB vs Qdrant Comparison

| Feature | ChromaDB | Qdrant |
|---------|----------|--------|
| **Simple filter** | `{"key": "value"}` | `Filter(must=[FieldCondition(...)])` |
| **AND logic** | `{"key1": "val1", "key2": "val2"}` | `Filter(must=[cond1, cond2])` |
| **OR logic** | `{"$or": [{"key": "val1"}, {"key": "val2"}]}` | `Filter(should=[cond1, cond2])` |
| **NOT logic** | `{"key": {"$ne": "value"}}` | `Filter(must_not=[cond1])` |

## 🔧 Complete Working Example

```python
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Initialize
embeddings = OllamaEmbeddings(model="nomic-embed-text")
client = QdrantClient(path="./qdrant_local")

# Create documents with metadata
docs = [
    Document(
        page_content="RAG combines retrieval and generation",
        metadata={"topic": "rag", "difficulty": "intermediate", "year": 2023}
    ),
    Document(
        page_content="LangChain simplifies LLM apps",
        metadata={"topic": "langchain", "difficulty": "beginner", "year": 2024}
    )
]

# Create vector store
vectorstore = QdrantVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    path="./qdrant_local",
    collection_name="test"
)

# Example 1: Single condition
filter1 = Filter(
    must=[FieldCondition(key="metadata.topic", match=MatchValue(value="rag"))]
)
results1 = vectorstore.similarity_search("tell me about RAG", k=2, filter=filter1)

# Example 2: Multiple conditions (AND)
filter2 = Filter(
    must=[
        FieldCondition(key="metadata.topic", match=MatchValue(value="rag")),
        FieldCondition(key="metadata.difficulty", match=MatchValue(value="intermediate"))
    ]
)
results2 = vectorstore.similarity_search("RAG info", k=2, filter=filter2)

# Example 3: OR logic
filter3 = Filter(
    should=[
        FieldCondition(key="metadata.topic", match=MatchValue(value="rag")),
        FieldCondition(key="metadata.topic", match=MatchValue(value="langchain"))
    ]
)
results3 = vectorstore.similarity_search("AI tools", k=2, filter=filter3)

print(f"Filter 1 results: {len(results1)}")
print(f"Filter 2 results: {len(results2)}")
print(f"Filter 3 results: {len(results3)}")
```

## 🎯 Quick Reference

### Required Imports
```python
from qdrant_client.models import Filter, FieldCondition, MatchValue
```

### Filter Template
```python
my_filter = Filter(
    must=[...],      # AND: all must match
    should=[...],    # OR: at least one must match
    must_not=[...]   # NOT: none must match
)
```

### FieldCondition Template
```python
FieldCondition(
    key="metadata.field_name",
    match=MatchValue(value="your_value")
)
```

## 📚 Additional Resources

- Qdrant Filtering Docs: https://qdrant.tech/documentation/concepts/filtering/
- LangChain Qdrant: https://python.langchain.com/docs/integrations/vectorstores/qdrant
- Qdrant Client Models: https://qdrant.tech/documentation/concepts/filtering/#filtering-conditions

## 🐛 Troubleshooting

### Error: AttributeError: 'dict' object has no attribute 'must'
**Fix**: Replace dict filter with Filter object

### Error: KeyError: 'topic'
**Fix**: Use "metadata.topic" instead of "topic"

### No results returned with filter
**Check**:
1. Field name is correct (case-sensitive)
2. Using "metadata." prefix
3. Value type matches (string, int, bool)
4. Documents actually have that metadata

---

**Updated**: 2025-01-22
**For**: Qdrant with LangChain 1.0+
