# Easy.py
# Create 5 LangChain Documents with metadata and print them

from langchain_core.documents import Document

# Create documents with meaningful metadata
documents = [
    Document(
        page_content="Python is a versatile programming language used for web, data science, and automation.",
        metadata={"topic": "Python", "category": "Programming", "level": "Beginner"}
    ),
    Document(
        page_content="Machine Learning allows computers to learn patterns from data and make predictions.",
        metadata={"topic": "Machine Learning", "category": "AI", "level": "Intermediate"}
    ),
    Document(
        page_content="Climate change refers to long-term shifts in temperatures and weather patterns.",
        metadata={"topic": "Climate Change", "category": "Environment", "level": "Beginner"}
    ),
    Document(
        page_content="Quantum computing uses the principles of quantum mechanics to perform complex computations.",
        metadata={"topic": "Quantum Computing", "category": "Technology", "level": "Advanced"}
    ),
    Document(
        page_content="Healthy nutrition involves balanced meals including proteins, carbs, fats, vitamins, and minerals.",
        metadata={"topic": "Nutrition", "category": "Health", "level": "Beginner"}
    ),
]

# Print documents
print("\n=== Created LangChain Documents ===\n")

for i, doc in enumerate(documents, start=1):
    print(f"Document {i}:")
    print(f"  Content  : {doc.page_content}")
    print(f"  Metadata : {doc.metadata}")
    print("-" * 60)
