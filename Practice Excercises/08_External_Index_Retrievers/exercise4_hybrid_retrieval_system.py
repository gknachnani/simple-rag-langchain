# Filename: exercise4_hybrid_retrieval_system.py

"""
Exercise 4: Hybrid Retrieval System (Intermediate)

Task:
- Classify queries into "academic", "general", or "current_events"
- Route to the appropriate retriever based on query type
- Return results from the most relevant source

Retrievers:
- academic       -> ArxivRetriever (research papers)
- general        -> WikipediaRetriever (encyclopedic info)
- current_events -> TavilySearchAPIRetriever (latest news/web)

Setup (install these first):
    pip install langchain langchain-community langchain-openai tavily-python wikipedia arxiv

Set your API keys as environment variables:
    export OPENAI_API_KEY="your_openai_key"    # macOS/Linux
    export TAVILY_API_KEY="your_tavily_key"

    # On Windows (PowerShell):
    setx OPENAI_API_KEY "your_openai_key"
    setx TAVILY_API_KEY "your_tavily_key"
"""

from typing import List

from langchain_community.retrievers import (
    ArxivRetriever,
    WikipediaRetriever,
    TavilySearchAPIRetriever,
)
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


# ---------- Utility: formatting ----------

def format_docs(docs: List[Document], max_chars: int = 800) -> str:
    """
    Convert a list of Documents into a readable string.
    Truncates each document's content for brevity.
    """
    parts = []
    for i, doc in enumerate(docs, start=1):
        title = (
            doc.metadata.get("title")
            or doc.metadata.get("source")
            or doc.metadata.get("page")
            or f"Result {i}"
        )
        content = doc.page_content.strip()
        if len(content) > max_chars:
            content = content[: max_chars - 3] + "..."
        parts.append(f"[{title}]\n{content}")
    return "\n\n".join(parts)


# ---------- 1) Query classification chain ----------

def build_query_classifier_chain():
    """
    Build an LLM chain that classifies a query into:
    - academic
    - general
    - current_events
    """

    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        """You are a routing classifier.

Classify the user's query into exactly ONE of these categories:
- academic      : research-level, papers, theory, algorithms, math, scientific topics
- general       : broad knowledge, definitions, history, how-things-work, biographies
- current_events: news, latest developments, recent incidents, 'today', 'this year', etc.

Return ONLY one word: academic, general, or current_events.

Query:
{query}
"""
    )

    chain = prompt | llm | StrOutputParser()
    return chain


def classify_query(query: str) -> str:
    """
    Run the classifier chain and normalize the result.
    Defaults to 'general' if classification is unclear.
    """
    chain = build_query_classifier_chain()
    raw = chain.invoke({"query": query}).strip().lower()

    if "academic" in raw:
        return "academic"
    if "current_events" in raw or "current" in raw or "news" in raw:
        return "current_events"
    if "general" in raw:
        return "general"

    # Fallback
    return "general"


# ---------- 2) Retrieve based on query type ----------

def get_retriever_for_type(query_type: str):
    """
    Return the appropriate retriever instance for a given query type.
    """
    if query_type == "academic":
        # Research papers from arXiv
        return ArxivRetriever(load_max_docs=3)

    if query_type == "current_events":
        # Real-time web/news search
        return TavilySearchAPIRetriever(
            k=5,
            search_type="news",  # focus search on news
        )

    # Default: general knowledge -> Wikipedia
    return WikipediaRetriever(
        top_k_results=3,
        lang="en",
    )


def hybrid_retrieve(query: str) -> str:
    """
    Full hybrid retrieval flow:
    - Classify query
    - Pick retriever
    - Retrieve and format results
    """
    # 1. Classify
    query_type = classify_query(query)

    # 2. Retrieve
    retriever = get_retriever_for_type(query_type)
    docs: List[Document] = retriever.invoke(query)

    if not docs:
        return (
            f"Query type: {query_type}\n"
            "No results found from the selected source."
        )

    # 3. Format results
    context = format_docs(docs)

    header = f"Query type: {query_type}\nSelected source: "
    if query_type == "academic":
        header += "arXiv (academic papers)\n"
    elif query_type == "current_events":
        header += "Tavily (current news/web)\n"
    else:
        header += "Wikipedia (general knowledge)\n"

    return header + "\n" + context


# ---------- Demo main ----------

def main():
    
    # Load .env file
    load_dotenv()
    
    print("🔀 Hybrid Retrieval System (academic / general / current_events)\n")

    # Try changing these example queries:
    # query = "Explain the transformer architecture in deep learning"
    # query = "Who is the current president of the United States?"
    # query = "History of the Eiffel Tower"
    query = "Latest developments in AI regulation"

    # Or uncomment to accept user input:
    # query = input("Enter your query: ")

    print(f"User query: {query}\n")
    result = hybrid_retrieve(query)
    print(result)


if __name__ == "__main__":
    main()
