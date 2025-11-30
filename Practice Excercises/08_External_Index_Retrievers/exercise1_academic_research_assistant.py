# Filename: exercise1_academic_research_assistant.py

"""
Exercise 1: Academic Research Assistant (Beginner)

Task:
- Use ArxivRetriever to find papers on "deep learning"
- Extract the top 3 paper titles and authors
- Summarize each paper's abstract using an LLM

Setup (install these first):
    pip install langchain langchain-community langchain-openai arxiv
And set your OpenAI API key:
    export OPENAI_API_KEY="your_key_here"   # macOS/Linux
    setx OPENAI_API_KEY "your_key_here"     # Windows (new terminal after)
"""

from typing import List, Dict, Any

from langchain_community.retrievers import ArxivRetriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv





def get_deep_learning_papers(k: int = 3) -> List[Document]:
    """
    Use ArxivRetriever to get top-k papers on 'deep learning'.
    """
    retriever = ArxivRetriever(
        load_max_docs=k,
        # filter: you can optionally filter by date/category, etc.
    )
    docs: List[Document] = retriever.invoke("deep learning")
    return docs


def build_summarization_chain() -> Any:
    """
    Build an LLM chain that summarizes a paper's abstract.
    """
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        """You are helping with an academic literature review.

Summarize the following paper abstract in 3–5 concise bullet points.
Focus on:
- Problem the paper addresses
- Main method/approach
- Key results or contributions
- Any notable applications

Title: {title}
Authors: {authors}

Abstract:
{abstract}

Return only the bullet-point summary."""
    )

    chain = prompt | llm | StrOutputParser()
    return chain


def extract_metadata(doc: Document) -> Dict[str, str]:
    """
    Extract title, authors, and abstract from an Arxiv document.
    Metadata keys can vary slightly, so this is defensive.
    """
    meta = doc.metadata or {}

    title = meta.get("Title") or meta.get("title") or "Unknown title"
    authors = meta.get("Authors") or meta.get("authors") or "Unknown authors"

    # For ArxivRetriever, page_content usually contains the abstract/summary.
    abstract = meta.get("Summary") or meta.get("summary") or doc.page_content

    return {
        "title": title,
        "authors": authors,
        "abstract": abstract,
    }


def main():
    # Load .env file
    load_dotenv()
    
    # 1. Retrieve top 3 deep learning papers
    print("🔎 Searching arXiv for papers on 'deep learning'...\n")
    docs = get_deep_learning_papers(k=3)

    if not docs:
        print("No papers found. Try again with a different query.")
        return

    # 2. Prepare summarization chain
    summarize_chain = build_summarization_chain()

    # 3. For each paper: show title/authors and summarize abstract
    for idx, doc in enumerate(docs, start=1):
        info = extract_metadata(doc)

        print(f"================ Paper {idx} ================\n")
        print(f"Title  : {info['title']}")
        print(f"Authors: {info['authors']}\n")

        print("Abstract summary:\n")
        summary = summarize_chain.invoke(
            {
                "title": info["title"],
                "authors": info["authors"],
                "abstract": info["abstract"],
            }
        )
        print(summary)
        print("\n")  # extra spacing between papers


if __name__ == "__main__":
    main()
