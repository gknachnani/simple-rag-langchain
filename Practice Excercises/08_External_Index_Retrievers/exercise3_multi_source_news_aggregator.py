# Filename: exercise3_multi_source_news_aggregator.py

"""
Exercise 3: Multi-Source News Aggregator (Intermediate)

Task:
- Use TavilySearchAPIRetriever to get latest AI news
- Use WikipediaRetriever to get background on AI topics
- Combine both sources to provide a comprehensive news summary

Setup (install these first):
    pip install langchain langchain-community langchain-openai tavily-python wikipedia

Set your API keys as environment variables:
    export OPENAI_API_KEY="your_openai_key"    # macOS/Linux
    export TAVILY_API_KEY="your_tavily_key"

    # On Windows (PowerShell):
    setx OPENAI_API_KEY "your_openai_key"
    setx TAVILY_API_KEY "your_tavily_key"
"""

from typing import List

from langchain_community.retrievers import TavilySearchAPIRetriever, WikipediaRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


def format_docs(docs: List[Document]) -> str:
    """
    Convert a list of Documents into a single context string.
    Each source is tagged for clarity.
    """
    parts = []
    for i, doc in enumerate(docs, start=1):
        title = doc.metadata.get("title") or doc.metadata.get("source") or f"Source {i}"
        parts.append(f"[{title}]\n{doc.page_content}")
    return "\n\n".join(parts)


def build_multi_source_news_chain():
    """
    Build a RAG chain that:
    - Uses TavilySearchAPIRetriever for latest AI news
    - Uses WikipediaRetriever for background
    - Combines both into a comprehensive summary
    """

    # News: Tavily retriever (real-time web search)
    tavily_retriever = TavilySearchAPIRetriever(
        k=5,  # number of results
        search_type="news"  # focus on news
    )

    # Background: Wikipedia retriever
    wiki_retriever = WikipediaRetriever(
        top_k_results=3,
        lang="en",
    )

    # LLM for summarization
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        """You are an AI news analyst.

User topic:
{topic}

You have two types of context:

[Latest AI news]
{news_context}

[Background from Wikipedia]
{wiki_context}

Write a clear, comprehensive summary that includes:

1) A short overview (2–3 sentences) of what's happening in AI related to this topic.
2) 3–5 bullet points describing the most important recent developments.
3) 2–3 bullet points giving background/context so a non-expert can understand why these developments matter.
4) If anything is uncertain or evolving, briefly mention it.

Stay factual and concise. Do not invent sources or dates."""
    )

    # RAG chain: given a topic string, run both retrievers + combine with LLM
    multi_source_news_chain = (
        {
            "topic": RunnablePassthrough(),
            "news_context": tavily_retriever | format_docs,
            "wiki_context": wiki_retriever | format_docs,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return multi_source_news_chain


def main():
    # Load .env file
    load_dotenv()
    
    print("🗞️ Multi-Source AI News Aggregator\n")

    # You can change this topic or read from input:
    topic = "latest developments in artificial intelligence"
    # topic = input("Enter an AI topic (e.g., 'AI regulation', 'LLM advancements'): ")

    chain = build_multi_source_news_chain()

    print(f"Topic: {topic}\n")
    summary = chain.invoke(topic)
    print(summary)


if __name__ == "__main__":
    main()
