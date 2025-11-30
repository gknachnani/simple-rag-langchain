# Filename: exercise2_wikipedia_fact_checker.py

"""
Exercise 2: Wikipedia Fact Checker (Beginner)

Task:
- Take a statement as input (e.g., "Python was created in 1991")
- Use WikipediaRetriever to search for relevant articles
- Use an LLM to verify if the statement is accurate

Setup (install these first):
    pip install langchain langchain-community langchain-openai wikipedia

And set your OpenAI API key:
    export OPENAI_API_KEY="your_key_here"   # macOS/Linux
    setx OPENAI_API_KEY "your_key_here"     # Windows (new terminal after)
"""

from typing import List

from langchain_community.retrievers import WikipediaRetriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


def format_docs(docs: List[Document]) -> str:
    """
    Convert a list of Documents into a single context string.
    """
    parts = []
    for i, doc in enumerate(docs, start=1):
        parts.append(f"[Source {i}]\n{doc.page_content}")
    return "\n\n".join(parts)


def build_fact_check_chain():
    """
    Build an LLM chain that:
    - Takes a statement and Wikipedia context
    - Returns True/False/Unknown + explanation
    """
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        """You are a careful fact-checking assistant.

Use ONLY the information in the context below to decide if the statement is accurate.
If the context does not clearly confirm or refute the statement, answer "Unknown".

Context:
{context}

Statement to verify:
"{statement}"

Respond in the following format:

Verdict: [True/False/Unknown]
Explanation: [brief explanation based only on the context]"""
    )

    chain = prompt | llm | StrOutputParser()
    return chain


def fact_check_statement(statement: str) -> str:
    """
    Fact-check a statement using WikipediaRetriever + LLM.
    """
    # 1. Retrieve relevant Wikipedia articles
    retriever = WikipediaRetriever(
        top_k_results=3,  # get top 3 relevant articles
        lang="en"
    )
    docs = retriever.invoke(statement)

    if not docs:
        return (
            "Verdict: Unknown\n"
            "Explanation: No relevant Wikipedia articles were found for this statement."
        )

    # 2. Build context from retrieved documents
    context = format_docs(docs)

    # 3. Run through the fact-checking chain
    fact_check_chain = build_fact_check_chain()
    result = fact_check_chain.invoke(
        {
            "context": context,
            "statement": statement,
        }
    )
    return result


def main():
    # Load .env file
    load_dotenv()
    
    # Example usage
    print("🔍 Wikipedia Fact Checker\n")
    # example_statement = "Python was created in 1991"
    example_statement = "Python was created in 1989"

    # You can change this or uncomment to read from input:
    # example_statement = input("Enter a statement to fact-check: ")

    print(f"Statement: {example_statement}\n")
    result = fact_check_statement(example_statement)
    print(result)


if __name__ == "__main__":
    main()
