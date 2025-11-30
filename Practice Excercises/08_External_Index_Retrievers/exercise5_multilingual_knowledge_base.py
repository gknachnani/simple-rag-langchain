# Filename: exercise5_multilingual_knowledge_base.py

"""
Exercise 5: Multilingual Knowledge Base (Advanced)

Task:
- Detect the language of the user's query
- Use WikipediaRetriever with the appropriate language setting
- Return answers in the user's language

Setup (install these first):
    pip install langchain langchain-community langchain-openai wikipedia langdetect

Set your OpenAI API key as an environment variable:
    export OPENAI_API_KEY="your_openai_key"    # macOS/Linux
    setx OPENAI_API_KEY "your_openai_key"     # Windows (new terminal after)
"""

from typing import List

from langdetect import detect  # simple language detection
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


# --------- Utilities --------- #

def format_docs(docs: List[Document], max_chars: int = 1000) -> str:
    """
    Convert a list of Documents into a single context string.
    Truncate each document for brevity.
    """
    parts = []
    for i, doc in enumerate(docs, start=1):
        title = doc.metadata.get("title") or f"Source {i}"
        content = doc.page_content.strip()
        if len(content) > max_chars:
            content = content[: max_chars - 3] + "..."
        parts.append(f"[{title}]\n{content}")
    return "\n\n".join(parts)


def detect_query_language(query: str) -> str:
    """
    Detect language of the query using langdetect.
    Returns a 2-letter ISO code suitable for Wikipedia (e.g., 'en', 'es', 'fr', 'de').
    Falls back to 'en' if detection fails.
    """
    try:
        lang_code = detect(query)
        # You can restrict to a whitelist of supported Wikipedia languages if needed.
        return lang_code
    except Exception:
        return "en"


def language_name_from_code(lang_code: str) -> str:
    """
    Simple mapping from language code to human-readable name.
    Used to instruct the LLM which language to answer in.
    """
    mapping = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "hi": "Hindi",
        "pt": "Portuguese",
        "it": "Italian",
        "nl": "Dutch",
        "sv": "Swedish",
        "ru": "Russian",
        "zh-cn": "Chinese (Simplified)",
        "zh-tw": "Chinese (Traditional)",
        "ja": "Japanese",
        "ko": "Korean",
    }
    return mapping.get(lang_code.lower(), "the same language as the question")


def get_wikipedia_retriever(lang_code: str) -> WikipediaRetriever:
    """
    Return a WikipediaRetriever configured for the given language.
    If the language is not supported, Wikipedia will still default sensibly,
    but we fall back to 'en' to be safe.
    """
    supported = {"en", "es", "fr", "de", "hi", "pt", "it", "nl", "sv", "ru", "ja", "ko"}
    wiki_lang = lang_code if lang_code in supported else "en"

    return WikipediaRetriever(
        top_k_results=3,
        lang=wiki_lang,
    )


# --------- LLM answer chain --------- #

def build_multilingual_answer_chain():
    """
    Build an LLM chain that:
    - Takes query, context, and a target language
    - Answers in the user's language
    """
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful encyclopedic assistant.

You are given:
- A user question
- Context from Wikipedia
- The target language to answer in

Rules:
- Use ONLY the information in the context when possible.
- If the answer is not clearly in the context, say you are not sure.
- Always answer in the target language: {target_language}.

Context:
{context}

Question (in original language):
{question}

Answer in {target_language}:"""
    )

    chain = prompt | llm | StrOutputParser()
    return chain


def answer_query_multilingual(query: str) -> str:
    """
    Full flow:
    - Detect language
    - Retrieve Wikipedia pages in that language
    - Answer using LLM in the same language
    """
    # 1. Detect language
    lang_code = detect_query_language(query)
    lang_name = language_name_from_code(lang_code)

    # 2. Retrieve from Wikipedia in that language
    retriever = get_wikipedia_retriever(lang_code)
    docs = retriever.invoke(query)

    if not docs:
        return (
            f"[Detected language: {lang_code} ({lang_name})]\n"
            "No relevant Wikipedia articles were found for this query."
        )

    context = format_docs(docs)

    # 3. LLM answer chain
    chain = build_multilingual_answer_chain()
    answer = chain.invoke(
        {
            "context": context,
            "question": query,
            "target_language": lang_name,
        }
    )

    # 4. Combine metadata + answer
    header = f"[Detected language: {lang_code} ({lang_name})]\n"
    return header + "\n" + answer


# --------- Demo main --------- #

def main():
    # Load .env file
    load_dotenv()
    
    print("🌐 Multilingual Knowledge Base (Wikipedia + LLM)\n")

    # Try queries in different languages:
    query = "Who is Alan Turing?"
    # query = "¿Quién inventó el lenguaje de programación Python?"
    # query = "भारत की राजधानी क्या है?"
    # query = "¿Qué es el aprendizaje profundo?"  # Spanish

    # Or uncomment to accept user input:
    # query = input("Enter your question (in any language): ")

    print(f"User question: {query}\n")
    response = answer_query_multilingual(query)
    print(response)


if __name__ == "__main__":
    main()
