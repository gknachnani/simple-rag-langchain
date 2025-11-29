# Advanced.py
# "Advanced": Create a chain that summarizes text in different styles
# (formal, casual, technical) and print relevant output.

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env (OPENAI_API_KEY, etc.)
load_dotenv()


def build_chain():
    """Build an LCEL chain: (text, style) -> styled summary string."""

    # Prompt template with variables {style} and {text}
    prompt = ChatPromptTemplate.from_template(
        """You are a summarization assistant.

Write a {style} summary of the following text.

Requirements:
- 3–4 sentences
- Preserve all key facts
- Use a clearly {style} tone
- Do not mention that you are an AI or that you are summarizing

Text:
{text}
"""
    )

    # LLM configuration (pick any available chat model in your setup)
    llm = ChatOpenAI(
        model="gpt-5-nano",  # or "gpt-4o-mini" / any other chat model you use
        temperature=0.3,      # slightly creative but mostly factual
    )

    # Parser to return plain string
    parser = StrOutputParser()

    # LCEL chain: Prompt -> LLM -> Text
    chain = prompt | llm | parser
    return chain


def main():
    chain = build_chain()

    # Example input text to summarize
    input_text = (
        "Retrieval-Augmented Generation (RAG) is a technique that combines a "
        "large language model with an external knowledge source, such as a "
        "vector database. Instead of relying only on what the model has seen "
        "during training, RAG first retrieves relevant documents based on the "
        "user's query and then uses those documents as context to generate a "
        "more accurate and grounded answer. This helps reduce hallucinations "
        "and keeps responses up to date with the latest information."
    )

    styles = ["formal", "casual", "technical"]

    print("\n=== ADVANCED: Styled Summaries (Formal, Casual, Technical) ===\n")
    print("Original Text:\n")
    print(input_text)
    print("\n" + "=" * 80)

    for i, style in enumerate(styles, start=1):
        print(f"\n{i}. STYLE: {style.upper()}")
        print("-" * 80)

        summary = chain.invoke({"style": style, "text": input_text})
        print(summary)


if __name__ == "__main__":
    main()
