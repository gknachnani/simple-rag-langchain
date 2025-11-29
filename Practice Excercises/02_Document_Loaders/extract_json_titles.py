# extract_json_titles.py

from pathlib import Path
from datetime import datetime
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document


def extract_article_titles(json_path: str) -> Document:
    """
    Load api_response.json and extract only article titles using jq.
    Returns a single summary Document containing all titles.
    """

    file_path = Path(json_path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    print(f"\n📄 Loading JSON file: {file_path.resolve()}\n")

    # jq schema explanation:
    #   .articles[].title  → iterate over each item in 'articles' array
    #                        and extract only the 'title' field
    loader = JSONLoader(
        file_path=str(file_path),
        jq_schema=".articles[].title",
        text_content=True
    )

    # Each returned Document contains ONLY the title text
    title_docs = loader.load()

    print(f"✅ Extracted {len(title_docs)} article titles\n")

    # Combine all titles into one string
    all_titles_text = "\n".join(
        f"- {doc.page_content.strip()}" for doc in title_docs
    )

    # Create a single summary Document
    summary_doc = Document(
        page_content=f"Summary of Article Titles:\n\n{all_titles_text}",
        metadata={
            "source": str(file_path),
            "file_type": "json",
            "loaded_date": datetime.now().isoformat(),
            "total_titles": len(title_docs)
        }
    )

    return summary_doc


def main():
    # json_file = "sample_data/api_response.json"
    json_file = r"C:\UltimateRAG\simple-rag-langchain\sample_data\api_response.json"

    try:
        summary_document = extract_article_titles(json_file)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    print("📝 Summary Document Created:\n")
    print("=" * 80)
    print(summary_document.page_content)
    print("=" * 80)

    print("\n📌 Metadata:\n", summary_document.metadata)


if __name__ == "__main__":
    main()
