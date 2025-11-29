# lazy_pdf_keyword_filter.py

from pathlib import Path
from datetime import datetime
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def lazy_load_and_filter_pdfs(
    pdf_dir: str,
    keywords: List[str]
) -> List[Document]:
    """
    Lazily load all PDFs from a directory and filter pages
    that contain any of the specified keywords.

    Args:
        pdf_dir: Directory containing PDF files.
        keywords: List of keywords to search for (case-insensitive).

    Returns:
        A list of Document objects, each representing a page that matched.
    """
    base_path = Path(pdf_dir)

    if not base_path.exists() or not base_path.is_dir():
        raise FileNotFoundError(f"PDF directory not found or not a folder: {pdf_dir}")

    # Normalize keywords to lowercase for case-insensitive search
    keywords = [kw.lower() for kw in keywords]

    filtered_pages: List[Document] = []

    print(f"\n📂 Scanning PDF directory: {base_path.resolve()}\n")

    pdf_files = list(base_path.glob("*.pdf"))
    if not pdf_files:
        print("⚠️ No PDF files found in the directory.")
        return filtered_pages

    print(f"Found {len(pdf_files)} PDF file(s):")
    for f in pdf_files:
        print(f"  - {f.name}")
    print()

    for pdf_file in pdf_files:
        print(f"🔄 Lazy loading pages from: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))
        page_iter = loader.lazy_load()

        for page_index, page_doc in enumerate(page_iter):
            text = page_doc.page_content or ""
            text_lower = text.lower()

            # Check if any keyword is present in this page
            if any(kw in text_lower for kw in keywords):
                # Enrich metadata
                if page_doc.metadata is None:
                    page_doc.metadata = {}

                page_doc.metadata.setdefault("source", str(pdf_file))
                page_doc.metadata["page_number"] = page_doc.metadata.get(
                    "page", page_index
                )
                page_doc.metadata["matched_keywords"] = [
                    kw for kw in keywords if kw in text_lower
                ]
                page_doc.metadata["loaded_date"] = datetime.now().isoformat()

                filtered_pages.append(page_doc)

        print(f"    ✅ Finished scanning {pdf_file.name}\n")

    print(f"📊 Total matching pages: {len(filtered_pages)}\n")
    return filtered_pages


def main():
    # pdf_directory = "pdfs"  # Folder containing your 2 PDF files (rag.pdf, ragsurvey.pdf)
    pdf_directory = r"C:\UltimateRAG\simple-rag-langchain\pdfs"  # Folder containing your 2 PDF files (rag.pdf, ragsurvey.pdf)

    # RAG-specific keywords based on the two PDFs
    keywords = [
        "retrieval-augmented generation",
        "rag",
        "non-parametric memory",
        "parametric memory",
        "dense retriever",
        "dense passage retriever",
        "wikipedia",
        "vector index",
        "naive rag",
        "advanced rag",
        "modular rag",
        "llm",
        "large language model",
    ]

    try:
        matching_pages = lazy_load_and_filter_pdfs(pdf_directory, keywords)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    if not matching_pages:
        print("No pages matched the given keywords.")
        return

    print("🔎 Preview of matching pages:\n")
    for i, doc in enumerate(matching_pages[:5], start=1):
        print("=" * 80)
        print(f"Match {i}")
        print("-" * 80)
        print(f"Source file      : {doc.metadata.get('source')}")
        print(f"Page number      : {doc.metadata.get('page_number')}")
        print(f"Matched keywords : {doc.metadata.get('matched_keywords')}")
        print("\nContent preview:")
        print(doc.page_content[:400], "...\n")  # show first 400 chars

    if len(matching_pages) > 5:
        print(f"... and {len(matching_pages) - 5} more matching page(s).")


if __name__ == "__main__":
    main()
