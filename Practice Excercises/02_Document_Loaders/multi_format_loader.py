# multi_format_loader.py

from pathlib import Path
from datetime import datetime
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    JSONLoader,
    TextLoader,
)
from langchain_core.documents import Document


def load_documents_from_directory(directory_path: str) -> List[Document]:
    """
    Load all supported files from a directory and add custom metadata.

    Supported types: .pdf, .csv, .json, .txt

    Metadata added:
      - file_type: extension without dot (e.g., 'pdf', 'csv')
      - loaded_date: ISO timestamp when the file was loaded
    """
    dir_path = Path(directory_path)

    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found or not a folder: {directory_path}")

    documents: List[Document] = []

    # Supported extensions and their loaders
    supported_exts = {".pdf", ".csv", ".json", ".txt"}

    print(f"\n📂 Scanning directory: {dir_path.resolve()}\n")

    for file_path in dir_path.iterdir():
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        if suffix not in supported_exts:
            print(f"  ⚠️ Skipping unsupported file type: {file_path.name}")
            continue

        print(f"  📄 Loading file: {file_path.name}")

        # Choose the appropriate loader based on file extension
        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif suffix == ".csv":
            loader = CSVLoader(file_path=str(file_path))
        elif suffix == ".json":
            # Load the whole JSON as a single document (root object)
            loader = JSONLoader(
                file_path=str(file_path),
                jq_schema=".",      # Take the root JSON
                text_content=False  # Keep it as structured JSON string
            )
        elif suffix == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
        else:
            # Should not reach here because of supported_exts check
            print(f"  ⚠️ No loader configured for: {file_path.name}")
            continue

        # Load documents for this file
        file_docs = loader.load()
        loaded_date = datetime.now().isoformat()

        # Enrich each document with custom metadata
        for doc in file_docs:
            if doc.metadata is None:
                doc.metadata = {}

            # Preserve existing metadata and add our own
            doc.metadata.setdefault("source", str(file_path))
            doc.metadata["file_type"] = suffix.lstrip(".")  # e.g. 'pdf', 'csv'
            doc.metadata["loaded_date"] = loaded_date

        print(f"    ✅ Loaded {len(file_docs)} document(s) from {file_path.name}\n")

        documents.extend(file_docs)

    print(f"📊 Total documents loaded: {len(documents)}\n")
    return documents


def main():
    # Folder containing sample_data (json, html, txt, csv, etc.)
    # data_dir = "sample_data"
    data_dir = r"C:\UltimateRAG\simple-rag-langchain\sample_data"

    try:
        docs = load_documents_from_directory(data_dir)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    # Simple inspection of loaded documents
    print("🔎 Document preview:\n")
    for i, doc in enumerate(docs, start=1):
        print("=" * 80)
        print(f"Document {i}")
        print("-" * 80)
        print(f"Source     : {doc.metadata.get('source')}")
        print(f"File type  : {doc.metadata.get('file_type')}")
        print(f"Loaded date: {doc.metadata.get('loaded_date')}")
        print("\nContent preview:")
        print(doc.page_content[:300], "...\n")  # show first 300 chars

    if not docs:
        print("No documents were loaded. Check that sample_data contains supported files.")


if __name__ == "__main__":
    main()
