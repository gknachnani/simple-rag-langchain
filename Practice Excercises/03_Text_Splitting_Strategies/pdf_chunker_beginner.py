# pdf_chunker_beginner.py

import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def process_pdf(pdf_path: Path):
    print(f"\n📄 Processing PDF: {pdf_path.name}")

    # Step 1: Load PDF using PyPDFLoader
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    # Step 2: Split PDF text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # max characters per chunk
        chunk_overlap=100     # overlap between chunks
    )

    chunks = splitter.split_documents(documents)

    # Step 3: Print summary
    print(f"Total Chunks Created: {len(chunks)}\n")

    # Print first chunk
    if chunks:
        print("------ FIRST CHUNK ------")
        print(chunks[0].page_content)
        print("\n------ LAST CHUNK ------")
        print(chunks[-1].page_content)
    else:
        print("⚠️ No chunks produced for this PDF.")

def main():
    pdf_folder = Path(r"C:\UltimateRAG\simple-rag-langchain\pdfs")

    if not pdf_folder.exists():
        print("❌ Folder 'pdfs' does not exist.")
        return

    pdf_files = [f for f in pdf_folder.iterdir() if f.suffix.lower() == ".pdf"]

    if not pdf_files:
        print("❌ No PDF files found in 'pdfs' folder.")
        return

    print(f"🔍 Found {len(pdf_files)} PDF(s) in the folder.")

    for pdf_file in pdf_files:
        process_pdf(pdf_file)

if __name__ == "__main__":
    main()
