# pdf_chunker_intermediate.py

import os
from pathlib import Path

import matplotlib.pyplot as plt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


def load_first_pdf(pdf_folder: Path):
    """Return the first .pdf file in the folder, or None if not found."""
    if not pdf_folder.exists():
        print("❌ Folder 'pdfs' does not exist.")
        return None

    pdf_files = [f for f in pdf_folder.iterdir() if f.suffix.lower() == ".pdf"]
    if not pdf_files:
        print("❌ No PDF files found in 'pdfs' folder.")
        return None

    # Use the first PDF for comparison
    pdf_file = sorted(pdf_files)[0]
    print(f"📄 Using PDF for comparison: {pdf_file.name}")
    return pdf_file


def get_chunk_counts_for_settings(documents, chunk_sizes, overlap_percents):
    """
    For each chunk size and overlap percent, compute number of chunks.
    Returns a dict:
      { overlap_percent: { chunk_size: num_chunks, ... }, ... }
    """
    results = {}

    for overlap_pct in overlap_percents:
        results[overlap_pct] = {}
        for size in chunk_sizes:
            overlap = int(size * overlap_pct)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=overlap,
            )

            chunks = splitter.split_documents(documents)
            results[overlap_pct][size] = len(chunks)

    return results


def print_results_table(results, chunk_sizes, overlap_percents):
    """Print a table of chunk_size vs overlap% vs number of chunks."""
    print("\n📊 Chunk Size / Overlap Comparison:\n")
    header = f"{'Overlap%':<10} " + " ".join(f"{cs:<10}" for cs in chunk_sizes)
    print(header)
    print("-" * len(header))

    for overlap_pct in overlap_percents:
        row = f"{int(overlap_pct*100):<10}"
        for cs in chunk_sizes:
            count = results[overlap_pct][cs]
            row += f"{count:<10}"
        print(row)


def plot_chunks_vs_chunk_size(results, chunk_sizes, overlap_percents, output_path: Path):
    """
    Create a line chart:
      x-axis: chunk sizes
      y-axis: number of chunks
      one line per overlap %
    """
    plt.figure()

    for overlap_pct in overlap_percents:
        counts = [results[overlap_pct][cs] for cs in chunk_sizes]
        label = f"{int(overlap_pct * 100)}% overlap"
        plt.plot(chunk_sizes, counts, marker="o", label=label)

    plt.xlabel("Chunk Size (characters)")
    plt.ylabel("Number of Chunks")
    plt.title("Chunks vs Chunk Size for Different Overlaps")
    plt.grid(True)
    plt.legend()

    # Save chart next to script
    plt.savefig(output_path)
    print(f"\n📈 Chart saved to: {output_path}")
    # Show chart (optional)
    # plt.show()


def main():
    pdf_folder = Path(r"C:\UltimateRAG\simple-rag-langchain\pdfs")
    pdf_file = load_first_pdf(pdf_folder)
    if pdf_file is None:
        return

    # Load document
    loader = PyPDFLoader(str(pdf_file))
    documents = loader.load()

    # Settings to compare
    chunk_sizes = [500, 1000, 2000]
    overlap_percents = [0.10, 0.20, 0.30]  # 10%, 20%, 30%

    # Compute number of chunks for each combination
    results = get_chunk_counts_for_settings(documents, chunk_sizes, overlap_percents)

    # Print comparison table
    print_results_table(results, chunk_sizes, overlap_percents)

    # Plot chart
    output_chart = Path("chunks_vs_chunk_size.png")
    plot_chunks_vs_chunk_size(results, chunk_sizes, overlap_percents, output_chart)


if __name__ == "__main__":
    main()
