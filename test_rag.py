import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#!/usr/bin/env python3
"""
Simple RAG (Retrieval-Augmented Generation) Test Script
This script tests the RAG pipeline to verify FAISS is working correctly
"""

import os
import sys
from pathlib import Path

print("=" * 80)
print("RAG Pipeline Test Script")
print("=" * 80)

# ============================================================================
# 1. IMPORTS
# ============================================================================
print("\n[1/9] Importing required libraries...")
try:
    from dotenv import load_dotenv
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    print("✓ All imports successful!")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nPlease install required packages:")
    print("pip install -r requirements.txt")
    sys.exit(1)

# ============================================================================
# 2. ENVIRONMENT SETUP
# ============================================================================
print("\n[2/9] Loading environment variables...")
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("✗ WARNING: OPENAI_API_KEY not found!")
    print("Please set it in .env file or export it as an environment variable")
    sys.exit(1)
else:
    print(f"✓ OpenAI API Key loaded: {os.getenv('OPENAI_API_KEY')[:8]}...")

# ============================================================================
# 3. LOAD PDF DOCUMENTS
# ============================================================================
print("\n[3/9] Loading PDF documents...")
pdf_path = "attention.pdf"

if not os.path.exists(pdf_path):
    print(f"✗ ERROR: File '{pdf_path}' not found!")
    print("Please update the pdf_path variable with your PDF file location.")
    sys.exit(1)

try:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✓ Loaded {len(documents)} pages from '{pdf_path}'")
    print(f"  Total characters: {sum(len(doc.page_content) for doc in documents):,}")
except Exception as e:
    print(f"✗ Error loading PDF: {e}")
    sys.exit(1)

# ============================================================================
# 4. SPLIT DOCUMENTS INTO CHUNKS
# ============================================================================
print("\n[4/9] Splitting documents into chunks...")
try:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=128,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    avg_chunk_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)

    print(f"✓ Split {len(documents)} documents into {len(chunks)} chunks")
    print(f"  Average chunk size: {avg_chunk_size:.0f} characters")
except Exception as e:
    print(f"✗ Error splitting documents: {e}")
    sys.exit(1)

# ============================================================================
# 5. CREATE EMBEDDINGS
# ============================================================================
print("\n[5/9] Initializing embeddings model...")
try:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Test embeddings
    sample_text = "This is a test sentence."
    sample_embedding = embeddings.embed_query(sample_text)

    print(f"✓ Embeddings model initialized: text-embedding-3-small")
    print(f"  Embedding dimension: {len(sample_embedding)}")
except Exception as e:
    print(f"✗ Error initializing embeddings: {e}")
    sys.exit(1)

# ============================================================================
# 6. CREATE FAISS VECTOR STORE
# ============================================================================
print("\n[6/9] Creating FAISS vector store...")
print(f"  Processing {len(chunks)} chunks (this may take a minute)...")

try:
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    print(f"✓ FAISS vector store created successfully!")
    print(f"  Indexed {len(chunks)} document chunks")

    # Save to disk
    vectorstore_path = "./faiss_index"
    vectorstore.save_local(vectorstore_path)
    print(f"✓ Vector store saved to '{vectorstore_path}'")
except Exception as e:
    print(f"✗ Error creating FAISS vector store: {e}")
    print(f"\nError type: {type(e).__name__}")
    print(f"Python version: {sys.version}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 7. CREATE RETRIEVER AND TEST
# ============================================================================
print("\n[7/9] Creating retriever and testing...")
try:
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    print("✓ Retriever configured successfully")

    # Test retrieval
    test_query = "What is the main topic of this document?"
    print(f"\n  Testing retrieval with query: '{test_query}'")

    retrieved_docs = retriever.invoke(test_query)
    print(f"✓ Retrieved {len(retrieved_docs)} documents successfully!")

    # Show first retrieved document preview
    if retrieved_docs:
        print(f"\n  First retrieved document preview:")
        print(f"  {retrieved_docs[0].page_content[:200]}...")

except Exception as e:
    print(f"✗ Error during retrieval: {e}")
    print(f"\nError type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 8. CONFIGURE LLM
# ============================================================================
print("\n[8/9] Configuring Language Model...")
try:
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Using cheaper model for testing
        temperature=0,
        max_tokens=2000
    )
    print("✓ LLM configured: gpt-4o-mini")

    # Test LLM
    test_response = llm.invoke("Say 'Ready!'")
    print(f"  LLM test: {test_response.content}")
except Exception as e:
    print(f"✗ Error configuring LLM: {e}")
    sys.exit(1)

# ============================================================================
# 9. CREATE AND TEST RAG CHAIN
# ============================================================================
print("\n[9/9] Creating RAG chain and testing end-to-end...")
try:
    # Define prompt template
    system_prompt = (
        "You are a helpful assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer based on the context, say that you don't know. "
        "Keep the answer concise and accurate.\n\n"
        "Context: {context}\n\n"
        "Question: {question}"
    )

    prompt = ChatPromptTemplate.from_template(system_prompt)

    # Helper function to format documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Build RAG chain using LCEL
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✓ RAG chain created successfully!")

    # Test the full RAG pipeline
    test_question = "What is the main contribution of this paper?"
    print(f"\n  Testing full RAG pipeline with: '{test_question}'")
    print("  Generating answer...\n")

    answer = rag_chain.invoke(test_question)

    print("=" * 80)
    print("RAG PIPELINE TEST RESULT")
    print("=" * 80)
    print(f"Question: {test_question}")
    print(f"\nAnswer: {answer}")
    print("=" * 80)

except Exception as e:
    print(f"✗ Error in RAG chain: {e}")
    print(f"\nError type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# SUCCESS
# ============================================================================
print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED SUCCESSFULLY!")
print("=" * 80)
print("\nFAISS is working correctly on your system.")
print("The issue with Jupyter notebook kernel crash may be Jupyter-specific.")
print("\nNext steps:")
print("1. Try restarting your Jupyter kernel")
print("2. Consider using ChromaDB as an alternative (see previous suggestions)")
print("3. Or downgrade to Python 3.11 for better compatibility")
print("=" * 80)
