"""
ingest.py
for reading document in data/document, chunk, embedd, and stores the result in chroma for retrieval

"""

import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from src.llm import embeddings

#set paths 
DOCUMENTS_DIR = "data/documents"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "research_docs"


def load_documents():
    #laod all .pdf files in data/document as langchain document obj.. ( pypdfloader create 1 obj per page, with page number in metadata)
    
    loader = DirectoryLoader(
        DOCUMENTS_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,)
    
    documents = loader.load()

    print(f"Loaded {len(documents)} document(s) from {DOCUMENTS_DIR}")
    return documents


def split_documents(documents):
    #split each loaded obj into chunks (500 char with overlap 50) for better embedding and retrieval (manageble size and context)
    #a sentence sitting near a chunk boundary isn't fully lost from either chunk's context

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(documents)

    print(f"Split into {len(chunks)} chunk(s)")
    return chunks


def build_vector_store(chunks):
    #embedding chunk, store in chroma_path

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
    )
    print( f"Stored {len(chunks)} chunk(s) in Chroma at '{CHROMA_PATH}'")
    return vector_store

#execute ingestion if run as main
if __name__ == "__main__":
    if not os.path.isdir(DOCUMENTS_DIR) or not os.listdir(DOCUMENTS_DIR):
        print(
            f"No documents found in '{DOCUMENTS_DIR}'. "
            "Add at least one .pdf file there before running ingest.py."
        )
    else:
        docs = load_documents()
        chunks = split_documents(docs)
        build_vector_store(chunks)
        print("Ingestion complete.")