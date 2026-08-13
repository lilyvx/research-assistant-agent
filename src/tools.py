"""
tools.py
got two tools the agent can call:

retrieve_documents: searches the local chroma vector store
for chunks relevant to a query

web_search: searches the live web via ddg for external,
information.
"""

from langchain_core.tools import tool
from langchain_chroma import Chroma
from ddgs import DDGS
from src.llm import embeddings

#connect to same chroma collection or it will connect to different store and return nothing 
vector_store = Chroma(
    collection_name="research_docs",
    embedding_function=embeddings,
    persist_directory="chroma_db",
)

#tool 1: curated search...seacrch chroma vector store for relevant chunks related to query (covered in local documents)
@tool
def retrieve_documents(query: str) -> str:

    """Search the local knowledge store of curated research documents
    for information relevant to the query. Use this first for topics
    that might be covered in the local document."""
 
    try:
        results = vector_store.similarity_search(query, k=3)
    except Exception as error:
        return f"Document retrieval failed: {error}"

    if not results:
        return "No relevant documents found locally."

    formatted = []
    for doc in results:
        source = doc.metadata.get("source", "unknown source")
        page = doc.metadata.get("page", None)
        source_label = f"{source} (page {page})" if page is not None else source
        formatted.append(f"[Source: {source_label}]\n{doc.page_content}")

    return "\n\n".join(formatted)

#tool 2: web search...searches web for information not covered by chroma vector store
@tool
def web_search(query: str) -> str:

    """Search the live web for current or external information that is
    not covered by the local knowledge store. Use this when the local
    documents dont seem to have relevant information to match query."""
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
    except Exception as error:
        return f"Web search failed: {error}"

    if not results:
        return "No web results found."

    formatted = []
    for r in results:
        title = r.get("title", "Untitled")
        url = r.get("href", "no URL")
        snippet = r.get("body", "")
        formatted.append(f"[Source: {title} — {url}]\n{snippet}")

    return "\n\n".join(formatted)

#note:@tool for llm tool calling