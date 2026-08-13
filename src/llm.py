"""
llm.py
connect to local model (chat and embedding)
"""

from langchain_ollama import ChatOllama, OllamaEmbeddings

llm = ChatOllama(
    model="llama3.2",
    temperature=0.3, 
)

#embedding model (chunk and search query) 
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)