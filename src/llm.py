from langchain_ollama import ChatOllama, OllamaEmbeddings

llm = ChatOllama(
    model="llama3.2",
    temperature=0.5, 
)

#embedding model (chunk and search query) 
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)