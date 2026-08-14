# Research Assistant Agent

A LangGraph-based AI agent that takes a research topic, breaks it into
sub-questions, gathers information from two tools: local document
retrieval (RAG over ingested PDFs) and live web search, reflects on
whether it has enough information, and produces a final answer with
cited sources.

Built on a fully local stack: **Ollama** for the LLM and embeddings,
**ChromaDB** for vector storage, and **LangGraph** for orchestration.
No cloud API keys required.

## Architecture

The agent is a LangGraph `StateGraph` with four nodes:

- **Planner** — breaks the input topic into 2-3 concrete sub-questions.
- **Researcher** — calls two tools for each sub-question: local
  document retrieval (`retrieve_documents`) and live web search
  (`web_search`). Duplicate results are filtered out, and tool
  failures are caught and logged separately rather than crashing.
- **Reflect** — decides whether enough research has been done, or
  loops back to the Researcher for another pass (conditional edge,
  capped at `MAX_LOOPS` to guarantee the graph terminates).
- **Responder** — synthesizes everything gathered into a final answer
  with a cited sources list, noting any tool failures along the way.

```
Start → Planner → Researcher → Reflect ─┬─→ Responder → End
                       ↑                 │
                       └── loop back ────┘
                     (if research incomplete)
```

State persists across turns via LangGraph's `MemorySaver` checkpointer,
keyed by a `thread_id` generated once per session in `main.py` — this
means multiple questions asked in the same run share conversation
state rather than starting fresh each time.

## Tech Stack

| Piece | Tool |
|---|---|
| Orchestration | LangGraph |
| LLM | Ollama (`llama3.2`, local) |
| Embeddings | Ollama (`nomic-embed-text`, local) |
| Vector store | ChromaDB |
| Document loading | `PyPDFDirectoryLoader` (PDF ingestion) |
| Web search | DuckDuckGo (`ddgs`) |

## Prerequisites

- Python 3.10+ (developed on 3.14 — check your version with
  `python3 --version`)
- [Ollama](https://ollama.com) installed and running locally
- Pull the required models before running the agent:
```bash
  ollama pull llama3.2
  ollama pull nomic-embed-text
```

## Setup

1. Clone the repo and enter the folder:
```bash
   git clone git@github.com:lilyvx/research-assistant-agent.git
   cd research-assistant-agent
```

2. Create and activate a virtual environment:
```bash
   python3 -m venv venv
   source venv/bin/activate
```

3. Install dependencies:
```bash
   pip freeze > requirements.txt #for updates
   pip install -r requirements.txt
```

4. Add at least one PDF file to `data/documents/`. This is your local
   knowledge base — the agent's `retrieve_documents` tool searches
   whatever PDFs you place here.

5. Build the vector store from those PDFs (run once, and again any
   time you add or change files in `data/documents/`):
```bash
   python -m src.ingest
```
   This chunks each PDF, embeds the chunks locally via Ollama, and
   stores them in a persistent `chroma_db/` folder.

## Running

```bash
python main.py
```

You'll be prompted for a research topic. The agent plans, researches
using both tools, reflects on whether it has enough information, and
responds with a final cited answer. Since the whole session shares one
`thread_id`, you can keep entering new topics without restarting
conversation state persists across turns. Type `x` to quit.

## Project Structure

```
research-assistant-agent/
├── data/
│   └── documents/          # source PDFs for the local RAG knowledge base
├── chroma_db/                # gitignored — auto-generated vector store
├── src/
│   ├── state.py               # shared AgentState schema
│   ├── llm.py                  # Ollama chat + embedding model setup
│   ├── tools.py                 # retrieve_documents, web_search tools
│   ├── ingest.py                 # PDF chunking + embedding into Chroma
│   ├── nodes.py                   # Planner, Researcher, Reflect, Responder
│   └── graph.py                    # wires nodes/edges, compiles with checkpointer
├── tests/                         # standalone tests for tools, nodes, graph, persistence, error handling
├── writeup/
│   └── writeup.md                 # architecture, design decisions, challenges
├── main.py                        # entry point — interactive CLI
├── requirements.txt
└── .gitignore
```

## Testing

Each layer of the agent was tested in isolation before being wired
together — see `tests/`:
## Testing

Each piece was tested in isolation before being wired together, in
this order:

| # | What | Command |
|---|---|---|
| 1 | State schema imports | `python -c "from src.state import AgentState; print('ok')"` |
| 2 | LLM connects to Ollama | `python -c "from src.llm import llm; print(llm.invoke('hi').content)"` |
| 3 | Ingest PDFs into Chroma | `python -m src.ingest` |
| 4 | Both tools return real results | `python -m tests.test_tools` |
| 5 | Each node, alone and chained | `python -m tests.test_nodes` |
| 6 | Full graph compiles and runs | `python -m tests.test_graph` |
| 7 | State persists across turns | `python -m tests.test_persistance` |
| 8 | Graceful failure on tool error | `python -m tests.test_error_handling` |
| 9 | Full interactive run | `python main.py` |
