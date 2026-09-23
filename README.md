# AgentOS

Production-style AI Agent Platform built with LangGraph, FastAPI, and PostgreSQL.

## Overview

A stateful autonomous research agent that:
- Plans research tasks
- Searches the web and knowledge base
- Uses RAG for document retrieval
- Evaluates information and loops when needed
- Persists state and memory
- Requests human approval for dangerous actions

## Tech Stack

- **Backend**: FastAPI
- **Agent**: LangChain + LangGraph
- **LLM**: Ollama (qwen2.5:7b)
- **Database**: PostgreSQL 18
- **Vector Store**: pgvector
- **Tools**: Web search, calculator, file handling

## Architecture

See `docs/architecture.md` for details.

## Setup

See `docs/setup.md` for installation # 🤖 AgentOS — Autonomous AI Research Agent

> A production-style AI agent platform with pluggable LLM providers, full RAG pipeline, LangGraph orchestration, and human-in-the-loop.

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-FF6B6B.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 What It Does

AgentOS is not a chatbot. It's a **stateful research agent** that can:

- 📚 **Search the user's own documents** (PDF/TXT/MD) via semantic RAG
- 🌐 **Query the web, Wikipedia, and specific URLs**
- 🧮 **Perform calculations** using a sandboxed evaluator
- 🔍 **Search inside text** it already retrieved
- 💾 **Save reports** to disk (with human approval)
- 🧠 **Remember conversations** across sessions
- ⏸️ **Pause for human approval** before dangerous actions
- 📊 **Measure itself** with a 25-case evaluation framework
- 🔄 **Resume from checkpoints** if the server crashes

---

## 🏗️ Architecture

```text
                         USER
                           │
                           ▼
                    ┌─────────────┐
                    │   FastAPI   │
                    └──────┬──────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   LangGraph     │
                  │   Orchestrator  │
                  └────────┬────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
             LLM         State        Tools
              │            │            │
       ┌──────┴──────┐     │      ┌────┼──────────┐
       │             │     │      │    │          │
    Ollama        Groq    │   Search DB  Calculator
    Gemini     (Cloud)    │              │
                          ▼              ▼
                    SQLite + sqlite-vec  │
                          │              │
                          ▼              ▼
                     Vector Store    Human Approval
                     (RAG chunks)    (HITL via pause)
```

---

## ✨ Features

### Multi-Provider LLM
- **Ollama** (local, free) — for development and full evals
- **Groq** (cloud, ultra-fast) — for quick demos
- **Gemini** (cloud) — OpenAI-compatible fallback
- Swap with a single `.env` change

### Full RAG Pipeline
1. **Load** — PDF (PyMuPDF), TXT, Markdown
2. **Clean** — remove page headers, page numbers, noise
3. **Chunk** — recursive character splitting with overlap
4. **Embed** — `nomic-embed-text` via Ollama (768-dim)
5. **Store** — `sqlite-vec` (cosine similarity)
6. **Retrieve** — top-K with metadata for citations

### Agent Capabilities
- 7 tools: `calculator`, `wikipedia`, `web_search`, `fetch_page`, `find_in_page`, `knowledge_search`, `save_report`
- Grounding rules: **every fact must come from a tool call**
- Citation enforcement: `[filename, chunk N]`
- Invalid tool-call recovery
- Retry with backoff on transient errors

### Observability
- Per-run events (run_start, iteration, llm_call, tool_call, run_end)
- Latency tracking per node
- Streaming via SSE to a live UI

### Evaluation
- **25 test cases** across 5 categories
- **7 metrics**: tool accuracy, retrieval hit rate, citation rate, boundary compliance, hallucination rate, failure rate, latency
- JSON reports + visual dashboard

### LangGraph Orchestration
- **StateGraph** with typed state
- **Conditional routing** (needs tools → execute → LLM → ...)
- **Checkpoints** persisted to SQLite after every node
- **Human-in-the-loop** with `interrupt()` + `Command(resume=...)`
- **Resume from crash** — state survives restarts

---

## 📊 Evaluation Results

Latest run on `Python for Data Analysis` (Wes McKinney, 541 pages):

| Metric | Value |
|---|---|
| **Tool Accuracy** | 96.0% |
| **Retrieval Hit Rate** | 100.0% |
| **Citation Rate** | 92.0% |
| **Boundary Compliance** | 90.0% |
| **Hallucination Rate** | 10.0% |
| **Failure Rate** | **0.0%** |
| **Avg Latency** | 93.68s (Ollama, CPU) |

**By category:**
- factual: 100%
- multi_chunk: 100%
- adversarial: 100%
- boundary: 100%
- citation: 100%
- cross_tool: 50%

---

## 🚀 Quickstart

### Prerequisites
- Python 3.12
- [Ollama](https://ollama.com/) installed
- (Optional) Groq or Gemini API key

### Installation

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/agentos.git
cd agentos

# Setup
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt

# Pull embedding model
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
```

### Configure

Create `.env`:

```env
LLM_PROVIDER=ollama
EVAL_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
DATABASE_URL=sqlite:///./agentos.db
```

### Run

```bash
# Start server
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open:
- **Dashboard**: http://127.0.0.1:8000/
- **Chat**: http://127.0.0.1:8000/chat
- **Documents**: http://127.0.0.1:8000/documents
- **Eval**: http://127.0.0.1:8000/eval
- **API Docs**: http://127.0.0.1:8000/docs

### Run the Evaluation

```bash
python -m scripts.run_eval
```

---

## 📁 Project Structure

```
agentos/
├── app/
│   ├── agent/
│   │   ├── llm.py              # Multi-provider LLM abstraction
│   │   ├── loop.py             # Legacy while-loop agent
│   │   ├── memory.py           # Conversation persistence
│   │   ├── observability.py    # Tracer + events
│   │   ├── state.py            # AgentState dataclass
│   │   ├── graph/              # LangGraph orchestration
│   │   │   ├── builder.py      # Graph assembly + entrypoint
│   │   │   ├── nodes.py        # call_llm, execute_tools, finalize
│   │   │   ├── state.py        # GraphState + reducers
│   │   │   └── checkpointer.py # SQLite checkpoint saver
│   │   ├── rag/                # RAG pipeline
│   │   │   ├── loader.py       # PDF/TXT/MD readers
│   │   │   ├── cleaner.py      # Text normalization
│   │   │   ├── chunker.py      # Recursive splitting
│   │   │   ├── embedder.py     # nomic-embed-text wrapper
│   │   │   ├── store.py        # sqlite-vec backend
│   │   │   └── ingest.py       # End-to-end pipeline
│   │   └── tools/              # Tool definitions
│   │       ├── base.py         # Tool ABC
│   │       ├── registry.py     # Tool registry
│   │       ├── calculator.py
│   │       ├── wikipedia.py
│   │       ├── web_search.py
│   │       ├── fetch_page.py
│   │       ├── find_in_page.py
│   │       ├── knowledge_search.py
│   │       └── save_report.py  # requires_approval=True
│   ├── api/
│   │   ├── routes/             # FastAPI routers
│   │   └── schemas.py          # Pydantic models
│   ├── core/
│   │   └── config.py           # Settings
│   ├── db/
│   │   ├── base.py             # SQLAlchemy engine
│   │   └── models.py           # ORM models
│   ├── eval/
│   │   ├── metrics.py          # 7 evaluation metrics
│   │   └── runner.py           # Dataset runner
│   └── main.py                 # FastAPI entrypoint
├── data/
│   ├── static/                 # HTML/CSS/JS UI
│   ├── eval/dataset.json       # 25 test cases
│   └── reports/                # Generated (gitignored)
├── docs/
│   ├── architecture.md
│   ├── roadmap.md
│   └── setup.md
├── scripts/                    # CLI utilities
├── tests/                      # (future)
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Web** | FastAPI + Uvicorn |
| **Agent** | LangGraph + LangChain-style tools |
| **LLM** | Ollama, Groq, Gemini |
| **Embeddings** | `nomic-embed-text` |
| **Vector DB** | `sqlite-vec` (cosine) |
| **Relational** | SQLite + SQLAlchemy |
| **Checkpoints** | LangGraph `SqliteSaver` |
| **Frontend** | Vanilla HTML/CSS/JS |
| **RAG Parser** | PyMuPDF, BeautifulSoup |
| **Validation** | Pydantic v2 |

---

## 🗺️ Roadmap

See [`docs/roadmap.md`](docs/roadmap.md) for the full plan.

- [x] **Phase 0–6**: LLM, tools, state, memory
- [x] **Phase 7**: Full RAG pipeline
- [x] **Phase 8**: Agent + RAG integration
- [x] **Phase 9**: Provider abstractions
- [x] **Phase 10**: LangGraph fundamentals
- [x] **Phase 11**: Conditional routing + HITL
- [ ] **Phase 12**: Streaming with LangGraph
- [ ] **Phase 13**: Multi-agent collaboration
- [ ] **Phase 14**: Production deployment (Docker, CI/CD)

---

## 📄 License

MIT — see [LICENSE](LICENSE).instructions.

## License

MIT