# Architecture

## Design Principles

1. **Provider-agnostic LLM layer** — swap Ollama/Groq/Gemini via `.env`
2. **Grounding is mandatory** — no fact without a tool call in the same turn
3. **Everything is observable** — every node emits structured events
4. **Everything is persistent** — conversations, runs, tool calls, checkpoints
5. **Human-in-the-loop for dangerous actions** — write operations require approval
6. **RAG is a first-class citizen** — not an afterthought

---

## Component Diagram

```text
┌──────────────────────────────────────────────────────────────┐
│                        FastAPI (app.main)                    │
│  /chat  /chat/stream  /documents  /eval  /stats  /health     │
└──────────────────────────┬───────────────────────────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  app.agent   │   │  app.agent   │   │  app.eval    │
│  .graph      │   │  .rag        │   │  metrics +   │
│  (LangGraph) │   │  (RAG)       │   │  runner      │
└──────┬───────┘   └──────┬───────┘   └──────────────┘
       │                  │
       ▼                  ▼
┌──────────────┐   ┌──────────────┐
│  .nodes      │   │  .store      │
│  .builder    │   │  (sqlite-vec)│
│  .checkpointer│  │  .embedder   │
└──────┬───────┘   └──────┬───────┘
       │                  │
       ▼                  ▼
┌──────────────┐   ┌──────────────┐
│  .tools      │   │  SQLite      │
│  (registry)  │   │  (agentos.db)│
└──────────────┘   └──────────────┘
The Graph
text
START
  │
  ▼
call_llm  ◄────────────────────┐
  │                            │
  │ (conditional routing)      │
  │                            │
  ├── needs_tools ──► execute_tools ──┐
  │                                  │
  ├── done/failed ──► finalize ──► END
  │
  └── max_iterations ──► force_end ──► END
Nodes:

call_llm — sends messages to the LLM, sets needs_tools flag

execute_tools — runs each tool call; interrupt() if approval needed

finalize — extracts the final answer

force_end — graceful exit on iteration cap

State:

messages — accumulate via _append reducer

tool_calls — accumulate via _append reducer

Everything else replaced by node returns

Checkpointer: SqliteSaver persists after every node transition.

RAG Pipeline
text
                 ┌─────────────┐
   PDF/TXT/MD →  │   Loader    │  (PyMuPDF / open)
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │   Cleaner   │  remove headers, page numbers
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │   Chunker   │  recursive, with overlap
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │  Embedder   │  nomic-embed-text (768-dim)
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │    Store    │  sqlite-vec (cosine)
                 └─────────────┘
Query path:

text
   query → embed → vec search → top-K chunks → LLM
Human-in-the-Loop Flow
text
call_llm → execute_tools → [tool.requires_approval?]
                                 │
                                 ├── NO  → run tool → continue
                                 │
                                 └── YES → interrupt({action, tool, args})
                                            │
                                            ▼
                                     [GRAPH PAUSES]
                                            │
                                            ▼
                                       HUMAN REVIEW
                                       /          \
                                    approve      reject
                                       │            │
                                       ▼            ▼
                                    run tool    skip + inject error msg
                                       │            │
                                       └─────┬──────┘
                                             ▼
                                       call_llm (continue)
Persistence Layers
Data	Where	Purpose
Conversations	SQLite (conversations, messages)	Chat history
Agent runs	SQLite (agent_runs)	Analytics
Tool calls	SQLite (tool_calls)	Observability
RAG documents	SQLite (rag_documents, document_chunks)	Knowledge base
RAG vectors	SQLite (vec_chunks) via sqlite-vec	Semantic search
Graph checkpoints	SQLite (LangGraph tables)	Resume + HITL
All in agentos.db.

Error Handling
Error	Handler
LLM 429/5xx	_call_llm_with_manual_retry with backoff
Invalid tool name	Recovery hint injected, iteration not counted
Tool execution failure	Error string fed back to LLM
Context too long	_format_tool_result truncates to 2500 chars
Iteration cap	force_end node
Observability Events
Emitted via Tracer:

run_start, run_end

iteration_start

llm_call_start, llm_call_end (with duration + tool count)

tool_call_start, tool_call_end (with duration + error)

error

Handlers: pretty console, silent, SSE queue.