# Roadmap

## ✅ Phase 0–6 — Foundations
- LLM abstraction (messages, temperature, tokens)
- Tool registry + function calling
- State machine (AgentState)
- Conversation memory (SQLite)
- Observability events

## ✅ Phase 7 — Full RAG
- Document loaders (PDF, TXT, MD)
- Text cleaner
- Recursive chunker with overlap
- Embeddings via `nomic-embed-text`
- `sqlite-vec` cosine search
- Metadata + citations

## ✅ Phase 8 — Agent + RAG
- `knowledge_search` tool
- Agent decides when to use the KB vs web
- Grounding rules enforced in prompt

## ✅ Phase 9 — Provider Abstractions
- Ollama, Groq, Gemini behind one interface
- `LLM_PROVIDER` and `EVAL_LLM_PROVIDER` env vars

## ✅ Phase 10 — LangGraph Fundamentals
- `StateGraph` with typed state
- Node functions with reducers
- Compilation with optional checkpointer

## ✅ Phase 11 — Routing + HITL
- Conditional edges (`route_after_llm`)
- `interrupt()` + `Command(resume=...)`
- `requires_approval` on tools
- `save_report` as a demo dangerous tool

## 🔄 Phase 12 — Streaming with LangGraph
- `graph.stream()` for node-level streaming
- SSE bridge to the frontend
- Token streaming from the LLM

## 🔄 Phase 13 — Multi-Agent
- Researcher agent + Writer agent
- Supervisor routing
- Shared checkpointer

## 🔄 Phase 14 — Production
- Docker Compose (app + Postgres + pgvector)
- Alembic migrations
- Structured logging (JSON)
- Prometheus metrics
- CI/CD with GitHub Actions
- Deployment guide