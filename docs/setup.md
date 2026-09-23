# Setup

## Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.12 | Runtime |
| Ollama | Latest | Local LLM + embeddings |
| Git | Latest | Version control |

Optional: Groq or Gemini API key for faster inference.

## Step-by-Step

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/agentos.git
cd agentos
2. Virtual environment
powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
3. Dependencies
powershell
pip install -r requirements.txt
4. Ollama models
powershell
ollama pull qwen2.5:7b           # chat model (CPU-friendly)
ollama pull nomic-embed-text     # embeddings (required)
5. Environment file
Create .env:

env
APP_NAME=AgentOS
APP_ENV=development
DEBUG=true

LLM_PROVIDER=ollama
EVAL_LLM_PROVIDER=ollama

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

# Optional
# GROQ_API_KEY=gsk_...
# GROQ_MODEL=openai/gpt-oss-20b
# GEMINI_API_KEY=...
# GEMINI_MODEL=gemma-4-31b-it

DATABASE_URL=sqlite:///./agentos.db
6. Run
powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
7. Try it
Upload a PDF: http://127.0.0.1:8000/documents

Ask a question: http://127.0.0.1:8000/chat

Run the eval: python -m scripts.run_eval

See results: http://127.0.0.1:8000/eval

Troubleshooting
ModuleNotFoundError
Ensure the venv is activated: where.exe python should point to .venv\Scripts\python.exe.

Slow responses
Set LLM_PROVIDER=groq in .env (requires API key). Ollama on CPU is 20–40× slower.

Ollama connection refused
Run ollama serve in a separate terminal.

Groq 429 rate limit
Free tier = 200K tokens/day. Use EVAL_LLM_PROVIDER=ollama for evals.