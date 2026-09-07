# Trust-Verified Agentic LLM Negotiation

A local research MVP comparing rule-based recovery, unverified agent negotiation, and negotiation protected by a deterministic safety verifier.

## Run backend

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

## Run frontend

```powershell
cd frontend
npm install
npm run dev
```

Open the URL shown by Vite and click **Run Demo**. Mock agents are the default (`LLM_ENABLED=false`). Copy `.env.example` to `.env`, set `LLM_ENABLED=true`, `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` to enable an OpenAI-compatible API; failures safely fall back to mock agents.
