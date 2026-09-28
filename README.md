# Signal Chain

Signal Chain is a local supply-chain resilience sandbox. It combines a deterministic multi-echelon simulator, role-specific negotiation agents, a hard safety verifier, and the ARDN recovery forecaster in one workflow.

The application is designed to answer one question: **given a disruption, which recovery plan is worth considering, and is it safe to execute?**

## What the application does

1. You configure a disruption: supplier capacity loss, route closure, or demand spike.
2. The simulation advances the four-stage network: supplier → manufacturer → distributor → retailer.
3. Four agents negotiate a shipment proposal: Supplier, Manufacturer, Distributor, and Retailer.
4. The deterministic verifier checks inventory, capacity, production, route, service, and fairness constraints.
5. Failed proposals are sent back for revision; the verifier remains the execution gate.
6. ARDN scores six action-conditioned recovery options and reports predicted recovery time, cost, risk, uncertainty, and novelty/OOD signal.
7. The frontend presents the network state, negotiation trail, safety review, agent roster, ARDN recommendation, and strategy comparison.

The current agent stack supports real Qwen calls through an OpenAI-compatible endpoint. If Qwen is not configured or returns an invalid response, that individual proposal safely falls back to the deterministic local proposal and is marked `(Mock)` in the negotiation log.

## Requirements

- Python 3.12 or newer
- Node.js and npm
- Optional: Ollama for a fully local Qwen agent
- Optional: a hosted OpenAI-compatible Qwen API key

## Install the backend

From the repository root (`aisupplychain`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
```

The backend dependencies include FastAPI, Uvicorn, Pydantic, NumPy, SciPy, and python-dotenv. NumPy and SciPy are required by the ARDN model artifact.

## Run with local deterministic fallback

This is the quickest way to verify the complete app. Without a configured provider, the four agents use deterministic proposals.

```powershell
python -m uvicorn backend.main:app --reload
```

The backend is available at:

- API: http://127.0.0.1:8000
- Interactive API docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/api/health

## Run the frontend

In a second PowerShell window:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL, usually http://localhost:5173, and select **Run scenario**.

## Enable real local Qwen agents with Ollama

Install Ollama for Windows, then download a model. The 4B variant is a practical starting point for a laptop:

```powershell
ollama pull qwen3.5:4b
ollama run qwen3.5:4b
```

Exit the interactive test with `/bye`. Ollama continues serving its local API at `http://localhost:11434`.

Create a root `.env` file from the example:

```powershell
Copy-Item .env.example .env
notepad .env
```

Use this configuration:

```env
LLM_ENABLED=true
QWEN_API_KEY=local
QWEN_BASE_URL=http://localhost:11434/v1
QWEN_MODEL=qwen3.5:4b
LLM_TIMEOUT_SECONDS=60
```

Restart the backend after changing `.env`:

```powershell
python -m uvicorn backend.main:app --reload
```

Confirm that real agents are configured:

```powershell
Invoke-RestMethod http://localhost:8000/api/health | ConvertTo-Json
```

The response should include `"mock_agents": false` and `"agent_model": "qwen3.5:4b"`.

Larger local variants need more memory. The model size is a hardware decision, not a software limit; local inference avoids hosted token charges but still uses your CPU/GPU, memory, storage, and electricity.

## Enable a hosted Qwen-compatible endpoint

The same agent adapter works with a hosted OpenAI-compatible provider:

```env
LLM_ENABLED=true
QWEN_API_KEY=your_key_here
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3.5-plus
LLM_TIMEOUT_SECONDS=30
```

Do not commit `.env` or API keys. Hosted providers can have quotas or usage charges; check the provider’s current terms before using them for long runs.

## API flow

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Reports provider configuration and active agent model. |
| `GET /api/ardn/config` | Returns ARDN architecture metadata, local evaluation summary, and active runtime settings. |
| `PUT /api/ardn/config` | Updates bounded ARDN runtime forecast settings without changing weights. |
| `POST /api/simulation` | Creates a simulation from a disruption request. |
| `POST /api/simulation/{id}/step` | Advances one deterministic simulation day. |
| `POST /api/simulation/{id}/negotiate` | Runs agent negotiation, verifier review, execution, and ARDN forecast. |
| `GET /api/simulation/{id}/negotiation` | Returns the negotiation log and verifier result. |
| `POST /api/experiment` | Compares classical, unverified, and verified strategies. |

The negotiation response contains `meta`, `forecast`, `verification`, `negotiation`, and the updated network snapshot. `forecast.status` is `ready` when the included ARDN artifact loads successfully.

## Project map

```text
backend/
  main.py             FastAPI routes and orchestration
  agents.py           Qwen-compatible agents plus safe mock fallback
  negotiation.py      Multi-round proposal/revision loop
  verifier.py         Deterministic safety gate
  simulator.py        Live four-echelon simulation
  ardn_service.py     Adapter from live state to the trained ARDN model
  experiments.py      Classical/unverified/verified comparison

frontend/src/
  App.tsx             Scenario workflow and dashboard composition
  components/         Controls, network, metrics, agents, verifier, forecast
  components/ARDNLab.tsx  ARDN runtime tuning and model-evidence page
  style.css           Main responsive visual system

model/
  model.py            ARDN architecture
  simulator.py        Synthetic graph-training simulator
  autograd.py         NumPy reverse-mode autodiff engine
  trained_model.pkl   Included trained weights and OOD statistics
  evaluate.py         Evaluation and counterfactual demonstrations
  eval_output.txt     Captured evaluation output

results/
  model_comparison.svg   Comparison graph
  model_comparison.csv   Graph source data and metric provenance
  model_architecture_comparison.svg  Top-five architecture-fit graph
  model_architecture_comparison.csv  Top-five rubric source data
  research_benchmark_comparison.svg  Published benchmark values with ARDN shown separately
  research_benchmark_comparison.csv  Source-linked research comparison data
  ARDN_HONEST_ASSESSMENT.md  Limitations and defensible conclusion for ARDN
  ardn_honest_assessment.svg  Honest local-evaluation graph
  ardn_honest_assessment.csv  Honest graph source data
  README.md              Comparability notes and interpretation

PROJECT_FUNCTIONING.txt  Full technical execution reference for the project
```

## ARDN boundaries

ARDN is advisory. It ranks recovery actions and predicts outcomes from the trained synthetic graph distribution; it does not approve shipments. The deterministic verifier remains the hard gate before a negotiated agreement is executed. The model’s uncertainty and OOD values should be treated as decision-support signals, not guarantees.

The included model was trained on the synthetic ARDN simulator described in [model/README.md](model/README.md). Retraining is optional for running the website.

## Troubleshooting

- `ModuleNotFoundError: No module named 'fastapi'`: activate `.venv` and run `python -m pip install -r backend\requirements.txt`.
- `ModuleNotFoundError: No module named 'backend'`: run Uvicorn from the repository root, not `frontend`.
- The UI shows `(Mock)`: Qwen is disabled, credentials are missing, the local server is unreachable, or the response was not valid proposal JSON.
- `forecast.status` is `unavailable`: verify that `model\trained_model.pkl`, NumPy, and SciPy are present.
- Ollama is not responding: run `ollama list`, then start Ollama or run `ollama serve`.
