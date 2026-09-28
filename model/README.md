# ARDN: Action-Conditioned Recovery Dynamics Network

This directory contains the trained recovery forecaster used by Signal Chain. ARDN is a NumPy-based implementation of an action-conditioned graph and temporal model. It predicts how a disrupted supply network may recover under different interventions.

## Role in the application

ARDN runs after the negotiation/verifier stage as an advisory layer:

```text
live network + disruption
        ↓
four agent proposals (Qwen or deterministic fallback)
        ↓
deterministic verifier
        ↓
ARDN ranks six candidate recovery actions
        ↓
frontend forecast and alternatives
```

The backend adapter is [backend/ardn_service.py](../backend/ardn_service.py). It maps the live four-echelon state into ARDN’s topology-invariant feature contract, evaluates each action, and returns JSON-safe predictions. ARDN does not replace the verifier and does not approve or execute shipments.

## Model outputs

For each candidate action, the adapter exposes:

- predicted recovery time and recovery-time uncertainty;
- predicted cost and an evidential cost interval;
- predicted service loss;
- risk probability;
- trajectory of predicted mean service;
- Mahalanobis OOD/novelty score.

The six action types are `reallocate`, `reroute`, `delay`, `prioritize`, `share_capacity`, and `emergency_source`.

## Files

- `autograd.py` — small reverse-mode autodiff engine over NumPy arrays, including `Tensor`, `Adam`, `Linear`, `MLP`, `GRUCell`, and differentiable gamma functions.
- `simulator.py` — synthetic multi-echelon graph simulator and feature builder used to generate training episodes.
- `model.py` — type-aware node encoders, FiLM-conditioned heterogeneous graph attention, GRU recovery dynamics, hazard head, evidential cost/fairness heads, risk head, and OOD scoring.
- `train(1).py` — original training loop with horizon curriculum and multi-task losses.
- `train.py` — import-safe shim around `train(1).py`.
- `dataset.py` — fixed, reproducible dataset generation (train, val, test, 12/16-node topology transfer, severity buckets).
- `baselines.py` — unified baseline models (Persistence, GlobalMean, FlatMLP, GNN, RandomForest, Oracle) exposing `.predict_unified()`.
- `evaluate_all.py` — unified counterfactual tau ranking and top-1 accuracy routines.
- `analyze.py` — paired bootstrap significance tests and confidence intervals.
- `run_train.py` — model loader and caching orchestrator for diagnostics and benchmarks.
- `diagnostics.py` — targeted diagnostics (per-timestep trajectory error and high-power counterfactual ranking).
- `evaluate.py` — held-out quality, counterfactual action ranking, OOD, calibration, topology-transfer, and ablation checks.
- `trained_model.pkl` — included trained weights, horizon, topology metadata, and OOD statistics.
- `eval_output.txt` — captured output from the included evaluation run.

## Reported evaluation

The included evaluation output reports:

| Check | Result |
| --- | --- |
| Held-out trajectory MSE | 0.0058 |
| Recovery-time MAE | 2.17 steps, compared with 5.25 for a midpoint baseline |
| Unseen 12-node topology trajectory MSE | 0.0048 |
| Typical vs. extreme OOD score | 3.27 vs. 46.91 |
| Cost within predicted 1σ | 79% |
| Mean counterfactual Kendall’s τ | +0.30 |

The graph-attention ablation is reported honestly in `eval_output.txt`; on the small synthetic topology, the no-GAT variant performed better in that run. This is a measured result, not a claim that graph attention always improves performance.

## Run the model tools directly

The model scripts use NumPy and SciPy. From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
cd model
python simulator.py
python evaluate.py
```

`evaluate.py` loads `trained_model.pkl` and does not retrain the model. The saved artifact is all the website needs at runtime.

The original training file is named `train(1).py` in this project. Training from scratch is optional and is substantially slower than loading the included artifact; inspect that file before launching a new run because its imports assume the working directory is `model`.

## Limitations

- ARDN was trained on a synthetic simulator, not historical supply-chain data.
- The live four-node application is a topology adaptation, not a retrained production model.
- OOD and uncertainty values are signals for human review, not calibrated guarantees for every real network.
- The model ranks candidate interventions; the deterministic verifier remains responsible for hard safety constraints.
- The model itself has no LLM dependency. Qwen is used by the separate negotiation-agent layer.
