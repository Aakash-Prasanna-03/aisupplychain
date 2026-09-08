# Model and strategy comparison

This folder contains a snapshot comparison for the current project. The chart is intentionally split into two sections:

1. **ARDN prediction quality** compares ARDN with the baselines reported by the ARDN evaluation run.
2. **Recovery strategy behavior** compares the existing rules-only, unverified-agent, and verified-agent workflows from the project’s 12-day experiment.

These groups are related but not interchangeable. ARDN is a predictor/advisor that estimates recovery dynamics. The other three rows are execution strategies evaluated inside the live deterministic simulator.

## Files

- `model_comparison.svg` — standalone visual graph.
- `model_comparison.csv` — values used by the graph, with source and comparability notes.
- `model_architecture_comparison.svg` — top-five architecture-fit heatmap.
- `model_architecture_comparison.csv` — rubric scores and status for the top-five comparison.
- `research_benchmark_comparison.svg` — evidence-based graph using exact values from a published supply-chain benchmark, with ARDN shown in a separate non-comparable panel.
- `research_benchmark_comparison.csv` — source-linked values used by the evidence-based graph.
- `ARDN_HONEST_ASSESSMENT.md` — limitations, negative findings, and the defensible conclusion for ARDN.
- `ardn_honest_assessment.svg` — visual graph of the measured wins, negative ablation result, and unvalidated claims.
- `ardn_honest_assessment.csv` — source values used by that graph.
- `README.md` — definitions and interpretation guidance.

## Sources

- ARDN values come from `model/eval_output.txt` and `model/README.md`.
- Strategy values come from `backend.experiments.run_experiment(SimulationRequest(seed=42))` using the default 12-day scenario.

## Interpretation

- Lower trajectory MSE and recovery-time MAE are better.
- The no-GAT row is an ARDN ablation, not an independent production model.
- The naive midpoint baseline only reports recovery-time MAE; no trajectory MSE was reported for it.
- An invalid-agreement rate of zero does not mean the strategy is best overall; the verified strategy also records verifier interventions because it actively rejects unsafe proposals before execution.
- These are single-run or captured evaluation summaries, not confidence intervals or a benchmark across many seeds.

## Top-five architecture comparison

`model_architecture_comparison.svg` compares ARDN with four plausible alternatives for the same use case:

- CA-HGAT (context-aware heterogeneous graph attention);
- a spatio-temporal GNN;
- Temporal Fusion Transformer;
- XGBoost as a strong tabular baseline.

The 1–5 scores are an explicit design rubric for topology transfer, action conditioning, temporal dynamics, uncertainty/OOD support, heterogeneous graph support, and interpretability. They are not measured benchmark scores. Only ARDN has measured results in this repository; implementing the other four would be the next experiment. Treat this file as a design-planning artifact, not as an empirical leaderboard.

## Research-backed comparison

`research_benchmark_comparison.svg` replaces the speculative architecture scores with reported measurements. Its left panel reproduces the MAE ranking from Table 3 of the published HAF-DS supply-chain forecasting study: Proposed Hybrid 15.02, Hybrid TFT-X 15.29, Informer 15.44, TFT 15.67, PPO 15.78, GRU 17.23, LSTM 17.89, Prophet 20.15, and ARIMA 22.47. The paper also reports the corresponding RMSE, MAPE, and sMAPE values in the CSV.

ARDN is deliberately placed in a separate panel because its available results are trajectory MSE 0.0058 and recovery-time MAE 2.17 on the project's local evaluation. Those targets, units, dataset, and evaluation protocol differ from the published demand-forecasting benchmark, so a direct “ARDN beats TFT” claim would be inaccurate. A fair leaderboard requires running every baseline on the same ARDN data split and scoring script.

For the blunt interpretation of the local evidence—including the no-GAT ablation outperforming full ARDN in the captured run and the modest mean action-ranking agreement—see [`ARDN_HONEST_ASSESSMENT.md`](ARDN_HONEST_ASSESSMENT.md).

The web research also did not identify a standard supply-chain paper/model named “CA-HGAT” with directly comparable results. It should therefore remain an unbenchmarked candidate until implemented and evaluated on the ARDN protocol.

Published source: [Hybrid Deep Learning Approach for Coupled Demand Forecasting and Supply Chain Optimization](https://www.techscience.com/cmc/v88n3/68058.html), Table 3.
