# ARDN: honest assessment

## Bottom line

There is currently no valid evidence that ARDN is the best model. It has not been evaluated against the published baselines on the same data, target, split, forecast horizon, and scoring code. The published numbers in `research_benchmark_comparison.csv` are useful reference results, but they are not an ARDN-vs-TFT/Informer/GRU experiment.

## What the included run actually shows

| Check | Result | Honest interpretation |
| --- | ---: | --- |
| Held-out trajectory MSE | 0.0058 | Good only relative to this project's synthetic simulator and split; MSE cannot be compared with another paper's MAE. |
| Recovery-time MAE | 2.17 steps | Better than the included naive midpoint baseline (5.25) on the same local evaluation. |
| Unseen 12-node topology MSE | 0.0048 | Promising transfer signal, but still synthetic and not an external validation set. |
| Mean action-ranking Kendall tau | +0.30 (std 0.25) | Modest agreement with the simulator's actual ranking; not close to a consistently correct decision policy. |
| Cost interval coverage | 79% within 1-sigma; 99% within 2-sigma | Intervals are wider/conservative relative to the usual 68%/95% reference points; this does not prove calibrated uncertainty. |
| Full ARDN vs no-GAT ablation | 0.0153 vs 0.0077 MSE | The no-GAT ablation performed better in the captured matched run. This is evidence against claiming that the graph-attention component is already helping. |

The counterfactual example also shows predicted costs around 90–113 while simulator costs are roughly 25–60. That is a material warning that the cost head needs further calibration before operational use.

## Verdict

ARDN is a promising research prototype with a useful local result, not a proven state-of-the-art model. The strongest defensible claim is: **ARDN beats the included naive midpoint baseline on its synthetic recovery-time test and shows some topology-transfer signal.** It is not defensible to claim that ARDN beats TFT, Informer, GRU, CA-HGAT, or the published hybrid model from the current evidence.

## What would make the comparison fair

Run every baseline on the same ARDN-generated train/validation/test episodes, with the same action set, horizon, topology split, seeds, and metrics. Report mean and standard deviation across multiple seeds, include the no-GAT ablation, and evaluate cost calibration and action-ranking agreement—not just trajectory error. CA-HGAT should remain “not benchmarked” until that experiment exists; no directly comparable paper result was found under that exact model name.
