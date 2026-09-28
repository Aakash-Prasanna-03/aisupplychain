"""
Two targeted diagnostics.

(1) Per-timestep trajectory error. ARDN decodes autoregressively (residual decoder),
    the strong baselines decode all steps in one shot. If ARDN's deficit is error
    ACCUMULATION, its error should grow with k relative to the one-shot decoders.

(2) High-power counterfactual ranking. The headline claim -- ARDN ranks actions better
    than the baselines -- rests on only 25 disruptions in the main table, where the
    ARDN-vs-FlatMLP gap did NOT reach significance. Re-run with 100 disruptions for the
    models that matter, with a paired bootstrap over the SAME disruptions.
"""
import pickle

import numpy as np

from dataset import load_all, HORIZON
from run_train import load_model
from baselines import pooled_targets
from evaluate_all import counterfactual_tau, top1_action_accuracy
from analyze import paired_bootstrap

MODELS_TRAJ = ["Oracle", "RandomForest", "GNN", "FlatMLP", "ARDN_v2_full",
               "ARDN_no_residual", "ARDN_no_graph"]
MODELS_CF = ["ARDN_v2_full", "FlatMLP", "GNN", "RandomForest",
             "ARDN_v1", "ARDN_no_action"]
N_CF = 100


def per_timestep(data):
    out = {}
    for name in MODELS_TRAJ:
        m = load_model(name)
        errs = []
        for ep in data["test"]:
            p = m.predict_unified(ep, HORIZON)["traj_pooled"][:HORIZON]
            errs.append(((p - pooled_targets(ep, HORIZON)) ** 2).mean(axis=1))  # (T,)
        out[name] = np.stack(errs).mean(axis=0)
    return out


def high_power_cf(data):
    res = {}
    for name in MODELS_CF:
        m = load_model(name)
        mean, std, taus, dm, dt = counterfactual_tau(m, data["net"], n_disruptions=N_CF,
                                                     horizon=HORIZON)
        top1 = top1_action_accuracy(m, data["net"], n_disruptions=N_CF, horizon=HORIZON)
        res[name] = {"tau": mean, "taus": np.array(taus), "top1": top1,
                     "degen_model": dm, "degen_truth": dt}
        print(f"  {name:18s} tau={mean:+.3f}  top1={top1:.2f}  "
              f"degenerate(model)={dm}/{N_CF}", flush=True)
    return res


if __name__ == "__main__":
    data = load_all()

    import sys
    print("=== (1) per-timestep pooled trajectory MSE ===")
    pt = per_timestep(data)
    ks = [0, 2, 4, 8, 12, 16, 20, 24]
    print("model              " + "".join(f"  k={k:<2d}" for k in ks) + "   slope(k>=4)")
    for name, e in pt.items():
        sl = np.polyfit(np.arange(4, HORIZON), e[4:], 1)[0]
        print(f"{name:18s}" + "".join(f" {e[k]:.4f}" for k in ks) + f"   {sl:+.2e}")

    print(f"\n=== (2) counterfactual ranking, n={N_CF} disruptions ===")
    cf = high_power_cf(data)

    print(f"\n=== paired bootstrap on tau, ARDN_v2_full vs others (n={N_CF}) ===")
    ref = cf["ARDN_v2_full"]["taus"]
    for name in MODELS_CF:
        if name == "ARDN_v2_full":
            continue
        d, lo, hi, p = paired_bootstrap(ref, cf[name]["taus"])
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not significant"
        print(f"  vs {name:18s} dtau={d:+.3f} [{lo:+.3f}, {hi:+.3f}]  p={p:.4f}  {sig}")

    pickle.dump({"per_timestep": pt, "cf": cf}, open("diagnostics.pkl", "wb"))
    print("\nsaved diagnostics.pkl")
