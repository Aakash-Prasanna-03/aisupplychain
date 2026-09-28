"""
Unified evaluation routines for all models (ARDN and all baselines).
Exposes counterfactual_tau and top1_action_accuracy used by diagnostics.py.
"""
import numpy as np
from simulator import ACTION_TYPES, SupplyChainSim


def _kendall_tau(rank_a, rank_b):
    pos_a = {x: i for i, x in enumerate(rank_a)}
    pos_b = {x: i for i, x in enumerate(rank_b)}
    items = list(pos_a.keys())
    n = len(items)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_order = pos_a[items[i]] - pos_a[items[j]]
            b_order = pos_b[items[i]] - pos_b[items[j]]
            if a_order * b_order > 0:
                concordant += 1
            elif a_order * b_order < 0:
                discordant += 1
    total = n * (n - 1) / 2
    return (concordant - discordant) / total if total else 0.0


def counterfactual_tau(model, net, n_disruptions=100, horizon=15, seed_base=100):
    """
    Evaluates action ranking across n_disruptions.
    Returns:
        mean_tau: float
        std_tau: float
        taus: list[float]
        dm: int, count of disruptions where the model output was degenerate (all values tied)
        dt: int, count of disruptions where the simulator output was degenerate (all values tied)
    """
    node_types, edges, echelons = net
    taus = []
    dm, dt = 0, 0
    for d_idx in range(n_disruptions):
        rng = np.random.default_rng(seed_base + d_idx)
        sim0 = SupplyChainSim(node_types, edges, echelons, rng=rng)
        disruption = sim0._sample_disruption()
        target_node = disruption["nodes"][0]

        pred_trecs = []
        true_trecs = []
        for a_type in ACTION_TYPES:
            rng2 = np.random.default_rng(seed_base + d_idx)
            sim = SupplyChainSim(node_types, edges, echelons, rng=rng2)
            action = sim._sample_action(forced_type=a_type)
            action["target"] = target_node
            sim.reset(disruption=dict(disruption), action=action)
            result = sim.rollout(T=horizon)
            X = sim.build_node_features()
            edge_index, edge_feat = sim.build_edge_index_and_features()
            d_vec, a_vec = sim.build_global_descriptors()
            hist = sim.history_summary()
            ep = {
                "node_types": np.array(node_types), "X": X, "edge_index": edge_index,
                "edge_feat": edge_feat, "d_vec": d_vec, "a_vec": a_vec, "hist": hist,
                "disruption_severity": float(disruption["severity"]), **result,
            }
            if hasattr(model, "predict_unified"):
                p = model.predict_unified(ep, horizon=horizon)
                pred_trec = float(p["T_rec"])
            else:
                pred = model.predict(ep)
                pred_trec = float(pred["T_rec"].data.reshape(-1)[0])
            pred_trecs.append((a_type, pred_trec))
            true_trecs.append((a_type, float(result["t_rec"])))

        pred_vals = [pt[1] for pt in pred_trecs]
        true_vals = [tt[1] for tt in true_trecs]
        is_dm = (max(pred_vals) - min(pred_vals)) < 1e-4
        is_dt = (max(true_vals) - min(true_vals)) < 1e-4
        if is_dm:
            dm += 1
        if is_dt:
            dt += 1

        if not is_dm and not is_dt:
            pred_rank = [pt[0] for pt in sorted(pred_trecs, key=lambda x: x[1])]
            true_rank = [tt[0] for tt in sorted(true_trecs, key=lambda x: x[1])]
            taus.append(_kendall_tau(pred_rank, true_rank))
        else:
            taus.append(0.0)

    mean_tau = float(np.mean(taus)) if taus else 0.0
    std_tau = float(np.std(taus)) if taus else 0.0
    return mean_tau, std_tau, taus, dm, dt


def top1_action_accuracy(model, net, n_disruptions=100, horizon=15, seed_base=100):
    """
    Computes fraction of disruptions where model's top-1 action matches true best action(s).
    """
    node_types, edges, echelons = net
    correct = 0
    for d_idx in range(n_disruptions):
        rng = np.random.default_rng(seed_base + d_idx)
        sim0 = SupplyChainSim(node_types, edges, echelons, rng=rng)
        disruption = sim0._sample_disruption()
        target_node = disruption["nodes"][0]

        pred_trecs = []
        true_trecs = []
        for a_type in ACTION_TYPES:
            rng2 = np.random.default_rng(seed_base + d_idx)
            sim = SupplyChainSim(node_types, edges, echelons, rng=rng2)
            action = sim._sample_action(forced_type=a_type)
            action["target"] = target_node
            sim.reset(disruption=dict(disruption), action=action)
            result = sim.rollout(T=horizon)
            X = sim.build_node_features()
            edge_index, edge_feat = sim.build_edge_index_and_features()
            d_vec, a_vec = sim.build_global_descriptors()
            hist = sim.history_summary()
            ep = {
                "node_types": np.array(node_types), "X": X, "edge_index": edge_index,
                "edge_feat": edge_feat, "d_vec": d_vec, "a_vec": a_vec, "hist": hist,
                "disruption_severity": float(disruption["severity"]), **result,
            }
            if hasattr(model, "predict_unified"):
                p = model.predict_unified(ep, horizon=horizon)
                pred_trec = float(p["T_rec"])
            else:
                pred = model.predict(ep)
                pred_trec = float(pred["T_rec"].data.reshape(-1)[0])
            pred_trecs.append((a_type, pred_trec))
            true_trecs.append((a_type, float(result["t_rec"])))

        best_true_val = min(tt[1] for tt in true_trecs)
        best_true_actions = {tt[0] for tt in true_trecs if abs(tt[1] - best_true_val) < 1e-4}
        best_pred_action = sorted(pred_trecs, key=lambda x: x[1])[0][0]
        if best_pred_action in best_true_actions:
            correct += 1

    return float(correct / n_disruptions) if n_disruptions else 0.0
