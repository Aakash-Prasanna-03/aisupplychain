"""
Evaluation / demonstration script for ARDN.

Runs three things a reviewer would actually ask for (Sections 15, 19, 20, 27):
  1. Held-out prediction quality (trajectory MSE, T_rec MAE) vs. a naive baseline.
  2. A counterfactual ranking demo (Section 15): fix (S_t, D_t), vary A_t across
     all six action types, and show ARDN's predicted ranking against the
     simulator's *actual* ranking for the same fixed disruption.
  3. A minimal ablation (Section 20): "no graph attention" vs. full model, on
     trajectory error, to show the graph-structured design earns its keep.
"""
import pickle
import numpy as np

from simulator import make_episode, default_network, ACTION_TYPES, SupplyChainSim
from model import ARDN
from train import train, forward_and_losses


def load(path="trained_model.pkl"):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    model = ARDN(horizon=obj["horizon"], tau=obj["tau"])
    model.load_state_dict(obj["state_dict"])
    model.ood_mean = obj["ood_mean"]
    model.ood_cov_inv = obj["ood_cov_inv"]
    return model, obj["net"], obj["history"]


def held_out_eval(model, net, n=60, seed=999):
    rng = np.random.default_rng(seed)
    dataset = [make_episode(net=net, T=model.horizon, rng=rng) for _ in range(n)]
    traj_errs, trec_errs, naive_trec_errs = [], [], []
    for ep in dataset:
        pred = model.predict(ep)
        pred_traj = np.concatenate([s.data for s in pred["s_hat_seq"]], axis=1).T  # (T,N)
        traj_errs.append(np.mean((pred_traj - ep["trajectory"]) ** 2))
        trec_errs.append(abs(float(pred["T_rec"].data.reshape(-1)[0]) - min(ep["t_rec"], model.horizon)))
        naive_trec_errs.append(abs(model.horizon / 2 - min(ep["t_rec"], model.horizon)))  # naive: predict midpoint
    print(f"[held-out, n={n}] trajectory MSE={np.mean(traj_errs):.4f}  "
          f"T_rec MAE(model)={np.mean(trec_errs):.2f} steps  "
          f"T_rec MAE(naive midpoint baseline)={np.mean(naive_trec_errs):.2f} steps")
    return dataset


def counterfactual_demo(model, net, seed=7):
    """Fix one disruption. Score every action type with ONE model (same forward pass
    family, Section 15), and compare ARDN's ranking to the simulator's ground-truth
    ranking under the same fixed disruption -- the exact test the design doc's
    'Predictor vs Advisor' framing requires."""
    rng = np.random.default_rng(seed)
    node_types, edges, echelons = net
    sim_for_disruption = SupplyChainSim(node_types, edges, echelons, rng=rng)
    disruption = sim_for_disruption._sample_disruption()
    target_node = disruption["nodes"][0]

    print(f"\n[counterfactual ranking] fixed disruption: nodes={disruption['nodes']} "
          f"severity={disruption['severity']:.2f} type={disruption['type']}")
    print(f"{'action':16s} {'ARDN T_rec':>11s} {'ARDN cost':>10s} | {'true T_rec':>10s} {'true cost':>10s}")

    rows = []
    for a_type in ACTION_TYPES:
        rng2 = np.random.default_rng(seed)  # re-seed so only action type varies, all else fixed
        sim = SupplyChainSim(node_types, edges, echelons, rng=rng2)
        action = sim._sample_action(forced_type=a_type)
        action["target"] = target_node
        sim.reset(disruption=dict(disruption), action=action)
        result = sim.rollout(T=model.horizon)
        X = sim.build_node_features()
        edge_index, edge_feat = sim.build_edge_index_and_features()
        d_vec, a_vec = sim.build_global_descriptors()
        hist = sim.history_summary()
        ep = {"node_types": np.array(node_types), "X": X, "edge_index": edge_index,
              "edge_feat": edge_feat, "d_vec": d_vec, "a_vec": a_vec, "hist": hist, **result}

        pred = model.predict(ep)
        pred_trec = float(pred["T_rec"].data.reshape(-1)[0])
        pred_cost = float(pred["cost"][0].data.reshape(-1)[0]) * 20.0  # undo the /20 scaling used in training
        true_trec = result["t_rec"]
        true_cost = result["cum_cost"]
        rows.append((a_type, pred_trec, pred_cost, true_trec, true_cost))
        print(f"{a_type:16s} {pred_trec:11.2f} {pred_cost:10.2f} | {true_trec:10d} {true_cost:10.2f}")

    pred_rank = sorted(rows, key=lambda r: r[1])
    true_rank = sorted(rows, key=lambda r: r[3])
    print("\nARDN-predicted fastest-to-slowest:", [r[0] for r in pred_rank])
    print("Simulator's actual fastest-to-slowest:", [r[0] for r in true_rank])
    tau_corr = _kendall_tau([r[0] for r in pred_rank], [r[0] for r in true_rank])
    print(f"Kendall's tau rank agreement: {tau_corr:.2f}  (1.0 = perfect agreement, 0 = random, -1 = reversed)")


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


def ablation_no_gat(net, episodes_per_epoch=60, n_epochs=15, seed=1):
    """Section 20: remove graph attention (flat pooled-MLP over node features) and
    compare trajectory error to the full model, trained under identical conditions."""
    print("\n[ablation] training FULL model vs. NO-GAT model, matched budget/data...")

    def run(use_gat):
        rng = np.random.default_rng(seed)
        model = ARDN(horizon=10, use_gat=use_gat)
        from autograd import Adam
        opt = Adam(model.params(), lr=3e-3)
        net_local = net
        for epoch in range(n_epochs):
            dataset = [make_episode(net=net_local, T=10, rng=rng) for _ in range(episodes_per_epoch)]
            for i in range(0, len(dataset), 4):
                batch = dataset[i:i + 4]
                opt.zero_grad()
                total = None
                for ep in batch:
                    losses, _ = forward_and_losses(model, ep)
                    from train import total_loss
                    t = total_loss(model, losses)
                    total = t if total is None else total + t
                (total * (1.0 / len(batch))).backward()
                opt.step()
        # held-out
        rng_eval = np.random.default_rng(seed + 500)
        eval_set = [make_episode(net=net_local, T=10, rng=rng_eval) for _ in range(40)]
        errs = []
        for ep in eval_set:
            pred = model.predict(ep)
            pred_traj = np.concatenate([s.data for s in pred["s_hat_seq"]], axis=1).T
            errs.append(np.mean((pred_traj - ep["trajectory"]) ** 2))
        return np.mean(errs)

    full_err = run(use_gat=True)
    nogat_err = run(use_gat=False)
    print(f"FULL model (with heterogeneous graph attention) trajectory MSE: {full_err:.4f}")
    print(f"NO-GAT ablation (flat pooled-MLP)              trajectory MSE: {nogat_err:.4f}")
    print(f"-> graph attention {'helps' if full_err < nogat_err else 'does NOT help'} "
          f"({'ratio' if full_err < nogat_err else 'ratio'}: {nogat_err / full_err:.2f}x)")


def ood_demo(model, net, seed=42):
    """Show OOD scores are higher for a disruption combination unlike anything trained on
    (Section 16/21: unseen disruption *combinations* / severities)."""
    rng = np.random.default_rng(seed)
    in_dist_ep = make_episode(net=net, T=model.horizon, rng=rng)

    # construct an extreme, unseen-style scenario: max severity, all nodes disrupted at once
    node_types, edges, echelons = net
    sim = SupplyChainSim(node_types, edges, echelons, rng=rng)
    extreme_disruption = {"nodes": list(range(len(node_types))), "severity": 0.95,
                           "duration": 8, "type": "capacity", "cooccurrence": True}
    action = sim._sample_action()
    sim.reset(disruption=extreme_disruption, action=action)
    result = sim.rollout(T=model.horizon)
    X = sim.build_node_features()
    edge_index, edge_feat = sim.build_edge_index_and_features()
    d_vec, a_vec = sim.build_global_descriptors()
    hist = sim.history_summary()
    ood_ep = {"node_types": np.array(node_types), "X": X, "edge_index": edge_index,
              "edge_feat": edge_feat, "d_vec": d_vec, "a_vec": a_vec, "hist": hist, **result}

    print("\n[OOD novelty check]")
    print(f"typical episode novelty score:  {model.ood_score(in_dist_ep):.2f}")
    print(f"network-wide extreme-disruption novelty score: {model.ood_score(ood_ep):.2f}  "
          f"(should be notably higher)")


def counterfactual_demo_averaged(model, net, n_disruptions=8, seed_base=100):
    """Runs counterfactual_demo's ranking test over several disruptions and reports the
    average Kendall's tau, which is a much less noisy signal than a single disruption."""
    print(f"\n[counterfactual ranking, averaged over {n_disruptions} disruptions]")
    taus = []
    node_types, edges, echelons = net
    for d_idx in range(n_disruptions):
        rng = np.random.default_rng(seed_base + d_idx)
        sim0 = SupplyChainSim(node_types, edges, echelons, rng=rng)
        disruption = sim0._sample_disruption()
        target_node = disruption["nodes"][0]
        rows = []
        for a_type in ACTION_TYPES:
            rng2 = np.random.default_rng(seed_base + d_idx)
            sim = SupplyChainSim(node_types, edges, echelons, rng=rng2)
            action = sim._sample_action(forced_type=a_type)
            action["target"] = target_node
            sim.reset(disruption=dict(disruption), action=action)
            result = sim.rollout(T=model.horizon)
            X = sim.build_node_features()
            edge_index, edge_feat = sim.build_edge_index_and_features()
            d_vec, a_vec = sim.build_global_descriptors()
            hist = sim.history_summary()
            ep = {"node_types": np.array(node_types), "X": X, "edge_index": edge_index,
                  "edge_feat": edge_feat, "d_vec": d_vec, "a_vec": a_vec, "hist": hist, **result}
            pred = model.predict(ep)
            pred_trec = float(pred["T_rec"].data.reshape(-1)[0])
            rows.append((a_type, pred_trec, result["t_rec"]))
        pred_rank = [r[0] for r in sorted(rows, key=lambda r: r[1])]
        true_rank = [r[0] for r in sorted(rows, key=lambda r: r[2])]
        tau = _kendall_tau(pred_rank, true_rank)
        taus.append(tau)
        print(f"  disruption {d_idx}: severity={disruption['severity']:.2f} tau={tau:+.2f}")
    print(f"mean Kendall's tau over {n_disruptions} disruptions: {np.mean(taus):+.2f} "
          f"(std {np.std(taus):.2f})")


def topology_generalization_check(model, seed=55):
    """Section 21: evaluate the trained model, UNCHANGED, on a larger/rewired network
    (3-per-echelon instead of 2) it never trained on. Dims are topology-invariant by
    design (type-aware encoders + global pooling), so this tests transfer, not just
    that the code runs."""
    print("\n[topology generalization] evaluating on an unseen, larger network "
          "(3 nodes/echelon vs. the trained 2 nodes/echelon)...")
    bigger_net = default_network(n_per_echelon=3)
    rng = np.random.default_rng(seed)
    dataset = [make_episode(net=bigger_net, T=model.horizon, rng=rng) for _ in range(40)]
    errs = []
    for ep in dataset:
        pred = model.predict(ep)
        pred_traj = np.concatenate([s.data for s in pred["s_hat_seq"]], axis=1).T
        errs.append(np.mean((pred_traj - ep["trajectory"]) ** 2))
    print(f"trajectory MSE on unseen 12-node topology: {np.mean(errs):.4f} "
          f"(compare to held-out MSE on the trained 8-node topology above)")


def calibration_check(model, net, n=100, seed=321):
    """Section 27: report empirical coverage of the evidential uncertainty intervals
    for the cost head, split as an overall reliability check (a fuller split by
    in-distribution vs. OOD would need many more samples than this demo affords)."""
    rng = np.random.default_rng(seed)
    dataset = [make_episode(net=net, T=model.horizon, rng=rng) for _ in range(n)]
    within_1sigma = 0
    within_2sigma = 0
    for ep in dataset:
        pred = model.predict(ep)
        mu, v, alpha, beta = pred["cost"]
        mu_v = float(mu.data.reshape(-1)[0])
        # aleatoric + epistemic variance of the NIG predictive distribution
        var = float(beta.data.reshape(-1)[0]) / (float(v.data.reshape(-1)[0]) * (float(alpha.data.reshape(-1)[0]) - 1) + 1e-6)
        sigma = max(1e-3, var) ** 0.5
        true = ep["cum_cost"] / 20.0
        z = abs(true - mu_v) / sigma
        within_1sigma += z <= 1.0
        within_2sigma += z <= 2.0
    print(f"\n[calibration, cost head, n={n}] within predicted 1-sigma: {within_1sigma/n:.0%} "
          f"(ideal ~68%)  within 2-sigma: {within_2sigma/n:.0%} (ideal ~95%)")


if __name__ == "__main__":
    model, net, history = load()
    held_out_eval(model, net)
    counterfactual_demo(model, net)
    counterfactual_demo_averaged(model, net)
    ood_demo(model, net)
    calibration_check(model, net)
    topology_generalization_check(model)
    ablation_no_gat(net)
