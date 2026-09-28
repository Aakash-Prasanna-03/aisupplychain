"""
Fixed dataset generation for fair, reproducible model comparison.

Every model in this study is trained and evaluated on EXACTLY the same episodes
(same seeds -> same disruptions/actions/trajectories), so differences in results
reflect the models and not sampling luck. Regenerating with the same seeds
reproduces the dataset exactly (the simulator is deterministic given its RNG).
"""
import pickle
import numpy as np

from simulator import make_episode, default_network

SEED_TRAIN, SEED_VAL, SEED_TEST = 1000, 2000, 3000
SEED_GEN_3, SEED_GEN_4, SEED_SEVERITY = 4000, 5000, 6000

HORIZON = 25
N_TRAIN, N_VAL, N_TEST = 500, 100, 150
N_GEN = 100
N_SEVERITY_PER_BUCKET = 60


def _gen(n, seed, net, T=HORIZON):
    rng = np.random.default_rng(seed)
    return [make_episode(net=net, T=T, rng=rng) for _ in range(n)]


def _gen_severity_bucket(n, seed, net, lo, hi, T=HORIZON):
    """Rejection-samples episodes whose disruption severity falls in [lo, hi), keeping the
    action/disruption-type distribution intact and constraining only severity."""
    rng = np.random.default_rng(seed)
    out, attempts = [], 0
    while len(out) < n and attempts < n * 80:
        ep = make_episode(net=net, T=T, rng=rng)
        attempts += 1
        sev = ep.get("disruption_severity", ep["d_vec"][1])
        if lo <= sev < hi:
            out.append(ep)
    return out


def build_all(path="dataset.pkl"):
    net = default_network()                       # 8 nodes -- the training topology
    net3 = default_network(n_per_echelon=3)       # 12 nodes -- unseen
    net4 = default_network(n_per_echelon=4)       # 16 nodes -- unseen
    data = {
        "net": net,
        "train": _gen(N_TRAIN, SEED_TRAIN, net),
        "val": _gen(N_VAL, SEED_VAL, net),
        "test": _gen(N_TEST, SEED_TEST, net),
        "gen_3x": _gen(N_GEN, SEED_GEN_3, net3),
        "gen_4x": _gen(N_GEN, SEED_GEN_4, net4),
        "severity_low": _gen_severity_bucket(N_SEVERITY_PER_BUCKET, SEED_SEVERITY + 1, net, 0.30, 0.50),
        "severity_med": _gen_severity_bucket(N_SEVERITY_PER_BUCKET, SEED_SEVERITY + 2, net, 0.50, 0.70),
        "severity_high": _gen_severity_bucket(N_SEVERITY_PER_BUCKET, SEED_SEVERITY + 3, net, 0.70, 0.95),
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return data


def load_all(path="dataset.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    data = build_all()
    for k, v in data.items():
        if isinstance(v, list):
            pos = np.mean([e["violated"] for e in v])
            trec = np.mean([min(e["t_rec"], HORIZON) for e in v])
            cens = np.mean([e["t_rec"] >= HORIZON for e in v])
            print(f"{k:14s} n={len(v):4d}  violation_rate={pos:.2f}  mean_T_rec={trec:.2f}  censored={cens:.2f}")
