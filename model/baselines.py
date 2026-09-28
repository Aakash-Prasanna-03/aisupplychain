"""
Baselines, all exposing  .predict_unified(ep, horizon) -> dict with keys
    traj_pooled (T,2)  [mean-over-nodes, min-over-nodes] service per step
    traj_node   (T,N) or None
    T_rec       float   predicted recovery step, target = min(t_rec, T)
    cost        float   in TRAINING units (cum_cost / 20)
    fair        float   in TRAINING units (fairness_gap * 5)
    risk        float   probability of constraint violation
so that ONE set of metric code scores every model on identical footing.

  PersistenceBaseline  state never changes                      (sanity floor)
  GlobalMeanBaseline   dataset-mean trajectory / targets        (sanity floor)
  FlatMLP              pooled + role features -> MLP            (conventional NN; no graph, no recurrence)
  GNNBaseline          GraphSAGE-style GNN + direct heads       (graph baseline WITHOUT ARDN's FiLM/GRU/hazard/evidential)
  RandomForestBaseline gradient-free tree ensemble, same features (classical ML)
  OracleBaseline       the true simulator marginalised over the one hidden variable;
                       a Bayes reference = irreducible-error floor, NOT a competitor
"""
import numpy as np
from autograd import Tensor, Linear, MLP, Adam, concat
from simulator import (NODE_FEAT_DIM, GLOBAL_D_DIM, GLOBAL_A_DIM, ACTION_TYPES,
                       DISRUPTION_TYPES, SupplyChainSim)

C_SERVICE, C_ISDISR, C_ISSRC, C_ISTGT = 6, 9, 16, 17
COST_SCALE, FAIR_SCALE = 1.0 / 20.0, 5.0


def pooled_targets(ep, T=None):
    traj = ep["trajectory"] if T is None else ep["trajectory"][:T]
    return np.stack([traj.mean(axis=1), traj.min(axis=1)], axis=1)


def pooled_input(ep):
    """Topology-invariant flat feature vector: node-feature mean/max, plus the ROLE rows
    the action/disruption point at (target, source, disrupted), plus global descriptors.
    This gives non-graph baselines access to the same information ARDN has -- they simply
    cannot use the wiring pattern."""
    X = ep["X"]
    tgt_rows = X[X[:, C_ISTGT] > 0.5]
    src_rows = X[X[:, C_ISSRC] > 0.5]
    dis_rows = X[X[:, C_ISDISR] > 0.5]
    z = np.zeros(X.shape[1])
    tgt = tgt_rows.mean(axis=0) if len(tgt_rows) else z
    src = src_rows.mean(axis=0) if len(src_rows) else z
    dis = dis_rows.mean(axis=0) if len(dis_rows) else z
    return np.concatenate([X.mean(axis=0), X.max(axis=0), tgt, src, dis, ep["d_vec"], ep["a_vec"]])


def pooled_input_dim():
    return NODE_FEAT_DIM * 5 + GLOBAL_D_DIM + GLOBAL_A_DIM


def _targets(ep, T):
    return {"traj": pooled_targets(ep, T), "t_rec": float(min(ep["t_rec"], T)),
            "cost": ep["cum_cost"] * COST_SCALE, "fair": ep["fairness_gap"] * FAIR_SCALE,
            "risk": 1.0 if ep["violated"] else 0.0}


# ------------------------------------------------------------------ trivial floors
class PersistenceBaseline:
    """Assumes the disrupted state simply persists. The floor any dynamics model must beat."""

    def fit(self, train, T=15):
        self.mean_cost = float(np.mean([e["cum_cost"] * COST_SCALE for e in train]))
        self.mean_fair = float(np.mean([e["fairness_gap"] * FAIR_SCALE for e in train]))
        self.mean_risk = float(np.mean([1.0 if e["violated"] else 0.0 for e in train]))
        return self

    def predict_unified(self, ep, horizon=15, tau=0.9):
        s0 = ep["X"][:, C_SERVICE]
        return {"traj_pooled": np.tile([s0.mean(), s0.min()], (horizon, 1)),
                "traj_node": np.tile(s0, (horizon, 1)),
                "T_rec": 0.0 if s0.min() >= tau else float(horizon),
                "cost": self.mean_cost, "fair": self.mean_fair, "risk": self.mean_risk}


class GlobalMeanBaseline:
    """Predicts the training-set mean for everything, ignoring the input entirely.
    Any model that fails to beat this has learned nothing conditional."""

    def fit(self, train, T=15):
        self.T = T
        self.mean_traj = np.mean([pooled_targets(e, T) for e in train], axis=0)
        self.mean_trec = float(np.mean([min(e["t_rec"], T) for e in train]))
        self.mean_cost = float(np.mean([e["cum_cost"] * COST_SCALE for e in train]))
        self.mean_fair = float(np.mean([e["fairness_gap"] * FAIR_SCALE for e in train]))
        self.mean_risk = float(np.mean([1.0 if e["violated"] else 0.0 for e in train]))
        return self

    def predict_unified(self, ep, horizon=15):
        return {"traj_pooled": self.mean_traj[:horizon], "traj_node": None,
                "T_rec": self.mean_trec, "cost": self.mean_cost,
                "fair": self.mean_fair, "risk": self.mean_risk}


# ------------------------------------------------------------------ shared loss pieces
def _mse(pred, tgt):
    d = pred - Tensor(np.asarray(tgt, dtype=np.float64).reshape(pred.shape), requires_grad=False)
    return (d * d).mean()


def _bce(p, y):
    eps = 1e-7
    y_t = Tensor(np.asarray(y, dtype=np.float64).reshape(p.shape), requires_grad=False)
    one = Tensor(np.ones(p.shape), requires_grad=False)
    return -(y_t * (p + eps).log() + (one - y_t) * (one - p + eps).log()).mean()


def _weighted(log_sigma, traj, trec, cost, fair, risk):
    """Identical homoscedastic multi-task weighting to ARDN's, so the neural baselines
    are not handicapped by a worse loss-balancing scheme."""
    tot = None
    for i, term in enumerate([traj, trec, cost, fair]):
        w = term * (log_sigma[:, i:i + 1] * -1.0).exp() + log_sigma[:, i:i + 1] * 0.5
        tot = w if tot is None else tot + w
    return (tot + risk * 0.3).mean()


# ------------------------------------------------------------------ FlatMLP
class FlatMLP:
    """Conventional feed-forward net on pooled features. No graph, no recurrence, no
    hazard model: it regresses the whole trajectory in one shot."""

    def __init__(self, horizon=15, hidden=64, seed=0):
        np.random.seed(seed)
        self.horizon = horizon
        D = pooled_input_dim()
        self.trunk = MLP([D, hidden, hidden])
        self.h_traj = Linear(hidden, horizon * 2)
        self.h_trec = Linear(hidden, 1)
        self.h_cost, self.h_fair, self.h_risk = (Linear(hidden, 1), Linear(hidden, 1), Linear(hidden, 1))
        self.log_sigma = Tensor(np.zeros((1, 4)))

    def params(self):
        p = self.trunk.params()
        for h in [self.h_traj, self.h_trec, self.h_cost, self.h_fair, self.h_risk]:
            p += h.params()
        return p + [self.log_sigma]

    def state_dict(self):
        return [p.data.copy() for p in self.params()]

    def load_state_dict(self, arrs):
        for p, a in zip(self.params(), arrs):
            p.data = a.copy(); p.grad = np.zeros_like(p.data)

    def forward(self, Xb):
        z = self.trunk(Tensor(Xb, requires_grad=False)).relu()
        return {"z": z, "traj": self.h_traj(z).sigmoid(),
                "T_rec": self.h_trec(z).sigmoid() * float(self.horizon),
                "cost": self.h_cost(z), "fair": self.h_fair(z), "risk": self.h_risk(z).sigmoid()}

    def predict_unified(self, ep, horizon=15):
        o = self.forward(pooled_input(ep).reshape(1, -1))
        return {"traj_pooled": o["traj"].data.reshape(self.horizon, 2)[:horizon], "traj_node": None,
                "T_rec": float(o["T_rec"].data[0, 0]), "cost": float(o["cost"].data[0, 0]),
                "fair": float(o["fair"].data[0, 0]), "risk": float(o["risk"].data[0, 0]),
                "z": o["z"].data.reshape(-1)}

    def fit_ood(self, episodes):
        zs = np.stack([self.predict_unified(e)["z"] for e in episodes])
        self.ood_mean = zs.mean(axis=0)
        self.ood_cov_inv = np.linalg.inv(np.cov(zs.T) + np.eye(zs.shape[1]) * 1e-3)

    def ood_score(self, ep):
        d = self.predict_unified(ep)["z"] - self.ood_mean
        return float(np.sqrt(d @ self.ood_cov_inv @ d))


def train_flatmlp(train, horizon=15, n_epochs=20, batch_size=6, lr=3e-3, seed=0, verbose=False):
    rng = np.random.default_rng(seed)
    m = FlatMLP(horizon=horizon, seed=seed)
    opt = Adam(m.params(), lr=lr)
    X = np.stack([pooled_input(e) for e in train])
    tg = [_targets(e, horizon) for e in train]
    Ytraj = np.stack([t["traj"].reshape(-1) for t in tg])
    Ytrec = np.array([t["t_rec"] for t in tg])
    Ycost = np.array([t["cost"] for t in tg])
    Yfair = np.array([t["fair"] for t in tg])
    Yrisk = np.array([t["risk"] for t in tg])
    for _ in range(n_epochs):
        order = rng.permutation(len(train))
        for i in range(0, len(order), batch_size):
            b = order[i:i + batch_size]
            opt.zero_grad()
            o = m.forward(X[b])
            loss = _weighted(m.log_sigma, _mse(o["traj"], Ytraj[b]),
                             _mse(o["T_rec"] * (1.0 / horizon), Ytrec[b] / horizon),
                             _mse(o["cost"], Ycost[b].reshape(-1, 1)),
                             _mse(o["fair"], Yfair[b].reshape(-1, 1)),
                             _bce(o["risk"], Yrisk[b].reshape(-1, 1)))
            loss.backward(); opt.step()
            m.log_sigma.data = np.clip(m.log_sigma.data, -3, 3)
    m.fit_ood(train[:150])
    return m


# ------------------------------------------------------------------ plain GNN
class GNNBaseline:
    """GraphSAGE-mean GNN with node-type one-hot inputs and direct multi-horizon decoding.
    It has the graph, but none of ARDN's FiLM conditioning, recurrent dynamics, hazard
    head or evidential uncertainty. This isolates 'does ARDN beat a standard GNN'."""

    def __init__(self, horizon=15, hidden=32, K=2, seed=0):
        np.random.seed(seed)
        self.horizon, self.K = horizon, K
        self.enc = Linear(NODE_FEAT_DIM + 4, hidden)
        self.W_self = [Linear(hidden, hidden) for _ in range(K)]
        self.W_nb = [Linear(hidden, hidden, bias=False) for _ in range(K)]
        g = 2 * hidden + GLOBAL_D_DIM + GLOBAL_A_DIM
        self.gdim = g
        self.dec = MLP([hidden + g, 64, horizon], out_act="sigmoid")
        self.g_trec, self.g_cost = MLP([g, 32, 1]), MLP([g, 32, 1])
        self.g_fair, self.g_risk = MLP([g, 32, 1]), MLP([g, 32, 1])
        self.log_sigma = Tensor(np.zeros((1, 4)))

    def params(self):
        p = self.enc.params()
        for l in self.W_self + self.W_nb:
            p += l.params()
        return (p + self.dec.params() + self.g_trec.params() + self.g_cost.params() +
                self.g_fair.params() + self.g_risk.params() + [self.log_sigma])

    def state_dict(self):
        return [p.data.copy() for p in self.params()]

    def load_state_dict(self, arrs):
        for p, a in zip(self.params(), arrs):
            p.data = a.copy(); p.grad = np.zeros_like(p.data)

    def forward(self, ep):
        N = len(ep["node_types"])
        onehot = np.eye(4)[np.asarray(ep["node_types"]).astype(int)]
        H = self.enc(Tensor(np.concatenate([ep["X"], onehot], axis=1), requires_grad=False)).relu()
        src, dst = ep["edge_index"]
        A = np.zeros((N, N)); A[dst, src] = 1.0
        A = A / np.maximum(1.0, A.sum(axis=1, keepdims=True))
        At = Tensor(A, requires_grad=False)
        for k in range(self.K):
            H = (self.W_self[k](H) + self.W_nb[k](At @ H)).relu()
        g = concat([H.mean(axis=0, keepdims=True), H.max(axis=0, keepdims=True),
                    Tensor(ep["d_vec"].reshape(1, -1), requires_grad=False),
                    Tensor(ep["a_vec"].reshape(1, -1), requires_grad=False)], axis=-1)
        g_rep = Tensor(np.ones((N, 1)), requires_grad=False) @ g
        traj_node = self.dec(concat([H, g_rep], axis=-1))      # (N,T)
        return {"traj_node": traj_node, "T_rec": self.g_trec(g).sigmoid() * float(self.horizon),
                "cost": self.g_cost(g), "fair": self.g_fair(g),
                "risk": self.g_risk(g).sigmoid(), "g": g}

    def predict_unified(self, ep, horizon=15):
        o = self.forward(ep)
        tn = o["traj_node"].data.T[:horizon]
        return {"traj_pooled": np.stack([tn.mean(axis=1), tn.min(axis=1)], axis=1), "traj_node": tn,
                "T_rec": float(o["T_rec"].data[0, 0]), "cost": float(o["cost"].data[0, 0]),
                "fair": float(o["fair"].data[0, 0]), "risk": float(o["risk"].data[0, 0]),
                "z": o["g"].data.reshape(-1)}

    def fit_ood(self, episodes):
        zs = np.stack([self.predict_unified(e)["z"] for e in episodes])
        self.ood_mean = zs.mean(axis=0)
        self.ood_cov_inv = np.linalg.inv(np.cov(zs.T) + np.eye(zs.shape[1]) * 1e-3)

    def ood_score(self, ep):
        d = self.predict_unified(ep)["z"] - self.ood_mean
        return float(np.sqrt(d @ self.ood_cov_inv @ d))


def train_gnn(train, horizon=15, n_epochs=20, batch_size=6, lr=3e-3, seed=0, verbose=False):
    rng = np.random.default_rng(seed)
    m = GNNBaseline(horizon=horizon, seed=seed)
    opt = Adam(m.params(), lr=lr)
    for _ in range(n_epochs):
        order = rng.permutation(len(train))
        for i in range(0, len(order), batch_size):
            batch = [train[j] for j in order[i:i + batch_size]]
            opt.zero_grad()
            acc = None
            for ep in batch:
                o = m.forward(ep)
                t = _weighted(m.log_sigma, _mse(o["traj_node"], ep["trajectory"][:horizon].T),
                              _mse(o["T_rec"] * (1.0 / horizon), min(ep["t_rec"], horizon) / horizon),
                              _mse(o["cost"], ep["cum_cost"] * COST_SCALE),
                              _mse(o["fair"], ep["fairness_gap"] * FAIR_SCALE),
                              _bce(o["risk"], 1.0 if ep["violated"] else 0.0))
                acc = t if acc is None else acc + t
            (acc * (1.0 / len(batch))).backward(); opt.step()
            m.log_sigma.data = np.clip(m.log_sigma.data, -3, 3)
    m.fit_ood(train[:150])
    return m


# ------------------------------------------------------------------ RandomForest
class RandomForestBaseline:
    """Classical ML on the same pooled features. Strong on tabular regression and a fair
    test of whether the deep architecture is needed at all."""

    def __init__(self, horizon=15, seed=0, n_estimators=300, min_samples_leaf=3):
        try:
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        except ImportError:
            raise ImportError(
                "RandomForestBaseline requires scikit-learn. Install it with: pip install scikit-learn"
            )
        self.horizon = horizon
        kw = dict(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                  max_features=0.5, random_state=seed, n_jobs=1)
        self.reg, self.clf = RandomForestRegressor(**kw), RandomForestClassifier(**kw)

    def fit(self, train):
        T = self.horizon
        X = np.stack([pooled_input(e) for e in train])
        tg = [_targets(e, T) for e in train]
        Y = np.column_stack([np.stack([t["traj"].reshape(-1) for t in tg]),
                             [t["t_rec"] / T for t in tg], [t["cost"] for t in tg],
                             [t["fair"] for t in tg]])
        self.y_mu, self.y_sd = Y.mean(axis=0), Y.std(axis=0) + 1e-9
        self.reg.fit(X, (Y - self.y_mu) / self.y_sd)
        self.clf.fit(X, [t["risk"] for t in tg])
        self.has_pos = len(self.clf.classes_) == 2
        return self

    def predict_unified(self, ep, horizon=15):
        T = self.horizon
        x = pooled_input(ep).reshape(1, -1)
        y = self.reg.predict(x)[0] * self.y_sd + self.y_mu
        p = self.clf.predict_proba(x)[0, 1] if self.has_pos else 0.0
        return {"traj_pooled": np.clip(y[:2 * T].reshape(T, 2), 0, 1)[:horizon], "traj_node": None,
                "T_rec": float(np.clip(y[2 * T] * T, 0, T)), "cost": float(y[2 * T + 1]),
                "fair": float(y[2 * T + 2]), "risk": float(p)}


# ------------------------------------------------------------------ Bayes-reference oracle
class OracleBaseline:
    """Re-runs the TRUE simulator, marginalising over the only quantity not recoverable
    from the features (per-node base_rate ~ U(0.10, 0.22)). Its error is therefore the
    irreducible (aleatoric) floor for this data-generating process -- the number every
    learned model should be compared against to know how much headroom is left.
    It is a reference, not a competitor: it has privileged access to the simulator."""

    def __init__(self, K=40, seed=0):
        self.K, self.seed = K, seed

    def _rebuild(self, ep):
        X, d_vec, a_vec = ep["X"], ep["d_vec"], ep["a_vec"]
        types = [int(t) for t in ep["node_types"]]
        N = len(types)
        edges = [tuple(e) for e in ep["edge_index"].T]
        echelons = [[i for i in range(N) if types[i] == t] for t in range(4)]
        dis_nodes = [int(i) for i in np.where(X[:, C_ISDISR] > 0.5)[0]]
        oh = X[dis_nodes[0], 12:16] if dis_nodes else np.zeros(4)
        disruption = {"nodes": dis_nodes,
                      "severity": float(X[dis_nodes, 10].max()) if dis_nodes else 0.0,
                      "duration": int(round(d_vec[2] * 8)),
                      "type": DISRUPTION_TYPES[int(np.argmax(oh))],
                      "cooccurrence": len(dis_nodes) > 1}
        action = {"type": ACTION_TYPES[int(np.argmax(a_vec[:6]))],
                  "source": int(np.argmax(X[:, C_ISSRC])), "target": int(np.argmax(X[:, C_ISTGT])),
                  "magnitude": float(a_vec[6]), "duration": int(round(a_vec[7] * 6))}
        return types, edges, echelons, disruption, action

    def predict_unified(self, ep, horizon=15):
        types, edges, echelons, disruption, action = self._rebuild(ep)
        X, N = ep["X"], len(types)
        rng = np.random.default_rng(self.seed)
        sim = SupplyChainSim(node_types=types, edges=edges, echelons=echelons, rng=rng)
        trajs, trecs, costs, fairs, viol = [], [], [], [], []
        for _ in range(self.K):
            sim.base_rate = rng.uniform(0.10, 0.22, size=N)
            sim.capacity, sim.demand = X[:, 3] * 120, X[:, 1] * 100
            sim.unit_cost, sim.contract_floor, sim.inventory = X[:, 7] * 3, X[:, 8], X[:, 0] * 100
            sim.service, sim.backlog = X[:, C_SERVICE].copy(), np.zeros(N)
            sim.disruption, sim.action = disruption, action
            sim.disr_remaining = np.zeros(N)
            for n in disruption["nodes"]:
                sim.disr_remaining[n] = disruption["duration"]
            r = sim.rollout(T=horizon)
            trajs.append(r["trajectory"]); trecs.append(min(r["t_rec"], horizon))
            costs.append(r["cum_cost"] * COST_SCALE); fairs.append(r["fairness_gap"] * FAIR_SCALE)
            viol.append(float(r["violated"]))
        trajs = np.stack(trajs)
        pooled = np.stack([trajs.mean(axis=2), trajs.min(axis=2)], axis=2)
        return {"traj_pooled": pooled.mean(axis=0), "traj_node": trajs.mean(axis=0),
                "T_rec": float(np.mean(trecs)), "cost": float(np.mean(costs)),
                "fair": float(np.mean(fairs)), "risk": float(np.mean(viol))}
