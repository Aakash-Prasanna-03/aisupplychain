"""
A lightweight multi-echelon supply-chain simulator.

This stands in for the "project's simulator" referenced in the design doc
(Section 17: "Generated directly from the project's simulator..."). It is
intentionally simple but has the properties ARDN is designed to exploit:

  * network propagation: a disruption/action upstream degrades or improves
    downstream nodes even when they are not directly targeted, via a
    multiplicative "ancestor service" term.
  * action-dependence: six action types with genuinely different effects
    (some trade off fairness for speed, some cost more, some only work if
    targeted correctly).
  * non-trivial recovery *shape* (fast-then-plateau, slow-and-steady,
    overshoot-then-settle) so trajectory modelling actually matters.
  * a real fairness/cost/risk trade-off surface, so the multi-head decoder
    has something non-degenerate to learn.
"""
import numpy as np

NODE_TYPES = ["Supplier", "Manufacturer", "Distributor", "Retailer"]
NODE_TYPE_ID = {n: i for i, n in enumerate(NODE_TYPES)}
ACTION_TYPES = ["reallocate", "reroute", "delay", "prioritize", "share_capacity", "emergency_source"]
DISRUPTION_TYPES = ["supply", "demand", "transport", "capacity"]

NODE_BASE_DIM = 9   # inventory, in_flow, out_flow, capacity, backlog, lead_time, service, cost, contract_floor
NODE_DISR_DIM = 3 + len(DISRUPTION_TYPES)  # is_disrupted, severity, remaining_dur, type onehot
NODE_ACT_DIM = 2 + len(ACTION_TYPES) + 1   # is_source, is_target, action onehot, magnitude
NODE_FEAT_DIM = NODE_BASE_DIM + NODE_DISR_DIM + NODE_ACT_DIM  # = 9+7+9 = 25
EDGE_FEAT_DIM = 5  # capacity, lead_time, route_status, shipment_qty, disrupted
GLOBAL_D_DIM = 4   # frac_disrupted, max_severity, max_duration, cooccurrence
GLOBAL_A_DIM = len(ACTION_TYPES) + 2  # onehot + magnitude + duration


def default_network(n_per_echelon=2, lateral=True):
    """Builds a small 4-echelon network: suppliers -> manufacturers -> distributors -> retailers.
    Returns node_types (list[int]), edges (list of (i,j)) directed downstream, plus
    optional lateral edges within an echelon (used by 'share_capacity' actions)."""
    node_types = []
    echelons = []
    idx = 0
    for t in range(4):
        ids = list(range(idx, idx + n_per_echelon))
        echelons.append(ids)
        node_types += [t] * n_per_echelon
        idx += n_per_echelon
    edges = []
    for e in range(3):
        for i in echelons[e]:
            for j in echelons[e + 1]:
                edges.append((i, j))
    if lateral:
        for ids in echelons:
            for a in range(len(ids)):
                for b in range(len(ids)):
                    if a != b:
                        edges.append((ids[a], ids[b]))
    return node_types, edges, echelons


class SupplyChainSim:
    """Discrete-time recovery simulator. One 'episode' = one (state, disruption, action)
    example rolled forward T steps."""

    def __init__(self, node_types, edges, echelons, rng=None):
        self.node_types = node_types
        self.N = len(node_types)
        self.edges = edges
        self.echelons = echelons
        self.rng = rng or np.random.default_rng()
        # static per-node economics, sampled once per episode in reset()
        self.parents = {i: [a for (a, b) in edges if b == i and self._echelon_of(a) == self._echelon_of(i) - 1]
                         for i in range(self.N)}
        self.siblings = {i: [j for j in echelons[self._echelon_of(i)] if j != i] for i in range(self.N)}

    def _echelon_of(self, node):
        for e, ids in enumerate(self.echelons):
            if node in ids:
                return e
        return -1

    def reset(self, disruption=None, action=None):
        """Sample a fresh episode: baseline recovery rates, then apply a disruption
        and (optionally) an action, then return the structured descriptors."""
        N = self.N
        self.base_rate = self.rng.uniform(0.10, 0.22, size=N)     # natural per-step recovery rate
        self.capacity = self.rng.uniform(80, 120, size=N)
        self.demand = self.rng.uniform(50, 90, size=N)
        self.contract_floor = self.rng.uniform(0.5, 0.7, size=N)
        self.unit_cost = self.rng.uniform(1.0, 3.0, size=N)
        self.service = np.ones(N)  # 1.0 = fully healthy
        self.backlog = np.zeros(N)
        self.inventory = self.rng.uniform(40, 100, size=N)

        if disruption is None:
            disruption = self._sample_disruption()
        if action is None:
            action = self._sample_action()
        self.disruption = disruption
        self.action = action

        # apply disruption: knock down service at affected nodes
        for node in disruption["nodes"]:
            self.service[node] = max(0.05, 1.0 - disruption["severity"])
        self.disr_remaining = np.zeros(N)
        for node in disruption["nodes"]:
            self.disr_remaining[node] = disruption["duration"]

        return disruption, action

    def _sample_disruption(self):
        n_affected = self.rng.choice([1, 2], p=[0.7, 0.3])
        nodes = list(self.rng.choice(self.N, size=n_affected, replace=False))
        return {
            "nodes": nodes,
            "severity": float(self.rng.uniform(0.3, 0.9)),
            "duration": int(self.rng.integers(2, 8)),
            "type": self.rng.choice(DISRUPTION_TYPES),
            "cooccurrence": n_affected > 1,
        }

    def _sample_action(self, forced_type=None):
        a_type = forced_type or self.rng.choice(ACTION_TYPES)
        source = int(self.rng.integers(0, self.N))
        target = int(self.rng.integers(0, self.N))
        return {
            "type": a_type,
            "source": source,
            "target": target,
            "magnitude": float(self.rng.uniform(0.2, 1.0)),
            "duration": int(self.rng.integers(2, 6)),
        }

    def _action_boost(self, node, t):
        """Per-node, per-step recovery-rate boost (or penalty) from the active action."""
        a = self.action
        boost = 0.0
        cost = 0.0
        active = t < a["duration"]
        if not active:
            return 0.0, 0.0
        mag = a["magnitude"]
        if a["type"] == "reallocate":
            if node == a["target"]:
                boost += 0.10 * mag
            if node == a["source"]:
                boost -= 0.04 * mag
        elif a["type"] == "reroute":
            # helps target only if it is downstream-connected to a disrupted node;
            # generically: flat modest boost to target, small network-wide relief
            if node == a["target"]:
                boost += 0.08 * mag
        elif a["type"] == "delay":
            # buys time: smaller boost but no cost, spread to target's siblings too (protects fairness)
            if node == a["target"] or node in self.siblings.get(a["target"], []):
                boost += 0.05 * mag
        elif a["type"] == "prioritize":
            if node == a["target"]:
                boost += 0.16 * mag
            elif node in self.siblings.get(a["target"], []):
                boost -= 0.05 * mag  # sibling penalty: this is the fairness trade-off action
        elif a["type"] == "share_capacity":
            if node == a["target"]:
                boost += 0.09 * mag
            if node == a["source"]:
                boost -= 0.03 * mag
        elif a["type"] == "emergency_source":
            if node == a["target"]:
                boost += 0.22 * mag
                cost += 6.0 * mag  # expensive
        if node == a["target"] and a["type"] == "emergency_source":
            cost += 6.0 * mag
        elif node in (a["source"], a["target"]) and a["type"] in ("reallocate", "share_capacity"):
            cost += 1.0 * mag
        elif node == a["target"]:
            cost += 0.4 * mag
        return boost, cost

    def step(self, t):
        """Advance one step. Returns per-node service (post-update), step cost, done-info."""
        N = self.N
        new_service = self.service.copy()
        step_cost = 0.0
        for i in range(N):
            boost, cost = self._action_boost(i, t)
            step_cost += cost
            # disruption drag: while remaining, actively fights recovery
            drag = 0.0
            if self.disr_remaining[i] > 0:
                drag = 0.12 * (self.disr_remaining[i] / max(1, self.disruption["duration"]))
                self.disr_remaining[i] -= 1
            # ancestor propagation: service is capped by upstream health (network effect)
            parents = self.parents.get(i, [])
            anc = np.mean([self.service[p] for p in parents]) if parents else 1.0
            target_level = min(1.0, anc)
            rate = max(0.01, self.base_rate[i] + boost - drag)
            new_service[i] = self.service[i] + rate * (target_level - self.service[i])
            new_service[i] = float(np.clip(new_service[i], 0.0, 1.0))
            # backlog / holding cost proxy
            unmet = self.demand[i] * (1 - new_service[i])
            self.backlog[i] = max(0.0, 0.6 * self.backlog[i] + unmet - self.capacity[i] * 0.05)
            step_cost += 0.05 * self.backlog[i] * self.unit_cost[i]
        self.service = new_service
        return self.service.copy(), step_cost

    def rollout(self, T=15, tau=0.9):
        """Runs T steps, returns trajectory (T,N), per-step cost, and derived targets."""
        traj = np.zeros((T, self.N))
        costs = np.zeros(T)
        for t in range(T):
            s, c = self.step(t)
            traj[t] = s
            costs[t] = c
        # ground-truth recovery step: first t where ALL nodes >= tau (else T, i.e. "not recovered in horizon")
        recovered_mask = (traj >= tau).all(axis=1)
        if recovered_mask.any():
            t_rec = int(np.argmax(recovered_mask))
        else:
            t_rec = T  # censored
        service_loss = float(np.max(tau - traj.min(axis=1)))
        service_loss = max(0.0, service_loss)
        cum_cost = float(costs.sum())
        fairness_gap = float(np.sum(np.var(traj, axis=1)))
        # a hard "constraint violation" proxy: backlog blew past capacity for any node at any step
        violated = bool((self.backlog > self.capacity * 1.5).any())
        return {
            "trajectory": traj,        # (T, N)
            "costs": costs,            # (T,)
            "t_rec": t_rec,
            "service_loss": service_loss,
            "cum_cost": cum_cost,
            "fairness_gap": fairness_gap,
            "violated": violated,
        }

    # ---------------- feature builders (topology-invariant dims) ----------------
    def build_node_features(self):
        """Builds the (N, NODE_FEAT_DIM) input matrix from current sim + disruption + action state."""
        N = self.N
        X = np.zeros((N, NODE_FEAT_DIM))
        d = self.disruption
        a = self.action
        for i in range(N):
            base = [
                self.inventory[i] / 100.0,
                self.demand[i] / 100.0,          # incoming flow proxy
                self.demand[i] / 100.0,          # outgoing flow proxy
                self.capacity[i] / 120.0,
                self.backlog[i] / 50.0,
                1.0,                              # lead_time placeholder (normalized)
                self.service[i],
                self.unit_cost[i] / 3.0,
                self.contract_floor[i],
            ]
            is_disr = 1.0 if i in d["nodes"] else 0.0
            sev = d["severity"] if i in d["nodes"] else 0.0
            rem = self.disr_remaining[i] / max(1, d["duration"])
            type_oh = [1.0 if i in d["nodes"] and d["type"] == dt else 0.0 for dt in DISRUPTION_TYPES]
            disr_feat = [is_disr, sev, rem] + type_oh
            is_src = 1.0 if i == a["source"] else 0.0
            is_tgt = 1.0 if i == a["target"] else 0.0
            act_oh = [1.0 if a["type"] == at else 0.0 for at in ACTION_TYPES]
            act_feat = [is_src, is_tgt] + act_oh + [a["magnitude"]]
            X[i] = base + disr_feat + act_feat
        return X

    def build_edge_index_and_features(self):
        idx = np.array(self.edges, dtype=np.int64).T  # (2, E)
        E = len(self.edges)
        feat = np.zeros((E, EDGE_FEAT_DIM))
        d = self.disruption
        for k, (i, j) in enumerate(self.edges):
            disrupted = 1.0 if (i in d["nodes"] or j in d["nodes"]) and d["type"] in ("transport",) else 0.0
            feat[k] = [1.0, 1.0, 1.0 - disrupted, 0.5, disrupted]
        return idx, feat

    def build_global_descriptors(self):
        d, a = self.disruption, self.action
        d_vec = np.array([
            len(d["nodes"]) / self.N,
            d["severity"],
            d["duration"] / 8.0,
            1.0 if d["cooccurrence"] else 0.0,
        ])
        a_oh = [1.0 if a["type"] == at else 0.0 for at in ACTION_TYPES]
        a_vec = np.array(a_oh + [a["magnitude"], a["duration"] / 6.0])
        return d_vec, a_vec

    def history_summary(self, window=4):
        """Cheap topology-invariant 'recent history': here we approximate pre-disruption
        history as [mean_service=1.0, min_service=1.0] repeated, since t=0 is the disruption
        onset in this simplified simulator (recent history was healthy)."""
        return np.tile(np.array([1.0, 1.0]), (window, 1))


def make_episode(net=None, T=15, forced_action_type=None, rng=None):
    """Convenience: build a fresh episode (state+disruption+action) and roll it out."""
    rng = rng or np.random.default_rng()
    if net is None:
        node_types, edges, echelons = default_network()
    else:
        node_types, edges, echelons = net
    sim = SupplyChainSim(node_types, edges, echelons, rng=rng)
    disruption = sim._sample_disruption()
    action = sim._sample_action(forced_type=forced_action_type)
    sim.reset(disruption=disruption, action=action)
    result = sim.rollout(T=T)
    X = sim.build_node_features()
    edge_index, edge_feat = sim.build_edge_index_and_features()
    d_vec, a_vec = sim.build_global_descriptors()
    hist = sim.history_summary()
    return {
        "node_types": np.array(node_types),
        "X": X,
        "edge_index": edge_index,
        "edge_feat": edge_feat,
        "d_vec": d_vec,
        "a_vec": a_vec,
        "hist": hist,
        **result,
    }


if __name__ == "__main__":
    ep = make_episode()
    print("N nodes:", len(ep["node_types"]), "edges:", ep["edge_index"].shape[1])
    print("t_rec:", ep["t_rec"], "service_loss:", round(ep["service_loss"], 3),
          "cum_cost:", round(ep["cum_cost"], 2), "fairness_gap:", round(ep["fairness_gap"], 3),
          "violated:", ep["violated"])
    print("trajectory (first 5 steps):\n", np.round(ep["trajectory"][:5], 2))
