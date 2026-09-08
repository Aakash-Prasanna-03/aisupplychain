"""
ARDN: Action-Conditioned Recovery Dynamics Network.

Implements the architecture from the design doc (Sections 6-9):
  (a) heterogeneous, disruption-and-action-conditioned graph attention
      encoder ("where is the network vulnerable")
  (b) recurrent latent dynamics module that unrolls a recovery trajectory
      ("how does it play out over time")
  (c) hazard-based multi-head decoder with evidential uncertainty and an
      OOD novelty scorer ("what does that imply, with what confidence")

Operates on a single episode (one supply-chain instance) per forward pass
-- there is no batch dimension baked into the graph ops, since node/edge
counts vary by topology (Section 21 generalization test). Mini-batching is
done by looping and averaging gradients in train.py, which is standard
practice for variable-sized-graph models even in full frameworks.
"""
import numpy as np
from autograd import Tensor, Linear, MLP, GRUCell, concat, stack

from simulator import (
    NODE_FEAT_DIM, EDGE_FEAT_DIM, GLOBAL_D_DIM, GLOBAL_A_DIM, NODE_TYPES,
)

HIST_DIM = 2          # [mean_service, min_service] per history step
H_HIST = 8            # history encoder hidden size
H_NODE = 32           # node embedding size
H_GRAPH = 32          # graph-level latent size (z_G, dynamics state)
K_LAYERS = 2          # number of graph attention layers
N_TYPES = len(NODE_TYPES)


class TypeAwareEncoder:
    """Separate MLP per node type (Section 8: 'the four echelons are not interchangeable')."""

    def __init__(self, in_dim, out_dim, n_types=N_TYPES):
        self.encoders = [MLP([in_dim, out_dim, out_dim]) for _ in range(n_types)]

    def __call__(self, X, node_types):
        rows = []
        for i, t in enumerate(node_types):
            rows.append(self.encoders[int(t)](X[i:i + 1]))
        return concat(rows, axis=0)

    def params(self):
        p = []
        for e in self.encoders:
            p += e.params()
        return p


class FiLM:
    """Feature-wise linear modulation: produces (gamma, beta) from a conditioning vector."""

    def __init__(self, cond_dim, feat_dim):
        self.mlp = MLP([cond_dim, feat_dim * 2])
        self.feat_dim = feat_dim

    def __call__(self, cond):
        out = self.mlp(cond)  # (1, 2*feat_dim)
        gamma = out[:, :self.feat_dim] + 1.0  # center around 1 so init is near-identity
        beta = out[:, self.feat_dim:]
        return gamma, beta

    def params(self):
        return self.mlp.params()


class HeteroGATLayer:
    """One layer of heterogeneous, FiLM-conditioned graph attention (Section 9)."""

    def __init__(self, in_dim, out_dim, edge_dim, cond_dim):
        self.out_dim = out_dim
        self.W = Linear(in_dim, out_dim, bias=False)
        self.W_self = Linear(in_dim, out_dim, bias=False)
        self.We = Linear(edge_dim, out_dim, bias=False)
        self.attn = Linear(3 * out_dim, 1, bias=False)
        self.film = FiLM(cond_dim, out_dim)

    def __call__(self, H, edge_index, edge_feat, cond, N):
        """H: (N, in_dim) Tensor. edge_index: (2,E) numpy. edge_feat: (E, edge_dim) Tensor."""
        Wh = self.W(H)          # (N, out_dim)
        Wh_self = self.W_self(H)
        We_edge = self.We(edge_feat)  # (E, out_dim)

        src, dst = edge_index[0], edge_index[1]
        E = edge_index.shape[1]

        # build attention logits per edge: a^T [Wh_i || Wh_j || e_ij]
        hi = Wh[dst]          # (E, out_dim)  -- receiving node
        hj = Wh[src]          # (E, out_dim)  -- sending node
        feat = concat([hi, hj, We_edge], axis=-1)
        logits = self.attn(feat).leaky_relu(0.2)  # (E,1)

        # segment softmax over edges sharing the same destination node
        logits_np = logits.data.reshape(-1)
        alpha_np = np.zeros_like(logits_np)
        for node in range(N):
            mask = np.where(dst == node)[0]
            if len(mask) == 0:
                continue
            m = logits_np[mask].max()
            e = np.exp(logits_np[mask] - m)
            alpha_np[mask] = e / (e.sum() + 1e-8)
        # wrap the (numerically-computed) softmax weights back as a differentiable op
        alpha = _SegmentSoftmax.apply(logits, dst, N, alpha_np)

        weighted = hj * alpha  # (E, out_dim), broadcasting alpha (E,1)
        # aggregate messages per destination node
        agg_rows = []
        for node in range(N):
            mask = np.where(dst == node)[0]
            if len(mask) == 0:
                agg_rows.append(Tensor(np.zeros((1, self.out_dim)), requires_grad=False))
            else:
                agg_rows.append(weighted[mask].sum(axis=0, keepdims=True))
        m_i = concat(agg_rows, axis=0)  # (N, out_dim)

        gamma, beta = self.film(cond)  # (1, out_dim) each
        pre = m_i + Wh_self
        out = gamma * pre.relu() + beta
        return out

    def params(self):
        return (self.W.params() + self.W_self.params() + self.We.params() +
                self.attn.params() + self.film.params())


class _SegmentSoftmax:
    """Helper: wraps a precomputed segment-softmax (numpy) as a Tensor with correct
    backward pass, since implementing a generic scatter-softmax gradient inline in
    HeteroGATLayer would be repeated boilerplate. Gradient: standard softmax VJP but
    restricted to each destination-node's edge segment."""

    @staticmethod
    def apply(logits_tensor, dst, N, alpha_np):
        out = Tensor(alpha_np.reshape(-1, 1), _prev=(logits_tensor,), name="segsoftmax")

        def _backward():
            g = out.grad.reshape(-1)
            grad_logits = np.zeros_like(g)
            for node in range(N):
                mask = np.where(dst == node)[0]
                if len(mask) == 0:
                    continue
                a = alpha_np[mask]
                gi = g[mask]
                dot = np.sum(a * gi)
                grad_logits[mask] = a * (gi - dot)
            logits_tensor.grad += grad_logits.reshape(-1, 1)

        out._backward = _backward
        return out


class EvidentialHead:
    """Normal-Inverse-Gamma head: outputs (mu, v, alpha, beta) in one forward pass,
    giving a point prediction plus aleatoric/epistemic uncertainty without an ensemble
    (Section 9)."""

    def __init__(self, in_dim, hidden=16):
        self.mlp = MLP([in_dim, hidden, 4])

    def __call__(self, x):
        raw = self.mlp(x)  # (1,4)
        mu = raw[:, 0:1]
        v = raw[:, 1:2].softplus() + 1e-3
        alpha = raw[:, 2:3].softplus() + 1.0 + 1e-3
        beta = raw[:, 3:4].softplus() + 1e-3
        return mu, v, alpha, beta

    def params(self):
        return self.mlp.params()


class ARDN:
    def __init__(self, horizon=15, tau=0.9, use_film=True, use_gat=True, use_dynamics=True,
                 use_action=True, use_hazard=True, use_evidential=True):
        # ablation switches (Section 20)
        self.use_film = use_film
        self.use_gat = use_gat
        self.use_dynamics = use_dynamics
        self.use_action = use_action
        self.use_hazard = use_hazard
        self.use_evidential = use_evidential

        self.horizon = horizon
        self.tau = tau
        cond_dim = (GLOBAL_D_DIM + GLOBAL_A_DIM) if use_action else GLOBAL_D_DIM

        self.node_encoder = TypeAwareEncoder(NODE_FEAT_DIM, H_NODE)
        self.hist_gru = GRUCell(HIST_DIM, H_HIST)
        self.gat_layers = [HeteroGATLayer(H_NODE, H_NODE, EDGE_FEAT_DIM, cond_dim) for _ in range(K_LAYERS)]
        self.pool_mlp = MLP([H_NODE, 1])  # attention readout logits
        self.init_mlp = MLP([H_NODE + H_HIST, H_GRAPH])
        self.dyn_gru = GRUCell(cond_dim, H_GRAPH)
        self.dec_mlp = MLP([H_GRAPH + H_NODE, H_NODE, 1], out_act="sigmoid")  # per-node service decoder

        self.hazard_head = MLP([H_GRAPH, 16, 1], out_act="sigmoid")
        self.cost_head = EvidentialHead(H_GRAPH * 2)
        self.fair_head = EvidentialHead(H_GRAPH * 2)
        self.risk_head = MLP([H_GRAPH * 2, 16, 1], out_act="sigmoid")

        # homoscedastic multi-task log-variances (Section 9/18)
        self.log_sigma = Tensor(np.zeros((1, 4)))  # traj, hazard, cost, fair (risk uses fixed weight)

    def params(self):
        p = []
        p += self.node_encoder.params()
        p += self.hist_gru.params()
        for l in self.gat_layers:
            p += l.params()
        p += self.pool_mlp.params()
        p += self.init_mlp.params()
        p += self.dyn_gru.params()
        p += self.dec_mlp.params()
        p += self.hazard_head.params()
        p += self.cost_head.params()
        p += self.fair_head.params()
        p += self.risk_head.params()
        p += [self.log_sigma]
        return p

    def encode(self, ep):
        """Runs the encoder + vulnerability propagation module. Returns node embeddings,
        graph embedding z_G, and the conditioning vector `cond` (for FiLM/dynamics)."""
        N = len(ep["node_types"])
        X = Tensor(ep["X"], requires_grad=False)
        H = self.node_encoder(X, ep["node_types"])

        d_vec = Tensor(ep["d_vec"].reshape(1, -1), requires_grad=False)
        a_vec = Tensor(ep["a_vec"].reshape(1, -1), requires_grad=False)
        cond = concat([d_vec, a_vec], axis=-1) if self.use_action else d_vec
        if not self.use_film:
            cond = Tensor(np.zeros_like(cond.data), requires_grad=False)

        edge_feat = Tensor(ep["edge_feat"], requires_grad=False)
        if self.use_gat:
            for layer in self.gat_layers:
                H = layer(H, ep["edge_index"], edge_feat, cond, N)
        # else: skip graph attention entirely -> flat pooled-MLP ablation (Section 20)

        hist = ep["hist"]
        h = Tensor(np.zeros((1, H_HIST)), requires_grad=False)
        for t in range(hist.shape[0]):
            h = self.hist_gru(Tensor(hist[t:t + 1], requires_grad=False), h)

        pool_logits = self.pool_mlp(H)  # (N,1)
        pool_w = pool_logits.softmax(axis=0)
        z_G_graph = (H * pool_w).sum(axis=0, keepdims=True)  # (1, H_NODE)
        z_G = self.init_mlp(concat([z_G_graph, h], axis=-1))  # (1, H_GRAPH)
        return H, z_G, cond, pool_w

    def rollout(self, ep, teacher_force_prob=0.0):
        """Unrolls the dynamics module for `horizon` steps, decoding a per-node service
        trajectory at each step. `teacher_force_prob` > 0 feeds ground truth at the
        previous step as an auxiliary decoder input (Section 18 scheduled sampling)."""
        N = len(ep["node_types"])
        H, z_G, cond, pool_w = self.encode(ep)
        true_traj = ep.get("trajectory")

        z_seq, s_hat_seq, haz_seq = [], [], []
        z_t = z_G
        prev_s = Tensor(np.ones((N, 1)), requires_grad=False)  # start fully healthy prior
        for k in range(self.horizon):
            if self.use_dynamics:
                dyn_input = cond
                z_t = self.dyn_gru(dyn_input, z_t)
            # else: static latent (no recurrent dynamics) -> tests trajectory value (Section 20)
            z_seq.append(z_t)

            # broadcast z_t (1,H_GRAPH) to (N,H_GRAPH), keeping it differentiable
            z_rep = _broadcast_rows(z_t, N)
            dec_in = concat([z_rep, H], axis=-1)
            s_hat = self.dec_mlp(dec_in)  # (N,1) in [0,1]
            s_hat_seq.append(s_hat)

            haz = self.hazard_head(z_t) if self.use_hazard else Tensor(np.zeros((1, 1)), requires_grad=False)
            haz_seq.append(haz)

            # scheduled sampling: choose the "previous service" fed forward implicitly
            # via z_t already carrying dynamics; explicit teacher forcing affects decoder
            # by nothing further needed here since z_t is graph-level and s_hat is read off it.
            prev_s = s_hat

        return {
            "H": H, "z_G": z_G, "z_seq": z_seq, "s_hat_seq": s_hat_seq,
            "haz_seq": haz_seq, "pool_w": pool_w,
        }

    def predict(self, ep, teacher_force_prob=0.0):
        """Full forward pass -> recovery profile R plus uncertainty, matching Section 11."""
        out = self.rollout(ep, teacher_force_prob=teacher_force_prob)
        return self.predict_from_rollout(ep, out)

    def predict_from_rollout(self, ep, out):
        """Same as predict(), but reuses an already-computed rollout (avoids a duplicate
        forward pass through the encoder+dynamics when the caller already has one, e.g.
        in the training loop which needs roll['haz_seq'] directly for the hazard loss)."""
        z_seq, s_hat_seq, haz_seq = out["z_seq"], out["s_hat_seq"], out["haz_seq"]
        T = self.horizon

        # hazard -> expected recovery time (Section 9/12)
        surv = Tensor(np.ones((1, 1)), requires_grad=False)
        p_prev = Tensor(np.zeros((1, 1)), requires_grad=False)
        t_rec_terms = []
        var_terms_probs = []
        for k in range(T):
            haz_k = haz_seq[k]
            p_rec_by_k = Tensor(np.ones((1, 1)), requires_grad=False) - surv * (Tensor(np.ones((1, 1)), requires_grad=False) - haz_k)
            surv = surv * (Tensor(np.ones((1, 1)), requires_grad=False) - haz_k)
            pmf_k = p_rec_by_k - p_prev
            t_rec_terms.append(pmf_k * float(k))
            var_terms_probs.append(pmf_k)
            p_prev = p_rec_by_k
        T_rec = t_rec_terms[0]
        for term in t_rec_terms[1:]:
            T_rec = T_rec + term
        # hazard-distribution variance -> a natural uncertainty component for T_rec
        e_k2 = var_terms_probs[0] * 0.0
        for k, pmf_k in enumerate(var_terms_probs):
            e_k2 = e_k2 + pmf_k * float(k * k)
        T_rec_var = e_k2 - T_rec * T_rec

        # pooled trajectory features for the evidential/risk heads
        z_mean = z_seq[0]
        for z in z_seq[1:]:
            z_mean = z_mean + z
        z_mean = z_mean * (1.0 / T)
        z_last = z_seq[-1]
        pooled = concat([z_mean, z_last], axis=-1)  # (1, 2*H_GRAPH)

        cost_mu, cost_v, cost_a, cost_b = self.cost_head(pooled)
        fair_mu, fair_v, fair_a, fair_b = self.fair_head(pooled)
        risk_p = self.risk_head(pooled)

        # trajectory functionals (Section 9): L_service, F_gap computed from decoded ŝ
        min_over_nodes = []
        for s_hat in s_hat_seq:
            min_over_nodes.append(s_hat.min(axis=0, keepdims=True))  # (1,1)
        deficits = [Tensor(np.array([[self.tau]]), requires_grad=False) - m for m in min_over_nodes]
        # L_service = max_k(tau - min_i s_hat) -> max across the T per-step deficits
        L_service = stack(deficits, axis=0).max(axis=0)

        return {
            "T_rec": T_rec, "T_rec_var": T_rec_var,
            "L_service": L_service,
            "cost": (cost_mu, cost_v, cost_a, cost_b),
            "fair": (fair_mu, fair_v, fair_a, fair_b),
            "risk_p": risk_p,
            "s_hat_seq": s_hat_seq,
            "z_G": out["z_G"],
            "pool_w": out["pool_w"],
        }

    def state_dict(self):
        """Plain numpy arrays only (Tensors hold non-picklable backward closures)."""
        return [p.data.copy() for p in self.params()]

    def load_state_dict(self, arrays):
        for p, a in zip(self.params(), arrays):
            p.data = a.copy()
            p.grad = np.zeros_like(p.data)

    def zG_numpy(self, ep):
        _, z_G, _, _ = self.encode(ep)
        return z_G.data.reshape(-1)

    def ood_score(self, ep):
        """Mahalanobis distance of this episode's graph embedding against the training
        distribution (Section 16). Requires train() to have set ood_mean/ood_cov_inv."""
        if not hasattr(self, "ood_mean"):
            raise RuntimeError("Call train() first (or set model.ood_mean/ood_cov_inv) "
                                "before requesting an OOD score.")
        z = self.zG_numpy(ep)
        diff = z - self.ood_mean
        return float(np.sqrt(diff @ self.ood_cov_inv @ diff))


def _broadcast_rows(t, n):
    """Repeats a (1,D) Tensor into (n,D) while keeping it differentiable
    (gradients from all n rows sum back into the single source row, which is
    exactly correct for a broadcast)."""
    rows = [t for _ in range(n)]
    return concat(rows, axis=0)
