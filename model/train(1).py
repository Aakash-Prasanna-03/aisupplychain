"""
Training loop for ARDN.

Implements, from the design doc:
  - Section 17: dataset generated from the simulator. Actions here are all
    randomized (see README "Scope notes" for why -- there is no LLM
    negotiation loop in this standalone build), which is exactly what
    Section 17 says is needed to give the action-conditioning capability
    something informative to learn from.
  - Section 18: horizon curriculum (short rollouts first, extending to the
    full horizon) and homoscedastic multi-task loss weighting.
  - Section 9: exact evidential (Normal-Inverse-Gamma) NLL loss with an
    evidence regularizer, using a differentiable lgamma (autograd.lgamma).
"""
import argparse
import numpy as np

from simulator import make_episode, default_network
from model import ARDN
from autograd import Tensor, Adam, lgamma


def evidential_nll(y, mu, v, alpha, beta, reg_weight=0.01):
    """Exact Normal-Inverse-Gamma NLL (Amini et al., 2020) + evidence regularizer."""
    one = lambda shape: Tensor(np.ones(shape), requires_grad=False)
    y_t = Tensor(np.array([[y]]), requires_grad=False)
    err = y_t - mu
    omega = beta * (one(v.shape) + v) * 2.0  # 2*beta*(1+v)

    nll = (0.5 * (Tensor(np.full(v.shape, np.pi), requires_grad=False) / v).log()
           - alpha * omega.log()
           + (alpha + 0.5) * ((err * err) * v + omega).log()
           + lgamma(alpha) - lgamma(alpha + 0.5))

    reg = (err * err).sum() ** 0.5 * (v * 2.0 + alpha)  # |err| * (2v + alpha), evidence penalty
    return nll.sum() + reg_weight * reg.sum()


def bce(p, y):
    y_t = Tensor(np.array([[float(y)]]), requires_grad=False)
    eps = 1e-7
    one = Tensor(np.ones_like(p.data), requires_grad=False)
    return -(y_t * (p + eps).log() + (one - y_t) * (one - p + eps).log()).sum()


def mse(pred, target_np):
    t = Tensor(target_np, requires_grad=False)
    diff = pred - t
    return (diff * diff).mean()


def make_dataset(n, T, rng, net=None):
    return [make_episode(net=net, T=T, rng=rng) for _ in range(n)]


def forward_and_losses(model, ep):
    """One forward pass (encode -> rollout -> heads) plus every loss term."""
    roll = model.rollout(ep)
    pred = model.predict_from_rollout(ep, roll)
    T = model.horizon
    true_traj = ep["trajectory"]

    traj_loss = None
    for k in range(T):
        term = mse(pred["s_hat_seq"][k], true_traj[k].reshape(-1, 1))
        traj_loss = term if traj_loss is None else traj_loss + term
    traj_loss = traj_loss * (1.0 / T)

    hazard_loss = None
    for k in range(T):
        y = 1.0 if (ep["t_rec"] < T and k == ep["t_rec"]) else 0.0
        term = bce(roll["haz_seq"][k], y)
        hazard_loss = term if hazard_loss is None else hazard_loss + term
    hazard_loss = hazard_loss * (1.0 / T)

    cost_mu, cost_v, cost_a, cost_b = pred["cost"]
    cost_loss = evidential_nll(ep["cum_cost"] / 20.0, cost_mu, cost_v, cost_a, cost_b)

    fair_mu, fair_v, fair_a, fair_b = pred["fair"]
    fair_loss = evidential_nll(ep["fairness_gap"] * 5.0, fair_mu, fair_v, fair_a, fair_b)

    risk_loss = bce(pred["risk_p"], 1.0 if ep["violated"] else 0.0)
    t_rec_loss = mse(pred["T_rec"], np.array([[float(min(ep["t_rec"], T))]]))

    losses = {"traj": traj_loss, "hazard": hazard_loss, "cost": cost_loss,
              "fair": fair_loss, "risk": risk_loss, "t_rec": t_rec_loss}
    return losses, pred


def total_loss(model, losses):
    """Homoscedastic multi-task uncertainty weighting (Section 9/18) for the four
    heads that share the trajectory; risk + the auxiliary t_rec regression get small
    fixed weights (they are secondary signals on top of the hazard formulation)."""
    ls = model.log_sigma  # (1,4): [traj, hazard, cost, fair]
    precision = (ls * -1.0).exp()
    terms = [losses["traj"], losses["hazard"], losses["cost"], losses["fair"]]
    total = None
    for i, term in enumerate(terms):
        weighted = term * precision[:, i:i + 1] + ls[:, i:i + 1] * 0.5
        total = weighted if total is None else total + weighted
    total = total + losses["risk"] * 0.3 + losses["t_rec"] * 0.1
    return total


def train(n_epochs=8, episodes_per_epoch=40, horizon_schedule=(4, 7, 10, 15), lr=3e-3, seed=0,
          batch_size=4, net=None, verbose=True):
    rng = np.random.default_rng(seed)
    net = net or default_network()
    model = ARDN(horizon=horizon_schedule[-1])
    opt = Adam(model.params(), lr=lr)

    history = []
    dataset = None
    for epoch in range(n_epochs):
        cur_horizon = horizon_schedule[min(epoch * len(horizon_schedule) // n_epochs, len(horizon_schedule) - 1)]
        model.horizon = cur_horizon
        dataset = make_dataset(episodes_per_epoch, T=cur_horizon, rng=rng, net=net)

        running = {"traj": 0.0, "hazard": 0.0, "cost": 0.0, "fair": 0.0, "risk": 0.0, "t_rec": 0.0}
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            opt.zero_grad()
            batch_total = None
            for ep in batch:
                losses, _ = forward_and_losses(model, ep)
                tot = total_loss(model, losses)
                batch_total = tot if batch_total is None else batch_total + tot
                for k in running:
                    running[k] += float(losses[k].data)
            (batch_total * (1.0 / len(batch))).backward()
            opt.step()

        avg = {k: v / len(dataset) for k, v in running.items()}
        history.append(avg)
        if verbose:
            print(f"epoch {epoch+1}/{n_epochs} horizon={cur_horizon}  "
                  + "  ".join(f"{k}={v:.4f}" for k, v in avg.items()))

    # fit OOD Mahalanobis stats (Section 16) over the final epoch's z_G embeddings
    zs = np.stack([model.zG_numpy(ep) for ep in dataset])
    model.ood_mean = zs.mean(axis=0)
    cov = np.cov(zs.T) + np.eye(zs.shape[1]) * 1e-3
    model.ood_cov_inv = np.linalg.inv(cov)

    return model, history, net


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--episodes-per-epoch", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    train(n_epochs=args.epochs, episodes_per_epoch=args.episodes_per_epoch, seed=args.seed)
