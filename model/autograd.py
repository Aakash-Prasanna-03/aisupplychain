"""
Minimal reverse-mode autodiff over NumPy arrays.

Why this exists: the sandbox this was built in has no network access, so
torch/jax could not be installed. ARDN (see model.py) needs real gradients
to actually train, not just a static architecture diagram, so this file
provides a small but general Tensor autograd engine (think: a tiny,
array-valued micrograd) with the handful of ops ARDN needs:
add/mul/matmul (with broadcasting), activations, softmax, concatenation,
reductions, and indexing/gather. It is not meant to be a general-purpose
framework -- just enough to make the model in model.py differentiable
end-to-end.
"""
import numpy as np
from scipy.special import gammaln, digamma

np.random.seed(0)


def _unbroadcast(grad, shape):
    """Sum-reduce `grad` down to `shape`, undoing NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    __slots__ = ("data", "grad", "_backward", "_prev", "requires_grad", "name")

    def __init__(self, data, requires_grad=True, _prev=(), name=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = _prev
        self.requires_grad = requires_grad
        self.name = name

    @property
    def shape(self):
        return self.data.shape

    # ---------- graph plumbing ----------
    def backward(self):
        topo, visited = [], set()

        def build(t):
            if id(t) not in visited:
                visited.add(id(t))
                for p in t._prev:
                    build(p)
                topo.append(t)

        build(self)
        self.grad = np.ones_like(self.data)
        for t in reversed(topo):
            t._backward()

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    # ---------- elementwise / linear algebra ----------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, _prev=(self, other), name="add")

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        out = Tensor(-self.data, _prev=(self,), name="neg")

        def _backward():
            self.grad += -out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, _prev=(self, other), name="mul")

        def _backward():
            self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data / other.data, _prev=(self, other), name="div")

        def _backward():
            self.grad += _unbroadcast(out.grad / other.data, self.data.shape)
            other.grad += _unbroadcast(-out.grad * self.data / (other.data ** 2), other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, _prev=(self, other), name="matmul")

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __pow__(self, p):
        out = Tensor(self.data ** p, _prev=(self,), name="pow")

        def _backward():
            self.grad += p * (self.data ** (p - 1)) * out.grad

        out._backward = _backward
        return out

    # ---------- reductions ----------
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), _prev=(self,), name="sum")

        def _backward():
            g = out.grad
            if not keepdims and axis is not None:
                g = np.expand_dims(g, axis)
            self.grad += np.broadcast_to(g, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def max(self, axis=None, keepdims=False):
        out_data = self.data.max(axis=axis, keepdims=True)
        out = Tensor(out_data if keepdims else out_data.squeeze(axis), _prev=(self,), name="max")
        mask = (self.data == out_data).astype(np.float64)
        mask = mask / mask.sum(axis=axis, keepdims=True)  # split credit on ties

        def _backward():
            g = out.grad
            if not keepdims and axis is not None:
                g = np.expand_dims(g, axis)
            self.grad += mask * g

        out._backward = _backward
        return out

    def min(self, axis=None, keepdims=False):
        out_data = self.data.min(axis=axis, keepdims=True)
        out = Tensor(out_data if keepdims else out_data.squeeze(axis), _prev=(self,), name="min")
        mask = (self.data == out_data).astype(np.float64)
        mask = mask / mask.sum(axis=axis, keepdims=True)

        def _backward():
            g = out.grad
            if not keepdims and axis is not None:
                g = np.expand_dims(g, axis)
            self.grad += mask * g

        out._backward = _backward
        return out

    # ---------- activations ----------
    def relu(self):
        out = Tensor(np.maximum(self.data, 0), _prev=(self,), name="relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def leaky_relu(self, slope=0.2):
        out = Tensor(np.where(self.data > 0, self.data, slope * self.data), _prev=(self,), name="lrelu")

        def _backward():
            self.grad += np.where(self.data > 0, 1.0, slope) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -30, 30)))
        out = Tensor(s, _prev=(self,), name="sigmoid")

        def _backward():
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, _prev=(self,), name="tanh")

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(np.clip(self.data, -30, 30))
        out = Tensor(e, _prev=(self,), name="exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        eps = 1e-8
        out = Tensor(np.log(self.data + eps), _prev=(self,), name="log")

        def _backward():
            self.grad += out.grad / (self.data + eps)

        out._backward = _backward
        return out

    def softplus(self):
        # log(1+exp(x)), numerically stable, used for evidential nu/alpha/beta
        return (self.exp() + 1.0).log()

    def softmax(self, axis=-1):
        m = self.data.max(axis=axis, keepdims=True)
        e = np.exp(self.data - m)
        s = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(s, _prev=(self,), name="softmax")

        def _backward():
            # jacobian-vector product for softmax
            g = out.grad
            dot = (g * s).sum(axis=axis, keepdims=True)
            self.grad += s * (g - dot)

        out._backward = _backward
        return out

    # ---------- shape ops ----------
    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), _prev=(self,), name="reshape")

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, *axes):
        axes = axes if axes else None
        out = Tensor(self.data.transpose(axes), _prev=(self,), name="transpose")
        inv = np.argsort(axes) if axes else None

        def _backward():
            self.grad += out.grad.transpose(inv) if inv is not None else out.grad.transpose()

        out._backward = _backward
        return out

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], _prev=(self,), name="getitem")

        def _backward():
            g = np.zeros_like(self.data)
            g[idx] += out.grad
            self.grad += g

        out._backward = _backward
        return out


def lgamma(t):
    """log-gamma, differentiable via digamma (needed for the exact evidential/NIG NLL)."""
    out = Tensor(gammaln(t.data), _prev=(t,), name="lgamma")

    def _backward():
        t.grad += digamma(t.data) * out.grad

    out._backward = _backward
    return out


def concat(tensors, axis=-1):
    datas = [t.data for t in tensors]
    out = Tensor(np.concatenate(datas, axis=axis), _prev=tuple(tensors), name="concat")
    sizes = [d.shape[axis] for d in datas]

    def _backward():
        splits = np.split(out.grad, np.cumsum(sizes)[:-1], axis=axis)
        for t, g in zip(tensors, splits):
            t.grad += g

    out._backward = _backward
    return out


def stack(tensors, axis=0):
    return concat([t.reshape(*_insert(t.shape, axis, 1)) for t in tensors], axis=axis)


def _insert(shape, axis, val):
    shape = list(shape)
    axis = axis if axis >= 0 else len(shape) + axis + 1
    shape.insert(axis, val)
    return tuple(shape)


# ---------------- layers ----------------
class Linear:
    def __init__(self, in_dim, out_dim, bias=True):
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        self.W = Tensor(np.random.uniform(-limit, limit, (in_dim, out_dim)))
        self.b = Tensor(np.zeros((1, out_dim))) if bias else None

    def __call__(self, x):
        out = x @ self.W
        if self.b is not None:
            out = out + self.b
        return out

    def params(self):
        return [self.W] + ([self.b] if self.b is not None else [])


class MLP:
    def __init__(self, dims, act="relu", out_act=None):
        self.layers = [Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        self.act = act
        self.out_act = out_act

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = x.relu() if self.act == "relu" else x.tanh()
            elif self.out_act == "sigmoid":
                x = x.sigmoid()
            elif self.out_act == "tanh":
                x = x.tanh()
        return x

    def params(self):
        p = []
        for l in self.layers:
            p += l.params()
        return p


class GRUCell:
    """Standard GRU cell operating on a single (1, hidden) state row."""

    def __init__(self, in_dim, hid_dim):
        self.hid_dim = hid_dim
        self.Wz = Linear(in_dim + hid_dim, hid_dim)
        self.Wr = Linear(in_dim + hid_dim, hid_dim)
        self.Wh = Linear(in_dim + hid_dim, hid_dim)

    def __call__(self, x, h):
        xh = concat([x, h], axis=-1)
        z = self.Wz(xh).sigmoid()
        r = self.Wr(xh).sigmoid()
        xrh = concat([x, r * h], axis=-1)
        h_tilde = self.Wh(xrh).tanh()
        h_new = z * h + (Tensor(np.ones_like(z.data), requires_grad=False) - z) * h_tilde
        return h_new

    def params(self):
        return self.Wz.params() + self.Wr.params() + self.Wh.params()


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self, clip=5.0):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            if clip is not None:
                g = np.clip(g, -clip, clip)
            if self.wd:
                g = g + self.wd * p.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g ** 2)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
