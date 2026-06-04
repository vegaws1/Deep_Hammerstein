"""Synthetic MIMO Hammerstein plants, noise regimes, excitation and scaling.

Two plant families are provided:

* :class:`StaticHammersteinPlant` -- a *memoryless* nonlinear block
  ``n_t = g(u_t)`` followed by a stable MIMO ARX block.  Because the
  nonlinearity has no memory, the latent/linear split is identifiable up to a
  latent coordinate transformation (Proposition 1 in the manuscript), so this
  plant is used for the identification / parameter-recovery experiments.

* :class:`RecurrentHammersteinPlant` -- a memoryful (GRU-like) nonlinear block
  ``n_t = f(u_{1:t})``, used for the prediction-only ("Hammerstein-compatible")
  regime.

Both share a configurable colored / heavy-tailed measurement-noise process.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np


# ----------------------------------------------------------------------
# Configs
# ----------------------------------------------------------------------
@dataclass
class NoiseConfig:
    kind: str = "studentt"        # "gaussian" | "studentt"
    colored: bool = True          # AR(1) residual dynamics if True
    df: float = 5.0               # Student-t degrees of freedom
    outlier_prob: float = 0.0     # probability of an additive outlier per sample
    outlier_scale: float = 6.0
    chol: tuple = None            # optional lower-tri Cholesky of the innovation scatter


@dataclass
class PlantConfig:
    in_dim: int = 2
    latent_dim: int = 2
    out_dim: int = 2
    na: int = 2
    nb: int = 2
    noise_ar_order: int = 1
    hidden_dim: int = 12          # used by the recurrent plant only
    noise: NoiseConfig = field(default_factory=NoiseConfig)


# ----------------------------------------------------------------------
# Stability helper
# ----------------------------------------------------------------------
def companion_matrix(A_list):
    """Block companion matrix of x_t = -sum_i A_i x_{t-i}."""
    na = len(A_list)
    s = A_list[0].shape[0]
    top = np.concatenate([-A_list[i] for i in range(na)], axis=1)
    if na == 1:
        return top
    eye = np.eye(s * (na - 1))
    zeros = np.zeros((s * (na - 1), s))
    bottom = np.concatenate([eye, zeros], axis=1)
    return np.concatenate([top, bottom], axis=0)


def spectral_radius(A_list):
    Ac = companion_matrix(A_list)
    return float(np.max(np.abs(np.linalg.eigvals(Ac))))


def _stabilize(A_list, margin=0.05):
    """Scale A_i <- kappa^i A_i so the companion radius is <= 1 - margin.

    Uses the exact eigenvalue-scaling identity (see manuscript, Sec. 4.5):
    replacing A_i by kappa^i A_i multiplies every companion eigenvalue by
    kappa.
    """
    rho = spectral_radius(A_list)
    target = 1.0 - margin
    if rho <= target:
        return A_list
    kappa = target / rho
    return [A_list[i] * (kappa ** (i + 1)) for i in range(len(A_list))]


# ----------------------------------------------------------------------
# Shared noise generator
# ----------------------------------------------------------------------
class _NoiseModel:
    def __init__(self, cfg: PlantConfig, rng: np.random.Generator):
        s = cfg.out_dim
        self.cfg = cfg
        self.rng = rng
        # contemporaneous innovation scatter Sigma = chol chol^T
        if cfg.noise.chol is not None:
            self.chol = np.array(cfg.noise.chol, dtype=np.float64)
        elif s == 2:
            self.chol = np.array([[0.12, 0.0], [0.05, 0.10]], dtype=np.float64)
        else:
            self.chol = 0.1 * np.tril(np.ones((s, s)))
        # residual AR(1) coefficient (stable)
        self.Phi = np.array([[0.55, 0.10], [0.08, 0.45]], dtype=np.float64)
        if s != 2:
            self.Phi = 0.4 * np.eye(s)

    def sample(self, T):
        s = self.cfg.out_dim
        nc = self.cfg.noise
        z = self.rng.standard_normal(size=(T, s))
        if nc.kind == "studentt":
            g = self.rng.chisquare(nc.df, size=(T, 1)) / nc.df
            eps = (z / np.sqrt(g)) @ self.chol.T
        else:
            eps = z @ self.chol.T
        eta = np.zeros((T, s), dtype=np.float64)
        for t in range(T):
            cur = eps[t].copy()
            if nc.colored and t >= 1:
                cur = cur + self.Phi @ eta[t - 1]
            eta[t] = cur
        if nc.outlier_prob > 0:
            mask = self.rng.random(size=(T, s)) < nc.outlier_prob
            eta = eta + mask * nc.outlier_scale * self.chol[0, 0] * self.rng.standard_normal(size=(T, s))
        return eta


# ----------------------------------------------------------------------
# Static (memoryless) Hammerstein plant -- identification-friendly
# ----------------------------------------------------------------------
class StaticHammersteinPlant:
    def __init__(self, cfg: PlantConfig, stability_margin: float = 0.05):
        self.cfg = cfg
        rng = np.random.default_rng(2026)
        self._rng = rng
        # well-excited stable dynamics (companion spectral radius ~0.86)
        A = [
            np.array([[0.70, -0.12], [0.10, 0.62]]),
            np.array([[-0.18, 0.05], [0.05, -0.16]]),
        ]
        B = [
            np.array([[0.90, 0.15], [-0.10, 0.70]]),
            np.array([[0.25, -0.12], [0.10, 0.20]]),
        ]
        self.A = _stabilize(A, margin=stability_margin)
        self.B = B
        self.noise = _NoiseModel(cfg, rng)

    # memoryless coupled nonlinear map g(u_t): saturation, oscillatory, bilinear
    # and quadratic terms. Nonlinear enough that latent recovery is non-trivial,
    # while the strong stable dynamics keep A well excited and identifiable.
    def nonlinear_block(self, u_seq):
        u1, u2 = u_seq[:, 0], u_seq[:, 1]
        n1 = 0.9 * np.tanh(1.6 * u1) + 0.7 * np.sin(1.4 * u2) + 0.5 * u1 * u2
        n2 = 0.9 * np.tanh(1.5 * u2) + 0.6 * (u1 ** 2 - 1.0) + 0.35 * u1
        return np.stack([n1, n2], axis=1)

    def linear_dynamic_block(self, n_seq):
        T = n_seq.shape[0]
        s = self.cfg.out_dim
        x = np.zeros((T, s), dtype=np.float64)
        for t in range(T):
            cur = np.zeros(s)
            for i in range(self.cfg.na):
                if t - i - 1 >= 0:
                    cur -= self.A[i] @ x[t - i - 1]
            for j in range(self.cfg.nb):
                if t - j - 1 >= 0:
                    cur += self.B[j] @ n_seq[t - j - 1]
            x[t] = cur
        return x

    def simulate(self, u_seq):
        n = self.nonlinear_block(u_seq)
        x = self.linear_dynamic_block(n)
        eta = self.noise.sample(len(u_seq))
        return n, x, eta, x + eta


# ----------------------------------------------------------------------
# Recurrent (memoryful) Hammerstein plant -- prediction regime
# ----------------------------------------------------------------------
class RecurrentHammersteinPlant:
    def __init__(self, cfg: PlantConfig, stability_margin: float = 0.05):
        self.cfg = cfg
        p, m, h = cfg.in_dim, cfg.latent_dim, cfg.hidden_dim
        rng = np.random.default_rng(2026)
        self.Wu = rng.normal(0.0, 0.6, size=(h, p))
        self.Wh = rng.normal(0.0, 0.35, size=(h, h))
        self.bh = rng.normal(0.0, 0.15, size=(h,))
        self.Wn1 = rng.normal(0.0, 0.5, size=(h, h))
        self.bn1 = rng.normal(0.0, 0.1, size=(h,))
        self.Wn2 = rng.normal(0.0, 0.5, size=(m, h))
        self.bn2 = rng.normal(0.0, 0.1, size=(m,))
        A = [
            np.array([[0.70, -0.12], [0.10, 0.62]]),
            np.array([[-0.18, 0.05], [0.05, -0.16]]),
        ]
        B = [
            np.array([[0.80, 0.12], [-0.05, 0.55]]),
            np.array([[0.20, -0.10], [0.08, 0.15]]),
        ]
        self.A = _stabilize(A, margin=stability_margin)
        self.B = B
        self.noise = _NoiseModel(cfg, rng)

    def nonlinear_block(self, u_seq):
        T, p = u_seq.shape
        h = self.cfg.hidden_dim
        m = self.cfg.latent_dim
        h_t = np.zeros(h)
        n_seq = np.zeros((T, m))
        for t in range(T):
            u_t = u_seq[t]
            z = np.tanh(self.Wu @ u_t + self.Wh @ h_t + self.bh)
            h_t = 0.70 * h_t + 0.30 * z
            z2 = np.tanh(self.Wn1 @ h_t + self.bn1)
            coupled = np.zeros(m)
            if m >= 2 and p >= 2:
                coupled[0] = 0.18 * np.sin(u_t[0]) * u_t[1]
                coupled[1] = 0.16 * np.cos(u_t[1]) * u_t[0]
            n_seq[t] = self.Wn2 @ z2 + self.bn2 + coupled
        return n_seq

    linear_dynamic_block = StaticHammersteinPlant.linear_dynamic_block
    simulate = StaticHammersteinPlant.simulate


def make_plant(kind: str, cfg: PlantConfig, stability_margin: float = 0.05):
    if kind == "static":
        return StaticHammersteinPlant(cfg, stability_margin)
    if kind == "recurrent":
        return RecurrentHammersteinPlant(cfg, stability_margin)
    raise ValueError(f"unknown plant kind: {kind}")


# ----------------------------------------------------------------------
# Excitation signals
# ----------------------------------------------------------------------
def generate_step_sequence(T, in_dim=2, amp_low=-1.5, amp_high=1.5, min_hold=8, max_hold=20, rng=None):
    rng = rng or np.random
    u = np.zeros((T, in_dim))
    t = 0
    while t < T:
        hold = rng.randint(min_hold, max_hold + 1) if hasattr(rng, "randint") else int(rng.integers(min_hold, max_hold + 1))
        level = (rng.uniform(amp_low, amp_high, size=(in_dim,)))
        u[t:min(T, t + hold)] = level
        t += hold
    return u


def generate_random_sequence(T, in_dim=2, smooth=True, rng=None):
    rng = rng or np.random
    u = rng.uniform(-1.6, 1.6, size=(T, in_dim))
    if smooth:
        for k in range(in_dim):
            for t in range(1, T):
                u[t, k] = 0.82 * u[t - 1, k] + 0.18 * u[t, k]
    return u


def generate_prbs_sequence(T, in_dim=2, amp=1.2, min_hold=4, max_hold=12, rng=None):
    rng = rng or np.random
    u = np.zeros((T, in_dim))
    for k in range(in_dim):
        t = 0
        sign = 1.0
        while t < T:
            hold = int(rng.integers(min_hold, max_hold + 1)) if hasattr(rng, "integers") else rng.randint(min_hold, max_hold + 1)
            u[t:min(T, t + hold), k] = sign * amp
            sign *= -1.0
            t += hold
    return u


def generate_multisine_sequence(T, in_dim=2, n_tones=5, rng=None):
    rng = rng or np.random
    t = np.arange(T)
    u = np.zeros((T, in_dim))
    for k in range(in_dim):
        freqs = rng.uniform(0.02, 0.25, size=n_tones)
        phases = rng.uniform(0, 2 * np.pi, size=n_tones)
        amps = rng.uniform(0.4, 1.0, size=n_tones)
        sig = sum(a * np.sin(2 * np.pi * f * t + p) for a, f, p in zip(amps, freqs, phases))
        u[:, k] = sig / (np.max(np.abs(sig)) + 1e-8) * 1.4
    return u


_GENERATORS = {
    "step": generate_step_sequence,
    "random": generate_random_sequence,
    "prbs": generate_prbs_sequence,
    "multisine": generate_multisine_sequence,
}


def make_dataset(plant, n_sequences, T, mode="random", seed=None):
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    gen = _GENERATORS[mode]
    U, Y, X, N = [], [], [], []
    for _ in range(n_sequences):
        u = gen(T, in_dim=plant.cfg.in_dim, rng=rng)
        n_seq, x_seq, _eta, y_seq = plant.simulate(u)
        U.append(u); Y.append(y_seq); X.append(x_seq); N.append(n_seq)
    return (np.stack(U), np.stack(Y), np.stack(X), np.stack(N))


# ----------------------------------------------------------------------
# Scaling
# ----------------------------------------------------------------------
class StandardScalerSeq:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, arr):
        self.mean = arr.mean(axis=(0, 1), keepdims=True)
        self.std = arr.std(axis=(0, 1), keepdims=True) + 1e-6
        return self

    def transform(self, arr):
        return (arr - self.mean) / self.std

    def inverse_transform(self, arr):
        return arr * self.std + self.mean
