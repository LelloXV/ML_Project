"""
algorithms.py – Vectorised multi-agent Q-learning with EMA policy trajectories.

Instead of tracking time-averaged action counts, every algorithm records an
exponential moving average (EMA) of the *instantaneous policy* at each step:

  • Boltzmann / Lenient: instantaneous policy = softmax(Q / τ)
  • ε-greedy:           instantaneous policy = ε-greedy distribution over Q

Benefits vs. action-count averaging
──────────────────────────────────────
  1. Trajectory starts EXACTLY at the intended (x₀, y₀) initial policy.
  2. Endpoint reflects where the policy CONVERGED, not a washed-out average
     dragged toward the corner by thousands of post-convergence steps.
  3. EMA smoothing (time-constant ≈ 2000 steps) removes stochastic noise
     while keeping the transition region clearly visible.

All batch functions return  traj : (n_steps, n_starts, 2, n_actions)
  traj[t, s, 0, :]  EMA policy of player 1 at step t, start s
  traj[t, s, 1, :]  EMA policy of player 2 at step t, start s
"""
from __future__ import annotations
import numpy as np
from config import N_STEPS, ALPHA, TAU, KAPPA, EPSILON, SEED, EMA_DECAY


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _softmax(Q: np.ndarray, tau: float) -> np.ndarray:
    """Row-wise softmax.  Q: (n_starts, n_actions)"""
    Q = Q - Q.max(axis=1, keepdims=True)
    e = np.exp(Q / max(tau, 1e-10))
    return e / e.sum(axis=1, keepdims=True)


def _sample(pi: np.ndarray, rg: np.random.Generator) -> np.ndarray:
    """Sample one action per row via inverse-CDF (no Python loop)."""
    cdf = np.cumsum(pi, axis=1)
    u   = rg.random(pi.shape[0])[:, None]
    return (u > cdf).sum(axis=1).astype(np.intp).clip(0, pi.shape[1] - 1)


def _egreedy_policy(Q: np.ndarray, epsilon: float, na: int,
                    idx: np.ndarray) -> np.ndarray:
    """ε-greedy policy as probability vector (n_starts, na)."""
    p = np.full(Q.shape, epsilon / na)
    p[idx, Q.argmax(axis=1)] += 1.0 - epsilon
    return p


def _rng(seed) -> np.random.Generator:
    return np.random.default_rng(seed if seed is not None else SEED)


# ─────────────────────────────────────────────────────────────────────────────
# (a)  ε-greedy Q-learning
# ─────────────────────────────────────────────────────────────────────────────

def egreedy_batch(
    A, B, Q1_batch, Q2_batch,
    n_steps=N_STEPS, alpha=ALPHA, epsilon=EPSILON,
    seed=None, **_,
) -> np.ndarray:
    """
    Independent ε-greedy Q-learning.
    Trajectory = EMA of the ε-greedy policy distribution at each step.
    """
    rg = _rng(seed)
    n_s, na = Q1_batch.shape
    Q1 = Q1_batch.copy().astype(float)
    Q2 = Q2_batch.copy().astype(float)
    idx = np.arange(n_s)
    traj = np.empty((n_steps, n_s, 2, na))

    # Initialise EMA to the starting ε-greedy policy
    ema1 = _egreedy_policy(Q1, epsilon, na, idx)
    ema2 = _egreedy_policy(Q2, epsilon, na, idx)

    for t in range(n_steps):
        pi1 = _egreedy_policy(Q1, epsilon, na, idx)
        pi2 = _egreedy_policy(Q2, epsilon, na, idx)

        a1 = _sample(pi1, rg)
        a2 = _sample(pi2, rg)

        r1 = A[a1, a2]; r2 = B[a1, a2]

        Q1[idx, a1] += alpha * (r1 - Q1[idx, a1])
        Q2[idx, a2] += alpha * (r2 - Q2[idx, a2])

        # EMA update with CURRENT (pre-update) policy
        ema1 = EMA_DECAY * ema1 + (1 - EMA_DECAY) * pi1
        ema2 = EMA_DECAY * ema2 + (1 - EMA_DECAY) * pi2

        traj[t, :, 0, :] = ema1
        traj[t, :, 1, :] = ema2

    return traj


# ─────────────────────────────────────────────────────────────────────────────
# (b)  Boltzmann Q-learning
# ─────────────────────────────────────────────────────────────────────────────

def boltzmann_batch(
    A, B, Q1_batch, Q2_batch,
    n_steps=N_STEPS, alpha=ALPHA, tau=TAU,
    seed=None, **_,
) -> np.ndarray:
    """
    Boltzmann (softmax) Q-learning.
    Trajectory = EMA of softmax(Q / τ) at each step.
    """
    rg = _rng(seed)
    n_s, na = Q1_batch.shape
    Q1 = Q1_batch.copy().astype(float)
    Q2 = Q2_batch.copy().astype(float)
    idx = np.arange(n_s)
    traj = np.empty((n_steps, n_s, 2, na))

    ema1 = _softmax(Q1, tau)
    ema2 = _softmax(Q2, tau)

    for t in range(n_steps):
        pi1 = _softmax(Q1, tau)
        pi2 = _softmax(Q2, tau)

        a1 = _sample(pi1, rg)
        a2 = _sample(pi2, rg)

        r1 = A[a1, a2]; r2 = B[a1, a2]

        Q1[idx, a1] += alpha * (r1 - Q1[idx, a1])
        Q2[idx, a2] += alpha * (r2 - Q2[idx, a2])

        ema1 = EMA_DECAY * ema1 + (1 - EMA_DECAY) * pi1
        ema2 = EMA_DECAY * ema2 + (1 - EMA_DECAY) * pi2

        traj[t, :, 0, :] = ema1
        traj[t, :, 1, :] = ema2

    return traj


# ─────────────────────────────────────────────────────────────────────────────
# (c)  Lenient Boltzmann Q-learning
# ─────────────────────────────────────────────────────────────────────────────

def lenient_boltzmann_batch(
    A, B, Q1_batch, Q2_batch,
    n_steps=N_STEPS, alpha=ALPHA, tau=TAU, kappa=KAPPA,
    seed=None, **_,
) -> np.ndarray:
    """
    Lenient Boltzmann Q-learning (Bloembergen et al., JAIR 2015).
    Leniency probability: L(a) = exp(−κ · visits(a)).
    Bad news (r < Q) suppressed early; accepted as visits grow.
    Trajectory = EMA of softmax(Q / τ).
    """
    rg = _rng(seed)
    n_s, na = Q1_batch.shape
    Q1  = Q1_batch.copy().astype(float)
    Q2  = Q2_batch.copy().astype(float)
    vis1 = np.zeros_like(Q1)
    vis2 = np.zeros_like(Q2)
    idx  = np.arange(n_s)
    traj = np.empty((n_steps, n_s, 2, na))

    ema1 = _softmax(Q1, tau)
    ema2 = _softmax(Q2, tau)

    for t in range(n_steps):
        pi1 = _softmax(Q1, tau)
        pi2 = _softmax(Q2, tau)

        a1 = _sample(pi1, rg)
        a2 = _sample(pi2, rg)

        r1 = A[a1, a2]; r2 = B[a1, a2]

        vis1[idx, a1] += 1;  vis2[idx, a2] += 1
        len1 = np.exp(-kappa * vis1[idx, a1])
        len2 = np.exp(-kappa * vis2[idx, a2])

        upd1 = (r1 >= Q1[idx, a1]) | (rg.random(n_s) > len1)
        upd2 = (r2 >= Q2[idx, a2]) | (rg.random(n_s) > len2)

        Q1[idx, a1] += alpha * (r1 - Q1[idx, a1]) * upd1
        Q2[idx, a2] += alpha * (r2 - Q2[idx, a2]) * upd2

        ema1 = EMA_DECAY * ema1 + (1 - EMA_DECAY) * pi1
        ema2 = EMA_DECAY * ema2 + (1 - EMA_DECAY) * pi2

        traj[t, :, 0, :] = ema1
        traj[t, :, 1, :] = ema2

    return traj


# ─────────────────────────────────────────────────────────────────────────────
# Single-start wrappers
# ─────────────────────────────────────────────────────────────────────────────

def _single(batch_fn):
    def fn(A, B, n_steps=N_STEPS, alpha=ALPHA, tau=TAU,
           q1_init=None, q2_init=None, seed=None, **kw):
        na = A.shape[0]
        q1 = (np.zeros((1, na)) if q1_init is None else q1_init[None]).astype(float)
        q2 = (np.zeros((1, na)) if q2_init is None else q2_init[None]).astype(float)
        out = batch_fn(A, B, q1, q2, n_steps=n_steps, alpha=alpha, tau=tau,
                       seed=seed, **kw)
        return out[:, 0, :, :]
    return fn


egreedy_qlearning           = _single(egreedy_batch)
boltzmann_qlearning         = _single(boltzmann_batch)
lenient_boltzmann_qlearning = _single(lenient_boltzmann_batch)


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm registry
# ─────────────────────────────────────────────────────────────────────────────

ALGORITHMS = [
    {
        "name":     r"$\varepsilon$-greedy Q-learning",
        "slug":     "egreedy",
        "batch_fn": egreedy_batch,
        "kwargs":   {},
    },
    {
        "name":     "Boltzmann Q-learning",
        "slug":     "boltzmann",
        "batch_fn": boltzmann_batch,
        "kwargs":   {},
    },
    {
        "name":     "Lenient Boltzmann Q-learning",
        "slug":     "lenient",
        "batch_fn": lenient_boltzmann_batch,
        "kwargs":   {},
    },
]
