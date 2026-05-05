# Task_3/models/ppo_module.py
"""
Two classes for the archer policy:

  ArcherPolicy   – pure nn.Module, no RLlib dependency.
                   Used for inference in submission.py and on dept computers.
                   Saved/loaded with torch.save/load.

  KAZPPOModule   – TorchRLModule wrapping ArcherPolicy for RLlib PPO training.
                   Identical trunk + pi_head weights; adds a value head for PPO.
                   After training, call export_policy() to get an ArcherPolicy.

Input:  40-dim vector — 8 zombie slots × [present, cx, cy, w, h] in [0,1]
Hidden: 64 → 64, ReLU
Output: action logits (pi_head) and V(s) (vf_head, training only)

Theory: Schulman et al. 2017 (PPO), Sutton & Barto Ch.13.5 (actor-critic).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.apis import ValueFunctionAPI

OBS_DIM = 40   # 8 zombies × 5 features
N_HIDDEN = 64


# ── standalone policy (no RLlib) ──────────────────────────────────────────────

class ArcherPolicy(nn.Module):
    """Pure PyTorch policy for inference. No RLlib required."""

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = 6):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, N_HIDDEN), nn.ReLU(),
            nn.Linear(N_HIDDEN, N_HIDDEN), nn.ReLU(),
        )
        self.pi_head = nn.Linear(N_HIDDEN, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.pi_head(self.trunk(obs))

    @torch.no_grad()
    def act(self, obs_np) -> int:
        import numpy as np
        obs_t = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).unsqueeze(0)
        return int(self(obs_t).argmax(1).item())


# ── RLlib module (training only) ──────────────────────────────────────────────

class KAZPPOModule(TorchRLModule, ValueFunctionAPI):
    """
    RLlib TorchRLModule used during PPO training.
    Shares trunk + pi_head architecture with ArcherPolicy so weights
    can be transferred after training via export_policy().
    """

    def setup(self):
        obs_dim = self.observation_space.shape[0]
        n_actions = self.action_space.n

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, N_HIDDEN), nn.ReLU(),
            nn.Linear(N_HIDDEN, N_HIDDEN), nn.ReLU(),
        )
        self.pi_head = nn.Linear(N_HIDDEN, n_actions)
        self.vf_head = nn.Linear(N_HIDDEN, 1)

    def _encode(self, batch: dict) -> torch.Tensor:
        return self.trunk(batch["obs"].float())

    def _forward(self, batch: dict, **_) -> dict:
        features = self._encode(batch)
        return {"action_dist_inputs": self.pi_head(features), "embeddings": features}

    def _forward_inference(self, batch, **kw):  return self._forward(batch)
    def _forward_exploration(self, batch, **kw): return self._forward(batch)
    def _forward_train(self, batch, **kw):       return self._forward(batch)

    def compute_values(self, batch: dict, embeddings=None) -> torch.Tensor:
        if embeddings is None:
            embeddings = self._encode(batch)
        return self.vf_head(embeddings).squeeze(-1)

    def export_policy(self) -> ArcherPolicy:
        """Copy trunk + pi_head into a standalone ArcherPolicy."""
        obs_dim = self.observation_space.shape[0]
        n_actions = self.action_space.n
        policy = ArcherPolicy(obs_dim=obs_dim, n_actions=n_actions)
        policy.trunk.load_state_dict(self.trunk.state_dict())
        policy.pi_head.load_state_dict(self.pi_head.state_dict())
        return policy
