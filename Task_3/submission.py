# Task_3/submission.py
"""
Tournament submission for KAZ.

Architecture: privileged-learning + bridge (Chen et al. 2020)
  Pixel frame → frozen Faster R-CNN detector → 40-dim zombie vector → MLP policy → action

Required interface elements:
  ENV_SETTINGS                  – dict consumed by create_environment() in utils.py
  CustomWrapper                 – env wrapper; produces 40-dim observations
  CustomPredictFunction         – callable; returns action given (obs, agent)
  CustomZombieDetectorFunction  – callable; returns Nx4 boxes given a raw image
"""
import math
import os
import sys
from typing import Callable

import numpy as np
import torch
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType

# Ensure models/ is importable when this file is the entry point.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


# ── env settings ─────────────────────────────────────────────────────────────

ENV_SETTINGS = {
    "distortion_level": 0,
}


# ── constants ─────────────────────────────────────────────────────────────────

_MAX_ZOMBIES = 8
_OBS_DIM     = _MAX_ZOMBIES * 5   # 40: 8 slots × [present, cx, cy, w, h]
_ANGLE_DIM   = 2                  # sin(angle), cos(angle) of archer facing direction
_ARCHER_DIM  = 2                  # archer position (cx/W, cy/H) — needed to compute relative distances to zombies
_THRESHOLD   = 0.3                # detection score cutoff (low → high recall)

_IMG_W = 1280.0
_IMG_H = 720.0


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_detector(device: str):
    from models.detector import build_detector
    path = os.path.join(_HERE, "checkpoints", "detector_best.pt")
    model = build_detector(frozen_backbone=True)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval().to(device)
    return model


def _boxes_to_vector(boxes_ltwh: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """Convert Nx4 [l,t,w,h] detector output to 40-dim feature vector."""
    vec = np.zeros((_MAX_ZOMBIES, 5), dtype=np.float32)
    if len(boxes_ltwh):
        cx = (boxes_ltwh[:, 0] + boxes_ltwh[:, 2] / 2.0) / img_w
        cy = (boxes_ltwh[:, 1] + boxes_ltwh[:, 3] / 2.0) / img_h
        bw =  boxes_ltwh[:, 2] / img_w
        bh =  boxes_ltwh[:, 3] / img_h
        order = np.argsort(-cy)           # lowest on screen (nearest archers) first
        for i, idx in enumerate(order[:_MAX_ZOMBIES]):
            vec[i] = [1.0, cx[idx], cy[idx], bw[idx], bh[idx]]
    return vec.flatten()


# ── CustomWrapper ─────────────────────────────────────────────────────────────

class CustomWrapper(BaseWrapper):
    """
    Replaces the raw pixel observation with a 40-dim zombie feature vector.

    Two modes:
      use_ground_truth=True  — training only: reads zombie rects directly from
                               the raw KAZ game state via env.unwrapped.zombie_list.
                               ~1000x faster than running the detector; same 40-dim
                               output format; zero detection error during training.
      use_ground_truth=False — submission / evaluation: runs the frozen Faster R-CNN
                               detector on the pixel frame produced by VisualWrapper.

    Both archers share the same full-frame view and therefore the same
    feature vector — appropriate because they use a shared policy.
    """

    def __init__(self, env, detector_path: str | None = None,
                 threshold: float = _THRESHOLD, device: str | None = None,
                 target_size=None, frame_stack=None,
                 use_ground_truth: bool = False):
        super().__init__(env)
        self._threshold        = threshold
        self._device           = device or _get_device()
        self._use_ground_truth = use_ground_truth

        if not use_ground_truth:
            det_path = detector_path or os.path.join(_HERE, "checkpoints", "detector_best.pt")
            from models.detector import build_detector
            self._detector = build_detector(frozen_backbone=True)
            self._detector.load_state_dict(
                torch.load(det_path, map_location=self._device, weights_only=True)
            )
            self._detector.eval().to(self._device)
        else:
            self._detector = None

        # Per-step cache: aec_to_parallel collects both archers' observations
        # after a single joint step — both archers see the same frame/game state.
        # Computing once and reusing halves the work.
        self._det_step: int = -1
        self._det_vec: np.ndarray = np.zeros(_OBS_DIM, dtype=np.float32)
        self._n_steps: int = 0

    # -- AEC state sync overrides ------------------------------------------
    # VisualWrapper.reset() and .step() both override BaseWrapper without
    # calling super(), so _cumulative_rewards is never set on VisualWrapper's
    # instance dict.  BaseWrapper.__getattr__ blocks private attrs, so
    # self.env._cumulative_rewards raises AttributeError.  We bypass this by
    # fetching _cumulative_rewards from the raw KAZ env (self.env.env), which
    # is NOT a BaseWrapper and sets the attribute directly during reset/step.

    def _fetch_cumul(self) -> dict:
        # _accumulate_rewards is a METHOD on AECEnv (not a dict), so we must
        # check isinstance(val, dict) to avoid calling dict() on a bound method.
        for env in (self.env, getattr(self.env, 'env', None)):
            if env is None:
                continue
            val = getattr(env, '_cumulative_rewards', None)
            if isinstance(val, dict):
                return dict(val)
        return {a: 0.0 for a in self.agents}

    def _sync_state(self) -> None:
        self.agents             = self.env.agents
        self.rewards            = self.env.rewards
        try:
            self._cumulative_rewards = self.env._cumulative_rewards
        except AttributeError:
            self._cumulative_rewards = self._fetch_cumul()
        self.terminations       = self.env.terminations
        self.truncations        = self.env.truncations
        self.infos              = self.env.infos
        self.agent_selection    = self.env.agent_selection

    def reset(self, seed=None, options=None):
        self._n_steps = 0
        self._det_step = -1
        result = self.env.reset(seed=seed, options=options)
        self._sync_state()
        return result

    def step(self, action) -> None:
        if (self.terminations.get(self.agent_selection, False)
                or self.truncations.get(self.agent_selection, False)):
            # Propagate the dead-step to the inner env so raw_kaz removes the
            # dead agent from its own agents/terminations.  BaseWrapper's
            # _was_dead_step does NOT delegate to self.env, so a later
            # _sync_state() call would restore the dead archer from raw_kaz,
            # resurrecting it and causing aec_to_parallel to KeyError.
            self.env.step(None)
            self._sync_state()
            return
        self.env.step(action)
        self._n_steps += 1
        self._sync_state()

    # -- observation -------------------------------------------------------

    def _archer_features(self, agent: AgentID) -> np.ndarray:
        """Return [sin(angle), cos(angle), cx/W, cy/H] for the given archer."""
        angle_deg, cx, cy = 0.0, 0.5, 1.0
        try:
            for a in self.env.unwrapped.archer_list:
                if a.agent_name == agent:
                    angle_deg = float(a.angle)
                    cx = a.rect.centerx / _IMG_W
                    cy = a.rect.centery / _IMG_H
                    break
        except Exception:
            pass
        angle_rad = math.radians(angle_deg)
        return np.array(
            [math.sin(angle_rad), math.cos(angle_rad), cx, cy],
            dtype=np.float32,
        )

    def observation_space(self, agent: AgentID) -> spaces.Box:
        return spaces.Box(low=-1.0, high=1.0, shape=(_OBS_DIM + _ANGLE_DIM + _ARCHER_DIM,), dtype=np.float32)

    def observe(self, agent: AgentID) -> ObsType | None:
        # Cache hit: zombie vec is shared; archer state is per-agent so always recomputed.
        if self._n_steps == self._det_step:
            return np.concatenate([self._det_vec, self._archer_features(agent)])

        if self._use_ground_truth:
            # Training fast path: read zombie positions directly from game state.
            # Bypasses rendering and detector inference entirely.
            try:
                raw = self.env.unwrapped          # KnightsArchersZombiesEnv
                zombies = list(raw.zombie_list)
                if zombies:
                    boxes = np.array(
                        [[z.rect.x, z.rect.y, z.rect.width, z.rect.height]
                         for z in zombies],
                        dtype=np.float32,
                    )
                else:
                    boxes = np.zeros((0, 4), dtype=np.float32)
            except Exception:
                boxes = np.zeros((0, 4), dtype=np.float32)
            vec = _boxes_to_vector(boxes, 720, 1280)
        else:
            # Submission path: run detector on the VisualWrapper pixel frame.
            frame = super().observe(agent)     # (H, W, 3) uint8 from VisualWrapper
            if frame is None or not np.asarray(frame).any():
                return np.zeros(_OBS_DIM + _ANGLE_DIM + _ARCHER_DIM, dtype=np.float32)

            frame = np.asarray(frame, dtype=np.uint8)
            h, w  = frame.shape[:2]

            from models.detector import predict
            boxes = predict(self._detector, frame, self._threshold, self._device)
            vec = _boxes_to_vector(boxes, h, w)

        self._det_step = self._n_steps
        self._det_vec  = vec
        return np.concatenate([self._det_vec, self._archer_features(agent)])


# ── CustomPredictFunction ─────────────────────────────────────────────────────

# RL action semantics (raw_env.step adds +1, so RL N → game N+1):
#   RL 0 → game 1 → move forward in facing direction
#   RL 1 → game 2 → move backward
#   RL 2 → game 3 → rotate CCW: angle += ang_rate → faces more LEFT on screen
#   RL 3 → game 4 → rotate CW:  angle -= ang_rate → faces more RIGHT on screen
#   RL 4 → game 5 → fire arrow along current heading
#   RL 5 → game 6 → noop

_ACT_FWD   = 0
_ACT_ROT_L = 2
_ACT_ROT_R = 3
_ACT_FIRE  = 4
_ACT_NOOP  = 5

_ANG_TOL   = 10.0   # degrees: fire when within this of the target angle
_ANG_RATE  = 10.0   # degrees per rotation step (PLAYER_ANG_RATE)


class CustomPredictFunction(Callable):
    """
    Primary path: load ArcherPolicy from checkpoints/policy_best.pt and use it
    for greedy action selection.

    Fallback (no checkpoint): aim-and-shoot heuristic.
    - Reads the archer's live angle from env.unwrapped.archer_list.
    - Rotates to face the nearest zombie in the archer's assigned screen zone
      (archer_0 → left half, archer_1 → right half), then fires when aligned.
    - Falls back to any visible zombie when the assigned zone is empty.
    - Returns to facing up (angle→0) when no zombies are visible.
    """

    def __init__(self, env):
        self._env = env
        self._policy = None
        policy_path = os.path.join(_HERE, "checkpoints", "policy_best.pt")
        if os.path.exists(policy_path):
            try:
                from models.ppo_module import ArcherPolicy
                pol = ArcherPolicy()
                pol.load_state_dict(
                    torch.load(policy_path, map_location="cpu", weights_only=True)
                )
                pol.eval()
                # Sanity check: a well-trained policy should strongly prefer
                # FIRE (action 4) when a zombie is directly above the archer.
                # If the logit spread is < 0.5 the weights are near-initialisation.
                with torch.no_grad():
                    probe = torch.zeros(1, _OBS_DIM + _ANGLE_DIM + _ARCHER_DIM)
                    probe[0, 0] = 1.0   # zombie present
                    probe[0, 1] = 0.5   # cx = center
                    probe[0, 2] = 0.3   # cy = high up
                    probe[0, _OBS_DIM]     = 0.0   # sin(0°) — facing straight up
                    probe[0, _OBS_DIM + 1] = 1.0   # cos(0°)
                    probe[0, _OBS_DIM + 2] = 0.5   # archer cx = center
                    probe[0, _OBS_DIM + 3] = 0.95  # archer cy = near bottom
                    logits = pol(probe)[0]
                    spread = float(logits.max() - logits.min())
                if spread >= 0.5:
                    self._policy = pol
                    print(f"[CustomPredictFunction] loaded policy (logit spread={spread:.2f})")
                else:
                    print(f"[CustomPredictFunction] policy looks undertrained "
                          f"(logit spread={spread:.2f} < 0.5); using heuristic")
            except Exception as exc:
                print(f"[CustomPredictFunction] policy load failed ({exc}); using heuristic")

    # -- heuristic helpers ---------------------------------------------------

    def _archer_state(self, agent: str):
        """Return (cx_px, cy_px, angle_deg) from the live raw env, or None."""
        try:
            for a in self._env.unwrapped.archer_list:
                if a.agent_name == agent:
                    return float(a.rect.centerx), float(a.rect.centery), float(a.angle)
        except Exception:
            pass
        return None

    def _heuristic(self, observation, agent: str) -> int:
        obs_2d = np.asarray(observation, dtype=np.float32)[:_OBS_DIM].reshape(_MAX_ZOMBIES, 5)
        state  = self._archer_state(agent)

        if state is None:
            # Can't read env state – just fire whenever zombies visible
            return _ACT_FIRE if bool(obs_2d[:, 0].any()) else _ACT_NOOP

        ax, ay, cur_angle = state
        visible = obs_2d[obs_2d[:, 0] > 0.5]

        if len(visible) == 0:
            # No zombies: return to facing up (angle → 0)
            if cur_angle > 5:
                return _ACT_ROT_R
            if cur_angle < -5:
                return _ACT_ROT_L
            return _ACT_NOOP

        # Zone split so the two archers cover different parts of the screen
        zone = (visible[visible[:, 1] < 0.5] if agent == "archer_0"
                else visible[visible[:, 1] >= 0.5])
        if len(zone) == 0:
            zone = visible   # fall back to any visible zombie

        # Nearest zombie in zone (Euclidean distance in pixel coords)
        zx_px = zone[:, 1] * _IMG_W
        zy_px = zone[:, 2] * _IMG_H
        dists = np.hypot(zx_px - ax, zy_px - ay)
        best  = int(np.argmin(dists))
        bx, by = float(zx_px[best]), float(zy_px[best])

        # Required facing angle: atan2(-dx, -dy) gives 0° for "directly above"
        dx = bx - ax
        dy = by - ay   # negative: zombie is above archer
        target_angle = math.degrees(math.atan2(-dx, -dy))

        # Shortest signed angular difference
        diff = (target_angle - cur_angle + 180.0) % 360.0 - 180.0

        if abs(diff) <= _ANG_TOL:
            return _ACT_FIRE
        return _ACT_ROT_L if diff > 0 else _ACT_ROT_R

    # -- public interface ----------------------------------------------------

    def __call__(self, observation, agent, *args, **kwargs):
        if self._policy is not None:
            return self._policy.act(observation)
        return self._heuristic(observation, agent)


# ── CustomZombieDetectorFunction ──────────────────────────────────────────────

class CustomZombieDetectorFunction(Callable):
    """
    Runs the frozen detector on a raw (H, W, 3) uint8 image.
    Returns (N, 4) float32 array of [left, top, width, height] boxes
    in pixel coordinates — the format expected by utils.iou().
    """

    def __init__(self, env):
        self._device   = _get_device()
        self._detector = _load_detector(self._device)
        self._threshold = _THRESHOLD

    def __call__(self, observation, *args, **kwargs):
        from models.detector import predict
        img = np.asarray(observation, dtype=np.uint8)
        if img.ndim == 1:          # flattened (H*W*3,) from np.load
            img = img.reshape(720, 1280, 3)
        return predict(self._detector, img, self._threshold, self._device)
