"""
PettingZoo KAZ → RLlib MultiAgentEnv wrapper.

Theory: this is just plumbing. The MARL setting is captured by RLlib's
MultiAgentEnv abstraction (reference [6] Albrecht et al., Ch. 9 on CTDE
and parameter sharing).
"""
from pettingzoo.butterfly import knights_archers_zombies_v10
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import supersuit as ss
import numpy as np


def make_kaz_env(distortion_level: int = 0, num_archers: int = 2,
                 num_knights: int = 0, max_zombies: int = 10,
                 max_cycles: int = 900, frame_stack: int = 4,
                 seed: int | None = None):
    """Create a PettingZoo KAZ env, apply preprocessing, return it."""
    env = knights_archers_zombies_v10.parallel_env(
        spawn_rate=20,
        num_archers=num_archers,
        num_knights=num_knights,
        max_zombies=max_zombies,
        max_arrows=10,
        killable_knights=False,
        killable_archers=True,
        pad_observation=True,
        line_death=True,
        max_cycles=max_cycles,
        vector_state=False,            # PIXEL observations (required)
        use_typemasks=False,
        sequence_space=False,
        render_mode=None,
    )

    # IMPORTANT: apply distortions BEFORE preprocessing so we train on
    # observations that match what the tournament will throw at us.
    env = apply_distortions(env, level=distortion_level)

    # Standard preprocessing pipeline (DQN-style)
    env = ss.color_reduction_v0(env, mode="B")          # grayscale
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, frame_stack)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    return env


def apply_distortions(env, level: int):
    """
    Levels: 0 none, 1 stars, 2 clouds, 3 colour shift,
            4 zombie pixel distortion, 5 waves over screen.
    The actual distortion code comes from the KU Leuven template;
    plug it in here as a SuperSuit observation wrapper.
    """
    if level == 0:
        return env
    # from template.distortions import StarOverlay, CloudOverlay, ...
    # env = ObservationWrapper(env, fn=...)
    return env  # TODO: wire to template's distortion classes