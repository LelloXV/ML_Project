# Task_3/training/train_ppo.py
"""
PPO training for KAZ with:
    - parameter sharing (one policy for both archers)
    - ground-truth zombie obs (no Faster R-CNN during training)
    - CPU-friendly defaults

Run modes:
    python -m training.train_ppo --smoke           # 3 iterations, verifies pipeline
    python -m training.train_ppo                   # full training run
    python -m training.train_ppo --iters 200       # custom length

Theory anchors:
    - PPO: Schulman et al. 2017, arXiv:1707.06347
    - Parameter sharing: Albrecht, Christianos, Schäfer 2024 (marl-book.com), Ch. 9
    - Generalized Advantage Estimation (lambda=0.95):
      Schulman et al. 2016, "High-Dimensional Continuous Control Using GAE",
      arXiv:1506.02438
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pettingzoo
import torch
from pettingzoo.butterfly import knights_archers_zombies_v10
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

# Make Task_3/ importable regardless of where we run from.
TASK3_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TASK3_DIR))

from submission import CustomWrapper           # noqa: E402
from models.ppo_module import KAZPPOModule    # noqa: E402


SHARED_POLICY = "shared_policy"
ENV_ID = "kaz_shared"


# ------------------------------------------------------------------ env factory

def _env_factory(env_config: dict | None = None):
    """Build the training env — no pixel rendering or detector inference needed.

    CustomWrapper(use_ground_truth=True) reads zombie rects directly from the
    game's sprite list. This is ~1000x faster than running FasterRCNN per step,
    produces the exact same 40-dim obs format as at submission time, and avoids
    the previous bug where train_batch_size (in agent transitions) was smaller
    than 2 * max_cycles, causing n_ep=0 and NaN losses.
    """
    # Kaggle / any headless server: Pygame needs a dummy video driver.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    # render_mode=None: no pixel frames needed (ground-truth obs bypass rendering).
    raw_kaz = knights_archers_zombies_v10.raw_env(
        max_cycles=900, num_archers=2, num_knights=0,
        max_zombies=4, vector_state=False, render_mode=None,
    )
    env = CustomWrapper(raw_kaz, use_ground_truth=True)
    # RLlib's parallel API expects a parallel env — convert AEC→parallel.
    return ParallelPettingZooEnv(pettingzoo.utils.conversions.aec_to_parallel(env))


# ------------------------------------------------------------------ PPO config

def build_config(num_env_runners: int = 2,
                 train_batch_size: int = 8000,
                 minibatch_size: int = 512,
                 num_epochs: int = 6,
                 lr: float = 3e-4) -> PPOConfig:
    """
    PPO hyperparameters for ground-truth-obs training (no detector in env runners).

    With max_cycles=900 parallel steps per episode and 2 agents:
      train_batch_size=8000 agent transitions = 4000 parallel steps → ~4 episodes
      per iteration → n_ep > 0 from iteration 0.

    Steps per iteration (est.): 8000 × ~0.5ms (raw KAZ step) ≈ 4 seconds.
    Kaggle: 200 iterations ≈ 13 minutes with 2 runners.
    """
    return (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(env=ENV_ID, disable_env_checking=True)
        .env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=1,
            rollout_fragment_length="auto",
            num_gpus_per_env_runner=0,  # No FasterRCNN — env runners are CPU-only
        )
        .framework("torch")
        .learners(
            num_learners=0,          # learner in driver process
            num_gpus_per_learner=0,  # MLP is tiny; CPU gradient updates are fine
        )
        .multi_agent(
            policies={SHARED_POLICY},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: SHARED_POLICY,
            policies_to_train=[SHARED_POLICY],
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    SHARED_POLICY: RLModuleSpec(module_class=KAZPPOModule),
                },
            ),
        )
        .training(
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            num_epochs=num_epochs,
            lr=lr,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            grad_clip=0.5,
            grad_clip_by="global_norm",
        )
        .debugging(log_level="ERROR")
    )


# ------------------------------------------------------------------ training

def train(iters: int = 200,
          checkpoint_dir: Path = TASK3_DIR / "results",
          checkpoint_every: int = 10,
          early_stop_return: float = 5.0,
          smoke: bool = False,
          num_env_runners: int | None = None) -> Path | None:

    # Force CPU on Apple Silicon — RLlib + MPS is unstable.
    os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
    if torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    np.random.seed(42)
    torch.manual_seed(42)

    register_env(ENV_ID, _env_factory)

    if smoke:
        cfg = build_config(num_env_runners=0,
                           train_batch_size=512,
                           minibatch_size=128,
                           num_epochs=2)
        iters = 3
        checkpoint_every = 999
    else:
        runners = num_env_runners if num_env_runners is not None else 2
        cfg = build_config(num_env_runners=runners)

    print("Building algorithm...", flush=True)
    algo = cfg.build()
    print("Algorithm built. Starting training.\n", flush=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint: Path | None = None

    for i in range(iters):
        result = algo.train()

        # Pull the headline metric: mean episode return for our shared policy.
        runners = result.get("env_runners", {})
        agent_returns = runners.get("agent_episode_returns_mean", {})
        per_agent = {a: round(r, 3) for a, r in agent_returns.items()}
        if agent_returns:
            mean_return = float(np.mean(list(agent_returns.values())))
        else:
            mean_return = runners.get("episode_return_mean", float("nan"))

        ep_len = runners.get("episode_len_mean", float("nan"))
        n_ep   = runners.get("num_episodes", 0)

        # Loss metrics.
        learners = result.get("learners", {})
        pol_stats = learners.get(SHARED_POLICY, {})
        if not pol_stats:
            for v in learners.values():
                if isinstance(v, dict) and SHARED_POLICY in v:
                    pol_stats = v[SHARED_POLICY]
                    break
        p_loss = pol_stats.get("policy_loss", float("nan"))
        v_loss = pol_stats.get("vf_loss", float("nan"))
        ent    = pol_stats.get("entropy", float("nan"))

        print(f"iter {i:3d} | return={mean_return:7.3f} | ep_len={ep_len:7.1f} | n_ep={n_ep} "
              f"| p_loss={p_loss:.4f} | v_loss={v_loss:.4f} | ent={ent:.3f} | per_agent={per_agent}",
              flush=True)

        # Checkpoint
        if (i + 1) % checkpoint_every == 0:
            save_result = algo.save(str(checkpoint_dir))
            last_checkpoint = Path(save_result.checkpoint.path)
            print(f"           checkpoint -> {last_checkpoint}", flush=True)

        # Early stop
        if not np.isnan(mean_return) and mean_return >= early_stop_return:
            print(f"Early stop: mean_return {mean_return:.3f} >= {early_stop_return}", flush=True)
            break

    # Always save final checkpoint
    save_result = algo.save(str(checkpoint_dir))
    last_checkpoint = Path(save_result.checkpoint.path)
    print(f"\nFinal checkpoint: {last_checkpoint}", flush=True)

    # Export pure-PyTorch policy so submission.py can load it without RLlib.
    policy_path = checkpoint_dir / "policy_best.pt"
    module = algo.get_module(SHARED_POLICY)
    torch.save(module.export_policy().state_dict(), str(policy_path))
    print(f"Policy weights  : {policy_path}", flush=True)

    algo.stop()
    return last_checkpoint


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Run a 3-iteration smoke test (no checkpoints)")
    parser.add_argument("--iters", type=int, default=200,
                        help="Number of PPO iterations to train")
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="Save a checkpoint every N iterations")
    parser.add_argument("--early-stop", type=float, default=5.0,
                        help="Stop training when mean episode return reaches this")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Where to save checkpoints (default: <task3_dir>/results)")
    parser.add_argument("--num-env-runners", type=int, default=None,
                        help="Number of parallel env runners (default: 2)")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else TASK3_DIR / "results"
    train(iters=args.iters,
          checkpoint_dir=ckpt_dir,
          checkpoint_every=args.checkpoint_every,
          early_stop_return=args.early_stop,
          smoke=args.smoke,
          num_env_runners=args.num_env_runners)


if __name__ == "__main__":
    main()
