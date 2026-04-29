# training/train_ppo.py
"""
Parameter sharing: both archers use the same policy. RLlib does this
via a single policy in policies={} mapped to by all agent_ids.
Theory anchor: Albrecht, Christianos & Schäfer (2024), Ch. 9.
"""
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from env.kaz_wrapper import make_kaz_env
from models.ppo_model import PPOAtariModel


ModelCatalog.register_custom_model("ppo_atari", PPOAtariModel)


def env_creator(cfg):
    return ParallelPettingZooEnv(make_kaz_env(
        distortion_level=cfg.get("distortion_level", 0),
        frame_stack=4,
    ))


tune.register_env("kaz", env_creator)


config = (
    PPOConfig()
    .environment("kaz", env_config={"distortion_level": 0})
    .framework("torch")
    .training(
        model={"custom_model": "ppo_atari"},
        train_batch_size=4000,
        sgd_minibatch_size=256,
        num_sgd_iter=10,
        lr=2.5e-4,
        gamma=0.99,
        lambda_=0.95,           # GAE
        clip_param=0.2,         # PPO-clip
        entropy_coeff=0.01,     # exploration
        vf_loss_coeff=0.5,
    )
    .multi_agent(
        # PARAMETER SHARING: one policy used by all archers
        policies={"shared_policy"},
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
    )
    .resources(num_gpus=1)
    .rollouts(num_rollout_workers=4, rollout_fragment_length=200)
)


def main():
    ray.init()
    tune.run(
        "PPO",
        config=config.to_dict(),
        stop={"timesteps_total": 5_000_000},
        checkpoint_freq=10,
        checkpoint_at_end=True,
        local_dir="./checkpoints",
    )


if __name__ == "__main__":
    main()