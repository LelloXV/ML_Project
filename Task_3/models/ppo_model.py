# models/ppo_model.py
"""
Custom RLlib model. Single shared trunk → policy head + value head.
This is standard actor-critic with a shared backbone (Sutton & Barto
Ch. 13.5; Szepesvari §4.4 on actor-critic).
"""
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from .cnn_encoder import CNNEncoder


class PPOAtariModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs,
                 model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        in_ch = obs_space.shape[-1]   # frame stack channels (e.g. 4)
        self.encoder = CNNEncoder(in_channels=in_ch, out_dim=512)
        self.policy_head = nn.Linear(512, num_outputs)
        self.value_head = nn.Linear(512, 1)
        self._features = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # RLlib gives us NHWC; PyTorch wants NCHW
        x = input_dict["obs"].permute(0, 3, 1, 2).float()
        self._features = self.encoder(x)
        return self.policy_head(self._features), state

    @override(TorchModelV2)
    def value_function(self):
        return self.value_head(self._features).squeeze(-1)