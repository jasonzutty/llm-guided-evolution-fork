# --PROMPT LOG--

import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

# --OPTION--

# -- NOTE --
# Note: The class GenePolicy inherits from ActorCriticPolicy (Stable-Baselines3).
# It must always accept *args and **kwargs and pass them to super().__init__().
# get_policy_kwargs() must return a dict with "policy_class" key.
# get_ppo_kwargs() must return a dict of valid PPO hyperparameters.
# Do not override forward(), _predict(), or evaluate_actions(); let SB3 handle those.
# Prefer mutating HIDDEN_PI, HIDDEN_VF, ACTIVATION, and PPO kwargs.
# -- NOTE --

# ===============================
# === Architecture Gene Space ===
# ===============================

HIDDEN_PI = [64, 64]
HIDDEN_VF = [64, 64]
ACTIVATION = nn.Tanh   # Tanh is strong for MuJoCo


class GenePolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            activation_fn=ACTIVATION,
            net_arch=dict(
                pi=HIDDEN_PI,
                vf=HIDDEN_VF
            ),
        )


def get_policy_kwargs():
    return {
        "policy_class": GenePolicy
    }

# --OPTION--

# ===============================
# === PPO Hyperparameter Gene ===
# ===============================

def get_ppo_kwargs():
    return dict(
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        normalize_advantage=True,
        verbose=0,
    )
