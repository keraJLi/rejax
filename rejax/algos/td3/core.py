import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import struct

from rejax.algos.base_config import BaseConfig
from rejax.algos.mixins import ReplayBufferMixin
from rejax.algos.networks import DeterministicPolicy, QNetwork
from rejax.algos.td3.td3 import TD3
from rejax.evaluate import make_evaluate
from rejax.normalize import FloatObsWrapper


class TD3Config(ReplayBufferMixin, BaseConfig):
    # TD3-specific parameters
    actor: nn.Module = struct.field(pytree_node=False, default=None)
    critic: nn.Module = struct.field(pytree_node=False, default=None)
    tau: chex.Scalar = struct.field(pytree_node=True, default=0.95)
    exploration_noise: chex.Scalar = struct.field(pytree_node=True, default=0.3)
    target_noise: chex.Scalar = struct.field(pytree_node=True, default=0.2)
    target_noise_clip: chex.Scalar = struct.field(pytree_node=True, default=0.5)
    policy_delay: int = struct.field(pytree_node=False, default=2)
    num_envs: int = struct.field(pytree_node=False, default=1)
    gradient_steps: int = struct.field(pytree_node=False, default=1)
    normalize_observations: bool = struct.field(pytree_node=False, default=False)

    @property
    def agent(self):
        """So that config.agent.apply exists"""
        return self.actor

    @property
    def action_low(self):
        return self.env.action_space(self.env_params).low

    @property
    def action_high(self):
        return self.env.action_space(self.env_params).high

    @staticmethod
    def initialize_agent(config, env, env_params):
        actor_kwargs = config.pop("actor_kwargs", {})
        activation = actor_kwargs.pop("activation", "swish")
        actor_kwargs["activation"] = getattr(nn, activation)
        action_range = (
            env.action_space(env_params).low,
            env.action_space(env_params).high,
        )
        action_dim = np.prod(env.action_space(env_params).shape)
        actor = DeterministicPolicy(
            action_dim, action_range, hidden_layer_sizes=(64, 64), **actor_kwargs
        )

        critic_kwargs = config.pop("critic_kwargs", {})
        activation = critic_kwargs.pop("activation", "swish")
        critic_kwargs["activation"] = getattr(nn, activation)
        critic = QNetwork(hidden_layer_sizes=(64, 64), **critic_kwargs)

        return {"actor": actor, "critic": critic}

    @staticmethod
    def initialize_eval_callback(config, env, env_params):
        return make_evaluate(TD3.make_act, env, env_params, 200)

    @classmethod
    def create(cls, **kwargs):
        config = super().create(**kwargs)
        if config.normalize_observations:
            config = config.replace(env=FloatObsWrapper(config.env))
        return config
