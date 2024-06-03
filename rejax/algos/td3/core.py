from copy import deepcopy
from typing import Any, Callable

import chex
import gymnax
import numpy as np
from flax import linen as nn
from flax import struct
from gymnax.environments.environment import Environment
from jax import numpy as jnp

from rejax.algos.networks import DeterministicPolicy, QNetwork
from rejax.algos.td3.td3 import TD3
from rejax.brax2gymnax import Brax2GymnaxEnv
from rejax.evaluate import make_evaluate


class TD3Config(struct.PyTreeNode):
    # fmt: off
    # Non-static parameters
    env: Environment                = struct.field(pytree_node=False)
    env_params: Any                 = struct.field(pytree_node=True)
    actor: nn.Module                = struct.field(pytree_node=False)
    critic: nn.Module               = struct.field(pytree_node=False)
    eval_callback: Callable         = struct.field(pytree_node=False)

    learning_rate: chex.Scalar      = struct.field(pytree_node=True, default=0.001)
    gamma: chex.Scalar              = struct.field(pytree_node=True, default=0.99)
    tau: chex.Scalar                = struct.field(pytree_node=True, default=0.95)
    exploration_noise: chex.Scalar  = struct.field(pytree_node=True, default=0.3)
    target_noise: chex.Scalar       = struct.field(pytree_node=True, default=0.2)
    target_noise_clip: chex.Scalar  = struct.field(pytree_node=True, default=0.5)
    max_grad_norm: chex.Scalar      = struct.field(pytree_node=True, default=jnp.inf)

    # Static parameters
    total_timesteps: int            = struct.field(pytree_node=False, default=10_000)
    policy_delay: int               = struct.field(pytree_node=False, default=2)
    eval_freq: int                  = struct.field(pytree_node=False, default=1_000)
    num_envs: int                   = struct.field(pytree_node=False, default=1)
    gradient_steps: int             = struct.field(pytree_node=False, default=1)
    buffer_size: int                = struct.field(pytree_node=False, default=10_000)
    fill_buffer: int                = struct.field(pytree_node=False, default=1_000)
    batch_size: int                 = struct.field(pytree_node=False, default=100)
    normalize_observations: bool    = struct.field(pytree_node=False, default=False)
    skip_initial_evaluation: bool   = struct.field(pytree_node=False, default=False)
    # fmt: on

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

    @classmethod
    def create(cls, **kwargs):
        """Create a config object from keyword arguments."""
        return cls.from_dict(kwargs)

    @classmethod
    def from_dict(cls, config):
        """Create a config object from a dictionary. Exists mainly for backwards
        compatibility and will be deprecated in the future."""

        config = deepcopy(config)  # Because we're popping from it

        if isinstance(config["env"], str):
            # Get env id and convert to gymnax environment and parameters
            env_kwargs = config.pop("env_kwargs", {})
            env_id = config.pop("env")
            if env_id.startswith("brax"):
                env = Brax2GymnaxEnv(env_id.split("/")[1], **env_kwargs)
                env_params = env.default_params
            else:
                env, env_params = gymnax.make(env_id, **env_kwargs)
        else:
            env = config.pop("env")
            env_params = config.pop("env_params", env.default_params)

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

        evaluate = make_evaluate(TD3.make_act, env, env_params, 200)
        return cls(
            env_params=env_params,
            actor=actor,
            critic=critic,
            env=env,
            eval_callback=evaluate,
            **config,
        )
