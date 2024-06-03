from copy import deepcopy
from typing import Any, Callable

import chex
import gymnax
import numpy as np
from flax import linen as nn
from flax import struct
from gymnax.environments.environment import Environment

from rejax.algos.networks import (
    DiscretePolicy,
    DiscreteQNetwork,
    QNetwork,
    SquashedGaussianPolicy,
)
from rejax.algos.sac.sac import SAC
from rejax.brax2gymnax import Brax2GymnaxEnv
from rejax.evaluate import make_evaluate


class SACConfig(struct.PyTreeNode):
    # fmt: off
    # Non-static parameters
    env: Environment                = struct.field(pytree_node=False)
    env_params: Any                 = struct.field(pytree_node=True)
    actor: nn.Module                = struct.field(pytree_node=False)
    critic: nn.Module               = struct.field(pytree_node=False)
    eval_callback: Callable         = struct.field(pytree_node=False)

    learning_rate: chex.Scalar      = struct.field(pytree_node=True, default=0.001)
    gamma: chex.Scalar              = struct.field(pytree_node=True, default=0.99)
    tau: float                      = struct.field(pytree_node=True, default=0.95)
    target_entropy_ratio: float     = struct.field(pytree_node=True, default=0.98)
    reward_scaling: chex.Scalar     = struct.field(pytree_node=True, default=1.0)

    # Static parameters
    total_timesteps: int            = struct.field(pytree_node=False, default=100_000)
    eval_freq: int                  = struct.field(pytree_node=False, default=10_000)
    num_envs: int                   = struct.field(pytree_node=False, default=1)
    buffer_size: int                = struct.field(pytree_node=False, default=100_000)
    fill_buffer: int                = struct.field(pytree_node=False, default=10_000)
    batch_size: int                 = struct.field(pytree_node=False, default=256)
    gradient_steps: int             = struct.field(pytree_node=False, default=1)
    normalize_observations: bool    = struct.field(pytree_node=False, default=False)
    skip_initial_evaluation: bool   = struct.field(pytree_node=False, default=False)
    # fmt: on

    @property
    def discrete(self):
        action_space = self.env.action_space(self.env_params)
        return isinstance(action_space, gymnax.environments.spaces.Discrete)

    @property
    def action_dim(self):
        action_space = self.env.action_space(self.env_params)
        if self.discrete:
            return action_space.n
        return np.prod(action_space.shape)

    @property
    def target_entropy(self):
        if self.discrete:
            return -self.target_entropy_ratio * np.log(1 / self.action_dim)
        return -self.action_dim

    @property
    def agent(self):
        """Backward compatibility with old evaluation creation"""
        return self.actor

    @classmethod
    def create(cls, **kwargs):
        """Create a config object from keyword arguments."""
        return cls.from_dict(kwargs)

    @classmethod
    def from_dict(cls, config: dict):
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

        action_space = env.action_space(env_params)
        discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)
        agent_kwargs = config.pop("agent_kwargs", {})

        # Convert activation from str to Callable
        activation = agent_kwargs.pop("activation", "swish")
        agent_kwargs["activation"] = getattr(nn, activation)

        # Convert hidden layer sizes to tuple
        hidden_layer_sizes = agent_kwargs.pop("hidden_layer_sizes", (64, 64))
        agent_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

        if discrete:
            actor = DiscretePolicy(action_space.n, **agent_kwargs)
            critic = DiscreteQNetwork(action_dim=action_space.n, **agent_kwargs)
        else:
            actor = SquashedGaussianPolicy(
                np.prod(action_space.shape),
                (action_space.low, action_space.high),
                log_std_range=(-10, 2),
                **agent_kwargs,
            )
            critic = QNetwork(**agent_kwargs)

        evaluate = make_evaluate(SAC.make_act, env, env_params, 200)
        return cls(
            env=env,
            env_params=env_params,
            actor=actor,
            critic=critic,
            eval_callback=evaluate,
            **config,
        )
