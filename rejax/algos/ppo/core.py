from copy import deepcopy
from typing import Any, Callable

import chex
import gymnax
import numpy as np
from flax import linen as nn
from flax import struct
from gymnax.environments.environment import Environment
from jax import numpy as jnp

from rejax.algos.networks import DiscretePolicy, GaussianPolicy, VNetwork
from rejax.algos.ppo.ppo import PPO
from rejax.brax2gymnax import Brax2GymnaxEnv
from rejax.evaluate import make_evaluate


class PPOConfig(struct.PyTreeNode):
    # fmt: off
    # Non-static parameters
    env: Environment                = struct.field(pytree_node=False)
    env_params: Any                 = struct.field(pytree_node=True)
    actor: nn.Module                = struct.field(pytree_node=False)
    critic: nn.Module               = struct.field(pytree_node=False)
    eval_callback: Callable         = struct.field(pytree_node=False)

    learning_rate: chex.Scalar      = struct.field(pytree_node=True, default=0.0005)
    gamma: chex.Scalar              = struct.field(pytree_node=True, default=0.99)
    gae_lambda: chex.Scalar         = struct.field(pytree_node=True, default=0.95)
    clip_eps: chex.Scalar           = struct.field(pytree_node=True, default=0.2)
    vf_coef: chex.Scalar            = struct.field(pytree_node=True, default=0.5)
    ent_coef: chex.Scalar           = struct.field(pytree_node=True, default=0.01)
    max_grad_norm: chex.Scalar      = struct.field(pytree_node=True, default=jnp.inf)

    # Static parameters
    total_timesteps: int            = struct.field(pytree_node=False, default=100_000)
    eval_freq: int                  = struct.field(pytree_node=False, default=10_000)
    num_envs: int                   = struct.field(pytree_node=False, default=100)
    num_steps: int                  = struct.field(pytree_node=False, default=50)
    num_epochs: int                 = struct.field(pytree_node=False, default=5)
    num_minibatches: int            = struct.field(pytree_node=False, default=10)
    normalize_observations: bool    = struct.field(pytree_node=False, default=False)
    skip_initial_evaluation: bool   = struct.field(pytree_node=False, default=False)
    # fmt: on

    @property
    def minibatch_size(self):
        assert (self.num_envs * self.num_steps) % self.num_minibatches == 0
        return (self.num_envs * self.num_steps) // self.num_minibatches

    @property
    def agent(self):
        """Backward compatibility with old evaluation creation"""
        return self.actor

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

        action_space = env.action_space(env_params)
        discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)

        agent_kwargs = config.pop("agent_kwargs", {})
        activation = agent_kwargs.pop("activation", "swish")
        agent_kwargs["activation"] = getattr(nn, activation)

        # Convert hidden layer sizes to tuple
        hidden_layer_sizes = agent_kwargs.pop("hidden_layer_sizes", (64, 64))
        agent_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

        if discrete:
            actor = DiscretePolicy(action_space.n, **agent_kwargs)
        else:
            actor = GaussianPolicy(
                np.prod(action_space.shape),
                (action_space.low, action_space.high),
                **agent_kwargs,
            )
        critic = VNetwork(**agent_kwargs)

        evaluate = make_evaluate(PPO.make_act, env, env_params, 200)

        return cls(
            env=env,
            env_params=env_params,
            actor=actor,
            critic=critic,
            eval_callback=evaluate,
            **config,
        )
