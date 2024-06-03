from typing import Any, Callable, Tuple

import chex
from flax import linen as nn
from flax import struct
from gymnax.environments.environment import Environment
from jax import numpy as jnp

from rejax.algos.ddpg.ddpg import DDPG


class DDPGConfig(struct.PyTreeNode):
    # fmt: off
    # Non-static parameters
    env: Environment                = struct.field(pytree_node=False)
    env_params: Any                 = struct.field(pytree_node=True)
    eval_callback: Callable         = struct.field(pytree_node=False)
    agent: nn.Module                = struct.field(pytree_node=False)

    learning_rate: chex.Scalar      = struct.field(pytree_node=True, default=0.001)
    gamma: chex.Scalar              = struct.field(pytree_node=True, default=0.99)
    tau: chex.Scalar                = struct.field(pytree_node=True, default=0.95)
    exploration_noise: chex.Scalar  = struct.field(pytree_node=True, default=0.3)
    max_grad_norm: chex.Scalar      = struct.field(pytree_node=True, default=jnp.inf)

    # Static parameters
    total_timesteps: int            = struct.field(pytree_node=False, default=10_000)
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
        from copy import deepcopy

        import gymnax
        import numpy as np

        from rejax.brax2gymnax import Brax2GymnaxEnv
        from rejax.evaluate import make_evaluate

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

        agent_kwargs = config.pop("agent_kwargs", {})
        activation = agent_kwargs.pop("activation", "relu")
        agent_kwargs["activation"] = getattr(nn, activation)

        action_range = (
            env.action_space(env_params).low,
            env.action_space(env_params).high,
        )
        action_dim = np.prod(env.action_space(env_params).shape)
        agent = DDPGAgent(action_dim, action_range, **agent_kwargs)

        def make_act(config, ts):
            from rejax.normalize import normalize_obs

            def act(obs, rng):
                if getattr(config, "normalize_observations", False):
                    obs = normalize_obs(ts.rms_state, obs)

                obs = jnp.expand_dims(obs, 0)
                action = config.actor.apply(ts.params, obs, rng, method="act")
                return jnp.squeeze(action)

            return act

        evaluate = make_evaluate(make_act, env, env_params, 200)
        return cls(
            env_params=env_params,
            agent=agent,
            env=env,
            eval_callback=evaluate,
            **config,
        )


class MLPQFunction(nn.Module):
    hidden_layer_sizes: Tuple[int]
    activation: Callable

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs.reshape(obs.shape[0], -1), action], axis=-1)
        for size in self.hidden_layer_sizes:
            x = nn.Dense(size)(x)
            x = self.activation(x)
        q = nn.Dense(1)(x)
        return jnp.squeeze(q, axis=-1)


class DDPGActor(nn.Module):
    action_dim: int
    action_range: Tuple[float, float]
    hidden_layer_sizes: Tuple[int]
    activation: Callable

    @property
    def action_loc(self):
        return (self.action_range[1] + self.action_range[0]) / 2

    @property
    def action_scale(self):
        return (self.action_range[1] - self.action_range[0]) / 2

    @nn.compact
    def __call__(self, x):
        for size in self.hidden_layer_sizes:
            x = nn.Dense(size)(x)
            x = self.activation(x)
        x = nn.Dense(self.action_dim)(x)
        x = jnp.tanh(x)

        action = self.action_loc + x * self.action_scale
        return action

    def act(self, obs, rng):
        action = self(obs)
        return action


class DDPGAgent(nn.Module):
    action_dim: int
    action_range: Tuple[float, float]
    hidden_layer_sizes: Tuple[int] = (64, 64)
    activation: Callable = nn.relu

    def setup(self):
        self.actor = DDPGActor(
            self.action_dim, self.action_range, self.hidden_layer_sizes, self.activation
        )
        self.q = MLPQFunction(self.hidden_layer_sizes, self.activation)

    def __call__(self, obs):
        action = self.actor(obs)
        q = self.q(obs, action)
        return action, q

    def act(self, obs, rng):
        action = self.actor(obs)
        return action

    def pi(self, obs):
        action = self.actor(obs)
        return action

    def q(self, obs, action):
        q = self.q(obs, action)
        return q
