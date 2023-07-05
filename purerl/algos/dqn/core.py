import chex
import distrax
from jax import numpy as jnp
from typing import Callable, Any
from optax import linear_schedule
from flax import struct, linen as nn
from gymnax.environments.environment import Environment


class DQNConfig(struct.PyTreeNode):
    # Non-static parameters
    env_params: Any
    gamma: chex.Scalar
    ddqn: bool
    max_grad_norm: chex.Scalar
    learning_rate: chex.Scalar
    target_update_freq: int

    # Static parameters
    total_timesteps: int = struct.field(pytree_node=False)
    eval_freq: int = struct.field(pytree_node=False)
    agent: nn.Module = struct.field(pytree_node=False)
    env: Environment = struct.field(pytree_node=False)
    evaluate: Callable = struct.field(pytree_node=False)
    exploration_fraction: chex.Scalar = struct.field(pytree_node=False)
    eps_start: chex.Scalar = struct.field(pytree_node=False)
    eps_end: chex.Scalar = struct.field(pytree_node=False)
    num_envs: int = struct.field(pytree_node=False)
    buffer_size: int = struct.field(pytree_node=False)
    fill_buffer: int = struct.field(pytree_node=False)
    batch_size: int = struct.field(pytree_node=False)
    gradient_steps: int = struct.field(pytree_node=False)

    @property
    def epsilon_schedule(self):
        return linear_schedule(
            self.eps_start,
            self.eps_end,
            int(self.exploration_fraction * self.total_timesteps),
        )

    @classmethod
    def from_dict(cls, config):
        import gymnax
        from purerl.evaluate import make_evaluate

        # Get env id and convert to gymnax environment and parameters
        env_kwargs = config.pop("env_kwargs", None) or {}
        env, env_params = gymnax.make(config.pop("env"), **env_kwargs)

        agent_name = config.pop("agent", "QNetwork")
        agent_cls = {
            "QNetwork": QNetwork,
            "DuelingQNetwork": DuelingQNetwork,
            "ConvQNetwork": ConvQNetwork,
        }[agent_name]
        agent_kwargs = config.pop("agent_kwargs", None) or {}
        activation = agent_kwargs.pop("activation", "relu")
        agent_kwargs["activation"] = getattr(nn, activation)

        action_dim = env.action_space(env_params).n
        agent = agent_cls(action_dim, **agent_kwargs)

        evaluate = make_evaluate(env, env_params, 200)
        return cls(
            env_params=env_params, agent=agent, env=env, evaluate=evaluate, **config
        )


class QNetwork(nn.Module):
    action_dim: int
    activation: Callable = nn.tanh

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(64)(x)
        x = self.activation(x)
        x = nn.Dense(64)(x)
        x = self.activation(x)
        x = nn.Dense(self.action_dim)(x)
        return x

    def act(self, obs, rng, epsilon=0):
        q_values = self(obs)
        action_dist = distrax.EpsilonGreedy(q_values, epsilon)
        action = action_dist.sample(seed=rng)
        return action

    def q_of(self, obs, action):
        q_values = self(obs)
        return jnp.take_along_axis(q_values, action[:, None], axis=1).squeeze(axis=1)


class DuelingQNetwork(nn.Module):
    action_dim: int
    activation: Callable = nn.tanh

    def setup(self):
        self.encoder = [nn.Dense(64), nn.Dense(64)]
        self.value_ = nn.Dense(1)
        self.advantage_ = nn.Dense(self.action_dim)

    def encode(self, x):
        x = x.reshape((x.shape[0], -1))
        for layer in self.encoder:
            x = self.activation(layer(x))
        return x

    def value(self, x_encoded):
        return self.value_(x_encoded)

    def advantage(self, x_encoded):
        advantage = self.advantage_(x_encoded)
        advantage = advantage - jnp.mean(advantage, axis=-1, keepdims=True)
        return advantage

    def __call__(self, x):
        x_encoded = self.encode(x)
        value = self.value(x_encoded)
        advantage = self.advantage(x_encoded)
        return value + advantage

    def act(self, obs, rng, epsilon=0):
        q_values = self(obs)
        action_dist = distrax.EpsilonGreedy(q_values, epsilon)
        action = action_dist.sample(seed=rng)
        return action

    def q_of(self, obs, action):
        q_values = self(obs)
        return jnp.take_along_axis(q_values, action[:, None], axis=1).squeeze(axis=1)


class ConvQNetwork(nn.Module):
    action_dim: int
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(16, (3, 3))(x)
        x = self.activation(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128)(x)
        x = self.activation(x)
        x = nn.Dense(self.action_dim)(x)
        return x

    def act(self, obs, rng, epsilon=0):
        q_values = self(obs)
        action_dist = distrax.EpsilonGreedy(q_values, epsilon)
        action = action_dist.sample(seed=rng)
        return action
