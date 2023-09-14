import jax
import chex
import gymnax
import distrax
import numpy as np
from jax import numpy as jnp
from flax import struct, linen as nn
from typing import Callable, Any, Tuple
from flax.linen.initializers import constant
from gymnax.environments.environment import Environment

# TODO: clean up q_all, q_of, q functions


class SACConfig(struct.PyTreeNode):
    # Non-static parameters
    env_params: Any
    gamma: chex.Scalar
    tau: float
    target_entropy_ratio: float
    learning_rate: chex.Scalar

    # Static parameters
    total_timesteps: int = struct.field(pytree_node=False)
    eval_freq: int = struct.field(pytree_node=False)
    agent: nn.Module = struct.field(pytree_node=False)
    env: Environment = struct.field(pytree_node=False)
    eval_callback: Callable = struct.field(pytree_node=False)
    num_envs: int = struct.field(pytree_node=False)
    buffer_size: int = struct.field(pytree_node=False)
    fill_buffer: int = struct.field(pytree_node=False)
    batch_size: int = struct.field(pytree_node=False)
    gradient_steps: int = struct.field(pytree_node=False)
    normalize_observations: bool = struct.field(pytree_node=False, default=False)
    skip_initial_evaluation: bool = struct.field(pytree_node=False, default=False)

    @property
    def discrete(self):
        action_space = self.env.action_space(self.env_params)
        return isinstance(action_space, gymnax.environments.spaces.Discrete)

    @property
    def action_dim(self):
        action_space = self.env.action_space(self.env_params)
        if self.discrete:
            return action_space.n
        else:
            return np.prod(action_space.shape)

    @property
    def target_entropy(self):
        if self.discrete:
            return -self.target_entropy_ratio * np.log(1 / self.action_dim)
        else:
            return -self.action_dim

    @classmethod
    def from_dict(cls, config: dict):
        from copy import deepcopy
        from purerl.evaluate import make_evaluate
        from purerl.brax2gymnax import Brax2GymnaxEnv

        config = deepcopy(config)  # Because we're popping from it

        # Get env id and convert to gymnax environment and parameters
        env_kwargs = config.pop("env_kwargs", None) or {}
        env_id = config.pop("env")
        if env_id.startswith("brax"):
            env = Brax2GymnaxEnv(env_id.split("/")[1], **env_kwargs)
            env_params = env.default_params
        else:
            env, env_params = gymnax.make(env_id, **env_kwargs)

        action_space = env.action_space(env_params)
        discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)
        agent_kwargs = config.pop("agent_kwargs", None) or {}

        if discrete:
            agent_cls = SACAgentDiscrete
            action_dim = env.action_space(env_params).n
        else:
            # Get action range to pass to agent for action normalization
            agent_cls = SACAgentContinuous
            action_range = (
                env.action_space(env_params).low,
                env.action_space(env_params).high,
            )
            agent_kwargs["action_range"] = action_range
            action_dim = np.prod(env.action_space(env_params).shape)

        # Convert activation from str to Callable
        activation = agent_kwargs.pop("activation", "relu")
        agent_kwargs["activation"] = getattr(nn, activation)

        # Convert hidden layer sizes to tuple
        hidden_layer_sizes = agent_kwargs.pop("hidden_layer_sizes", None)
        if hidden_layer_sizes is not None:
            agent_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

        agent = agent_cls(action_dim, **agent_kwargs)

        evaluate = make_evaluate(env, env_params, 200)
        return cls(
            env=env,
            env_params=env_params,
            agent=agent,
            eval_callback=evaluate,
            **config,
        )


class SquashedGaussianActor(nn.Module):
    action_dim: int
    action_range: Tuple[float, float]
    hidden_layer_sizes: Tuple[int]
    activation: Callable
    log_std_range: Tuple[float, float] = (-20, 2)

    def setup(self):
        features = []
        for size in self.hidden_layer_sizes:
            features.append(nn.Dense(size))
            features.append(self.activation)
        self.features = nn.Sequential(features)

        self.action_mean = nn.Dense(self.action_dim)
        self.action_log_std = nn.Dense(self.action_dim)
        self.bij = distrax.Tanh()

    @property
    def action_loc(self):
        return (self.action_range[1] + self.action_range[0]) / 2

    @property
    def action_scale(self):
        return (self.action_range[1] - self.action_range[0]) / 2

    def __call__(self, x, rng):
        features = self.features(x)
        action_mean = self.action_mean(features)
        action_log_std = self.action_log_std(features)
        action_log_std = jnp.clip(
            action_log_std, *self.log_std_range
        )  # TODO: tanh transform

        action_dist = distrax.MultivariateNormalDiag(
            loc=action_mean, scale_diag=jnp.exp(action_log_std)
        )

        action = action_dist.sample(seed=rng)
        action_log_prob = action_dist.log_prob(action)
        action, log_det_j = self.bij.forward_and_log_det(action)
        action = self.action_loc + action * self.action_scale
        action_log_prob -= log_det_j.sum(axis=-1)

        return action, action_log_prob

    def act(self, obs, rng):
        action, _ = self(obs, rng)
        return action


class MLPQFunction(nn.Module):
    hidden_layer_sizes: Tuple[int]
    activation: Callable

    @nn.compact
    def __call__(self, obs, action):
        seq = nn.Sequential(
            [nn.Dense(64), self.activation, nn.Dense(64), self.activation, nn.Dense(1)]
        )
        q = seq(jnp.concatenate([obs.reshape(obs.shape[0], -1), action], axis=-1))
        return jnp.squeeze(q, axis=-1)


class SACAgentContinuous(nn.Module):
    action_dim: int
    action_range: Tuple[float, float]
    hidden_layer_sizes: Tuple[int] = (64, 64)
    activation: Callable = nn.relu
    log_std_range: Tuple[float, float] = (-20, 2)

    def setup(self):
        self.actor = SquashedGaussianActor(
            self.action_dim,
            self.action_range,
            self.hidden_layer_sizes,
            self.activation,
            self.log_std_range,
        )
        self.q1 = MLPQFunction(self.hidden_layer_sizes, self.activation)
        self.q2 = MLPQFunction(self.hidden_layer_sizes, self.activation)
        self.log_alpha = self.param("log_alpha", constant(0.0), ())

    def __call__(self, obs, rng):
        action, action_log_prob = self.pi(obs, rng)
        q1, q2 = self.q(obs, action)
        return action, action_log_prob, q1, q2

    def act(self, obs, rng):
        action, _ = self.pi(obs, rng)
        return action

    def pi(self, obs, rng):
        action, action_log_prob = self.actor(obs, rng)
        return action, action_log_prob

    def q(self, obs, action):
        q1 = self.q1(obs, action)
        q2 = self.q2(obs, action)
        return q1, q2

    def log_alpha(self):
        return self.log_alpha


class DiscreteActor(nn.Module):
    action_dim: int
    hidden_layer_sizes: Tuple[int]
    activation: Callable

    def setup(self):
        features = []
        for size in self.hidden_layer_sizes:
            features.append(nn.Dense(size))
            features.append(self.activation)
        self.features = nn.Sequential(features)

        self.action_logits = nn.Dense(self.action_dim)

    def __call__(self, x, rng):
        features = self.features(x)
        action_logits = self.action_logits(features)
        action_dist = distrax.Categorical(logits=action_logits)
        action = action_dist.sample(seed=rng)

        action_logprobs = jax.nn.log_softmax(action_logits)
        return action, action_logprobs

    def act(self, obs, rng):
        action, _ = self(obs, rng)
        return action


class DuelingQNetwork(nn.Module):
    action_dim: int
    hidden_layer_sizes: Tuple[int]
    activation: Callable

    def setup(self):
        encoder = []
        for size in self.hidden_layer_sizes:
            encoder.append(nn.Dense(size))
            encoder.append(self.activation)
        self.encoder = nn.Sequential(encoder)

        self.value_ = nn.Dense(1)
        self.advantage_ = nn.Dense(self.action_dim)

    def encode(self, x):
        x = x.reshape((x.shape[0], -1))
        return self.encoder(x)

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

    def q_of(self, obs, action):
        q_values = self(obs)
        return jnp.take_along_axis(q_values, action[:, None], axis=1).squeeze(axis=1)


class SACAgentDiscrete(nn.Module):
    action_dim: int
    hidden_layer_sizes: Tuple[int] = (64, 64)
    activation: Callable = nn.relu

    def setup(self):
        self.actor = DiscreteActor(
            self.action_dim, self.hidden_layer_sizes, self.activation
        )

        # TODO: give type of Q network as argument
        self.q1 = DuelingQNetwork(
            self.action_dim, self.hidden_layer_sizes, self.activation
        )
        self.q2 = DuelingQNetwork(
            self.action_dim, self.hidden_layer_sizes, self.activation
        )
        self.log_alpha = self.param("log_alpha", constant(0.0), ())

    def __call__(self, obs, rng):
        action, action_logprobs = self.actor(obs, rng)
        q1, q2 = self.q_all(obs)
        # TODO: do you use action_logits or action?
        return action, action_logprobs, q1, q2

    def pi(self, obs, rng):
        action, action_logprobs = self.actor(obs, rng)
        return action, action_logprobs

    def q_all(self, obs, *args):
        # action is passed by the algorithm since it works for both discrete and
        # continuous. This should be fixed later
        q1 = self.q1(obs)
        q2 = self.q2(obs)
        return q1, q2

    def q(self, obs, action):
        q1 = jnp.take_along_axis(self.q1(obs), action[:, None], axis=1).squeeze()
        q2 = jnp.take_along_axis(self.q2(obs), action[:, None], axis=1).squeeze()
        return q1, q2

    def act(self, obs, rng):
        action = self.actor.act(obs, rng)
        return action

    def log_alpha(self):
        return self.log_alpha
