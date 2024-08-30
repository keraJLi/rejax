from collections.abc import Sequence
from typing import Callable, Tuple, Type

import distrax
import jax
from flax import linen as nn
from flax.linen.initializers import constant
from jax import numpy as jnp


class MLP(nn.Module):
    hidden_layer_sizes: Sequence[int]
    activation: Callable

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        for size in self.hidden_layer_sizes:
            x = nn.Dense(size)(x)
            x = self.activation(x)
        return x


# Policy networks


class DiscretePolicy(nn.Module):
    action_dim: int
    hidden_layer_sizes: Sequence[int]
    activation: Callable

    def setup(self):
        self.features = MLP(self.hidden_layer_sizes, self.activation)
        self.action_logits = nn.Dense(self.action_dim)

    def _action_dist(self, obs):
        features = self.features(obs)
        action_logits = self.action_logits(features)
        return distrax.Categorical(logits=action_logits)

    def __call__(self, obs, rng):
        action_dist = self._action_dist(obs)
        action = action_dist.sample(seed=rng)
        return action, action_dist.log_prob(action), action_dist.entropy()

    def act(self, obs, rng):
        action, _, _ = self(obs, rng)
        return action

    def log_prob_entropy(self, obs, action):
        action_dist = self._action_dist(obs)
        return action_dist.log_prob(action), action_dist.entropy()

    def action_log_prob(self, obs, rng):
        action_dist = self._action_dist(obs)
        action = action_dist.sample(seed=rng)
        return action, action_dist.log_prob(action)


def EpsilonGreedyPolicy(qnet: nn.Module) -> Type[nn.Module]:
    class EpsilonGreedyPolicy(qnet):
        def _action_dist(self, obs, epsilon):
            q = self(obs)
            return distrax.EpsilonGreedy(q, epsilon=epsilon)

        def act(self, obs, rng, epsilon=0.05):
            action_dist = self._action_dist(obs, epsilon)
            action = action_dist.sample(seed=rng)
            return action

    return EpsilonGreedyPolicy


class GaussianPolicy(nn.Module):
    action_dim: int
    action_range: Tuple[int, int]
    hidden_layer_sizes: Sequence[int]
    activation: Callable

    def setup(self):
        self.features = MLP(self.hidden_layer_sizes, self.activation)
        self.action_mean = nn.Dense(self.action_dim)
        self.action_log_std = self.param(
            "action_log_std", constant(0.0), (self.action_dim,)
        )

    def _action_dist(self, obs):
        features = self.features(obs)
        action_mean = self.action_mean(features)
        return distrax.MultivariateNormalDiag(
            loc=action_mean, scale_diag=jnp.exp(self.action_log_std)
        )

    def __call__(self, obs, rng):
        action_dist = self._action_dist(obs)
        action = action_dist.sample(seed=rng)
        return action, action_dist.log_prob(action), action_dist.entropy()

    def act(self, obs, rng):
        action, _, _ = self(obs, rng)
        return jnp.clip(action, self.action_range[0], self.action_range[1])

    def log_prob_entropy(self, obs, action):
        action_dist = self._action_dist(obs)
        return action_dist.log_prob(action), action_dist.entropy()

    def action_log_prob(self, obs, rng):
        action_dist = self._action_dist(obs)
        action = action_dist.sample(seed=rng)
        return action, action_dist.log_prob(action)


class SquashedGaussianPolicy(nn.Module):
    action_dim: int
    action_range: Tuple[float, float]
    hidden_layer_sizes: Sequence[int]
    activation: Callable
    log_std_range: Tuple[float, float]

    def setup(self):
        self.features = MLP(self.hidden_layer_sizes, self.activation)
        self.action_mean = nn.Dense(self.action_dim)
        self.action_log_std = nn.Dense(self.action_dim)
        self.bij = distrax.Tanh()

    @property
    def action_loc(self):
        return (self.action_range[1] + self.action_range[0]) / 2

    @property
    def action_scale(self):
        return (self.action_range[1] - self.action_range[0]) / 2

    def _action_dist(self, obs):
        features = self.features(obs)
        action_mean = self.action_mean(features)
        action_log_std = self.action_log_std(features)
        action_log_std = jnp.clip(
            action_log_std, *self.log_std_range
        )  # TODO: tanh transform?

        return distrax.MultivariateNormalDiag(
            loc=action_mean, scale_diag=jnp.exp(action_log_std)
        )

    def __call__(self, obs, rng):
        action_dist = self._action_dist(obs)
        action = action_dist.sample(seed=rng)
        action_log_prob = action_dist.log_prob(action)
        action, log_det_j = self.bij.forward_and_log_det(action)
        action = self.action_loc + action * self.action_scale
        action_log_prob -= log_det_j.sum(axis=-1)
        return action, action_log_prob

    def action_log_prob(self, obs, rng):
        return self(obs, rng)

    def log_prob(self, obs, action):
        action_dist = self._action_dist(obs)
        action = (action - self.action_loc) / self.action_scale
        action, log_det_j = self.bij.inverse_and_log_det(action)
        action_log_prob = action_dist.log_prob(action)
        action_log_prob += log_det_j.sum(axis=-1)
        return action_log_prob

    def act(self, obs, rng):
        action, _ = self(obs, rng)
        return action


class BetaPolicy(nn.Module):
    action_dim: int
    action_range: Tuple[float, float]
    hidden_layer_sizes: Sequence[int]
    activation: Callable

    @property
    def action_loc(self):
        return self.action_range[0]

    @property
    def action_scale(self):
        return self.action_range[1] - self.action_range[0]

    def __call__(self, obs, rng):
        action, _ = self.action_log_prob(obs, rng)
        return action, *self.log_prob_entropy(obs, action)

    def setup(self):
        self.features = MLP(self.hidden_layer_sizes, self.activation)
        self.alpha = nn.Dense(self.action_dim)
        self.beta = nn.Dense(self.action_dim)

    def _action_dist(self, obs):
        x = self.features(obs)
        alpha = 1 + nn.softplus(self.alpha(x))
        beta = 1 + nn.softplus(self.beta(x))
        return distrax.Beta(alpha, beta)

    def action_log_prob(self, obs, rng):
        action_dist = self._action_dist(obs)
        action = action_dist.sample(seed=rng)
        log_prob = action_dist.log_prob(action)
        action = self.action_loc + action * self.action_scale
        return action, log_prob.squeeze(1)

    def act(self, obs, rng):
        action, _ = self.action_log_prob(obs, rng)
        return action

    def log_prob_entropy(self, obs, action):
        action_dist = self._action_dist(obs)
        action = (action - self.action_loc) / self.action_scale
        return action_dist.log_prob(action).squeeze(1), action_dist.entropy()


class DeterministicPolicy(nn.Module):
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


# Value networks


class VNetwork(MLP):
    @nn.compact
    def __call__(self, obs):
        x = super().__call__(obs)
        return nn.Dense(1)(x).squeeze(1)


class QNetwork(MLP):
    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs.reshape(obs.shape[0], -1), action], axis=-1)
        x = super().__call__(x)
        return nn.Dense(1)(x).squeeze(1)


class DiscreteQNetwork(MLP):
    action_dim: int

    @nn.compact
    def __call__(self, obs):
        x = super().__call__(obs)
        return nn.Dense(self.action_dim)(x)

    def take(self, obs, action):
        q_values = self(obs)
        return jnp.take_along_axis(q_values, action[:, None], axis=1).squeeze(1)


class DuelingQNetwork(nn.Module):
    hidden_layer_sizes: Sequence[int]
    activation: Callable
    action_dim: int

    @nn.compact
    def __call__(self, obs):
        x = MLP(self.hidden_layer_sizes, self.activation)(obs)
        value = nn.Dense(1)(x)
        advantage = nn.Dense(self.action_dim)(x)
        advantage = advantage - jnp.mean(advantage, axis=-1, keepdims=True)
        return value + advantage

    def take(self, obs, action):
        q_values = self(obs)
        return jnp.take_along_axis(q_values, action[:, None], axis=1).squeeze(1)


class ImplicitQuantileNetwork(nn.Module):
    hidden_layer_sizes: Sequence[int]
    activation: Callable
    action_dim: int

    risk_distortion: Callable = lambda tau: tau
    # risk_distortion: Callable = lambda tau: 0.8 * tau
    # Or e.g.: tau ** 0.71 / (tau ** 0.71 + (1 - tau) ** 0.71) ** (1 / 0.71)

    @property
    def embedding_dim(self):
        return self.hidden_layer_sizes[-1]

    @nn.compact
    def __call__(self, obs, rng):
        x = obs.reshape(obs.shape[0], -1)
        psi = MLP(self.hidden_layer_sizes, self.activation)(x)

        tau = distrax.Uniform(0, 1).sample(seed=rng, sample_shape=obs.shape[0])
        tau = self.risk_distortion(tau)
        phi_input = jnp.cos(jnp.pi * jnp.outer(tau, jnp.arange(self.embedding_dim)))
        phi = nn.relu(nn.Dense(self.embedding_dim)(phi_input))

        x = nn.swish(nn.Dense(64)(psi * phi))
        return nn.Dense(self.action_dim)(x), tau

    def q(self, obs, rng, num_samples=32):
        rng = jax.random.split(rng, num_samples)
        zs, _ = jax.vmap(self, in_axes=(None, 0))(obs, rng)
        return zs.mean(axis=0)

    def best_action(self, obs, rng, num_samples=32):
        q = self.q(obs, rng, num_samples)
        best_action = jnp.argmax(q, axis=1)
        return best_action
