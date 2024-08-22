# imports
from collections.abc import Sequence
from typing import Callable, Tuple

import distrax
from flax import linen as nn


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
        # import jax
        # jax.debug.print("{}, {}", alpha, beta)
        return distrax.Beta(alpha, beta)

    def action_log_prob(self, obs, rng):
        action_dist = self._action_dist(obs)
        action = action_dist.sample(seed=rng)
        log_prob = action_dist.log_prob(action)
        action = self.action_loc + action * self.action_scale
        # import jax
        # jax.debug.print("{}", action)
        return action, log_prob

    def act(self, obs, rng):
        action, _ = self.action_log_prob(obs, rng)
        return action

    def log_prob_entropy(self, obs, action):
        action_dist = self._action_dist(obs)
        action = (action - self.action_loc) / self.action_scale
        # import jax
        # jax.debug.print("{}", jnp.concatenate((action_dist.alpha, action_dist.beta), axis=1))
        return action_dist.log_prob(action), action_dist.entropy()


if __name__ == "__main__":
    import jax
    import optax
    from flax.training.train_state import TrainState
    from jax import numpy as jnp

    b = BetaPolicy(1, (-3, 3), (64, 64), nn.swish)
    ts = TrainState.create(
        apply_fn=b.apply,
        params=b.init(jax.random.PRNGKey(0), jnp.ones((1, 4))),
        tx=optax.adam(1e-3),
    )

    def loss(ts):
        log_prob, _ = ts.apply_fn(
            ts.params, jnp.ones((1, 4)), jnp.ones((1,)), method="log_prob_entropy"
        )
        return -log_prob

    @jax.jit
    def update(ts):
        grads = jax.grad(loss)(ts)
        return ts.apply_gradients(grads=grads)

    for _ in range(1000):
        ts = update(ts)
        print(loss(ts))
