import jax
import chex
import distrax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
from typing import Callable, Tuple
from flax.linen.initializers import constant, orthogonal


class Agent(nn.Module):
    def act(self, obs: chex.Array, rng: chex.PRNGKey) -> chex.Array:
        raise NotImplementedError

    def squeezed_act(self, obs: chex.Array, rng: chex.PRNGKey) -> chex.Array:
        obs = jnp.expand_dims(obs, axis=0)
        unsqueezed_action = self.act(obs, rng)
        return jnp.squeeze(unsqueezed_action, axis=0)


class ActorCritic(Agent):
    action_dim: int
    discrete: bool
    activation: Callable = nn.tanh

    def setup(self):
        self.value_ = [
            nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)),
            nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)),
            nn.Dense(
                1,
                kernel_init=orthogonal(jnp.sqrt(0.01)),
                bias_init=constant(0.0),
            ),
        ]
        self.action_ = [
            nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)),
            nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)),
            nn.Dense(
                self.action_dim,
                kernel_init=orthogonal(1),
                bias_init=constant(0.0),
            ),
        ]
        self.action_log_std = self.param(
            "action_log_std", constant(0.0), (self.action_dim,)
        )

    def value(self, x):
        x = x.reshape((x.shape[0], -1))
        for i, layer in enumerate(self.value_, start=1):
            x = layer(x)
            if i < len(self.value_):
                x = self.activation(x)
        return x.squeeze()

    def action(self, x):
        x = x.reshape((x.shape[0], -1))
        for i, layer in enumerate(self.action_, start=1):
            x = layer(x)
            if i < len(self.action_):
                x = self.activation(x)
        return x

    def __call__(self, x):
        value = self.value(x)
        action = self.action(x)
        return action, self.action_log_std, value

    def _action_dist(self, obs):
        action = self.action(obs)
        if self.discrete:
            action_dist = distrax.Categorical(logits=action)
        else:
            action_dist = distrax.MultivariateNormalDiag(
                loc=action, scale_diag=jnp.exp(self.action_log_std)
            )
        return action_dist

    def act(self, obs, rng):
        action_dist = self._action_dist(obs)
        action = action_dist.sample(seed=rng)
        return action

    def action_log_prob(self, obs, rng):
        action_dist = self._action_dist(obs)
        action = action_dist.sample(seed=rng)
        return action, action_dist.log_prob(action)

    def log_prob_entropy(self, obs, action):
        action_dist = self._action_dist(obs)
        return action_dist.log_prob(action), action_dist.entropy()


class QNetwork(Agent):
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


class DuelingQNetwork(Agent):
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


class ConvQNetwork(Agent):
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


class SquashedGaussianActor(Agent):
    action_dim: int
    action_range: Tuple[float, float]
    activation: Callable = nn.relu
    log_std_range: Tuple[float, float] = (-20, 2)

    def setup(self):
        self.features = nn.Sequential(
            [nn.Dense(64), self.activation, nn.Dense(64), self.activation]
        )
        self.action_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.action_log_std = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )

        self.bij = distrax.Tanh()

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

        action_loc = (self.action_range[1] + self.action_range[0]) / 2
        action_scale = (self.action_range[1] - self.action_range[0]) / 2
        action = action_loc + action * action_scale
        action_log_prob -= log_det_j.sum(axis=-1)

        return action, action_log_prob

    def act(self, obs, rng):
        action, _ = self(obs, rng)
        return action


class MLPQFunction(nn.Module):
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, obs, action):
        seq = nn.Sequential(
            [nn.Dense(64), self.activation, nn.Dense(64), self.activation, nn.Dense(1)]
        )
        q = seq(jnp.concatenate([obs.reshape(obs.shape[0], -1), action], axis=-1))
        return jnp.squeeze(q, axis=-1)


class SACAgentContinuous(Agent):
    action_dim: int
    action_range: Tuple[float, float]
    activation: Callable = nn.tanh
    log_std_range: Tuple[float, float] = (-20, 2)

    def setup(self):
        self.actor = SquashedGaussianActor(
            self.action_dim,
            self.action_range,
            self.activation,
            self.log_std_range,
        )
        self.q1 = MLPQFunction()
        self.q2 = MLPQFunction()
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


class DiscreteActor(Agent):
    action_dim: int
    activation: Callable = nn.relu

    def setup(self):
        self.features = nn.Sequential(
            [nn.Dense(64), self.activation, nn.Dense(64), self.activation]
        )
        self.action_logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )

    def __call__(self, x, rng):
        features = self.features(x)
        action_logits = self.action_logits(features)
        action_dist = distrax.Categorical(logits=action_logits)  # normalizes logits
        action = action_dist.sample(seed=rng)

        action_logprobs = jax.nn.log_softmax(action_logits)
        return action, action_logprobs

    def act(self, obs, rng):
        action, _ = self(obs, rng)
        return action


class SACAgentDiscrete(Agent):
    action_dim: int
    activation: Callable = nn.relu

    def setup(self):
        self.actor = DiscreteActor(self.action_dim, self.activation)
        self.q1 = DuelingQNetwork(self.action_dim, self.activation)
        self.q2 = DuelingQNetwork(self.action_dim, self.activation)
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
        # action is int array
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



@partial(
    nn.vmap, variable_axes={"params": 0}, split_rngs={"params": True}, axis_name="q"
)
class MLPQFunctionEnsemble(nn.Module):
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, obs, action):
        seq = nn.Sequential(
            [nn.Dense(64), self.activation, nn.Dense(64), self.activation, nn.Dense(1)]
        )
        q = seq(jnp.concatenate([obs.reshape(obs.shape[0], -1), action], axis=-1))
        return jnp.squeeze(q, axis=-1)


class TD3Agent(Agent):
    action_dim: int
    action_range: Tuple[float, float]
    activation: Callable = nn.relu

    def setup(self):
        self.actor = DDPGActor(
            self.action_dim, self.action_range, activation=self.activation
        )
        self.q = nn.vmap(
            MLPQFunction,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_name="q",
            axis_size=2,
        )
        # self.q1 = MLPQFunction(activation=self.activation)
        # self.q2 = MLPQFunction(activation=self.activation)

    def __call__(self, obs):
        action = self.actor(obs)
        q1 = self.q1(obs, action)
        q2 = self.q2(obs, action)
        return action, q1, q2

    def act(self, obs, rng):
        action = self.actor(obs)
        return action

    def pi(self, obs):
        action = self.actor(obs)
        return action

    def q(self, obs, action):
        q1 = self.q1(obs, action)
        q2 = self.q2(obs, action)
        return q1, q2
