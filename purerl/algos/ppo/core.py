import chex
import distrax
from jax import numpy as jnp
from flax import struct, linen as nn
from typing import Any, Callable, Tuple, Union
from gymnax.environments.environment import Environment
from flax.linen.initializers import constant, orthogonal


class PPOConfig(struct.PyTreeNode):
    # Non-static parameters
    env_params: Any
    gamma: chex.Scalar
    gae_lambda: chex.Scalar
    clip_eps: chex.Scalar
    vf_coef: chex.Scalar
    ent_coef: chex.Scalar
    learning_rate: chex.Scalar

    # Static parameters
    total_timesteps: int = struct.field(pytree_node=False)
    eval_freq: int = struct.field(pytree_node=False)
    agent: nn.Module = struct.field(pytree_node=False)
    env: Environment = struct.field(pytree_node=False)
    eval_callback: Callable = struct.field(pytree_node=False)
    num_envs: int = struct.field(pytree_node=False)
    num_steps: int = struct.field(pytree_node=False)
    num_epochs: int = struct.field(pytree_node=False)
    num_minibatches: int = struct.field(pytree_node=False)
    max_grad_norm: chex.Scalar = struct.field(pytree_node=False)
    normalize_observations: bool = struct.field(pytree_node=False, default=False)
    skip_initial_evaluation: bool = struct.field(pytree_node=False, default=False)

    @property
    def minibatch_size(self):
        assert (self.num_envs * self.num_steps) % self.num_minibatches == 0
        return (self.num_envs * self.num_steps) // self.num_minibatches

    @classmethod
    def from_dict(cls, config):
        import gymnax
        import numpy as np
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

        agent_kwargs = config.pop("agent_kwargs", {})
        agent_kwargs["discrete"] = discrete

        if discrete:
            action_dim = env.action_space(env_params).n
        else:
            action_dim = np.prod(env.action_space(env_params).shape)
            action_range = (
                env.action_space(env_params).low,
                env.action_space(env_params).high,
            )
            agent_kwargs["action_range"] = action_range

        activation = agent_kwargs.pop("activation", "relu")
        agent_kwargs["activation"] = getattr(nn, activation)

        # Convert hidden layer sizes to tuple
        hidden_layer_sizes = agent_kwargs.pop("hidden_layer_sizes", None)
        if hidden_layer_sizes is not None:
            agent_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

        agent = PPOAgent(action_dim, **agent_kwargs)
        evaluate = make_evaluate(env, env_params, 200)

        return cls(
            env=env,
            env_params=env_params,
            agent=agent,
            eval_callback=evaluate,
            **config,
        )


class PPOAgent(nn.Module):
    action_dim: int
    discrete: bool
    action_range: Union[Tuple[float, float], None] = None
    hidden_layer_sizes: Tuple[int] = (64, 64)
    activation: Callable = nn.tanh

    def setup(self):
        if not self.discrete and self.action_range is None:
            raise ValueError(
                "Continuous action space requires action_range to be specified"
            )

        value_ = [
            nn.Dense(s, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0))
            for s in self.hidden_layer_sizes
        ]
        value_.append(nn.Dense(1, kernel_init=orthogonal(0.01), bias_init=constant(0)))
        self.value_ = value_

        action_ = [
            nn.Dense(s, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0))
            for s in self.hidden_layer_sizes
        ]
        action_.append(
            nn.Dense(
                self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0)
            )
        )
        self.action_ = action_

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
        if not self.discrete:
            action = jnp.clip(action, self.action_range[0], self.action_range[1])
        return action

    def action_log_prob(self, obs, rng):
        action_dist = self._action_dist(obs)
        action = action_dist.sample(seed=rng)
        return action, action_dist.log_prob(action)

    def log_prob_entropy(self, obs, action):
        action_dist = self._action_dist(obs)
        return action_dist.log_prob(action), action_dist.entropy()
