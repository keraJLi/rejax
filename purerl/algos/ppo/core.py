import chex
import distrax
from flax import struct
from flax import linen as nn
from jax import numpy as jnp
from typing import Any, Callable
from flax.linen.initializers import constant
from gymnax.environments.environment import Environment


class PPOConfig(struct.PyTreeNode):
    # Non-static parameters
    env_params: Any
    gamma: chex.Scalar
    gae_lambda: chex.Scalar
    clip_eps: chex.Scalar
    vf_coef: chex.Scalar
    ent_coef: chex.Scalar

    # Static parameters
    total_timesteps: int = struct.field(pytree_node=False)
    eval_freq: int = struct.field(pytree_node=False)
    agent: nn.Module = struct.field(pytree_node=False)
    env: Environment = struct.field(pytree_node=False)
    evaluate: Callable = struct.field(pytree_node=False)
    num_envs: int = struct.field(pytree_node=False)
    num_steps: int = struct.field(pytree_node=False)
    num_epochs: int = struct.field(pytree_node=False)
    num_minibatches: int = struct.field(pytree_node=False)
    learning_rate: chex.Scalar = struct.field(pytree_node=False)
    max_grad_norm: chex.Scalar = struct.field(pytree_node=False)

    @property
    def minibatch_size(self):
        assert self.num_steps % self.num_minibatches == 0
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

        agent_kwargs = config.pop("agent_kwargs", None) or {}
        agent_kwargs["discrete"] = discrete

        if discrete:
            action_dim = env.action_space(env_params).n
        else:
            action_dim = np.prod(env.action_space(env_params).shape)

        activation = agent_kwargs.pop("activation", "relu")
        agent_kwargs["activation"] = getattr(nn, activation)

        agent = PPOAgent(action_dim, **agent_kwargs)
        evaluate = make_evaluate(env, env_params, 200)

        return cls(
            env=env, env_params=env_params, agent=agent, evaluate=evaluate, **config
        )


class PPOAgent(nn.Module):
    action_dim: int
    discrete: bool
    activation: Callable = nn.tanh

    def setup(self):
        self.value_ = [nn.Dense(64), nn.Dense(64), nn.Dense(1)]
        self.action_ = [nn.Dense(64), nn.Dense(64), nn.Dense(self.action_dim)]
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
