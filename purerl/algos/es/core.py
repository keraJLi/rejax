from typing import Any, Callable, Sequence

import chex
import jax
from evosax import Strategies
from flax import linen as nn
from flax import struct
from gymnax.environments.environment import Environment
from jax import numpy as jnp

# TODO: Support for custom evo_params


class ESConfig(struct.PyTreeNode):
    # Non-static parameters
    env_params: Any

    # Static parameters
    num_generations: int = struct.field(pytree_node=False)
    eval_freq: int = struct.field(pytree_node=False)
    agent: nn.Module = struct.field(pytree_node=False)
    env: Environment = struct.field(pytree_node=False)
    eval_callback: Callable = struct.field(pytree_node=False)
    strategy: Callable = struct.field(pytree_node=False)
    strategy_params: chex.ArrayTree = struct.field(pytree_node=False)
    num_rollouts: int = struct.field(pytree_node=False)

    @property
    def minibatch_size(self):
        assert self.num_steps % self.num_minibatches == 0
        return (self.num_envs * self.num_steps) // self.num_minibatches

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

        from purerl.brax2gymnax import Brax2GymnaxEnv
        from purerl.evaluate import make_evaluate

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

        if discrete:
            action_dim = env.action_space(env_params).n
        else:
            action_dim = np.prod(env.action_space(env_params).shape)

        agent_kwargs = config.pop("agent_kwargs", None) or {}
        if "features" in agent_kwargs:
            agent_kwargs["features"] = tuple(agent_kwargs["features"])
        if "activation" in agent_kwargs:
            agent_kwargs["activation"] = getattr(nn, agent_kwargs["activation"])

        agent = MLPPolicy(action_dim, discrete, **agent_kwargs)
        evaluate = make_evaluate(env, env_params, 200)

        strategy_kwargs = config.pop("strategy_kwargs", None) or {}
        obs, _ = env.reset(jax.random.PRNGKey(0), env_params)
        pholder_params = agent.init(jax.random.PRNGKey(0), obs)
        strategy_kwargs["pholder_params"] = pholder_params
        strategy_kwargs["maximize"] = True
        strategy_kwargs["n_devices"] = 1  # TODO: How to behave with multiple devices?

        strategy = Strategies[config.pop("strategy")](**strategy_kwargs)

        return cls(
            env=env,
            env_params=env_params,
            agent=agent,
            eval_callback=evaluate,
            strategy=strategy,
            strategy_params=strategy.default_params,
            **config
        )


class MLPPolicy(nn.Module):
    action_dim: int
    discrete: bool
    use_bias: bool = False
    features: Sequence[int] = ()
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, obs):
        x = obs.reshape(-1)
        for feature in self.features:
            x = nn.Dense(feature, use_bias=self.use_bias)(x)
            x = self.activation(x)
        action = nn.Dense(self.action_dim, use_bias=self.use_bias)(x)

        if self.discrete:
            action = jnp.argmax(action)

        return action

    def act(self, obs, rng):
        return self(obs)
