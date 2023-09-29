import jax
import chex
import dataclasses
from flax import struct
from jax import numpy as jnp
from functools import partial
from typing import Union, Optional, Tuple
from gymnax.environments import environment


class GymnaxWrapper(environment.Environment):
    """Base wrapper for gymnax environments. Can be subclassed to modify the behavior of
    the wrapped environment.
    The wrapper assumes that the wrapped environment did not overwrite the original
    `step` and `reset` methods.

    Args:
        environment (environment.Environment): The environment to wrap.
    """

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def unwrapped(self) -> environment.Environment:
        """Return the unwrapped environment."""
        return self.env

    @property
    def default_params(self) -> environment.EnvParams:
        return self.env.default_params

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        # return super().step(key, state, action, params)
        return self.env.step(key, state, action, params)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        # return super().reset(key, params)
        return self.env.reset(key, params)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: environment.EnvParams,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        return self.env.step_env(key, state, action, params)

    def reset_env(
        self, key: chex.PRNGKey, params: environment.EnvParams
    ) -> Tuple[chex.Array, environment.EnvState]:
        return self.env.reset_env(key, params)

    def get_obs(self, state: environment.EnvState) -> chex.Array:
        """Applies observation function to state."""
        return self.env.get_obs(state)

    def is_terminal(
        self, state: environment.EnvState, params: environment.EnvParams
    ) -> bool:
        """Check whether state transition is terminal."""
        return self.env.is_terminal(state, params)

    def discount(
        self, state: environment.EnvState, params: environment.EnvParams
    ) -> float:
        """Return a discount of zero if the episode has terminated."""
        return self.env.discount(state, params)

    @property
    def name(self) -> str:
        """Environment name."""
        return self.env.name

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.env.num_actions

    def action_space(self, params: environment.EnvParams):
        """Action space of the environment."""
        return self.env.action_space(params)

    def observation_space(self, params: environment.EnvParams):
        """Observation space of the environment."""
        return self.env.observation_space(params)

    def state_space(self, params: environment.EnvParams):
        """State space of the environment."""
        return self.env.state_space(params)


@struct.dataclass
class RMSState:
    mean: chex.Array
    var: chex.Array
    count: chex.Numeric
    env_state: environment.EnvState


def rms_update(rms_state, obs):
    """
    Adapted from https://github.com/openai/gym/blob/master/gym/wrappers/normalize.py
    """
    delta = obs - rms_state.mean
    tot_count = rms_state.count + 1

    new_mean = rms_state.mean + delta / tot_count
    m_a = rms_state.var * rms_state.count
    m_b = 0
    M2 = m_a + m_b + delta ** 2 * rms_state.count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return rms_state.replace(mean=new_mean, var=new_var, count=new_count)


class NormalizeObservations(GymnaxWrapper):
    """
    Note that this wrapper is not suitable for vectorized environments. When vmapping
    over the step function, every vmapped environment will have its own normalization
    statistics. In general, a good implementation should solve this within the training
    algorithm, but this in not implemented in pureRL.
    """
    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        """Performs step transitions in the environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params

        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(
            key, state.env_state, action, params
        )
        obs_re, state_re = self.reset_env(key_reset, params)

        # Auto-reset environment based on termination
        # Only reset env_state if done, not other RMSState fields
        env_state = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = jax.lax.select(done, obs_re, obs_st)
        state = state.replace(env_state=env_state)
        obs, state = self.normalize(obs, state)

        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        """Performs resetting of environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        obs, state = self.reset_env(key, params)
        state = RMSState(
            mean=jnp.zeros_like(obs),
            var=jnp.ones_like(obs),
            count=1e-8,
            env_state=state,
        )
        obs, state = self.normalize(obs, state)
        return obs, state

    def normalize(self, obs, state):
        state = rms_update(state, obs)
        obs = (obs - state.mean) / jnp.sqrt(state.var + 1e-8)
        return obs, state
