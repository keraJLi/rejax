import warnings
from copy import copy

from brax.envs import Env as BraxEnv
from brax.envs import create
from flax import struct
from gymnax.environments import spaces
from gymnax.environments.environment import Environment as GymnaxEnv
from jax import numpy as jnp


def create_brax(env_name, **kwargs):
    env = create(env_name, **kwargs)
    env = Brax2GymnaxEnv(env)
    return env, env.default_params


@struct.dataclass
class EnvParams:
    # CAUTION: Passing params with a different value than on init has no effect
    max_steps_in_episode: int = 1000


class Brax2GymnaxEnv(GymnaxEnv):
    def __init__(self, env: BraxEnv):
        self.env = env
        self.max_steps_in_episode = env.episode_length

    @property
    def default_params(self):
        return EnvParams(max_steps_in_episode=self.max_steps_in_episode)

    def step_env(self, key, state, action, params):
        state = self.env.step(state, action)
        return state.obs, state, state.reward, state.done.astype(bool), state.info

    def reset_env(self, key, params):
        state = self.env.reset(key)
        return state.obs, state

    def get_obs(self, state):
        return state.obs

    def is_terminal(self, state):
        return state.done.astype(bool)

    @property
    def name(self):
        return self.env.unwrapped.__class__.__name__

    def action_space(self, params):
        # All brax evironments have action limit of -1 to 1
        return spaces.Box(low=-1, high=1, shape=(self.env.action_size,))

    def observation_space(self, params):
        # All brax evironments have observation limit of -inf to inf
        return spaces.Box(
            low=-jnp.inf, high=jnp.inf, shape=(self.env.observation_size,)
        )

    @property
    def num_actions(self) -> int:
        return self.env.action_size

    def __deepcopy__(self, memo):
        warnings.warn(
            f"Trying to deepcopy {type(self).__name__}, which contains a brax env. "
            "Brax envs throw an error when deepcopying, so a shallow copy is returned."
        )
        return copy(self)
