from functools import partial

import envpool
import jax
from flax import struct
from gymnax.environments import environment, spaces


def create_envpool(env_name, **kwargs):
    env = envpool.make(env_name, "gymnasium", **kwargs)
    env = Envpool2GymnaxEnv(env)
    return env, env.default_params


# CAUTION: Passing params with a different value than on init has no effect
@struct.dataclass
class EnvParams:
    num_envs: int
    max_steps_in_episode: int


class Envpool2GymnaxEnv(environment.Environment):
    def __init__(self, env):
        env.reset()

        self.env = env
        self.num_envs = env.spec.config.num_envs
        self.max_steps_in_episode = env.spec.config.max_episode_steps

        handle, recv, send, step = env.xla()
        self._handle = handle
        self._recv = recv
        self._send = send
        self._step = step

    @property
    def default_params(self):
        return EnvParams(
            num_envs=self.num_envs, max_steps_in_episode=self.max_steps_in_episode
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        if params is None:
            params = self.default_params

        state, (obs, reward, term, trunc, info) = self._step(state, action)
        done = term | trunc
        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        if params is None:
            params = self.default_params

        state, (obs, reward, term, trunc, info) = self._recv(self._handle)
        return obs, state

    def get_obs(self, state):
        state, (obs, reward, term, trunc, info) = self._recv(state)
        return obs

    def is_terminal(self, state):
        return state.done.astype(bool)

    @property
    def name(self):
        return self.env.unwrapped.__class__.__name__

    def action_space(self, params):
        return spaces.Discrete(self.env.action_space.n)

    def observation_space(self, params):
        return spaces.Box(
            low=self.env.observation_space.low,
            high=self.env.observation_space.high,
            shape=self.env.observation_space.low.shape,
        )

    @property
    def num_actions(self) -> int:
        return self.env.action_space.n

    def __deepcopy__(self, memo):
        import warnings
        from copy import copy

        warnings.warn(
            f"Trying to deepcopy {type(self).__name__}, which contains an envpool env. "
            "Envpool envs throw an error when deepcopying, so a shallow copy is "
            "returned."
        )
        return copy(self)
