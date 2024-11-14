import dataclasses
from functools import partial

import chex
import jax
import jumanji
from gymnax.environments import spaces
from gymnax.environments.environment import Environment as GymnaxEnv
from gymnax.environments.environment import EnvParams
from jax import numpy as jnp
from jumanji.env import Environment as JumanjiEnv
from jumanji.specs import Array, BoundedArray, DiscreteArray
from jumanji.types import StepType


def create_jumanji(env_name, flatten_obs=True, **kwargs):
    env = jumanji.make(env_name, **kwargs)
    env = Jumanji2GymnaxEnv(env)
    if flatten_obs:
        env = FlattenObsWrapper(env)
    return env, env.default_params


def num_entries(space):
    if isinstance(space, spaces.Discrete):
        return space.num_categories
    elif isinstance(space, spaces.Box):
        return jnp.prod(jnp.array(space.shape))
    elif isinstance(space, spaces.Dict):
        return sum(num_entries(subspace) for subspace in space.spaces.values())
    raise ValueError(f"Unsupported space {space}")


def convert_spec(spec):
    if isinstance(spec, Array):
        return spaces.Box(
            low=-jnp.inf, high=jnp.inf, shape=spec.shape, dtype=spec.dtype
        )
    elif isinstance(spec, BoundedArray):
        return spaces.Box(
            low=spec.minimum, high=spec.maximum, shape=spec.shape, dtype=spec.dtype
        )
    elif isinstance(spec, DiscreteArray):
        return spaces.Discrete(num_categories=spec.num_values)
    return spaces.Dict({k: convert_spec(v) for k, v in spec._specs.items()})


def flatten_obs(obs):
    if isinstance(obs, (int, float, bool)):
        return jnp.array([obs], dtype=jnp.float32)
    elif isinstance(obs, chex.Array):
        return obs.reshape(-1).astype(jnp.float32)
    else:
        flat_obs = jax.tree.map(flatten_obs, obs)
        leaves = jax.tree.leaves(flat_obs)
        return jnp.concatenate(leaves)


def observation_to_dict(obs):
    if dataclasses.is_dataclass(obs):
        return dict(dataclasses.fields(obs))
    elif isinstance(obs, tuple):
        try:
            return obs._asdict()  # defined for namedtuples
        except:
            raise ValueError(
                f"I don't know how to convert observations of type {type(obs)}"
            )


class Jumanji2GymnaxEnv(GymnaxEnv):
    def __init__(self, env: JumanjiEnv):
        self.env = env
        self.max_steps_in_episode = getattr(env, "time_limit", 1000)

    @property
    def default_params(self):
        return EnvParams(max_steps_in_episode=self.max_steps_in_episode)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params):
        # We overwrite to tree map reset selection
        if params is None:
            params = self.default_params
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(key, state, action, params)
        obs_re, state_re = self.reset_env(key_reset, params)

        # Tree map over observation as well, to
        state, obs = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y),
            (state_re, obs_re),
            (state_st, obs_st),
        )
        return obs, state, reward, done, info

    def step_env(self, key, state, action, params):
        # Is this reasonable? Should we let Jumanji handle this?
        if "key" in dataclasses.fields(state):
            state = dataclasses.replace(state, key=key)

        state, obs = self.env.step(state, action)
        done = obs.step_type == StepType.LAST
        obs_obs = observation_to_dict(obs.observation)
        return obs_obs, state, obs.reward, done, obs.extras

    def reset_env(self, key, params):
        state, obs = self.env.reset(key)
        obs_obs = observation_to_dict(obs.observation)
        return obs_obs, state

    def get_obs(self, state):
        # We could implement it by including the obs in the state, but I will wait
        # until someone raises an issue...
        raise NotImplementedError

    def is_terminal(self, state):
        # see Jumanji2GymnaxEnv.get_obs
        raise NotImplementedError

    @property
    def name(self):
        return self.env.unwrapped.__class__.__name__

    def action_space(self, params):
        spec = self.env.action_spec
        return convert_spec(spec)

    def observation_space(self, params):
        spec = self.env.observation_spec
        # This is a little hacky since it accesses a private attribute, but the jumanji
        # specs do not expose the name of each spec. Using spec.generate_value is not an
        # option since we wouldn't know the minvals/maxvals of bounded arrays.
        return spaces.Dict({n: convert_spec(s) for n, s in spec._specs.items()})

    @property
    def num_entries(self) -> int:
        return num_entries(self.action_space(None))


class FlattenObsWrapper(GymnaxEnv):
    def __init__(self, env):
        self.env = env

    def __getattribute__(self, name):
        if name in ["env", "reset", "step", "observation_space", "get_obs"]:
            return super().__getattribute__(name)
        return self.env.__getattribute__(name)

    def observation_space(self, params):
        space = self.env.observation_space(params)
        size = num_entries(space)
        return spaces.Box(-jnp.inf, jnp.inf, (size,), dtype=jnp.float32)

    def get_obs(self, state):
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params):
        obs, state, reward, done, info = self.env.step(key, state, action, params)
        obs = flatten_obs(obs)
        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        obs, state = self.env.reset(key, params)
        obs = flatten_obs(obs)
        return obs, state
