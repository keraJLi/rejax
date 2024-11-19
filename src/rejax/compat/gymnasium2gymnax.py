import warnings
from dataclasses import asdict
from functools import cached_property

import gymnasium as gym
import jax
import numpy as np
from flax import struct
from gymnasium import Env as GymnasiumEnv
from gymnasium import spaces as gymnasium_spaces
from gymnasium.wrappers import Autoreset
from gymnax.environments import spaces as gymnax_spaces
from gymnax.environments.environment import Environment as GymnaxEnv
from gymnax.environments.environment import EnvParams as GymnaxEnvParams
from jax import ShapeDtypeStruct
from jax import numpy as jnp
from jax.dtypes import canonicalize_dtype

# Gymnasium currently has a functional API, but only few environments are supported
# (e.g. BlackJack, but not classic control). We want to use jax.pure_callback to call
# their methods, so we need to make their API functional. We do this by explicitly
# listing the attributes that are changed during the environment step, and overwriting
# them before stepping in our callback. We seperately handle the rng key.
# All of this nonsense means we lose all vectorization benefits from JAX, but we can
# extend the compat layer to support vectorized environments using multiprocessing
# pools in the future, or using gymnasium VectorizedEnvs with vmap_method broadcast_all


def create_gymnasium(env_name, **kwargs):
    env = gym.make(env_name, **kwargs)
    env = Autoreset(env)
    env = Gymnasium2GymnaxEnv(env)
    return env, env.default_params


# The state attributes are the ones that are changed during the step function
# Before stepping, we overwrite them with values of the state to step from
# The random key is sometimes changed as well, we take care of it seperately
STATE_ATTRIBUTES = {
    # Classic control
    "CartPole-v1": ("state",),
    "MountainCar-v0": ("state",),
    "Pendulum-v0": ("state",),
    "Acrobot-v1": ("state",),
    "MountainCarContinuous-v0": ("state",),
}

STATE_SEED_NAME = "_rejax_seed"


def load_state(env, state):
    # not very pretty...
    # gymnasium.Env.np_random_seed is a property without a setter. Maybe it would be
    # better to add one to our env object?
    env._np_random_seed = state[STATE_SEED_NAME]

    for attr in STATE_ATTRIBUTES[env.spec.id]:
        setattr(env.unwrapped, attr, state[attr])


def get_state(env):
    state = {}
    state[STATE_SEED_NAME] = env.np_random_seed

    for attr in STATE_ATTRIBUTES[env.spec.id]:
        state[attr] = getattr(env.unwrapped, attr)

    return state


def step_gymnasium_env(env: GymnasiumEnv, state, action):
    action = np.array(action)

    load_state(env, state)
    obs, reward, term, trunc, info = env.step(action)
    state = get_state(env)

    done = term or trunc
    info = {}
    return obs, state, reward, done, info


def reset_gymnasium_env(env: GymnasiumEnv, seed: int):
    seed = np.array(seed).item()
    obs, info = env.reset(seed=seed)
    state = get_state(env)
    return obs, state


def gymnasium_space_to_gymnax_space(gymnasium_space):
    if isinstance(gymnasium_space, gymnasium_spaces.Box):
        return gymnax_spaces.Box(
            low=gymnasium_space.low,
            high=gymnasium_space.high,
            shape=gymnasium_space.shape,
            dtype=gymnasium_space.dtype,
        )
    elif isinstance(gymnasium_space, gymnasium_spaces.Discrete):
        return gymnax_spaces.Discrete(gymnasium_space.n)
    elif isinstance(gymnasium_space, gymnasium_spaces.Dict):
        return gymnax_spaces.Dict(
            {
                k: gymnasium_space_to_gymnax_space(v)
                for k, v in gymnasium_space.spaces.items()
            }
        )


def num_entries(space):
    if isinstance(space, gymnax_spaces.Discrete):
        return space.n
    elif isinstance(space, gymnax_spaces.Box):
        return jnp.prod(jnp.array(space.shape))
    elif isinstance(space, gymnax_spaces.Dict):
        return sum(num_entries(subspace) for subspace in space.spaces.values())
    raise ValueError(f"Unsupported space {space}")


def shape_dtype_from_pytree(pytree):
    def get_shape_dtype(obj):
        if isinstance(obj, (bool, int, float)):
            dtype = canonicalize_dtype(type(obj))
            return ShapeDtypeStruct((), dtype)
        dtype = canonicalize_dtype(obj.dtype)
        return ShapeDtypeStruct(obj.shape, dtype)

    return jax.tree.map(get_shape_dtype, pytree)


class Gymnasium2GymnaxEnv(GymnaxEnv):
    def __init__(self, env: GymnasiumEnv):
        warnings.warn(
            "Wrapper for Gymnasium2GymnaxEnv is experimental, and may not correctly "
            "handle rng keys. Please report any issues by creating an issue. "
        )
        self.env = env
        self.env.reset()

    @property
    def default_params(self):
        spec = self.env.spec
        param_dict = asdict(spec)
        param_dict["max_steps_in_episode"] = param_dict.pop("max_episode_steps", 1000)
        param_cls = type("Gymnasium2GymnaxEnvParams", (), param_dict)
        param_cls = struct.dataclass(param_cls)
        return param_cls()

    @cached_property
    def reset_shape_dtype(self):
        seed = 0
        reset_ph = reset_gymnasium_env(self.env, seed)
        return shape_dtype_from_pytree(reset_ph)

    @cached_property
    def step_shape_dtype(self):
        state = get_state(self.env)
        action = self.env.action_space.sample()
        load_state(self.env, state)

        step_ph = step_gymnasium_env(self.env, state, action)
        return shape_dtype_from_pytree(step_ph)

    def step_env(self, key, state, action, params):
        def step(state, action):
            return step_gymnasium_env(self.env, state, action)

        seed = jax.random.randint(key, (), 0, 2**16)
        state[STATE_SEED_NAME] = seed
        return jax.pure_callback(
            step,
            self.step_shape_dtype,
            vmap_method="sequential",
            state=state,
            action=action,
        )

    def reset_env(self, key, params):
        def reset(seed):
            return reset_gymnasium_env(self.env, seed)

        seed = jax.random.randint(key, (), 0, 2**16)
        return jax.pure_callback(reset, self.reset_shape_dtype, seed=seed)

    def get_obs(self, state):
        raise NotImplementedError

    def is_terminal(self, state):
        return NotImplementedError

    @property
    def name(self):
        return self.env.unwrapped.__class__.__name__

    def action_space(self, params):
        return gymnasium_space_to_gymnax_space(self.env.action_space)

    def observation_space(self, params):
        return gymnasium_space_to_gymnax_space(self.env.observation_space)

    @property
    def num_actions(self) -> int:
        return num_entries(self.action_space(self.default_params))
