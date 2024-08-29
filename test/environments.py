import chex
import jax
from flax.struct import PyTreeNode
from gymnax.environments import spaces
from gymnax.environments.environment import Environment
from jax import numpy as jnp

"""
Test environments to debug RL algorithms as in
https://andyljones.com/posts/rl-debugging.html
"""


class TestEnvState1(PyTreeNode):
    pass


class TestEnvParams1(PyTreeNode):
    max_steps_in_episode: int = 1


class TestEnv1Continuous(Environment):
    """ One action, zero observation, one timestep long, +1 reward every timestep """
    def __init__(self):
        super().__init__()
        self.obs_shape = (1,)

    @property
    def default_params(self):
        return TestEnvParams1()

    def step_env(self, key, state, action, params):
        return self.get_obs(state), state, 1, self.is_terminal(state, params), {}

    def reset_env(self, key, params):
        state = TestEnvState1()
        return self.get_obs(state), state

    def get_obs(self, state):
        return jnp.array([0])

    def is_terminal(self, state, params):
        return True

    @property
    def name(self):
        return "TestEnv1"

    def num_actions(self):
        return 1

    def action_space(self, params):
        return spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32)

    def observation_space(self, params):
        return spaces.Box(low=0, high=0, shape=(1,), dtype=jnp.float32)


class TestEnv1Discrete(Environment):
    """ One action, zero observation, one timestep long, +1 reward every timestep """
    def __init__(self):
        super().__init__()
        self.obs_shape = (1,)

    @property
    def default_params(self):
        return TestEnvParams1()

    def step_env(self, key, state, action, params):
        return self.get_obs(state), state, 1, self.is_terminal(state, params), {}

    def reset_env(self, key, params):
        state = TestEnvState1()
        return self.get_obs(state), state

    def get_obs(self, state):
        return jnp.array([0])

    def is_terminal(self, state, params):
        return True

    @property
    def name(self):
        return "TestEnv1"

    def num_actions(self):
        return 1

    def action_space(self, params):
        return spaces.Discrete(1)

    def observation_space(self, params):
        return spaces.Box(low=0, high=0, shape=(1,), dtype=jnp.float32)


class TestEnvState2(PyTreeNode):
    obs: chex.Scalar


class TestEnvParams2(PyTreeNode):
    max_steps_in_episode: int = 1


class TestEnv2Continuous(Environment):
    """
    One action, random +1/-1 observation, one timestep long, obs-dependent +1/-1 reward
    every time
    """

    def __init__(self):
        super().__init__()
        self.obs_shape = (1,)

    @property
    def default_params(self):
        return TestEnvParams2()

    def step_env(self, key, state, action, params):
        rand = jax.random.uniform(key, minval=-1, maxval=1, shape=(1,))
        obs = state.obs
        next_state = state.replace(obs=rand)
        next_obs = self.get_obs(state)
        reward = obs.squeeze()
        return next_obs, next_state, reward, self.is_terminal(next_state, params), {}

    def reset_env(self, key, params):
        obs = jax.random.uniform(key, minval=-1, maxval=1, shape=(1,))
        state = TestEnvState2(obs=obs)
        return self.get_obs(state), state

    def get_obs(self, state):
        return state.obs

    def is_terminal(self, state, params):
        return True

    @property
    def name(self):
        return "TestEnv2"

    def num_actions(self):
        return 1

    def action_space(self, params):
        return spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32)

    def observation_space(self, params):
        return spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32)


class TestEnv2Discrete(Environment):
    """
    One action, random +1/-1 observation, one timestep long, obs-dependent +1/-1 reward
    every time
    """

    def __init__(self):
        super().__init__()
        self.obs_shape = (1,)

    @property
    def default_params(self):
        return TestEnvParams2()

    def step_env(self, key, state, action, params):
        rand = jax.random.uniform(key, minval=-1, maxval=1, shape=(1,))
        obs = state.obs
        next_state = state.replace(obs=rand)
        next_obs = self.get_obs(state)
        reward = obs.squeeze()
        return next_obs, next_state, reward, self.is_terminal(next_state, params), {}

    def reset_env(self, key, params):
        obs = jax.random.uniform(key, minval=-1, maxval=1, shape=(1,))
        state = TestEnvState2(obs=obs)
        return self.get_obs(state), state

    def get_obs(self, state):
        return state.obs

    def is_terminal(self, state, params):
        return True

    @property
    def name(self):
        return "TestEnv2"

    def num_actions(self):
        return 1

    def action_space(self, params):
        return spaces.Discrete(1)

    def observation_space(self, params):
        return spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32)


class TestEnvState3(PyTreeNode):
    t: int = 0


class TestEnvParams3(PyTreeNode):
    max_steps_in_episode: int = 100


class TestEnv3Continuous(Environment):
    """
    One action, zero-then-one observation, two timesteps long, +1 reward at the end
    """
    def __init__(self):
        super().__init__()
        self.obs_shape = (1,)

    @property
    def default_params(self):
        return TestEnvParams3()

    def step_env(self, key, state, action, params):
        t = (state.t + 1) % 3
        state = state.replace(t=t)
        reward = jax.lax.select(state.t >= 2, 1, 0)
        return self.get_obs(state), state, reward, self.is_terminal(state, params), {}

    def reset_env(self, key, params):
        state = TestEnvState3()
        return self.get_obs(state), state

    def get_obs(self, state):
        return jnp.array([2 * state.t - 1])

    def is_terminal(self, state, params):
        return state.t >= 2

    @property
    def name(self):
        return "TestEnv3"

    def num_actions(self):
        return 1

    def action_space(self, params):
        return spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32)

    def observation_space(self, params):
        return spaces.Box(low=0, high=1, shape=(1,), dtype=jnp.float32)


class TestEnv3Discrete(Environment):
    """
    One action, zero-then-one observation, two timesteps long, +1 reward at the end
    """
    def __init__(self):
        super().__init__()
        self.obs_shape = (1,)

    @property
    def default_params(self):
        return TestEnvParams3()

    def step_env(self, key, state, action, params):
        t = (state.t + 1) % 3
        state = state.replace(t=t)
        reward = jax.lax.select(state.t >= 2, 1, 0)
        return self.get_obs(state), state, reward, self.is_terminal(state, params), {}

    def reset_env(self, key, params):
        state = TestEnvState3()
        return self.get_obs(state), state

    def get_obs(self, state):
        return jnp.array([2 * state.t - 1])

    def is_terminal(self, state, params):
        return state.t >= 2

    @property
    def name(self):
        return "TestEnv3"

    def num_actions(self):
        return 1

    def action_space(self, params):
        return spaces.Discrete(1)

    def observation_space(self, params):
        return spaces.Box(low=0, high=1, shape=(1,), dtype=jnp.float32)


class TestEnvState4(PyTreeNode):
    pass


class TestEnvParams4(PyTreeNode):
    max_steps_in_episode: int = 1


class TestEnv4Continuous(Environment):
    """
    Two actions, zero observation, one timestep long, action-dependent reward
    """
    def __init__(self):
        super().__init__()
        self.obs_shape = (1,)

    @property
    def default_params(self):
        return TestEnvParams4()

    def step_env(self, key, state, action, params):
        reward = action.squeeze()
        return self.get_obs(state), state, reward, self.is_terminal(state, params), {}

    def reset_env(self, key, params):
        state = TestEnvState4()
        return self.get_obs(state), state

    def get_obs(self, state):
        return jnp.array([0])

    def is_terminal(self, state, params):
        return True

    @property
    def name(self):
        return "TestEnv4"

    def num_actions(self):
        return 1

    def action_space(self, params):
        return spaces.Box(low=-2, high=2, shape=(1,), dtype=jnp.float32)

    def observation_space(self, params):
        return spaces.Box(low=0, high=0, shape=(1,), dtype=jnp.float32)


class TestEnv4Discrete(Environment):
    """
    Two actions, zero observation, one timestep long, action-dependent reward
    """
    def __init__(self):
        super().__init__()
        self.obs_shape = (1,)

    @property
    def default_params(self):
        return TestEnvParams4()

    def step_env(self, key, state, action, params):
        reward = action.squeeze()
        return self.get_obs(state), state, reward, self.is_terminal(state, params), {}

    def reset_env(self, key, params):
        state = TestEnvState4()
        return self.get_obs(state), state

    def get_obs(self, state):
        return jnp.array([0])

    def is_terminal(self, state, params):
        return True

    @property
    def name(self):
        return "TestEnv4"

    def num_actions(self):
        return 1

    def action_space(self, params):
        return spaces.Discrete(2)

    def observation_space(self, params):
        return spaces.Box(low=0, high=0, shape=(1,), dtype=jnp.float32)


class TestEnvState5(PyTreeNode):
    obs: chex.Scalar


class TestEnvParams5(PyTreeNode):
    max_steps_in_episode: int = 1


class TestEnv5Continuous(Environment):
    """
    Two actions, random +1/-1 observation, one timestep long, action-and-obs dependent
    +1/-1 reward
    """
    def __init__(self):
        super().__init__()
        self.obs_shape = (1,)

    @property
    def default_params(self):
        return TestEnvParams5()

    def step_env(self, key, state, action, params):
        obs = self.get_obs(state)
        reward = - jnp.abs(action - obs).squeeze()
        state = state.replace(
            obs=jax.random.uniform(key, minval=-1, maxval=1, shape=(1,))
        )
        return obs, state, reward, self.is_terminal(state, params), {}

    def reset_env(self, key, params):
        state = TestEnvState5(
            obs=jax.random.uniform(key, minval=-1, maxval=1, shape=(1,))
        )
        return self.get_obs(state), state

    def get_obs(self, state):
        return state.obs

    def is_terminal(self, state, params):
        return True

    @property
    def name(self):
        return "TestEnv5"

    def num_actions(self):
        return 1

    def action_space(self, params):
        return spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32)

    def observation_space(self, params):
        return spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32)


class TestEnv5Discrete(Environment):
    """
    Two actions, random +1/-1 observation, one timestep long, action-and-obs dependent
    +1/-1 reward
    """
    def __init__(self):
        super().__init__()
        self.obs_shape = (1,)

    @property
    def default_params(self):
        return TestEnvParams5()

    def step_env(self, key, state, action, params):
        obs = self.get_obs(state)
        action = 2 * action - 1
        reward = - jnp.abs(action - obs).squeeze()
        state = state.replace(obs=2 * jax.random.bernoulli(key) - 1)
        return obs, state, reward, self.is_terminal(state, params), {}

    def reset_env(self, key, params):
        state = TestEnvState5(obs=2 * jax.random.bernoulli(key) - 1)
        return self.get_obs(state), state

    def get_obs(self, state):
        return state.obs

    def is_terminal(self, state, params):
        return True

    @property
    def name(self):
        return "TestEnv5"

    def num_actions(self):
        return 1

    def action_space(self, params):
        return spaces.Discrete(2)

    def observation_space(self, params):
        return spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32)
