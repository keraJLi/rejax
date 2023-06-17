import jax
import chex
from jax import numpy as jnp
from flax.struct import PyTreeNode
from gymnax.environments import spaces
from gymnax.environments.environment import Environment

class TestEnvState1(PyTreeNode):
    pass

class TestEnvParams1(PyTreeNode):
    pass

class TestEnv1(Environment):
    def __init__(self):
        super().__init__()
        self.obs_shape = (1, )
    
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
        return spaces.Box(low=-1, high=1, shape=(1, ), dtype=jnp.float32)
    
    def observation_space(self, params):
        return spaces.Box(low=-1, high=1, shape=(1, ), dtype=jnp.float32)
    
    def state_space(self, params):
        return spaces.Discrete(0)



class TestEnvState2(PyTreeNode):
    obs: chex.Scalar

class TestEnvParams2(PyTreeNode):
    pass

class TestEnv2(Environment):
    def __init__(self):
        super().__init__()
        self.obs_shape = (1, )
    
    def default_params(self):
        return TestEnvParams2()

    def step_env(self, key, state, action, params):
        rand = jax.random.uniform(key, minval=-1, maxval=1, shape=(1, ))
        obs = state.obs
        next_state = state.replace(obs=rand)
        next_obs = self.get_obs(state)
        reward = obs.squeeze()
        return next_obs, next_state, reward, self.is_terminal(next_state, params), {}
    
    def reset_env(self, key, params):
        state = TestEnvState2(obs=jax.random.uniform(key, minval=-1, maxval=1, shape=(1, )))
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
        return spaces.Box(low=-1, high=1, shape=(1, ), dtype=jnp.float32)
    
    def observation_space(self, params):
        return spaces.Box(low=-1, high=1, shape=(1, ), dtype=jnp.float32)
    
    def state_space(self, params):
        return spaces.Box(low=-1, high=1, shape=(1, ), dtype=jnp.float32)


class TestEnvState3(PyTreeNode):
    t: int = 0

class TestEnvParams3(PyTreeNode):
    pass

class TestEnv3(Environment):
    def __init__(self):
        super().__init__()
        self.obs_shape = (1, )
    
    def default_params(self):
        return TestEnvParams3()

    def step_env(self, key, state, action, params):
        t = (state.t + 1) % 3
        state = state.replace(t=t)
        reward = jax.lax.select(state.t == 2, 1, 0)
        return self.get_obs(state), state, reward, self.is_terminal(state, params), {}
        
    def reset_env(self, key, params):
        state = TestEnvState3()
        return self.get_obs(state), state
    
    def get_obs(self, state):
        return 2 * state.t - 1
    
    def is_terminal(self, state, params):
        return state.t == 2
    
    @property
    def name(self):
        return "TestEnv3"
    
    def num_actions(self):
        return 1
    
    def action_space(self, params):
        return spaces.Box(low=-1, high=1, shape=(1, ), dtype=jnp.float32)
    
    def observation_space(self, params):
        return spaces.Box(low=-1, high=1, shape=(1, ), dtype=jnp.float32)  # .Discrete(2)
    
    def state_space(self, params):
        return spaces.Box(low=-1, high=1, shape=(1, ), dtype=jnp.float32)


class TestEnvState4(PyTreeNode):
    pass

class TestEnvParams4(PyTreeNode):
    pass

class TestEnv4(Environment):
    def __init__(self):
        super().__init__()
        self.obs_shape = (1, )
    
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
        return spaces.Box(low=-1, high=1, shape=(1, ), dtype=jnp.float32)
    
    def observation_space(self, params):
        return spaces.Box(low=-1, high=1, shape=(1, ), dtype=jnp.float32)  # .Discrete(2)
    
    def state_space(self, params):
        return spaces.Box(low=-1, high=1, shape=(1, ), dtype=jnp.float32)



class TestEnvState5(PyTreeNode):
    obs: chex.Scalar

class TestEnvParams5(PyTreeNode):
    pass

class TestEnv5(Environment):
    def __init__(self):
        super().__init__()
        self.obs_shape = (1, )
    
    def default_params(self):
        return TestEnvParams5()

    def step_env(self, key, state, action, params):
        obs = self.get_obs(state)
        reward = (action - obs).squeeze()
        state = state.replace(obs=jax.random.uniform(key, minval=-1, maxval=1, shape=(1, )))
        return obs, state, reward, self.is_terminal(state, params), {}
        
    def reset_env(self, key, params):
        state = TestEnvState5(obs=jax.random.uniform(key, minval=-1, maxval=1, shape=(1, )))
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
        return spaces.Box(low=-1, high=1, shape=(1, ), dtype=jnp.float32)
    
    def observation_space(self, params):
        return spaces.Box(low=-1, high=1, shape=(1, ), dtype=jnp.float32)  # .Discrete(2)
    
    def state_space(self, params):
        return spaces.Box(low=-1, high=1, shape=(1, ), dtype=jnp.float32)
