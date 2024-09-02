import envpool
import jax
from jax import numpy as jnp

from rejax.evaluate import EvalState


class EnvpoolEvaluator:
    def __init__(self, env_name, env_params):
        self.env = envpool.make(env_name, "gymnasium", **env_params)
        self.num_envs = self.env.spec.config.num_envs
        self.max_steps_in_episode = self.env.spec.config.max_episode_steps

        handle, recv, send, step = self.env.xla()
        self._handle = handle
        self._recv = recv
        self._send = send
        self._step = step

    def evaluate(self, act, rng):
        def step(state):
            rng, rng_act = jax.random.split(state.rng)
            action = jax.vmap(act, in_axes=(0, None))(state.last_obs, rng_act)
            env_state, (obs, reward, term, trunc, info) = self._step(
                state.env_state, action
            )

            done = state.done | term | trunc
            reward = (1 - state.done) * reward.squeeze()
            step_length = (1 - state.done) * jnp.ones_like(done)

            state = EvalState(
                rng=rng,
                env_state=env_state,
                last_obs=obs,
                done=done,
                return_=state.return_ + reward,
                length=state.length + step_length,
            )
            return state

        obs_shape_dtype = jax.ShapeDtypeStruct(
            shape=(self.num_envs, *self.env.observation_space.shape),
            dtype=self.env.observation_space.dtype,
        )
        obs = jax.experimental.io_callback(lambda: self.env.reset()[0], obs_shape_dtype)

        zeros = jnp.zeros(self.num_envs)
        state = EvalState(
            rng, self._handle, obs, done=zeros.astype(bool), return_=zeros, length=zeros
        )
        state = jax.lax.while_loop(
            lambda s: jnp.logical_and(
                jnp.any(s.length < self.max_steps_in_episode),
                jnp.any(jnp.logical_not(s.done)),
            ),
            step,
            state,
        )
        return state.length, state.return_
