import chex
import jax
import numpy as np
from flax import struct
from jax import numpy as jnp
from optax import linear_schedule

from rejax.algos.algorithm import register_init
from rejax.buffers import ReplayBuffer


class EpsilonGreedyMixin(struct.PyTreeNode):
    eps_start: chex.Scalar = struct.field(pytree_node=True, default=1.0)
    eps_end: chex.Scalar = struct.field(pytree_node=True, default=0.05)
    exploration_fraction: chex.Scalar = struct.field(pytree_node=False, default=0.1)

    @property
    def epsilon_schedule(self):
        return linear_schedule(
            self.eps_start,
            self.eps_end,
            int(self.exploration_fraction * self.total_timesteps),
        )


class VectorizedEnvMixin(struct.PyTreeNode):
    num_envs: int = struct.field(pytree_node=False, default=1)

    @property
    def vmap_reset(self):
        return jax.vmap(self.env.reset, in_axes=(0, None))

    @property
    def vmap_step(self):
        return jax.vmap(self.env.step, in_axes=(0, 0, 0, None))

    @register_init
    def initialize_env_state(self, rng):
        rng, env_rng = jax.random.split(rng)
        obs, env_state = self.vmap_reset(
            jax.random.split(env_rng, self.num_envs), self.env_params
        )
        return {
            "env_state": env_state,
            "last_obs": obs,
            "global_step": 0,
            "last_done": jnp.zeros(self.num_envs, dtype=bool),
        }


class ReplayBufferMixin(VectorizedEnvMixin):
    buffer_size: int = struct.field(pytree_node=False, default=131_072)
    fill_buffer: int = struct.field(pytree_node=False, default=2_048)
    batch_size: int = struct.field(pytree_node=False, default=256)

    @register_init
    def initialize_replay_buffer(self, rng):
        buf = ReplayBuffer.empty(self.buffer_size, self.obs_space, self.action_space)
        return {"replay_buffer": buf}

    def train(self, rng=None, train_state=None):
        if train_state is None and rng is None:
            raise ValueError("Either train_state or rng must be provided")

        ts = train_state or self.init_state(rng)

        if not self.skip_initial_evaluation:
            initial_evaluation = self.eval_callback(self, ts, ts.rng)

        def eval_iteration(ts, unused):
            # Run a few trainig iterations
            ts = jax.lax.fori_loop(
                0,
                np.ceil(self.eval_freq / self.num_envs).astype(int),
                lambda _, ts: self.train_iteration(ts),
                ts,
            )

            # Run evaluation
            return ts, self.eval_callback(self, ts, ts.rng)

        ts, evaluation = jax.lax.scan(
            eval_iteration,
            ts,
            None,
            np.ceil(self.total_timesteps / self.eval_freq).astype(int),
        )

        if not self.skip_initial_evaluation:
            evaluation = jax.tree_map(
                lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
                initial_evaluation,
                evaluation,
            )

        return ts, evaluation


class OnPolicyMixin(VectorizedEnvMixin):
    num_envs: int = struct.field(pytree_node=False, default=64)  # overwrite default
    num_steps: int = struct.field(pytree_node=False, default=64)
    num_minibatches: int = struct.field(pytree_node=False, default=16)

    @property
    def minibatch_size(self):
        assert (self.num_envs * self.num_steps) % self.num_minibatches == 0
        return (self.num_envs * self.num_steps) // self.num_minibatches

    @property
    def iteration_size(self):
        return self.minibatch_size * self.num_minibatches

    def shuffle_and_split(self, data, rng):
        permutation = jax.random.permutation(rng, self.iteration_size)

        def _shuffle_and_split(x):
            x = x.reshape((self.iteration_size, *x.shape[2:]))
            x = jnp.take(x, permutation, axis=0)
            return x.reshape(self.num_minibatches, -1, *x.shape[1:])

        return jax.tree_util.tree_map(_shuffle_and_split, data)

    def train(self, rng=None, train_state=None):
        if train_state is None and rng is None:
            raise ValueError("Either train_state or rng must be provided")

        ts = train_state or self.init_state(rng)

        if not self.skip_initial_evaluation:
            initial_evaluation = self.eval_callback(self, ts, ts.rng)

        def eval_iteration(ts, unused):
            # Run a few training iterations
            iteration_steps = self.num_envs * self.num_steps
            num_iterations = np.ceil(self.eval_freq / iteration_steps).astype(int)
            ts = jax.lax.fori_loop(
                0,
                num_iterations,
                lambda _, ts: self.train_iteration(ts),
                ts,
            )

            # Run evaluation
            return ts, self.eval_callback(self, ts, ts.rng)

        num_evals = np.ceil(self.total_timesteps / self.eval_freq).astype(int)
        ts, evaluation = jax.lax.scan(eval_iteration, ts, None, num_evals)

        if not self.skip_initial_evaluation:
            evaluation = jax.tree_map(
                lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
                initial_evaluation,
                evaluation,
            )

        return ts, evaluation


class TargetNetworkMixin(struct.PyTreeNode):
    target_update_freq: int = struct.field(pytree_node=False, default=1)
    polyak: chex.Scalar = struct.field(pytree_node=True, default=0.99)

    def polyak_update(self, params, target_params):
        return jax.tree_map(
            lambda p, tp: tp * self.polyak + p * (1 - self.polyak),
            params,
            target_params,
        )


@struct.dataclass
class RMSState:
    mean: chex.Array
    var: chex.Array
    count: chex.Numeric

    @classmethod
    def create(cls, shape):
        return cls(
            mean=jnp.zeros(shape, dtype=jnp.float32),
            var=jnp.ones(shape, dtype=jnp.float32),
            count=1e-4,
        )


class NormalizeObservationsMixin(struct.PyTreeNode):
    normalize_observations: bool = struct.field(pytree_node=False, default=False)

    @register_init
    def initialize_rms_state(self, rng):
        obs_shape = (1, *self.env.observation_space(self.env_params).shape)
        return {"rms_state": RMSState.create(obs_shape)}

    def update_rms(self, rms_state, obs, batched=True):
        batch = obs if batched else jnp.expand_dims(obs, 0)

        batch_count = batch.shape[0]
        batch_mean, batch_var = batch.mean(axis=0), batch.var(axis=0)

        delta = batch_mean - rms_state.mean
        tot_count = rms_state.count + batch_count

        new_mean = rms_state.mean + delta * batch_count / tot_count
        m_a = rms_state.var * rms_state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * rms_state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return RMSState(mean=new_mean, var=new_var, count=new_count)

    def normalize_obs(self, rms_state, obs):
        return (obs - rms_state.mean) / jnp.sqrt(rms_state.var + 1e-8)

    def update_and_normalize(self, rms_state, obs, batched=True):
        rms_state = self.update_rms(rms_state, obs, batched)
        return rms_state, self.normalize_obs(rms_state, obs)
