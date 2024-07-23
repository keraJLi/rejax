"""
Adapted from https://github.com/mttga/purejaxql/blob/main/purejaxql/pqn_gymnax.py
by Matteo Gallici et. al.
Thanks!
"""


from typing import Any

import chex
import jax
import numpy as np
import optax
from flax.struct import PyTreeNode
from flax.training.train_state import TrainState
from jax import numpy as jnp

from rejax.algos.algorithm import Algorithm
from rejax.normalize import RMSState, normalize_obs, update_and_normalize, update_rms


class Trajectory(PyTreeNode):
    obs: chex.Array
    action: chex.Array
    next_q: chex.Array
    reward: chex.Array
    done: chex.Array


class TargetMinibatch(PyTreeNode):
    trajectories: Trajectory
    targets: chex.Array


class PQNTrainState(PyTreeNode):
    q_ts: TrainState
    env_state: Any
    last_obs: chex.Array
    last_done: chex.Array
    global_step: int
    rms_state: RMSState
    rng: chex.PRNGKey

    @property
    def params(self):
        return self.q_ts.params

    def get_rng(self):
        rng, rng_new = jax.random.split(self.rng)
        return self.replace(rng=rng), rng_new


class PQN(Algorithm):
    @classmethod
    def make_act(cls, config, ts):
        def act(obs, rng):
            if getattr(config, "normalize_observations", False):
                obs = normalize_obs(ts.rms_state, obs)

            obs = jnp.expand_dims(obs, 0)
            action = config.agent.apply(
                ts.params, obs, rng, epsilon=0.005, method="act"
            )
            return jnp.squeeze(action)

        return act

    @classmethod
    def initialize_train_state(cls, config, rng):
        rng, rng_agent = jax.random.split(rng)
        q_params = config.agent.init(
            rng_agent,
            jnp.zeros((1, *config.env.observation_space(config.env_params).shape)),
        )
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.learning_rate, eps=1e-5),
        )
        q_ts = TrainState.create(apply_fn=(), params=q_params, tx=tx)

        rng, rng_reset = jax.random.split(rng)
        rng_reset = jax.random.split(rng_reset, config.num_envs)
        vmap_reset = jax.vmap(config.env.reset, in_axes=(0, None))
        obs, env_state = vmap_reset(rng_reset, config.env_params)

        rms_state = RMSState.create(obs.shape[1:])
        if config.normalize_observations:
            rms_state = update_rms(rms_state, obs)

        train_state = PQNTrainState(
            q_ts=q_ts,
            env_state=env_state,
            last_obs=obs,
            last_done=jnp.zeros(config.num_envs).astype(bool),
            global_step=0,
            rms_state=rms_state,
            rng=rng,
        )

        return train_state

    @classmethod
    def train(cls, config, rng=None, train_state=None):
        if train_state is None and rng is None:
            raise ValueError("Either train_state or rng must be provided")

        ts = train_state or cls.initialize_train_state(config, rng)

        if not config.skip_initial_evaluation:
            initial_evaluation = config.eval_callback(config, ts, ts.rng)

        def eval_iteration(ts, unused):
            # Run a few training iterations
            iteration_steps = config.num_envs * config.num_steps
            num_iterations = np.ceil(config.eval_freq / iteration_steps).astype(int)
            ts = jax.lax.fori_loop(
                0,
                num_iterations,
                lambda _, ts: cls.train_iteration(config, ts),
                ts,
            )

            # Run evaluation
            return ts, config.eval_callback(config, ts, ts.rng)

        ts, evaluation = jax.lax.scan(
            eval_iteration,
            ts,
            None,
            np.ceil(config.total_timesteps / config.eval_freq).astype(int),
        )

        if not config.skip_initial_evaluation:
            evaluation = jax.tree_map(
                lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
                initial_evaluation,
                evaluation,
            )

        return ts, evaluation

    @classmethod
    def train_iteration(cls, config, ts):
        epsilon = config.epsilon_schedule(ts.global_step)
        ts, trajectories = cls.collect_trajectories(config, ts, epsilon)

        max_last_q = config.agent.apply(ts.q_ts.params, ts.last_obs).max(axis=1)
        max_last_q = (1 - ts.last_done) * max_last_q
        targets = cls.calculate_targets(config, trajectories, max_last_q)

        def update_epoch(ts: PQNTrainState, unused):
            ts, minibatch_rng = ts.get_rng()
            minibatches = cls.make_minibatches(
                config, trajectories, targets, minibatch_rng
            )
            ts, _ = jax.lax.scan(
                lambda ts, mbs: (cls.update(config, ts, mbs), None),
                ts,
                minibatches,
            )
            return ts, None

        ts, _ = jax.lax.scan(update_epoch, ts, None, config.num_epochs)
        return ts

    @classmethod
    def collect_trajectories(cls, config, ts, epsilon):
        def env_step(ts, unused):
            ts, rng = ts.get_rng()
            rng_action, rng_step = jax.random.split(rng)
            action = config.agent.apply(
                ts.params, ts.last_obs, rng_action, epsilon=epsilon, method="act"
            )

            rng_step = jax.random.split(rng_step, config.num_envs)
            vmap_step = jax.vmap(config.env.step, in_axes=(0, 0, 0, None))
            transition = vmap_step(rng_step, ts.env_state, action, config.env_params)
            next_obs, env_state, reward, done, _ = transition
            next_q = config.agent.apply(ts.q_ts.params, next_obs)

            if config.normalize_observations:
                rms_state, next_obs = update_and_normalize(ts.rms_state, next_obs)
                ts = ts.replace(rms_state=rms_state)

            # Return updated runner state and transition
            transition = Trajectory(ts.last_obs, action, next_q, reward, done)
            ts = ts.replace(
                env_state=env_state,
                last_obs=next_obs,
                last_done=done,
                global_step=ts.global_step + config.num_envs,
            )
            return ts, transition

        ts, trajectories = jax.lax.scan(env_step, ts, None, config.num_steps)
        return ts, trajectories

    @classmethod
    def make_minibatches(cls, config, trajectories, targets, rng):
        iteration_size = config.minibatch_size * config.num_minibatches
        permutation = jax.random.permutation(rng, iteration_size)

        def shuffle_and_split(x):
            x = x.reshape((iteration_size, *x.shape[2:]))
            x = jnp.take(x, permutation, axis=0)
            return x.reshape(config.num_minibatches, -1, *x.shape[1:])

        batch = TargetMinibatch(trajectories, targets)
        minibatches = jax.tree_util.tree_map(shuffle_and_split, batch)
        return minibatches

    @classmethod
    def calculate_targets(cls, config, trajectories, max_last_q):
        def get_target(lambda_return_and_next_q, transition):
            lambda_return, next_q = lambda_return_and_next_q
            return_bootstrap = next_q + config.lambda_ * (lambda_return - next_q)
            lambda_return = transition.reward + (1 - transition.done) * config.gamma * (
                return_bootstrap
            )
            max_next_q = transition.next_q.max(axis=1)
            return (lambda_return, max_next_q), lambda_return

        max_last_q = (1 - trajectories.done[-1]) * max_last_q
        lambda_returns = trajectories.reward[-1] + config.gamma * max_last_q
        _, targets = jax.lax.scan(
            get_target,
            (lambda_returns, max_last_q),
            jax.tree_util.tree_map(lambda x: x[:-1], trajectories),
            reverse=True,
        )
        targets = jnp.concatenate((targets, lambda_returns[None]))
        return targets

    @classmethod
    def update(cls, config, ts, minibatch):
        tr, ta = minibatch.trajectories, minibatch.targets

        def loss_fn(params):
            q_values = config.agent.apply(params, tr.obs, tr.action, method="take")
            return optax.l2_loss(q_values, ta).mean()

        grads = jax.grad(loss_fn)(ts.q_ts.params)
        ts = ts.replace(q_ts=ts.q_ts.apply_gradients(grads=grads))
        return ts
