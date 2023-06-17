import jax
import chex
import optax
import numpy as np
from typing import Any
from flax import struct
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from purerl.algos.buffers import ReplayBuffer, Minibatch


class TD3TrainState(struct.PyTreeNode):
    q_ts: TrainState
    q_target_params: FrozenDict
    pi_ts: TrainState
    pi_target_params: FrozenDict

    replay_buffer: ReplayBuffer
    env_state: Any
    last_obs: chex.Array
    global_step: int
    rng: chex.PRNGKey

    def get_rng(self):
        rng, rng_new = jax.random.split(self.rng)
        return self.replace(rng=rng), rng_new

    @property
    def params(self):
        """ So that train_states.params are the params for config.agent """
        return self.pi_ts.params


def evaluate(config, ts):
    def act(obs, rng):
        obs = jnp.expand_dims(obs, 0)
        action = ts.pi_ts.apply_fn(ts.pi_ts.params, obs, rng, method="act")
        return jnp.squeeze(action, 0)

    return config.evaluate(act, ts.rng)


@jax.jit
def train(config, rng):
    ts = initialize_train_state(config, rng)
    initial_evaluation = evaluate(config, ts)

    def eval_iteration(ts, unused):
        # Run a few training iterations
        ts = jax.lax.fori_loop(
            0,
            config.eval_freq,
            lambda _, ts: train_iteration(config, ts),
            ts,
        )

        # Run evaluation
        return ts, evaluate(config, ts)

    ts, evaluation = jax.lax.scan(
        eval_iteration,
        ts,
        None,
        np.ceil(config.total_timesteps / config.eval_freq).astype(int),
    )

    all_evaluations = jax.tree_map(
        lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
        initial_evaluation,
        evaluation,
    )
    return ts, all_evaluations


def initialize_train_state(config, rng):
    rng, rng_q, rng_pi = jax.random.split(rng, 3)
    q_params = config.critic.init(
        rng_q,
        jnp.zeros((2, 1, *config.env.observation_space(config.env_params).shape)),
        jnp.zeros((2, 1, *config.env.action_space(config.env_params).shape)),
    )
    q_target_params = q_params
    q_tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate, eps=1e-5),
    )

    def q_apply_fn(params, obs, action, **kwargs):
        obs = jnp.repeat(jnp.expand_dims(obs, 0), 2, axis=0)
        action = jnp.repeat(jnp.expand_dims(action, 0), 2, axis=0)
        return config.critic.apply(params, obs, action, **kwargs)

    q_ts = TrainState.create(
        apply_fn=q_apply_fn,
        params=q_params,
        tx=q_tx,
    )

    pi_params = config.actor.init(
        rng_pi,
        jnp.zeros((1, *config.env.observation_space(config.env_params).shape)),
    )
    pi_target_params = pi_params
    pi_tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate, eps=1e-5),
    )
    pi_ts = TrainState.create(
        apply_fn=config.actor.apply,
        params=pi_params,
        tx=pi_tx,
    )

    rng, rng_reset = jax.random.split(rng)
    obs, env_state = config.env.reset(rng_reset, config.env_params)

    replay_buffer = ReplayBuffer.empty(
        size=config.buffer_size,
        obs_space=config.env.observation_space(config.env_params),
        action_space=config.env.action_space(config.env_params),
    )

    train_state = TD3TrainState(
        q_ts=q_ts,
        q_target_params=q_target_params,
        pi_ts=pi_ts,
        pi_target_params=pi_target_params,
        replay_buffer=replay_buffer,
        env_state=env_state,
        last_obs=obs,
        global_step=0,
        rng=rng,
    )

    return train_state


def train_iteration(config, ts):
    start_training = ts.global_step > config.fill_buffer

    # Collect transition
    ts, rng = ts.get_rng()
    rng_uniform, rng_noise, rng_step = jax.random.split(rng, 3)

    action = jax.lax.cond(
        jnp.logical_not(start_training),
        lambda rng: config.env.action_space(config.env_params).sample(rng),
        lambda _: ts.pi_ts.apply_fn(ts.pi_ts.params, ts.last_obs),
        rng_uniform,
    )
    noise = config.exploration_noise * jax.random.normal(rng_noise, action.shape)
    action = jnp.clip(action + noise, config.action_low, config.action_high)

    next_obs, env_state, rewards, dones, _ = config.env.step(
        rng_step, ts.env_state, action, config.env_params
    )
    minibatch = Minibatch(
        obs=ts.last_obs,
        action=action,
        reward=rewards,
        next_obs=next_obs,
        done=dones,
    )
    ts = ts.replace(
        last_obs=next_obs,
        env_state=env_state,
        global_step=ts.global_step + 1,
        replay_buffer=ts.replay_buffer.append(minibatch),
    )

    # Perform updates to q network
    def update_iteration(ts):
        # Sample minibatch
        ts, rng_sample = ts.get_rng()
        minibatch = ts.replay_buffer.sample(config.batch_size, rng_sample)

        # Update network
        ts = update_q(config, ts, minibatch)
        ts = jax.lax.cond(
            (ts.global_step + 1) % config.policy_delay == 0,
            lambda ts: update_pi(config, ts, minibatch),
            lambda ts: ts,
            ts,
        )
        return ts

    # Do updates if buffer is sufficiently full
    # TODO: is predicate of cond vmapped, i.e. converted to select? Would introduce
    # unnecessary computation
    ts = jax.lax.cond(
        start_training,
        lambda ts: update_iteration(ts),
        lambda ts: ts,
        ts,
    )

    # Update target networks
    q_target_network = jax.tree_map(
        lambda q, qt: (1 - config.tau) * q + config.tau * qt,
        ts.q_ts.params,
        ts.q_target_params,
    )
    pi_target_network = jax.tree_map(
        lambda pi, pit: (1 - config.tau) * pi + config.tau * pit,
        ts.pi_ts.params,
        ts.pi_target_params,
    )

    ts = ts.replace(
        q_target_params=q_target_network, pi_target_params=pi_target_network
    )

    return ts


def update_q(config, ts, minibatch):
    def q_loss(params):
        action = ts.pi_ts.apply_fn(ts.pi_target_params, minibatch.next_obs)
        noise = jnp.clip(
            config.target_noise * jax.random.normal(ts.rng, action.shape),
            -config.target_noise_clip,
            config.target_noise_clip,
        )
        action = jnp.clip(action + noise, config.action_low, config.action_high)

        q1_target, q2_target = ts.q_ts.apply_fn(
            ts.q_target_params, minibatch.next_obs, action
        )
        q_target = jnp.minimum(q1_target, q2_target)
        target = minibatch.reward + (1 - minibatch.done) * config.gamma * q_target
        q1, q2 = ts.q_ts.apply_fn(params, minibatch.obs, minibatch.action)

        loss_q1 = optax.l2_loss(q1, target).mean()
        loss_q2 = optax.l2_loss(q2, target).mean()
        return loss_q1 + loss_q2

    grads = jax.grad(q_loss)(ts.q_ts.params)
    ts = ts.replace(q_ts=ts.q_ts.apply_gradients(grads=grads))
    return ts


def update_pi(config, ts, minibatch):
    def pi_loss(params):
        action = ts.pi_ts.apply_fn(params, minibatch.obs)
        q = ts.q_ts.apply_fn(ts.q_ts.params, minibatch.obs, action)
        return -q.mean()

    grads = jax.grad(pi_loss)(ts.pi_ts.params)
    ts = ts.replace(pi_ts=ts.pi_ts.apply_gradients(grads=grads))
    return ts
