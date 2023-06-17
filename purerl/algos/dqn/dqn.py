import jax
import chex
import optax
import numpy as np
from typing import Any
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from purerl.algos.buffers import Minibatch, ReplayBuffer


class DQNTrainState(TrainState):
    target_params: FrozenDict
    replay_buffer: ReplayBuffer
    env_state: Any
    last_obs: chex.Array
    global_step: int
    rng: chex.PRNGKey

    def get_rng(self):
        rng, rng_new = jax.random.split(self.rng)
        return self.replace(rng=rng), rng_new


def initialize_train_state(config, rng):
    rng, rng_agent = jax.random.split(rng)
    q_params = config.agent.init(
        rng_agent,
        jnp.zeros((1, *config.env.observation_space(config.env_params).shape)),
    )
    q_target_params = q_params
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate, eps=1e-5),
    )

    rng, rng_reset = jax.random.split(rng)
    rng_reset = jax.random.split(rng_reset, config.num_envs)
    vmap_reset = jax.vmap(config.env.reset, in_axes=(0, None))
    obs, env_state = vmap_reset(rng_reset, config.env_params)

    replay_buffer = ReplayBuffer.empty(
        size=config.buffer_size,
        obs_space=config.env.observation_space(config.env_params),
        action_space=config.env.action_space(config.env_params),
    )

    train_state = DQNTrainState.create(
        apply_fn=config.agent.apply,
        params=q_params,
        target_params=q_target_params,
        tx=tx,
        env_state=env_state,
        replay_buffer=replay_buffer,
        last_obs=obs,
        global_step=0,
        rng=rng,
    )

    return train_state


def evaluate(config, ts):
    def act(obs, rng):
        obs = jnp.expand_dims(obs, 0)
        action = ts.apply_fn(ts.params, obs, rng, method="act")
        return jnp.squeeze(action, 0)

    return config.evaluate(act, ts.rng)


@jax.jit
def train(config, rng):
    ts = initialize_train_state(config, rng)
    initial_evaluation = evaluate(config, ts)

    def eval_iteration(ts, unused):
        # Run a few trainig iterations
        ts = jax.lax.fori_loop(
            0,
            np.ceil(config.eval_freq / config.num_envs).astype(int),
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


def train_iteration(config, ts):
    # Calculate epsilon
    epsilon = config.epsilon_schedule(ts.global_step)

    # Collect transitions
    # old_global_step is to determine if we should update target network
    old_global_step = ts.global_step
    ts, batch = collect_transitions(config, ts, epsilon)
    ts = ts.replace(replay_buffer=ts.replay_buffer.extend(batch))

    # Perform updates to q network
    def update_iteration(ts):
        # Sample minibatch
        ts, rng_sample = ts.get_rng()
        minibatch = ts.replay_buffer.sample(config.batch_size, rng_sample)

        # Update network
        ts = update(config, ts, minibatch)
        return ts

    def do_updates(ts):
        return jax.lax.fori_loop(
            0, config.gradient_steps, lambda _, ts: update_iteration(ts), ts
        )

    ts = jax.lax.cond(
        ts.replay_buffer.num_entries > config.fill_buffer,
        lambda ts: do_updates(ts),
        lambda ts: ts,
        ts,
    )

    # Update target network
    update_target_params = (
        ts.global_step % config.target_update_freq
        <= old_global_step % config.target_update_freq
    )
    target_network = jax.tree_map(
        lambda q, qt: jax.lax.select(update_target_params, q, qt),
        ts.params,
        ts.target_params,
    )
    ts = ts.replace(target_params=target_network)

    return ts


def collect_transitions(config, ts, epsilon):
    # Sample actions
    ts, rng_action = ts.get_rng()
    actions = ts.apply_fn(
        ts.params, ts.last_obs, rng_action, epsilon=epsilon, method="act"
    )

    ts, rng_steps = ts.get_rng()
    rng_steps = jax.random.split(rng_steps, config.num_envs)
    vmap_step = jax.vmap(config.env.step, in_axes=(0, 0, 0, None))
    next_obs, env_state, rewards, dones, _ = vmap_step(
        rng_steps, ts.env_state, actions, config.env_params
    )
    minibatch = Minibatch(
        obs=ts.last_obs,
        action=actions,
        reward=rewards,
        next_obs=next_obs,
        done=dones,
    )
    ts = ts.replace(
        last_obs=next_obs,
        env_state=env_state,
        global_step=ts.global_step + config.num_envs,
    )
    return ts, minibatch


def update(config, ts, minibatch):
    action = jnp.expand_dims(minibatch.action, 1)
    next_q_target_values = ts.apply_fn(ts.target_params, minibatch.next_obs)

    def vanilla_targets(q_params):
        return jnp.max(next_q_target_values, axis=1)

    def ddqn_targets(q_params):
        next_q_values = ts.apply_fn(q_params, minibatch.next_obs)
        next_action = jnp.argmax(next_q_values, axis=1, keepdims=True)
        next_q_values_target = jnp.take_along_axis(
            next_q_target_values, next_action, axis=1
        ).squeeze(axis=1)
        return next_q_values_target

    def loss_fn(q_params):
        q_values = ts.apply_fn(q_params, minibatch.obs)
        q_values = jnp.take_along_axis(q_values, action, axis=1).squeeze()
        next_q_values_target = jax.lax.cond(
            config.ddqn, ddqn_targets, vanilla_targets, q_params
        )
        mask_done = jnp.logical_not(minibatch.done)
        targets = minibatch.reward + mask_done * config.gamma * next_q_values_target
        loss = optax.l2_loss(q_values, targets).mean()
        return loss

    grads = jax.grad(loss_fn)(ts.params)
    ts = ts.apply_gradients(grads=grads)
    return ts
