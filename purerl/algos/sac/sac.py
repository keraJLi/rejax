import jax
import chex
import optax
import numpy as np
from typing import Any
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from purerl.normalize import RMSState, update_rms, normalize_obs

from purerl.algos.buffers import ReplayBuffer, Minibatch


class SACTrainState(TrainState):
    # TODO: we are storing the actor params twice, unncessary
    target_params: FrozenDict
    replay_buffer: ReplayBuffer
    env_state: Any
    last_obs: chex.Array
    global_step: int
    rms_state: RMSState
    rng: chex.PRNGKey

    def get_rng(self):
        rng, rng_new = jax.random.split(self.rng)
        return self.replace(rng=rng), rng_new


@jax.jit
def train(config, rng):
    ts = initialize_train_state(config, rng)

    if not config.skip_initial_evaluation:
        initial_evaluation = config.eval_callback(config, ts, ts.rng)

    def eval_iteration(ts, unused):
        # Run a few training iterations
        ts = jax.lax.fori_loop(
            0,
            np.ceil(config.eval_freq / config.num_envs).astype(int),
            lambda _, ts: train_iteration(config, ts),
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


def initialize_train_state(config, rng):
    rng, rng_agent = jax.random.split(rng)

    # TODO: discrete observation spaces
    obs_shape = config.env.observation_space(config.env_params).shape

    params = config.agent.init(rng_agent, jnp.zeros((1, *obs_shape)), rng_agent)
    tx = optax.adam(config.learning_rate, eps=1e-5)

    rng, rng_reset = jax.random.split(rng)
    rng_reset = jax.random.split(rng_reset, config.num_envs)
    vmap_reset = jax.vmap(config.env.reset, in_axes=(0, None))
    obs, env_state = vmap_reset(rng_reset, config.env_params)

    replay_buffer = ReplayBuffer.empty(
        size=config.buffer_size,
        obs_space=config.env.observation_space(config.env_params),
        action_space=config.env.action_space(config.env_params),
    )

    rms_state = RMSState.create(obs_shape)
    if config.normalize_observations:
        rms_state = update_rms(rms_state, obs)

    train_state = SACTrainState.create(
        apply_fn=config.agent.apply,
        params=params,
        target_params=params,
        replay_buffer=replay_buffer,
        env_state=env_state,
        tx=tx,
        last_obs=obs,
        global_step=0,
        rms_state=rms_state,
        rng=rng,
    )

    return train_state


def train_iteration(config, ts):
    # Collect transitions
    ts, transitions = collect_transitions(config, ts)
    ts = ts.replace(replay_buffer=ts.replay_buffer.extend(transitions))

    # Perform updates to citics, actor and alpha
    def update_iteration(ts):
        ts, rng_sample = ts.get_rng()
        minibatch = ts.replay_buffer.sample(config.batch_size, rng_sample)
        if config.normalize_observations:
            minibatch = minibatch._replace(
                obs=normalize_obs(ts.rms_state, minibatch.obs),
                next_obs=normalize_obs(ts.rms_state, minibatch.next_obs),
            )
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
    target_params = jax.tree_map(
        lambda q, qt: (1 - config.tau) * q + config.tau * qt,
        ts.params,
        ts.target_params,
    )
    ts = ts.replace(target_params=target_params)

    return ts


def collect_transitions(config, ts):
    # Sample actions
    ts, rng_action = ts.get_rng()

    if config.normalize_observations:
        last_obs = normalize_obs(ts.rms_state, ts.last_obs)
    else:
        last_obs = ts.last_obs

    actions = ts.apply_fn(ts.params, last_obs, rng_action, method="act")

    # Step environment
    ts, rng_steps = ts.get_rng()
    rng_steps = jax.random.split(rng_steps, config.num_envs)
    vmap_step = jax.vmap(config.env.step, in_axes=(0, 0, 0, None))
    next_obs, env_state, rewards, dones, infos = vmap_step(
        rng_steps, ts.env_state, actions, config.env_params
    )

    if config.normalize_observations:
        ts = ts.replace(rms_state=update_rms(ts.rms_state, next_obs))

    # Create minibatch and update train state
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


def update(config, ts, mb):
    ts, rng = ts.get_rng()
    rng_target, rng_pi = jax.random.split(rng, 2)

    # In the discrete case, we need to compute the q values of all actions to compute
    # the full expectation
    q_all = "q_all" if config.discrete else "q"
    log_alpha = ts.apply_fn(ts.params, method="log_alpha")
    alpha = jnp.exp(log_alpha)

    def q_loss_fn(params):
        # Calculate target without gradient wrt `params``
        action, logprob = ts.apply_fn(ts.params, mb.next_obs, rng_target, method="pi")
        q1, q2 = ts.apply_fn(ts.target_params, mb.next_obs, action, method=q_all)
        q_target = jnp.minimum(q1, q2) - alpha * logprob
        if config.discrete:
            q_target = jnp.sum(jnp.exp(logprob) * q_target, axis=1)
        target = mb.reward + config.gamma * (1 - mb.done) * q_target

        # Calculate MSBE wrt `params`
        q1, q2 = ts.apply_fn(params, mb.obs, mb.action, method="q")
        loss_q1 = optax.l2_loss(q1, target).mean()
        loss_q2 = optax.l2_loss(q2, target).mean()
        return loss_q1 + loss_q2

    def pi_loss_fn(params):
        action, logprob = ts.apply_fn(params, mb.obs, rng_pi, method="pi")
        q1, q2 = ts.apply_fn(ts.params, mb.obs, action, method=q_all)
        loss_pi = alpha * logprob - jnp.minimum(q1, q2)
        if config.discrete:
            loss_pi = jnp.sum(jnp.exp(logprob) * loss_pi, axis=1)
        return loss_pi.mean(), jax.lax.stop_gradient(logprob)

    def alpha_loss_fn(params, logprob):
        log_alpha = ts.apply_fn(params, method="log_alpha")
        loss_alpha = -log_alpha * (logprob + config.target_entropy)
        if config.discrete:
            loss_alpha = jnp.sum(jnp.exp(logprob) * loss_alpha, axis=1)
        return loss_alpha.mean()

    def loss_fn(params):
        q_loss = q_loss_fn(params)
        pi_loss, logprob = pi_loss_fn(params)
        alpha_loss = alpha_loss_fn(params, logprob)
        return q_loss + pi_loss + alpha_loss

    grads = jax.grad(loss_fn)(ts.params)
    ts = ts.apply_gradients(grads=grads)
    return ts
