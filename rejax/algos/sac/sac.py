from typing import Any

import chex
import jax
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from flax.struct import PyTreeNode
from flax.training.train_state import TrainState
from jax import numpy as jnp

from rejax.algos.algorithm import Algorithm
from rejax.algos.buffers import Minibatch, ReplayBuffer
from rejax.normalize import RMSState, normalize_obs, update_rms


class SACTrainState(PyTreeNode):
    alpha_ts: TrainState
    actor_ts: TrainState
    critic_ts: TrainState
    critic_target_params: FrozenDict
    replay_buffer: ReplayBuffer
    env_state: Any
    last_obs: chex.Array
    global_step: int
    rms_state: RMSState
    rng: chex.PRNGKey

    @property
    def params(self):
        """Backward compatibility with old evaluation creation"""
        return self.actor_ts.params

    def get_rng(self):
        rng, rng_new = jax.random.split(self.rng)
        return self.replace(rng=rng), rng_new


class SAC(Algorithm):
    @classmethod
    def make_act(cls, config, ts):
        def act(obs, rng):
            if getattr(config, "normalize_observations", False):
                obs = normalize_obs(ts.rms_state, obs)

            obs = jnp.expand_dims(obs, 0)
            action = config.actor.apply(ts.params, obs, rng, method="act")
            return jnp.squeeze(action)

        return act

    @classmethod
    def initialize_train_state(cls, config, rng):
        # Initialize optimizer
        tx = optax.adam(config.learning_rate, eps=1e-5)

        # Initialize network parameters and train states
        rng, rng_actor, rng_critic = jax.random.split(rng, 3)
        rng_critic = jax.random.split(rng_critic, 2)
        obs_ph = jnp.empty((1, *config.env.observation_space(config.env_params).shape))
        actor_params = config.actor.init(rng_actor, obs_ph, rng_actor)
        actor_ts = TrainState.create(apply_fn=(), params=actor_params, tx=tx)

        if config.discrete:
            critic_params = jax.vmap(config.critic.init, in_axes=(0, None))(
                rng_critic, obs_ph
            )
        else:
            act_ph = jnp.empty((1, *config.env.action_space(config.env_params).shape))
            critic_params = jax.vmap(config.critic.init, in_axes=(0, None, None))(
                rng_critic, obs_ph, act_ph
            )
        critic_ts = TrainState.create(apply_fn=(), params=critic_params, tx=tx)

        alpha_params = FrozenDict({"log_alpha": jnp.array(0.0)})
        alpha_ts = TrainState.create(apply_fn=(), params=alpha_params, tx=tx)

        # Initialize environment
        rng, rng_reset = jax.random.split(rng)
        rng_reset = jax.random.split(rng_reset, config.num_envs)
        vmap_reset = jax.vmap(config.env.reset, in_axes=(0, None))
        obs, env_state = vmap_reset(rng_reset, config.env_params)

        # Initialize replay buffer
        replay_buffer = ReplayBuffer.empty(
            size=config.buffer_size,
            obs_space=config.env.observation_space(config.env_params),
            action_space=config.env.action_space(config.env_params),
        )

        # Initialize observation normalization
        rms_state = RMSState.create(obs.shape[1:])
        if config.normalize_observations:
            rms_state = update_rms(rms_state, obs)

        train_state = SACTrainState(
            alpha_ts=alpha_ts,
            actor_ts=actor_ts,
            critic_ts=critic_ts,
            critic_target_params=critic_params,
            env_state=env_state,
            last_obs=obs,
            replay_buffer=replay_buffer,
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
            ts = jax.lax.fori_loop(
                0,
                np.ceil(config.eval_freq / config.num_envs).astype(int),
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
        # Collect transitions
        ts, transitions = cls.collect_transitions(config, ts)
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
            ts = cls.update(config, ts, minibatch)
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
        critic_target_params = jax.tree_map(
            lambda q, qt: (1 - config.tau) * q + config.tau * qt,
            ts.critic_ts.params,
            ts.critic_target_params,
        )
        ts = ts.replace(critic_target_params=critic_target_params)

        return ts

    @classmethod
    def collect_transitions(cls, config, ts):
        # Normalize observations
        if config.normalize_observations:
            last_obs = normalize_obs(ts.rms_state, ts.last_obs)
        else:
            last_obs = ts.last_obs

        # Sample actions
        ts, rng_action = ts.get_rng()
        actions = config.actor.apply(
            ts.actor_ts.params, last_obs, rng_action, method="act"
        )

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

    @classmethod
    def udpate_actor(cls, config, ts, mb):
        ts, rng = ts.get_rng()
        alpha = jnp.exp(ts.alpha_ts.params["log_alpha"])

        def actor_loss_fn(params):
            if config.discrete:
                logprob = jnp.log(
                    config.actor.apply(params, mb.obs, method="_action_dist").probs
                )
                q1, q2 = jax.vmap(config.critic.apply, in_axes=(0, None))(
                    ts.critic_ts.params, mb.obs
                )
                loss_pi = alpha * logprob - jnp.minimum(q1, q2)
                loss_pi = jnp.sum(jnp.exp(logprob) * loss_pi, axis=1)
            else:
                action, logprob = config.actor.apply(
                    params, mb.obs, rng, method="action_log_prob"
                )
                q1, q2 = jax.vmap(config.critic.apply, in_axes=(0, None, None))(
                    ts.critic_ts.params, mb.obs, action
                )
                loss_pi = alpha * logprob - jnp.minimum(q1, q2)
            return loss_pi.mean(), logprob

        grads, logprob = jax.grad(actor_loss_fn, has_aux=True)(ts.actor_ts.params)
        ts = ts.replace(actor_ts=ts.actor_ts.apply_gradients(grads=grads))
        return ts, logprob

    @classmethod
    def update_critic(cls, config, ts, mb):
        ts, rng = ts.get_rng()
        alpha = jnp.exp(ts.alpha_ts.params["log_alpha"])

        def critic_loss_fn(params):
            # Calculate target without gradient wrt `params``
            if config.discrete:
                logprob = jnp.log(
                    config.actor.apply(
                        ts.actor_ts.params, mb.next_obs, method="_action_dist"
                    ).probs
                )
                q1, q2 = jax.vmap(config.critic.apply, in_axes=(0, None))(
                    ts.critic_target_params, mb.next_obs
                )
                q_target = jnp.minimum(q1, q2) - alpha * logprob
                q_target = jnp.sum(jnp.exp(logprob) * q_target, axis=1)

                q1, q2 = jax.vmap(
                    lambda *args: config.critic.apply(*args, method="take"),
                    in_axes=(0, None, None),
                )(params, mb.obs, mb.action)
            else:
                action, logprob = config.actor.apply(
                    ts.actor_ts.params, mb.next_obs, rng, method="action_log_prob"
                )
                q1, q2 = jax.vmap(config.critic.apply, in_axes=(0, None, None))(
                    ts.critic_target_params, mb.next_obs, action
                )
                q_target = jnp.minimum(q1, q2) - alpha * logprob
                q1, q2 = jax.vmap(config.critic.apply, in_axes=(0, None, None))(
                    params, mb.obs, mb.action
                )

            target = mb.reward + config.gamma * (1 - mb.done) * q_target
            loss_q1 = optax.l2_loss(q1, target).mean()
            loss_q2 = optax.l2_loss(q2, target).mean()
            return loss_q1 + loss_q2

        grads = jax.grad(critic_loss_fn)(ts.critic_ts.params)
        ts = ts.replace(critic_ts=ts.critic_ts.apply_gradients(grads=grads))
        return ts

    @classmethod
    def update_alpha(cls, config, ts, logprob):
        def alpha_loss_fn(params, logprob):
            log_alpha = params["log_alpha"]
            loss_alpha = -log_alpha * (logprob + config.target_entropy)
            if config.discrete:
                loss_alpha = jnp.sum(jnp.exp(logprob) * loss_alpha, axis=1)
            return loss_alpha.mean()

        grads = jax.grad(alpha_loss_fn)(ts.alpha_ts.params, logprob)
        ts = ts.replace(alpha_ts=ts.alpha_ts.apply_gradients(grads=grads))
        return ts

    @classmethod
    def update(cls, config, ts, mb):
        ts, logprob = cls.udpate_actor(config, ts, mb)
        ts = cls.update_critic(config, ts, mb)
        ts = cls.update_alpha(config, ts, logprob)
        return ts
