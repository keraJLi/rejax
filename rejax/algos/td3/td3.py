from typing import Any

import chex
import jax
import numpy as np
import optax
from flax import struct
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from jax import numpy as jnp

from rejax.algos.algorithm import Algorithm
from rejax.algos.buffers import Minibatch, ReplayBuffer
from rejax.normalize import RMSState, normalize_obs, update_rms

# Algorithm outline
# num_eval_iterations = total_timesteps / eval_freq
# num_train_iterations = eval_freq / (num_envs * policy_delay)
# for _ in range(num_eval_iterations):
#   for _ in range(num_train_iterations):
#     for _ in range(policy_delay):
#       M = collect num_gradient_steps minibatches
#       update critic using M
#     update actor using M
#     update target networks


class TD3TrainState(struct.PyTreeNode):
    actor_ts: TrainState
    actor_target_params: FrozenDict
    critic_ts: TrainState
    critic_target_params: FrozenDict

    replay_buffer: ReplayBuffer
    env_state: Any
    last_obs: chex.Array
    global_step: int
    rms_state: RMSState
    rng: chex.PRNGKey

    def get_rng(self):
        rng, rng_new = jax.random.split(self.rng)
        return self.replace(rng=rng), rng_new

    @property
    def params(self):
        """So that train_states.params are the params for config.agent"""
        return self.actor_ts.params


class TD3(Algorithm):
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
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.learning_rate, eps=1e-5),
        )

        # Initialize network parameters and train states
        rng, rng_critic, rng_pi = jax.random.split(rng, 3)
        rng_critic = jax.random.split(rng_critic, 2)
        obs_ph = jnp.zeros((1, *config.env.observation_space(config.env_params).shape))

        act_ph = jnp.zeros((1, *config.env.action_space(config.env_params).shape))
        actor_params = config.actor.init(rng_pi, obs_ph)
        actor_target_params = actor_params
        actor_ts = TrainState.create(
            apply_fn=config.actor.apply,
            params=actor_params,
            tx=tx,
        )

        critic_params = jax.vmap(config.critic.init, in_axes=(0, None, None))(
            rng_critic, obs_ph, act_ph
        )
        critic_target_params = critic_params
        critic_ts = TrainState.create(apply_fn=(), params=critic_params, tx=tx)

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

        train_state = TD3TrainState(
            actor_ts=actor_ts,
            actor_target_params=actor_target_params,
            critic_ts=critic_ts,
            critic_target_params=critic_target_params,
            replay_buffer=replay_buffer,
            env_state=env_state,
            last_obs=obs,
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
            # Run a few trainig iterations
            steps_per_train_it = config.num_envs * config.policy_delay
            num_train_its = np.ceil(config.eval_freq / steps_per_train_it).astype(int)
            ts = jax.lax.fori_loop(
                0,
                num_train_its,
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
        placeholder_minibatch = jax.tree_map(
            lambda sdstr: jnp.empty((config.gradient_steps, *sdstr.shape), sdstr.dtype),
            ts.replay_buffer.sample(config.batch_size, jax.random.PRNGKey(0)),
        )
        ts, minibatch = jax.lax.fori_loop(
            0,
            config.policy_delay,
            lambda _, ts_mb: cls.train_critic(config, ts_mb[0]),
            (ts, placeholder_minibatch),
        )
        ts = cls.train_policy(config, ts, minibatch)
        return ts

    @classmethod
    def train_critic(cls, config, ts):
        start_training = ts.global_step > config.fill_buffer

        # Collect transition
        uniform = jnp.logical_not(start_training)
        ts, transitions = cls.collect_transitions(config, ts, uniform=uniform)
        ts = ts.replace(replay_buffer=ts.replay_buffer.extend(transitions))

        def update_iteration(ts, unused):
            # Sample minibatch
            ts, rng_sample = ts.get_rng()
            minibatch = ts.replay_buffer.sample(config.batch_size, rng_sample)
            if config.normalize_observations:
                minibatch = minibatch._replace(
                    obs=normalize_obs(ts.rms_state, minibatch.obs),
                    next_obs=normalize_obs(ts.rms_state, minibatch.next_obs),
                )

            # Update network
            ts = cls.update_critic(config, ts, minibatch)
            return ts, minibatch

        def do_updates(ts):
            return jax.lax.scan(update_iteration, ts, None, config.gradient_steps)

        placeholder_minibatch = jax.tree_map(
            lambda sdstr: jnp.empty((config.gradient_steps, *sdstr.shape), sdstr.dtype),
            ts.replay_buffer.sample(config.batch_size, jax.random.PRNGKey(0)),
        )
        ts, minibatches = jax.lax.cond(
            start_training,
            do_updates,
            lambda ts: (ts, placeholder_minibatch),
            ts,
        )
        return ts, minibatches

    @classmethod
    def train_policy(cls, config, ts, minibatches):
        def do_updates(ts):
            ts, _ = jax.lax.scan(
                lambda ts, minibatch: (cls.update_actor(config, ts, minibatch), None),
                ts,
                minibatches,
            )
            return ts

        start_training = ts.global_step > config.fill_buffer
        ts = jax.lax.cond(start_training, do_updates, lambda ts: ts, ts)

        # Update target networks
        critic_target_network = jax.tree_map(
            lambda q, qt: (1 - config.tau) * q + config.tau * qt,
            ts.critic_ts.params,
            ts.critic_target_params,
        )
        actor_target_network = jax.tree_map(
            lambda pi, pit: (1 - config.tau) * pi + config.tau * pit,
            ts.actor_ts.params,
            ts.actor_target_params,
        )
        ts = ts.replace(
            critic_target_params=critic_target_network,
            actor_target_params=actor_target_network,
        )
        return ts

    @classmethod
    def collect_transitions(cls, config, ts, uniform=False):
        # Sample actions
        ts, rng_action = ts.get_rng()

        def sample_uniform(rng):
            sample_fn = config.env.action_space(config.env_params).sample
            return jax.vmap(sample_fn)(jax.random.split(rng, config.num_envs))

        def sample_policy(rng):
            if config.normalize_observations:
                last_obs = normalize_obs(ts.rms_state, ts.last_obs)
            else:
                last_obs = ts.last_obs

            actions = config.actor.apply(ts.actor_ts.params, last_obs)
            noise = config.exploration_noise * jax.random.normal(rng, actions.shape)
            return jnp.clip(actions + noise, config.action_low, config.action_high)

        actions = jax.lax.cond(uniform, sample_uniform, sample_policy, rng_action)

        # Step environment
        ts, rng_steps = ts.get_rng()
        rng_steps = jax.random.split(rng_steps, config.num_envs)
        vmap_step = jax.vmap(config.env.step, in_axes=(0, 0, 0, None))
        next_obs, env_state, rewards, dones, _ = vmap_step(
            rng_steps, ts.env_state, actions, config.env_params
        )

        if config.normalize_observations:
            ts = ts.replace(rms_state=update_rms(ts.rms_state, next_obs))

        # Return minibatch and updated train state
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
    def update_critic(cls, config, ts, minibatch):
        def critic_loss_fn(params):
            action = ts.actor_ts.apply_fn(ts.actor_target_params, minibatch.next_obs)
            noise = jnp.clip(
                config.target_noise * jax.random.normal(ts.rng, action.shape),
                -config.target_noise_clip,
                config.target_noise_clip,
            )
            action = jnp.clip(action + noise, config.action_low, config.action_high)

            q1_target, q2_target = jax.vmap(
                config.critic.apply, in_axes=(0, None, None)
            )(ts.critic_target_params, minibatch.next_obs, action)
            q_target = jnp.minimum(q1_target, q2_target)
            target = minibatch.reward + (1 - minibatch.done) * config.gamma * q_target
            q1, q2 = jax.vmap(config.critic.apply, in_axes=(0, None, None))(
                params, minibatch.obs, minibatch.action
            )

            loss_q1 = optax.l2_loss(q1, target).mean()
            loss_q2 = optax.l2_loss(q2, target).mean()
            return loss_q1 + loss_q2

        grads = jax.grad(critic_loss_fn)(ts.critic_ts.params)
        ts = ts.replace(critic_ts=ts.critic_ts.apply_gradients(grads=grads))
        return ts

    @classmethod
    def update_actor(cls, config, ts, minibatch):
        def actor_loss_fn(params):
            action = config.actor.apply(params, minibatch.obs)
            q = jax.vmap(config.critic.apply, in_axes=(0, None, None))(
                ts.critic_ts.params, minibatch.obs, action
            )
            return -q.mean()

        grads = jax.grad(actor_loss_fn)(ts.actor_ts.params)
        ts = ts.replace(actor_ts=ts.actor_ts.apply_gradients(grads=grads))
        return ts
