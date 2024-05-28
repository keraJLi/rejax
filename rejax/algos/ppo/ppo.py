from typing import Any

import chex
import gymnax
import jax
import numpy as np
import optax
from flax.struct import PyTreeNode
from flax.training.train_state import TrainState
from jax import numpy as jnp

from rejax.algos.algorithm import Algorithm
from rejax.normalize import RMSState, normalize_obs, update_and_normalize

# TODO: fix step ratios and so on by implementing an eval tracker


class Trajectory(PyTreeNode):
    obs: chex.Array
    action: chex.Array
    log_prob: chex.Array
    reward: chex.Array
    value: chex.Array
    done: chex.Array


class AdvantageMinibatch(PyTreeNode):
    trajectories: Trajectory
    advantages: chex.Array
    targets: chex.Array


class PPOTrainState(PyTreeNode):
    actor_ts: TrainState
    critic_ts: TrainState
    env_state: Any
    last_obs: chex.Array
    last_done: chex.Array
    global_step: chex.Array
    rms_state: RMSState
    rng: chex.PRNGKey

    @property
    def params(self):
        """Backward compatibility with old evaluation creation"""
        return self.actor_ts.params

    def get_rng(self):
        rng, rng_new = jax.random.split(self.rng)
        return self.replace(rng=rng), rng_new


class PPO(Algorithm):
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
        rng, rng_actor, rng_critic = jax.random.split(rng, 3)
        obs_ph = jnp.empty((1, *config.env.observation_space(config.env_params).shape))

        actor_params = config.actor.init(rng_actor, obs_ph, rng_actor)
        actor_ts = TrainState.create(apply_fn=(), params=actor_params, tx=tx)
        critic_params = config.critic.init(rng_critic, obs_ph)
        critic_ts = TrainState.create(apply_fn=(), params=critic_params, tx=tx)

        # Initialize environment
        rngs = jax.random.split(rng, config.num_envs + 1)
        rng, env_rng = rngs[0], rngs[1:]
        vmap_reset = jax.vmap(config.env.reset, in_axes=(0, None))
        obs, env_state = vmap_reset(env_rng, config.env_params)

        # Initialize observation normalization
        rms_state = RMSState.create(obs.shape[1:])
        if config.normalize_observations:
            rms_state, obs = update_and_normalize(rms_state, obs)

        train_state = PPOTrainState(
            actor_ts=actor_ts,
            critic_ts=critic_ts,
            env_state=env_state,
            last_obs=obs,
            last_done=jnp.zeros(config.num_envs, dtype=bool),
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
        ts, trajectories = cls.collect_trajectories(config, ts)

        last_val = config.critic.apply(ts.critic_ts.params, ts.last_obs)
        last_val = jnp.where(ts.last_done, jnp.zeros_like(last_val), last_val)
        advantages, targets = cls.calculate_gae(config, trajectories, last_val)

        def update_epoch(ts: PPOTrainState, unused):
            ts, minibatch_rng = ts.get_rng()
            minibatches = cls.make_minibatches(
                config, trajectories, advantages, targets, minibatch_rng
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
    def collect_trajectories(cls, config, ts):
        def env_step(ts, unused):
            # Get keys for sampling action and stepping environment
            ts, rng = ts.get_rng()
            rng_steps, rng_action = jax.random.split(rng, 2)
            rng_steps = jax.random.split(rng_steps, config.num_envs)

            # Sample action
            unclipped_action, log_prob = config.actor.apply(
                ts.actor_ts.params, ts.last_obs, rng_action, method="action_log_prob"
            )
            value = config.critic.apply(ts.critic_ts.params, ts.last_obs)

            if isinstance(
                config.env.action_space(config.env_params),
                gymnax.environments.spaces.Discrete,
            ):
                action = unclipped_action
            else:
                low = config.env.action_space(config.env_params).low
                high = config.env.action_space(config.env_params).high
                action = jnp.clip(unclipped_action, low, high)

            # Step environment
            vmap_step = jax.vmap(config.env.step, in_axes=(0, 0, 0, None))
            transition = vmap_step(rng_steps, ts.env_state, action, config.env_params)
            next_obs, env_state, reward, done, _ = transition

            if config.normalize_observations:
                rms_state, next_obs = update_and_normalize(ts.rms_state, next_obs)
                ts = ts.replace(rms_state=rms_state)

            # Return updated runner state and transition
            transition = Trajectory(
                ts.last_obs, unclipped_action, log_prob, reward, value, done
            )
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
    def calculate_gae(cls, config, trajectories, last_val):
        def get_advantages(advantage_and_next_value, transition):
            advantage, next_value = advantage_and_next_value
            delta = (
                transition.reward.squeeze()  # For gymnax envs that return shape (1, )
                + config.gamma * next_value * (1 - transition.done)
                - transition.value
            )
            advantage = (
                delta
                + config.gamma * config.gae_lambda * (1 - transition.done) * advantage
            )
            return (advantage, transition.value), advantage

        _, advantages = jax.lax.scan(
            get_advantages,
            (jnp.zeros_like(last_val), last_val),
            trajectories,
            reverse=True,
            unroll=16,
        )
        return advantages, advantages + trajectories.value

    @classmethod
    def make_minibatches(cls, config, trajectories, advantages, targets, rng):
        iteration_size = config.minibatch_size * config.num_minibatches
        permutation = jax.random.permutation(rng, iteration_size)

        def shuffle_and_split(x):
            x = x.reshape((iteration_size, *x.shape[2:]))
            x = jnp.take(x, permutation, axis=0)
            return x.reshape(config.num_minibatches, -1, *x.shape[1:])

        batch = AdvantageMinibatch(trajectories, advantages, targets)
        minibatches = jax.tree_util.tree_map(shuffle_and_split, batch)
        return minibatches

    @classmethod
    def update_actor(cls, config, ts, batch):
        def actor_loss_fn(params):
            log_prob, entropy = config.actor.apply(
                params,
                batch.trajectories.obs,
                batch.trajectories.action,
                method="log_prob_entropy",
            )
            entropy = entropy.mean()

            # Calculate actor loss
            ratio = jnp.exp(log_prob - batch.trajectories.log_prob)
            advantages = (batch.advantages - batch.advantages.mean()) / (
                batch.advantages.std() + 1e-8
            )
            clipped_ratio = jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps)
            pi_loss1 = ratio * advantages
            pi_loss2 = clipped_ratio * advantages
            pi_loss = -jnp.minimum(pi_loss1, pi_loss2).mean()
            return pi_loss - config.ent_coef * entropy

        grads = jax.grad(actor_loss_fn)(ts.actor_ts.params)
        return ts.replace(actor_ts=ts.actor_ts.apply_gradients(grads=grads))

    @classmethod
    def update_critic(cls, config, ts, batch):
        def critic_loss_fn(params):
            value = config.critic.apply(params, batch.trajectories.obs)
            value_pred_clipped = batch.trajectories.value + (
                value - batch.trajectories.value
            ).clip(-config.clip_eps, config.clip_eps)
            value_losses = jnp.square(value - batch.targets)
            value_losses_clipped = jnp.square(value_pred_clipped - batch.targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
            return config.vf_coef * value_loss

        grads = jax.grad(critic_loss_fn)(ts.critic_ts.params)
        return ts.replace(critic_ts=ts.critic_ts.apply_gradients(grads=grads))

    @classmethod
    def update(cls, config, ts, batch):
        ts = cls.update_actor(config, ts, batch)
        ts = cls.update_critic(config, ts, batch)
        return ts
