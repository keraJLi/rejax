from typing import Type

import chex
import distrax
import jax
import optax
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState
from jax import numpy as jnp

from rejax.algos.algorithm import Algorithm, register_init
from rejax.algos.mixins import (
    EpsilonGreedyMixin,
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    ReplayBufferMixin,
    TargetNetworkMixin,
)
from rejax.buffers import Minibatch
from rejax.networks import ImplicitQuantileNetwork


def EpsilonGreedyPolicy(iqn: nn.Module) -> Type[nn.Module]:
    class EpsilonGreedyPolicy(iqn):
        def _action_dist(self, obs, rng, epsilon):
            q = self.q(obs, rng)
            return distrax.EpsilonGreedy(q, epsilon=epsilon)

        def act(self, obs, rng, epsilon):
            rng_tau, rng_epsilon = jax.random.split(rng)
            action_dist = self._action_dist(obs, rng_tau, epsilon)
            action = action_dist.sample(seed=rng_epsilon)
            return action

    return EpsilonGreedyPolicy


class IQN(
    EpsilonGreedyMixin,
    ReplayBufferMixin,
    TargetNetworkMixin,
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    Algorithm,
):
    agent: nn.Module = struct.field(pytree_node=False, default=None)
    num_epochs: int = struct.field(pytree_node=False, default=1)
    num_tau_samples: int = struct.field(pytree_node=False, default=64)
    num_tau_prime_samples: int = struct.field(pytree_node=False, default=64)
    kappa: chex.Scalar = struct.field(pytree_node=True, default=1.0)

    def make_act(self, ts):
        def act(obs, rng):
            if self.normalize_observations:
                obs = self.normalize_obs(ts.obs_rms_state, obs)

            obs = jnp.expand_dims(obs, 0)
            action = self.agent.apply(
                ts.q_ts.params, obs, rng, epsilon=0.005, method="act"
            )
            return jnp.squeeze(action)

        return act

    @classmethod
    def create_agent(cls, config, env, env_params):
        agent_kwargs = config.pop("agent_kwargs", {})
        activation = agent_kwargs.pop("activation", "swish")
        agent_kwargs["activation"] = getattr(nn, activation)
        hidden_layer_sizes = agent_kwargs.pop("hidden_layer_sizes", (64, 64))
        agent_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

        action_dim = env.action_space(env_params).n
        agent = EpsilonGreedyPolicy(ImplicitQuantileNetwork)(
            action_dim=action_dim, **agent_kwargs
        )
        return {"agent": agent}

    @register_init
    def initialize_network_params(self, rng):
        obs_ph = jnp.empty([1, *self.env.observation_space(self.env_params).shape])
        q_params = self.agent.init(rng, obs_ph, rng)
        tx = optax.chain(
            optax.clip(self.max_grad_norm),
            optax.adam(learning_rate=self.learning_rate),
        )
        q_ts = TrainState.create(apply_fn=(), params=q_params, tx=tx)
        return {"q_ts": q_ts, "q_target_params": q_params}

    def train_iteration(self, ts):
        start_training = ts.global_step > self.fill_buffer
        old_global_step = ts.global_step

        # Calculate epsilon
        epsilon = self.epsilon_schedule(ts.global_step)

        # Collect transitions
        uniform = jnp.logical_not(start_training)
        ts, batch = self.collect_transitions(ts, epsilon, uniform=uniform)
        ts = ts.replace(replay_buffer=ts.replay_buffer.extend(batch))

        # Perform updates to q network
        def update_iteration(ts):
            # Sample minibatch
            rng, rng_sample = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            minibatch = ts.replay_buffer.sample(self.batch_size, rng_sample)
            if self.normalize_observations:
                minibatch = minibatch._replace(
                    obs=self.normalize_obs(ts.obs_rms_state, minibatch.obs),
                    next_obs=self.normalize_obs(ts.obs_rms_state, minibatch.next_obs),
                )

            # Update network
            ts = self.update(ts, minibatch)
            return ts

        def do_updates(ts):
            return jax.lax.fori_loop(
                0, self.num_epochs, lambda _, ts: update_iteration(ts), ts
            )

        ts = jax.lax.cond(start_training, lambda: do_updates(ts), lambda: ts)

        # Update target network
        if self.target_update_freq == 1:
            target_params = self.polyak_update(ts.q_ts.params, ts.q_target_params)
        else:
            update_target_params = (
                ts.global_step % self.target_update_freq
                <= old_global_step % self.target_update_freq
            )
            target_params = jax.tree.map(
                lambda q, qt: jax.lax.select(update_target_params, q, qt),
                self.polyak_update(ts.q_ts.params, ts.q_target_params),
                ts.q_target_params,
            )
        ts = ts.replace(q_target_params=target_params)
        return ts

    def collect_transitions(self, ts, epsilon, uniform=False):
        # Sample actions
        rng, rng_action = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)

        def sample_uniform(rng):
            sample_fn = self.env.action_space(self.env_params).sample
            return jax.vmap(sample_fn)(jax.random.split(rng, self.num_envs))

        def sample_policy(rng):
            if self.normalize_observations:
                last_obs = self.normalize_obs(ts.obs_rms_state, ts.last_obs)
            else:
                last_obs = ts.last_obs

            return self.agent.apply(
                ts.q_ts.params, last_obs, rng, epsilon=epsilon, method="act"
            )

        actions = jax.lax.cond(uniform, sample_uniform, sample_policy, rng_action)

        rng, rng_steps = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        rng_steps = jax.random.split(rng_steps, self.num_envs)
        next_obs, env_state, rewards, dones, _ = self.vmap_step(
            rng_steps, ts.env_state, actions, self.env_params
        )
        if self.normalize_observations:
            ts = ts.replace(
                obs_rms_state=self.update_obs_rms(ts.obs_rms_state, next_obs)
            )
        if self.normalize_rewards:
            ts = ts.replace(
                rew_rms_state=self.update_rew_rms(ts.rew_rms_state, rewards, dones)
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
            global_step=ts.global_step + self.num_envs,
        )
        return ts, minibatch

    def update(self, ts, mb):
        # Normalize rewards
        if self.normalize_rewards:
            rewards = self.normalize_rew(ts.rew_rms_state, mb.reward)
        else:
            rewards = mb.reward

        # Move tau to axis 1, leaving batch as leading axis
        vmapped_apply = jax.vmap(self.agent.apply, in_axes=(None, None, 0), out_axes=1)

        # Split off multiple keys for tau and tau_prime
        rng, rng_action, rng_tau, rng_tau_prime = jax.random.split(ts.rng, 4)
        ts = ts.replace(rng=rng)
        rng_tau = jax.random.split(rng_tau, self.num_tau_samples)
        rng_tau_prime = jax.random.split(rng_tau_prime, self.num_tau_prime_samples)

        best_action = self.agent.apply(
            ts.q_ts.params, mb.next_obs, rng_action, method="best_action"
        )
        zs, _ = vmapped_apply(ts.q_ts.params, mb.next_obs, rng_tau_prime)
        best_z = jnp.take_along_axis(zs, best_action[:, None, None], axis=2).squeeze(2)

        targets = rewards[:, None] + self.gamma * (1 - mb.done[:, None]) * best_z
        assert targets.shape == (
            self.batch_size,
            self.num_tau_prime_samples,
        )

        # Vmap over batch and sampled taus
        @jax.vmap
        @jax.vmap
        def rho(td_err, tau):
            l = optax.huber_loss(td_err, delta=self.kappa)
            return jnp.abs(tau - (td_err < 0)) * l / self.kappa

        def loss_fn(q_params):
            # tau has shape (batch, num_tau_samples)
            z, tau = vmapped_apply(q_params, mb.obs, rng_tau)

            # after taking actions, z has shape (batch, num_tau_samples, action_dim)
            z = jnp.take_along_axis(z, mb.action[:, None, None], axis=2).squeeze(2)

            # td_err has shape (batch, num_tau_samples, num_tau_prime_samples)
            td_err = jax.vmap(lambda x, y: x[None, :] - y[:, None])(targets, z)

            # before sum, loss has shape (batch, num_tau_samples, num_tau_prime_samples)
            loss = rho(td_err, tau).sum(axis=1)
            return loss.mean()

        grads = jax.grad(loss_fn)(ts.q_ts.params)
        ts = ts.replace(q_ts=ts.q_ts.apply_gradients(grads=grads))
        return ts
