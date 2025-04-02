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
from rejax.networks import DiscreteQNetwork, DuelingQNetwork, EpsilonGreedyPolicy


class DQN(
    EpsilonGreedyMixin,
    ReplayBufferMixin,
    TargetNetworkMixin,
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    Algorithm,
):
    agent: nn.Module = struct.field(pytree_node=False, default=None)
    num_epochs: int = struct.field(pytree_node=False, default=1)
    ddqn: bool = struct.field(pytree_node=True, default=True)

    def make_act(self, ts):
        def act(obs, rng):
            if getattr(self, "normalize_observations", False):
                obs = self.normalize_obs(ts.obs_rms_state, obs)

            obs = jnp.expand_dims(obs, 0)
            action = self.agent.apply(
                ts.q_ts.params, obs, rng, epsilon=0.005, method="act"
            )
            return jnp.squeeze(action)

        return act

    @classmethod
    def create_agent(cls, config, env, env_params):
        agent_name = config.pop("agent", "QNetwork")
        agent_cls = {
            "QNetwork": DiscreteQNetwork,
            "DuelingQNetwork": DuelingQNetwork,
        }[agent_name]
        agent_kwargs = config.pop("agent_kwargs", {})
        activation = agent_kwargs.pop("activation", "swish")
        agent_kwargs["activation"] = getattr(nn, activation)

        action_dim = env.action_space(env_params).n
        agent = EpsilonGreedyPolicy(agent_cls)(
            hidden_layer_sizes=(64, 64), action_dim=action_dim, **agent_kwargs
        )

        return {"agent": agent}

    @register_init
    def initialize_network_params(self, rng):
        obs_ph = jnp.empty([1, *self.env.observation_space(self.env_params).shape])
        q_params = self.agent.init(rng, obs_ph)
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
        next_q_target_values = self.agent.apply(ts.q_target_params, mb.next_obs)
        if self.normalize_rewards:
            rewards = self.normalize_rew(ts.rew_rms_state, mb.reward)
        else:
            rewards = mb.reward

        def vanilla_targets(q_params):
            return jnp.max(next_q_target_values, axis=1)

        def ddqn_targets(q_params):
            next_q_values = self.agent.apply(q_params, mb.next_obs)
            next_action = jnp.argmax(next_q_values, axis=1, keepdims=True)
            next_q_values_target = jnp.take_along_axis(
                next_q_target_values, next_action, axis=1
            ).squeeze(axis=1)
            return next_q_values_target

        def loss_fn(q_params):
            q_values = self.agent.apply(q_params, mb.obs, mb.action, method="take")
            next_q_values_target = jax.lax.cond(
                self.ddqn, ddqn_targets, vanilla_targets, q_params
            )
            mask_done = jnp.logical_not(mb.done)
            targets = rewards + mask_done * self.gamma * next_q_values_target
            loss = optax.l2_loss(q_values, targets).mean()
            return loss

        grads = jax.grad(loss_fn)(ts.q_ts.params)
        ts = ts.replace(q_ts=ts.q_ts.apply_gradients(grads=grads))
        return ts
