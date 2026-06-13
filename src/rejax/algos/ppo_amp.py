
import chex
import jax
import optax
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState
from jax import numpy as jnp
from rejax.algos.algorithm import register_init
from rejax.algos.ppo import PPO, Trajectory
from rejax.algos.mixins import RewardRMSState
from rejax.networks import VNetwork


class PPOAMP(PPO):
    amp_discriminator: nn.Module = struct.field(pytree_node=False, default=None)
    lambda_amp: chex.Scalar = struct.field(pytree_node=True, default=0.5)
    amp_data: chex.Array = struct.field(pytree_node=False, default=None)
    amp_max_grad_norm: chex.Scalar = struct.field(pytree_node=True, default=0.5)
    amp_learning_rate: chex.Scalar = struct.field(pytree_node=True, default=5e-4)
    gp_lambda: chex.Scalar = struct.field(pytree_node=True, default=10.0)

    @classmethod
    def create_agent(cls, config, env, env_params):
        agent_dict = super(PPOAMP, cls).create_agent(config, env, env_params)

        discriminator_kwargs = config.pop("discriminator_kwargs", {})
        discriminator_hidden_sizes = discriminator_kwargs.pop("hidden_layer_sizes", (1024, 512))
        discriminator_kwargs["hidden_layer_sizes"] = tuple(discriminator_hidden_sizes)
        activation = discriminator_kwargs.pop("activation", "swish")
        discriminator_kwargs["activation"] = getattr(nn, activation)
        amp_disc = VNetwork(**discriminator_kwargs)

        agent_dict["amp_discriminator"] = amp_disc
        return agent_dict

    @register_init
    def initialize_network_params(self, rng):
        rng_super, rng_amp = jax.random.split(rng, 2)
        params_dict = super().initialize_network_params(rng_super)
        dummy_amp_input = jnp.empty([1, *self.env.observation_space(self.env_params).shape])
        amp_disc_params = self.amp_discriminator.init(rng_amp, dummy_amp_input)
        amp_tx = optax.chain(
            optax.clip(self.amp_max_grad_norm),
            optax.adam(learning_rate=self.amp_learning_rate),
        )
        amp_disc_ts = TrainState.create(apply_fn=self.amp_discriminator.apply, params=amp_disc_params, tx=amp_tx)
        params_dict["amp_disc_ts"] = amp_disc_ts
        return params_dict
    
    @register_init
    def initialize_im_reward_rms_state(self, rng):
        batch_size = getattr(self, "num_envs", ())
        return {"im_rew_rms_state": RewardRMSState.create(batch_size)}

    def collect_trajectories(self, ts):
        def env_step(ts, unused):
            rng, new_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            rng_steps, rng_action = jax.random.split(new_rng, 2)
            rng_steps = jax.random.split(rng_steps, self.num_envs)
            unclipped_action, log_prob = self.actor.apply(
                ts.actor_ts.params, ts.last_obs, rng_action, method="action_log_prob"
            )
            value = self.critic.apply(ts.critic_ts.params, ts.last_obs)
            if self.discrete:
                action = unclipped_action
            else:
                low = self.env.action_space(self.env_params).low
                high = self.env.action_space(self.env_params).high
                action = jnp.clip(unclipped_action, low, high)
            t = self.vmap_step(rng_steps, ts.env_state, action, self.env_params)
            next_obs, env_state, reward, done, _ = t

            if self.normalize_observations:
                obs_rms_state, next_obs = self.update_and_normalize_obs(ts.obs_rms_state, next_obs)
                ts = ts.replace(obs_rms_state=obs_rms_state)

            im_reward = self.amp_discriminator.apply(ts.amp_disc_ts.params, next_obs)
            if self.normalize_rewards:
                rew_rms_state, reward = self.update_and_normalize_rew(ts.rew_rms_state, reward, done)
                ts = ts.replace(rew_rms_state=rew_rms_state)
                im_rew_rms_state, im_reward = self.update_and_normalize_rew(ts.im_rew_rms_state, im_reward, done)
                ts = ts.replace(im_rew_rms_state=im_rew_rms_state)
            reward = (1 - self.lambda_amp) * reward + self.lambda_amp * im_reward

            transition = Trajectory(
                ts.last_obs,
                unclipped_action,
                log_prob,
                reward,
                value,
                done
            )
            ts = ts.replace(
                env_state=env_state,
                last_obs=next_obs,
                last_done=done,
                global_step=ts.global_step + self.num_envs,
            )
            return ts, transition
        ts, trajectories = jax.lax.scan(env_step, ts, None, self.num_steps)
        return ts, trajectories

    def update_amp_discriminator(self, ts, batch): 
        rng, sample_rng, gp_rng = jax.random.split(ts.rng, 3)
        ts = ts.replace(rng=rng)
        batch_size = batch.trajectories.obs.shape[0]
        sample_idx = jax.random.randint(sample_rng, (batch_size,), 0, self.amp_data.shape[0])
        dataset_obs = self.amp_data[sample_idx]

        if getattr(self, "normalize_observations", False):
            dataset_obs = self.normalize_obs(ts.obs_rms_state, dataset_obs)

        def compute_gradient_penalty(params, real_data, fake_data):
            alpha = jax.random.uniform(gp_rng, shape=(batch_size, 1))
            interpolated_data = alpha * real_data + (1 - alpha) * fake_data
            def grad_fn(x):
                return self.amp_discriminator.apply(params, x[None]).squeeze(0)
            
            grad_interpolated = jax.vmap(jax.grad(grad_fn))(interpolated_data)
            grad_norm = jnp.linalg.norm(grad_interpolated, ord=2, axis=1)
            gradient_penalty = jnp.mean((grad_norm - 1.0) ** 2)
            return gradient_penalty

        def amp_loss_fn(params):
            policy_output = ts.amp_disc_ts.apply_fn(params, batch.trajectories.obs)
            expert_output = ts.amp_disc_ts.apply_fn(params, dataset_obs)
            wgan_loss = jnp.mean(policy_output) - jnp.mean(expert_output)
            
            gp = compute_gradient_penalty(params, dataset_obs, batch.trajectories.obs)
            
            total_loss = wgan_loss + self.gp_lambda * gp
            return total_loss
        
        grads = jax.grad(amp_loss_fn)(ts.amp_disc_ts.params)
        new_amp_disc_ts = ts.amp_disc_ts.apply_gradients(grads=grads)
        return ts.replace(amp_disc_ts=new_amp_disc_ts)

    def make_discriminator(self, ts):
        def amp_discriminator_fn(obs):
            if getattr(self, "normalize_observations", False):
                obs = self.normalize_obs(ts.obs_rms_state, obs)
            obs = jnp.atleast_2d(obs)
            im_rew = self.amp_discriminator.apply(ts.amp_disc_ts.params, obs)
            return im_rew
        return amp_discriminator_fn

    def update(self, ts, batch):
        ts = super().update(ts, batch)
        ts = self.update_amp_discriminator(ts, batch)
        return ts