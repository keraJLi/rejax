"""
This example demonstrates how to modify an existing algorithm. To this end, we change
the update of PPO to be the one from "Discovered Policy Optimisation" by Chris Lu et al.
(https://arxiv.org/abs/2210.05639). 
"""

import jax
from jax import numpy as jnp

from rejax import PPO


def dpo_drift(ratio, advantages, alpha=2.0, beta=0.6):
    """Calculate the drift term for DPO, which is part of the loss. Details in paper."""
    drift1 = jax.nn.relu(
        (ratio - 1) * advantages - alpha * jnp.tanh((ratio - 1) * advantages / alpha)
    )
    drift2 = jax.nn.relu(
        jnp.log(ratio) * advantages
        - beta * jnp.tanh(jnp.log(ratio) * advantages / beta)
    )
    drift = jnp.where(advantages >= 0, drift1, drift2)
    return drift


# Overwrite PPO to change actor update, modifying the loss function
class DPO(PPO):
    def update_actor(self, ts, batch):
        def actor_loss_fn(params):
            log_prob, entropy = self.actor.apply(
                params,
                batch.trajectories.obs,
                batch.trajectories.action,
                method="log_prob_entropy",
            )
            entropy = entropy.mean()

            # Calculate drift and finally actor loss as in DPO
            ratio = jnp.exp(log_prob - batch.trajectories.log_prob)
            advantages = batch.advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            drift = dpo_drift(ratio, advantages)
            pi_loss = -(ratio * advantages - drift).mean()

            return pi_loss - self.ent_coef * entropy

        grads = jax.grad(actor_loss_fn)(ts.actor_ts.params)
        return ts.replace(actor_ts=ts.actor_ts.apply_gradients(grads=grads))


algo = PPO.create(
    env="Pendulum-v1",
    total_timesteps=1_000_000,
    eval_freq=5000,
    num_envs=50,
    num_steps=100,
    num_epochs=5,
    num_minibatches=20,
    max_grad_norm=0.5,
    learning_rate=0.001,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
)

eval_callback = algo.eval_callback


def eval_with_print(c, ts, rng):
    _, returns = eval_callback(c, ts, rng)
    jax.debug.print("Step: {}, Mean return: {}", ts.global_step, returns.mean())
    return ()


algo = algo.replace(eval_callback=eval_with_print)
DPO.train(algo, jax.random.PRNGKey(0))
