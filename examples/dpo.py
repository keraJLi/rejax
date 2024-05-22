"""
This example demonstrates how to modify an existing algorithm. To this end, we change
the update of PPO to be the one from "Discovered Policy Optimisation" by Chris Lu et al.
(https://arxiv.org/abs/2210.05639). 
"""


import jax
from jax import numpy as jnp

from fastrl import PPO, PPOConfig


class DPO(PPO):
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

            # Calculate actor loss as in DPO
            alpha, beta = 2, 0.6
            ratio = jnp.exp(log_prob - batch.trajectories.log_prob)
            advantages = (batch.advantages - batch.advantages.mean()) / (
                batch.advantages.std() + 1e-8
            )
            drift1 = jax.nn.relu(
                (ratio - 1) * advantages
                - alpha * jnp.tanh((ratio - 1) * advantages / alpha),
            )
            drift2 = jax.nn.relu(
                jnp.log(ratio) * advantages
                - beta * jnp.tanh(jnp.log(ratio) * advantages / beta),
            )
            drift = jnp.where(advantages >= 0, drift1, drift2)
            pi_loss = -(ratio * advantages - drift).mean()

            return pi_loss - config.ent_coef * entropy

        grads = jax.grad(actor_loss_fn)(ts.actor_ts.params)
        return ts.replace(actor_ts=ts.actor_ts.apply_gradients(grads=grads))


config = PPOConfig.from_dict(
    {
        "env": "CartPole-v1",
        "total_timesteps": 250_000,
        "eval_freq": 5000,
        "num_envs": 20,
        "num_steps": 100,
        "num_epochs": 10,
        "num_minibatches": 10,
        "max_grad_norm": 10,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
    }
)

eval_callback = config.eval_callback


def eval_with_print(c, ts, rng):
    _, returns = eval_callback(c, ts, rng)
    jax.debug.print("Step: {}, Mean return: {}", ts.global_step, returns.mean())
    return ()


config = config.replace(eval_callback=eval_with_print)
DPO.train(config, jax.random.PRNGKey(0))
