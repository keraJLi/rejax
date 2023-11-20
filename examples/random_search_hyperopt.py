import jax
from jax import numpy as jnp

from purerl.algos import get_agent

"""
In this example we are fitting hyperparameters of a TD3 agent on Pendulum using random
search.
"""

# How many random configurations should we try?
POPULATION_SIZE = 10

train_fn, config_cls = get_agent("td3")

# Static parameters, cannot be vmapped
static_params = {
    "env": "Pendulum-v1",
    "total_timesteps": 10_000,
    "buffer_size": 2000,
    "fill_buffer": 1000,
    "batch_size": 100,
    "max_grad_norm": 10,
    "policy_delay": 2,
    "eval_freq": 10_000,  # We only care about the final performance
    "skip_initial_evaluation": True,
}


def log_uniform(rng, minval, maxval):
    """
    Samples from a uniform distribution in log space, meaning values close to minval
    are sampled more densely.
    """
    return jnp.exp(
        jax.random.uniform(rng, minval=jnp.log(minval), maxval=jnp.log(maxval))
    )


def exp_uniform(rng, minval, maxval):
    """
    Samples from a uniform distribution in exp space, meaning values close to maxval
    are sampled more densely.
    """
    return jnp.log(
        jax.random.uniform(rng, minval=jnp.exp(minval), maxval=jnp.exp(maxval))
    )


def sample_config_dict(rng):
    """ Samples a dictionary of vmappable hyperparameters. """
    rngs = jax.random.split(rng, 6)
    return {
        "learning_rate": log_uniform(rngs[0], minval=0.0001, maxval=0.01),
        "gamma": exp_uniform(rngs[1], minval=0.7, maxval=1.0),
        "tau": exp_uniform(rngs[2], minval=0.7, maxval=1.0),
        "exploration_noise": jax.random.uniform(rngs[3], minval=0.0, maxval=0.5),
        "target_noise": jax.random.uniform(rngs[4], minval=0.0, maxval=0.5),
        "target_noise_clip": jax.random.uniform(rngs[5], minval=0.0, maxval=0.5),
    }

# Vmap to create a population of configurations
global_rngs = jax.random.split(jax.random.PRNGKey(0), POPULATION_SIZE)
config_dicts = jax.vmap(sample_config_dict)(global_rngs)
configs = jax.vmap(lambda c: config_cls.from_dict({**static_params, **c}))(config_dicts)

# Vmap training to parallelize the evaluation of return, which was originally of shape
# (POPULATION_SIZE, 1, num_eval_seeds), where num_eval_seeds = 200 by default
_, (_, returns) = jax.vmap(train_fn)(configs, global_rngs)
returns = returns[:, 0].mean(axis=1)

# Find the best configuration and get its final return and hyperparameter configuration
best_index = returns.argmax()
best_return = returns[best_index]
best_config = jax.tree_map(lambda c: c[best_index], config_dicts)

print(f"Best config: {best_config}")
print(f"with return: {best_return}")
