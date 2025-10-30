"""
In this example we are fitting hyperparameters of an SAC agent playing pendulum using an
evolutionary strategy implemented in evosax (https://github.com/RobertTLange/evosax).
The strategy samples a population of hyperparameters, calculates a "fitness" for each
and updates its statistics as to maximize the fitness of the next generation. In our
case, the fitness will be the final return of our agent after training with the sampled
hyperparameters.
In order to efficiently calculate the fitness of a whole population, we are vmapping
both over all sampled hyperparameter combinations, as well as several random seeds per
combination to reduce variance in our fitness estimate.

Running this example takes some time, ~20 minutes on my laptop. Feel free to reduce the
population size and number of evaluation seeds to make it faster.
"""

from functools import partial
from time import time

import jax
from evosax import SNES

from rejax import get_algo


NUM_GENERATIONS = 10
POPULATION_SIZE = 10
EVAL_SEEDS = 5

# Load SAC
algo_cls = get_algo("sac")

# fmt: off
# Static parameters, cannot be vmapped
static_params = {
    "env": "Pendulum-v1",

    # Time steps: we only care about the final performance
    "total_timesteps": 10_000,
    "eval_freq": 10_000,
    "skip_initial_evaluation": True,

    # Other hyperparameters
    "buffer_size": 2000,
    "fill_buffer": 1000,
    "batch_size": 256,
    "num_envs": 1,    # Matching the original SAC formulation
    "num_epochs": 1,  # Matching the original SAC formulation
}
# fmt: on

# Parameters to optimize, can be vmapped. Initilize with educated guesses
optim_params = {
    "learning_rate": 0.001,
    "gamma": 1.0,
    "polyak": 0.95,
}


# vmap over parameters and seeds
@jax.jit
@partial(jax.vmap, in_axes=(0, None))
@partial(jax.vmap, in_axes=(None, 0))
def evaluate_fitness(meta_params, rng):
    algo = algo_cls.create(**static_params, **meta_params)
    train_state, (lenghts, returns) = algo.train(rng)  # noqa: RUF059

    # Take mean over evaluation seeds and calculate fitness as final return
    fitness = returns.mean(axis=1)[-1]
    return fitness


# Initialize ES with our educated guesses and set parameter bounds
rng = jax.random.PRNGKey(0)
strategy = SNES(
    popsize=POPULATION_SIZE,
    pholder_params=optim_params,
    sigma_init=0.005,
    maximize=True,
)
es_params = strategy.default_params.replace(clip_min=1e-10, clip_max=1)
state = strategy.initialize(rng, es_params, init_mean=optim_params)

# Run a few generations of ES
print(
    f"Running {NUM_GENERATIONS} generations with {POPULATION_SIZE} members each, "
    f"evaluating each member on {EVAL_SEEDS} training seeds."
)
for t in range(1, NUM_GENERATIONS + 1):
    start_time = time()
    rng, rng_gen, rng_eval = jax.random.split(rng, 3)

    # Ask strategy for new population
    meta_params, state = strategy.ask(rng_gen, state, es_params)

    # Evaluate fitness as mean over several rollouts
    rng_eval = jax.random.split(rng_eval, EVAL_SEEDS)
    fitness = evaluate_fitness(meta_params, rng_eval)
    fitness = fitness.mean(axis=1)

    # Update strategy with fitness
    state = strategy.tell(meta_params, fitness, state, es_params)

    # Print some information
    time_t = time() - start_time
    print(
        f"Gen {t:<2} - "
        f"mean fitness: {fitness.mean():.1f} ± {fitness.std():.1f}, "
        f"best fitness: {state.best_fitness:.1f}, "
        f"time: {time_t:.1f}s ({time_t / (POPULATION_SIZE * EVAL_SEEDS):.1f}s per fit)"
    )

# Print final results
print(f"Finished after {t} generations")
print(f"Best member: {strategy.param_reshaper.reshape_single(state.best_member)}")
print(f"Fitness of best member: {state.best_fitness}")
print(f"Population mean: {strategy.param_reshaper.reshape_single(state.mean)}")
