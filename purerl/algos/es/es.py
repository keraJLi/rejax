import jax
import chex
import numpy as np
import jax.numpy as jnp
from flax.core import FrozenDict
from flax.struct import PyTreeNode
from purerl.evaluate import evaluate as evaluate_act


def evaluate(config, evo_state, rng):
    # Mocks train state to ensure compatibility with evaluate.make_evaluate
    # TODO: rewrite to avoid this (e.g. use ESTrainState globally)
    class ESTrainState(PyTreeNode):
        params: FrozenDict
        evo_state: chex.ArrayTree

    params = config.strategy.param_reshaper.reshape_single(evo_state.best_member)
    return config.eval_callback(config, ESTrainState(params, evo_state), rng)


def evaluate_fitness(config, params, rng):
    def act(obs, rng):
        return config.agent.apply(params, obs, rng, method="act")

    rng = jax.random.split(rng, config.num_rollouts)
    _, fitness = jax.vmap(evaluate_act, in_axes=(None, 0, None, None, None, None))(
        act,
        rng,
        config.env,
        config.env_params,
        config.num_rollouts,
        config.env_params.max_steps_in_episode,
    )
    return fitness.mean()


@jax.jit
def train(config, rng):
    rng, rng_init, rng_init_eval = jax.random.split(rng, 3)
    evo_state = config.strategy.initialize(rng_init)
    initial_evaluation = evaluate(config, evo_state, rng_init_eval)

    def eval_iteration(evo_state, rng):
        # Run a few training iterations
        rng = jax.random.split(rng, config.eval_freq + 1)
        evo_state = jax.lax.fori_loop(
            0,
            config.eval_freq,
            lambda i, evo_state: train_iteration(config, evo_state, rng[i]),
            evo_state,
        )

        # Run evaluation
        return evo_state, evaluate(config, evo_state, rng[-1])

    num_eval_iterations = np.ceil(config.num_generations / config.eval_freq).astype(int)
    evo_state, evaluation = jax.lax.scan(
        eval_iteration,
        evo_state,
        jax.random.split(rng, num_eval_iterations),
    )

    all_evaluations = jax.tree_map(
        lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
        initial_evaluation,
        evaluation,
    )
    return evo_state, all_evaluations


def train_iteration(config, evo_state, rng):
    rng, rng_ask, rng_eval = jax.random.split(rng, 3)
    params, evo_state = config.strategy.ask(rng_ask, evo_state, config.strategy_params)
    vmap_evaluate_fitness = jax.vmap(evaluate_fitness, in_axes=(None, 0, None))
    fitness = vmap_evaluate_fitness(config, params, rng_eval)
    evo_state = config.strategy.tell(params, fitness, evo_state, config.strategy_params)
    return evo_state
