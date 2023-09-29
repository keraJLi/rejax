import jax
import chex
import numpy as np
from flax import struct
import jax.numpy as jnp
from flax.core import FrozenDict


def execute_eval_callback(config, evo_state, rng):
    # Mocks train state to ensure compatibility with evaluate.make_evaluate
    # TODO: rewrite to avoid this (e.g. use ESTrainState globally)
    @struct.dataclass
    class ESTrainState:
        params: FrozenDict
        evo_state: chex.ArrayTree

    params = config.strategy.param_reshaper.reshape_single(evo_state.best_member)
    # params = jax.tree_map(lambda x: jnp.expand_dims(x, 0), params)
    return config.eval_callback(config, ESTrainState(params, evo_state), rng)


def evaluate(config, params, rng):
    def step(state):
        (rng, env_state, last_obs, done, return_, length) = state
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        action = config.agent.apply(params, last_obs, rng_act, method="act")
        obs, env_state, reward, done, _ = config.env.step(
            rng_step, env_state, action, config.env_params
        )
        return_ = return_ + reward.squeeze()
        return (rng, env_state, obs, done, return_, length + 1)

    rng_reset, rng_eval = jax.random.split(rng)
    obs, env_state = config.env.reset(rng_reset, config.env_params)
    state = (rng_eval, env_state, obs, False, 0.0, 0)
    _, _, _, _, return_, _ = jax.lax.while_loop(
        lambda s: jnp.logical_and(
            s[5] < config.env_params.max_steps_in_episode, jnp.logical_not(s[3])
        ),
        step,
        state,
    )
    return return_


def evaluate_fitness(config, params, rng):
    rng = jax.random.split(rng, config.num_rollouts)
    evaluate_vmap = jax.vmap(evaluate, in_axes=(None, None, 0))
    fitness = evaluate_vmap(config, params, rng)
    return fitness.mean()


@jax.jit
def train(config, rng):
    rng, rng_init, rng_init_eval = jax.random.split(rng, 3)
    evo_state = config.strategy.initialize(rng_init)
    initial_evaluation = execute_eval_callback(config, evo_state, rng_init_eval)

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
        return evo_state, execute_eval_callback(config, evo_state, rng[-1])

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
