from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment


class EvalState(NamedTuple):
    rng: chex.PRNGKey
    env_state: Any
    last_obs: chex.Array
    done: bool = False
    return_: float = 0.0
    length: int = 0


def evaluate_single(
    act: Callable[[chex.Array, chex.PRNGKey], chex.Array],  # act(obs, rng) -> action
    env,
    env_params,
    rng,
    max_steps_in_episode,
):
    def step(state):
        rng, rng_act, rng_step = jax.random.split(state.rng, 3)
        action = act(state.last_obs, rng_act)
        obs, env_state, reward, done, info = env.step(
            rng_step, state.env_state, action, env_params
        )
        state = EvalState(
            rng=rng,
            env_state=env_state,
            last_obs=obs,
            done=done,
            return_=state.return_ + reward.squeeze(),
            length=state.length + 1,
        )
        return state

    rng_reset, rng_eval = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    state = EvalState(rng_eval, env_state, obs)
    state = jax.lax.while_loop(
        lambda s: jnp.logical_and(
            s.length < max_steps_in_episode, jnp.logical_not(s.done)
        ),
        step,
        state,
    )
    return state.length, state.return_


@partial(jax.jit, static_argnames=("act", "env", "num_seeds"))
def evaluate(
    act: Callable[[chex.Array, chex.PRNGKey], chex.Array],
    rng: chex.PRNGKey,
    env: environment.Environment,
    env_params: Any,
    num_seeds: int = 128,
    max_steps_in_episode: Optional[int] = None,
) -> Tuple[chex.Array, chex.Array]:
    """Evaluate a policy given by `act` on `num_seeds` environments.

    Args:
        act (Callable[[chex.Array, chex.PRNGKey], chex.Array]): A policy represented as
        a function of type (obs, rng) -> action.
        rng (chex.PRNGKey): Initial seed, will be split into `num_seeds` seeds for
        parallel evaluation.
        env (environment.Environment): The environment to evaluate on.
        env_params (Any): The parameters of the environment.
        num_seeds (int): Number of initializations of the environment.

    Returns:
        Tuple[chex.Array, chex.Array]: Tuple of episode length and cumultative reward
        for each seed.
    """
    if max_steps_in_episode is None:
        max_steps_in_episode = env_params.max_steps_in_episode

    seeds = jax.random.split(rng, num_seeds)
    vmap_collect = jax.vmap(evaluate_single, in_axes=(None, None, None, 0, None))
    return vmap_collect(act, env, env_params, seeds, max_steps_in_episode)
