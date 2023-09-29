import jax
import time
import numpy as np
import pandas as pd
import jax.numpy as jnp

from purerl.algos import get_agent
from purerl.evaluate import make_evaluate as make_evaluate_vanilla


class Logger:
    def __init__(self, mle_logger):
        self.last_step = 0
        self.last_time = 0
        self._log_step = []
        self.mle_logger = mle_logger

    def collect_log_step(self):
        def convert(x):
            if isinstance(x, (np.ndarray, jnp.ndarray)):
                return x.tolist()
            return x

        _log_step = jax.tree_map(convert, self._log_step)
        _log_step = pd.DataFrame(_log_step).mean().to_dict()

        self.mle_logger.update({"step": self.last_step}, _log_step)
        self._log_step = []

    def log(self, data, step):
        print(step)
        step = step.item()  # jax cpu callback returns numpy array

        # Because of vmapping the training function, self.log is called several times
        # sequentially. Therefore we only log once we reach a new global_step
        if step > self.last_step:
            self.collect_log_step()

        self._log_step.append(data)
        self.last_step = step


def make_evaluate(logger, env, env_params, num_seeds=50):
    evaluate_vanilla = make_evaluate_vanilla(env, env_params, num_seeds)

    def log_with_garbage_return(data, step):
        logger.log(data, step)
        return jnp.empty(()), jnp.empty(())  # garbage

    def evaluate(config, ts, rng):
        lengths, returns = evaluate_vanilla(config, ts, rng)
        garbage = jax.experimental.io_callback(
            log_with_garbage_return,
            (jnp.empty(()), jnp.empty(())),
            # Take mean over evaluation seeds
            {
                "episode_length": lengths.mean(axis=0),
                "episode_length_std": lengths.std(axis=0),
                "episode_length_min": lengths.min(axis=0),
                "episode_length_max": lengths.max(axis=0),
                "return": returns.mean(axis=0),
                "return_std": returns.std(axis=0),
                "return_min": returns.min(axis=0),
                "return_max": returns.max(axis=0),
            },
            ts.global_step,
        )
        return garbage

    return evaluate


def main(config, mle_logger):
    algo, num_seeds, seed_id = (
        config.pop("algo"),
        config.pop("num_seeds"),
        config.pop("seed_id"),
    )
    logger = Logger(mle_logger)

    # Prepare train function and config
    train_fn, config_cls = get_agent(algo)
    train_config = config_cls.from_dict(config)
    evaluate = make_evaluate(logger, train_config.env, train_config.env_params)
    train_config = train_config.replace(eval_callback=evaluate)

    key = jax.random.PRNGKey(seed_id)
    keys = jax.random.split(key, num_seeds)
    vmap_train = jax.jit(jax.vmap(train_fn, in_axes=(None, 0)))

    # Time compilation
    start = time.process_time()
    lowered = vmap_train.lower(train_config, keys)
    time_lower = time.process_time() - start
    compiled = lowered.compile()
    time_compile = time.process_time() - time_lower
    vmap_train = compiled

    mle_logger.update(
        {"step": 0}, {"time/lower": time_lower, "time/compile": time_compile}
    )
    train_state, _ = vmap_train(train_config, keys)
    train_state_0 = jax.tree_map(lambda x: x[0], train_state)
    logger.collect_log_step()
    mle_logger.update(
        {"step": train_state_0.global_step.item()}, {}, train_state_0, save=True
    )


if __name__ == "__main__":
    from mle_toolbox import MLExperiment

    mle = MLExperiment()
    main(mle.train_config, mle.log)
