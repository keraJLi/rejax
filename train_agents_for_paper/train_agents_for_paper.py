import jax
import time
import json
import numpy as np
from yaml import load, Loader
from purerl.algos import get_agent


def main(config):
    algo, env, num_seeds, seed_id = (
        config.pop("algo"),
        config.pop("env"),
        config.pop("num_seeds"),
        config.pop("seed_id"),
    )

    with open(f"../purerl/configs/{env}.yaml") as f:
        config = load(f, Loader=Loader)[algo]

    # Prepare train function and config
    train_fn, config_cls = get_agent(algo)
    train_config = config_cls.from_dict(config)
    key = jax.random.PRNGKey(seed_id)

    # Time compilation
    start = time.process_time()
    lowered_train_fn = train_fn.lower(train_config, key)
    time_lower = time.process_time() - start
    compiled_train_fn = lowered_train_fn.compile()
    time_compile = time.process_time() - time_lower

    start_training = time.process_time()
    compiled_train_fn(train_config, key)
    time_train = time.process_time() - start_training
    with open(f"{env}_{algo}_times.json", "w+") as f:
        json.dump(
            {
                "time/lower": time_lower,
                "time/compile": time_compile,
                "time/train": time_train,
                "config": config,
            },
            f,
        )

    # Collect curves
    keys = jax.random.split(key, num_seeds)
    _, (lengths, returns) = jax.vmap(train_fn, in_axes=(None, 0))(train_config, keys)
    np.savez_compressed(
        f"{env}_{algo}",
        lengths=lengths,
        returns=returns,
        steps=np.arange(0, lengths.shape[1]) * train_config.eval_freq,
    )


if __name__ == "__main__":
    from mle_toolbox import MLExperiment

    mle = MLExperiment()
    main(mle.train_config)
