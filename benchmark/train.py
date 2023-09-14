import os
import jax
import json
import time
import numpy as np
import pandas as pd
import jax.numpy as jnp
from flax import serialization

from purerl.algos import get_agent
from purerl.evaluate import make_evaluate as make_evaluate_vanilla


class Logger:
    def __init__(self, folder, name, metadata, use_wandb):
        self.folder = folder
        self.name = f"{name}_{time.time()}"
        self.metadata = metadata
        self.last_step = 0
        self._log = []
        self._log_step = []
        self.timer = None
        self.use_wandb = use_wandb

        if not os.path.exists(folder):
            os.makedirs(folder)

        print(f"Logging to {os.path.join(folder, name)}.{{json,ckpt}}")

    def log_once(self, data):
        self.metadata = {**self.metadata, **data}
        self.write_log()
        if self.use_wandb:
            for k, v in data.items():
                wandb.run.summary[k] = v

    def collect_log_step(self):
        def convert(x):
            if isinstance(x, (np.ndarray, jnp.ndarray)):
                return x.tolist()
            return x

        process_time = time.process_time() - self.timer

        # Compute mean over initial seeds for wandb, log all stuff for json
        _log_step = jax.tree_map(convert, self._log_step)
        _log_step = pd.DataFrame(_log_step)

        self._log.append(
            {
                "time/process_time": process_time,
                "step": self.last_step,
                **_log_step.to_dict("list"),
            }
        )
        self._log_step = []

        if self.use_wandb:
            wandb.log(
                {
                    "time/process_time": process_time,
                    **_log_step.mean(axis=0).to_dict(),
                },
                step=self.last_step,
            )

    def log(self, data, step):
        step = step.item()  # jax cpu callback returns numpy array

        # Because of vmapping the training function, self.log is called several times
        # sequentially. Therefore we only log once we reach a new global_step
        if step > self.last_step:
            self.collect_log_step()

        self._log_step.append(data)
        self.last_step = step

    def write_log(self):
        file = os.path.join(self.folder, f"{self.name}.json")
        with open(file, "w+") as f:
            data = {
                **self.metadata,
                **pd.DataFrame(self._log).to_dict("list"),
            }
            json.dump(data, f)

    def write_checkpoint(self, ckpt):
        file = os.path.join(self.folder, f"{self.name}.ckpt")
        with open(file, "wb+") as f:
            f.write(serialization.to_bytes(ckpt))

        if self.use_wandb:
            wandb.save(file)

    def reset_timer(self):
        self.timer = time.process_time()


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
                "return": returns.mean(axis=0),
                "return_std": returns.std(axis=0),
            },
            ts.global_step,
        )
        return garbage

    return evaluate


def main(args, config):
    # Initialize logging
    escaped_env = config["env"].replace("/", "_")
    log_name = f"{escaped_env}_{args.algorithm}_{args.num_seeds}_{args.global_seed}"
    metadata = {
        "environment": config["env"],
        "algorithm": args.algorithm,
        "num_seeds": args.num_seeds,
        "global_seed": args.global_seed,
        "config": config,
    }
    logger = Logger(args.log_dir, log_name, metadata, args.use_wandb)
    logger.write_log()
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=metadata,
            name=log_name,
        )

    # Prepare train function and config
    train_fn, config_cls = get_agent(args.algorithm)
    train_config = config_cls.from_dict(config)
    evaluate = make_evaluate(logger, train_config.env, train_config.env_params)
    train_config = train_config.replace(eval_callback=evaluate)

    key = jax.random.PRNGKey(args.global_seed)
    keys = jax.random.split(key, args.num_seeds)
    vmap_train = jax.jit(jax.vmap(train_fn, in_axes=(None, 0)))

    # Time compilation
    start = time.process_time()
    lowered = vmap_train.lower(train_config, keys)
    time_lower = time.process_time() - start
    compiled = lowered.compile()
    time_compile = time.process_time() - time_lower
    vmap_train = compiled

    logger.log_once(
        {
            "time/lower": time_lower,
            "time/compile": time_compile,
        }
    )
    logger.write_log()

    # Train
    logger.reset_timer()
    train_state, _ = vmap_train(train_config, keys)
    logger.collect_log_step()
    logger.write_log()
    if args.save_all_checkpoints:
        logger.write_checkpoint(train_state)
    else:
        train_state = jax.tree_map(lambda x: x[0], train_state)
        logger.write_checkpoint(train_state)


if __name__ == "__main__":
    import argparse
    from yaml import load, CLoader as Loader

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cartpole.yaml",
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of seeds to use.",
    )
    parser.add_argument(
        "--save-all-checkpoints",
        action="store_true",
        help="Save checkpoints of all seeds.",
    )
    parser.add_argument(
        "--global-seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="",
        help="Directory to store logs.",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Use wandb for logging.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="purerl",
        help="Wandb project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="purerl",
        help="Wandb entity name.",
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = load(f, Loader=Loader)

    if args.use_wandb:
        import wandb

    main(args, config[args.algorithm])
