# pureRL

Fully vectorizable reinforcement learning algorithms in jax!

## Vectorize training for incledible speedups!
`vmap` over initial seeds or hyperparameters to train a whole batch of agents in parallel! 

```python
from purerl.algos import get_agent
from mle_logging import load_config

# Load hyperparameter configuration as dict
config_dict = load_config("purerl/configs/cartpole.yaml")

# Get train function and initialize config for training
train_fn, config_cls = get_agent("sac")
train_config = config_cls.from_dict(config_dict)

# Vmap training function over 300 initial seeds
keys = jax.random.split(jax.random.PRNGKey(0), 300)
vmap_train = jax.vmap(jax.jit(train_fn), in_axes=(None, 0))

# Train 300 agents!
train_state, evaluation = vmap_train(train_config, keys)
```

![Speedup over cleanRL](img/speedup.svg)

Benchmark on an A100 and a Intel Xeon 4215R CPU.

## Algorithms
| Algorithm | Discrete | Continuous        | Notes                     |
|-----------|----------|-------------------|---------------------------|
| PPO       | ✔        | ✔                 |                           |
| SAC       | ✔        | ✔                 | discrete version as in [Christodoulou, 2019](https://arxiv.org/abs/1910.07207)                          |
| DQN       | ✔        |                   | incl. DDQN, Dueling DQN   |
| DDPG      |          | ✔                 |                           |
| TD3       |          | ✔                 |                           |


## Built for researchers
Easily modify the implemented algorithms by overwriting isolated parts, such as the loss function, trajectory generation or parameter updates.
Algorithms are implemented as stateless classes containing only class methods. This allows for an explicit differentiation between

- **Instructions** that make up the algorithm (stateless class)
- **Configuration** including environment, policy networks, hyperparameters, ... (flax.PyTreeNode, partly static)
- **Train state** including the current environment state, network parameters, global step, ... (flax.PyTreeNode)

All algorithms implement `algo.train(config, rng)`, where `config` is a PyTreeNode. This function can be jitted and vmapped over in both inputs. For a quick start, you can get the train function and config class from `get_agent(algorithm: str) -> Tuple[Callable, PyTreeNode]`, and use `config.from_dict` to load a dictionary config.

## Flexible callbacks
The config of all algorithms includes a callback function `callback(config, train_state, rng) -> PyTree`, which is called every `eval_freq` training steps with the config and current train state. The output of the callback will be aggregated over training and returned by the train function. The default callback runs a number of episodes in the training environment and returns their length and episodic return, such that the train function returns a training curve.

Importantly, this function is jit-compiled along with the rest of the algorithm. However, you can use one of Jax' callbacks such as `jax.experimental.io_callback` to implement model checkpoining, logging to wandb, and more, all while maintaining the advantages of a completely jittable training function.
