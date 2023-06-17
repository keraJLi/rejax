# pureRL

Minimal reinforcement learning algorithms in jax.
Leverage the full potential of jax with fully `jax.jit`able and `jax.vmap`pable training loops!

```python
from purerl.algos import get_agent
from mle_logging import load_config

config_dict = load_config("purerl/configs/cartpole.yaml")

train_fn, config_cls = get_agent("sac")
train_config = config_cls.from_dict(config_dict)

keys = jax.random.split(jax.random.PRNGKey(0), 10)
vmap_train = jax.vmap(jax.jit(train_fn), in_axes=(None, 0))
train_state, evaluation = vmap_train(train_config, keys)
```
