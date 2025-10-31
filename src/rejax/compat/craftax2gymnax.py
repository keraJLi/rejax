from craftax.craftax.craftax_state import EnvParams as CraftaxEnvParams
from craftax.craftax_env import make_craftax_env_from_name
from flax import struct


def create_craftax(env_name, auto_reset=True):
    env = make_craftax_env_from_name(env_name, auto_reset=auto_reset)
    # Wrap params to add max_steps_in_episode
    env_params = EnvParams(
        _base=env.default_params, max_steps_in_episode=env.default_params.max_timesteps
    )
    return env, env_params


@struct.dataclass
class EnvParams:
    _base: CraftaxEnvParams
    max_steps_in_episode: int = struct.field(pytree_node=False, default=None)

    def __getattr__(self, name):
        # Delegate to _base for attributes not found on wrapper
        try:
            return getattr(self._base, name)
        except AttributeError as err:
            raise AttributeError(
                f"{type(self).__name__} has no attribute {name!r}"
            ) from err

    def __repr__(self):
        base_repr = ", ".join(f"{k}={v!r}" for k, v in vars(self._base).items())
        return (
            f"{type(self).__name__}({base_repr}, "
            f"max_steps_in_episode={self.max_steps_in_episode!r})"
        )


# Override the replace method after the dataclass decorator runs
_original_replace = EnvParams.replace


def _custom_replace(self, **updates):
    """Custom replace with auto-sync between max_steps_in_episode and max_timesteps."""
    top = {"_base", "max_steps_in_episode"}
    base_updates = {k: v for k, v in updates.items() if k not in top}
    top_updates = {k: v for k, v in updates.items() if k in top}

    # Auto-sync: if max_steps_in_episode is updated, sync max_timesteps
    if "max_steps_in_episode" in top_updates:
        base_updates["max_timesteps"] = top_updates["max_steps_in_episode"]

    # Auto-sync: if max_timesteps is updated, sync max_steps_in_episode
    if "max_timesteps" in base_updates:
        top_updates["max_steps_in_episode"] = base_updates["max_timesteps"]

    # Update base with base_updates
    base = self._base
    if base_updates:
        if hasattr(base, "replace"):
            base = base.replace(**base_updates)
        else:
            d = dict(vars(base))
            d.update(base_updates)
            base = base.__class__(**d)

    # Use the original dataclass replace for top-level fields
    final_updates = top_updates.copy()
    final_updates["_base"] = base
    return _original_replace(self, **final_updates)


EnvParams.replace = _custom_replace
