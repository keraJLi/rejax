from flax import struct
from kinetix.environment import make_kinetix_env
from kinetix.environment.spaces import ActionType, ObservationType
from kinetix.environment.spaces import EnvParams as KinetixEnvParams
from kinetix.util import load_from_json_file


def create_kinetix(env_name, **kwargs):
    """Create a Kinetix environment. The `env_name` argument is the name of the
    level file, e.g. `"l/grasp_easy"`.
    """
    level, static_env_params, env_params = load_from_json_file(f"{env_name}.json")

    default_kwargs = {
        "action_type": ActionType.CONTINUOUS,
        "observation_type": ObservationType.SYMBOLIC_FLAT,
        "reset_fn": lambda rng: level,
        "env_params": env_params,
        "static_env_params": static_env_params,
    }
    default_kwargs.update(kwargs)

    env = make_kinetix_env(**default_kwargs)

    # Very hacky! But rejax generally assumes that max_steps_in_episode exists. Once we
    # try to replace this it will get dicey. This is on the Mikeys for not adhering to
    # the gymnax API!
    env_params = EnvParams(
        _base=env_params, max_steps_in_episode=env_params.max_timesteps
    )
    return env, env_params


@struct.dataclass
class EnvParams:
    _base: KinetixEnvParams
    max_steps_in_episode: int = struct.field(pytree_node=False, default=None)

    def __getattr__(self, name):
        # only called if attribute not found on wrapper
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
