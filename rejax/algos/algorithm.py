import jax
import optax
from flax.struct import PyTreeNode

from rejax.normalize import RMSState, update_rms


class Algorithm:
    @classmethod
    def initialize_train_state(cls, config, rng):
        # Initialize optimizer
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.learning_rate, eps=1e-5),
        )

        # Initialize environment
        rng, env_rng = jax.random.split(rng)
        vmap_reset = jax.vmap(config.env.reset, in_axes=(0, None))
        obs, env_state = vmap_reset(
            jax.random.split(env_rng, config.num_envs), config.env_params
        )

        # Initialize observation normalization
        rms_state = RMSState.create(obs.shape[1:])
        if config.normalize_observations:
            rms_state = update_rms(rms_state, obs)

        # Initialize network parameters and train states
        rng, network_rng = jax.random.split(rng)
        network_state = cls.initialize_network_params(config, network_rng, obs, tx)

        # Combine everything into a final train state
        return cls.create_train_state(
            config, network_state, env_state, obs, rms_state, rng
        )

    @classmethod
    def initialize_network_params(cls, config, rng, obs, tx):
        # This method should be implemented by each algorithm
        raise NotImplementedError

    @classmethod
    def create_train_state(cls, config, network_state, env_state, obs, rms_state, rng):
        # This method should be implemented by each algorithm
        raise NotImplementedError

    @classmethod
    def train(cls, config, rng):
        raise NotImplementedError
