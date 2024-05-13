from flax.struct import PyTreeNode


class Algorithm:
    @classmethod
    def train(cls, config, rng):
        raise NotImplementedError
