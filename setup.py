from setuptools import find_packages, setup

setup(
    name="pureRL",
    version="0.0.1",
    url="https://github.com/keraJLi/pureRL",
    author="Jarek Liesen",
    description="Minimal implementations of RL algorithms in jax",
    packages=find_packages(),
    install_requires=[
        "flax",
        "gymnax",
        "distrax",
        "optax",
        "numpy",
    ],
)
