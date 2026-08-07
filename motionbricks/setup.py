from setuptools import setup, find_packages

setup(
    name="motionbricks",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "numpy",
        "mujoco>=3.0",
        "scipy",
        "hydra-core",
        "omegaconf",
        "pytorch-lightning",
        "transformers",
        "pynput",
        "matplotlib",
        "vector-quantize-pytorch",
        "colorlog",
        "adam-atan2-pytorch",
    ],
)
