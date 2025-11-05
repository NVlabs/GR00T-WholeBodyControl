from setuptools import find_packages, setup


setup(
    name="robocasa",
    packages=[package for package in find_packages() if package.startswith("robocasa")],
    install_requires=[
        "numpy==1.26.4",
        "numba==0.61.0",
        "scipy>=1.2.3",
        "mujoco==3.2.6",
        "pygame",
        "Pillow",
        "opencv-python",
        "pyyaml",
        "pynput",
        "tqdm",
        "termcolor",
        "imageio",
        "h5py",
        "lxml",
        "hidapi",
        "tianshou==0.4.10",
    ],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
    description="Gr00t RoboCasa for loco-manipulation",
    version="0.2.0",
)
