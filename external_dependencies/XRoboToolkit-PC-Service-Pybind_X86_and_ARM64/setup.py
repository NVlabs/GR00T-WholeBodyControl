import os
import platform
import re
import shutil  # Added for shutil.rmtree
import subprocess
import sys
from distutils.version import LooseVersion

from setuptools import Command, Extension, find_packages, setup  # Added Command
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r"version\s*([\d.]+)", out.decode()).group(1))
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        # Get pybind11 include paths
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j2"]  # Adjust core count as needed

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)


# New Clean Command
class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Remove build directory
        if os.path.exists("build"):
            print("Removing 'build/' directory")
            shutil.rmtree("build")
        # Remove .egg-info directory
        for item in os.listdir("."):
            if item.endswith(".egg-info"):
                print(f"Removing '{item}' directory")
                shutil.rmtree(item)
        for item in os.listdir("."):
            if item.endswith(".eggs"):
                print(f"Removing '{item}' directory")
                shutil.rmtree(item)
        # Optionally, remove dist directory if you generate distributions
        if os.path.exists("dist"):
            print("Removing 'dist/' directory")
            shutil.rmtree("dist")


# New Uninstall Command
class UninstallCommand(Command):
    """Custom command to uninstall the package."""

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        package_name = self.distribution.get_name()
        print(f"Attempting to uninstall {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package_name])
            print(f"{package_name} uninstalled successfully.")
        except subprocess.CalledProcessError as e:
            print(
                f"Failed to uninstall {package_name}. It may not be installed or pip uninstall failed."
            )
            print(f"Error: {e}")
        except FileNotFoundError:
            print("pip command not found. Please ensure pip is installed and in your PATH.")


setup(
    name="xrobotoolkit_sdk",
    version="1.0.2",
    author="Zhigen Zhao",
    author_email="zhigen.zhao@bytedance.com",
    description="A Python binding for XRobotoolkit PC Service SDK using pybind11 and CMake",
    long_description="",  # Optionally, load from a README.md file
    ext_modules=[CMakeExtension("xrobotoolkit_sdk")],
    cmdclass=dict(
        build_ext=CMakeBuild,
        clean=CleanCommand,  # Add clean command
        uninstall=UninstallCommand,  # Add uninstall command
    ),
    zip_safe=False,
    python_requires=">=3.10",  # Specify your Python version requirement
    packages=find_packages(),  # If you have other Python packages in your project
)
