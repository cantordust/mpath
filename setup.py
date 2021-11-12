# ------------------------------------------------------------------------------
# Setup function
# ------------------------------------------------------------------------------
from distutils.core import setup

# ------------------------------------------------------------------------------
# Enum (used for versioning)
# ------------------------------------------------------------------------------
from enum import IntEnum


class Version(IntEnum):

    major = 0
    minor = 2
    patch = 0


setup(
    name="mpath",
    author="Alexander Hadjiivanov",
    version=f"{Version.major}.{Version.minor}.{Version.patch}",
    packages=["mpath"],
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
        "PyQt5",
    ],
    license="MIT",
    long_description=open("README.md").read(),
)
