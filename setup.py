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
    minor = 1
    patch = 9


setup(
    name="mpath",
    author="Alexander Hadjiivanov",
    version=f"{Version.major}.{Version.minor}.{Version.patch}",
    packages=["mpath"],
    install_requires=[
        "tensorflow",
        "numpy",
        "matplotlib",
        "PyQt5",
    ],
    license="MIT",
    long_description=open("README.md").read(),
)