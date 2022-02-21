# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
from distutils.core import setup

# --------------------------------------
from enum import IntEnum


class Version(IntEnum):

    major = 0
    minor = 3
    patch = 1


setup(
    name="mpath",
    author="Alexander Hadjiivanov",
    version=f"{Version.major}.{Version.minor}.{Version.patch}",
    packages=["mpath"],
    install_requires=[
        "torch",
        "numpy",
        "dotmap",
        "plotly",
        "python-opencv",
    ],
    license="MIT",
    long_description=open("README.md").read(),
)
