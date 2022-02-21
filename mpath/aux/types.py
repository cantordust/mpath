# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
import enum


class View(enum.Enum):
    """
    Frame plot views.
    """

    Original = enum.auto()
    LocalMean = enum.auto()
    Normalised = enum.auto()
