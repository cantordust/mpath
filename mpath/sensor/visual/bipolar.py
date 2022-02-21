# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
from typing import Optional

# --------------------------------------
import torch as pt


class BipolarLayer:

    """
    A field of bipolar cells.
    This transforms the image in the receptor
    and passes it to the RGC layer.
    """

    def __init__(
        self,
        _source_width: int,
        _source_height: int,
    ):

        # Bipolar cell field.
        # The field itself is _width * _height, but it is
        # padded to allow the fovea to be albe to look at
        # any part of the input image, all the way to the edge.
        self.bp = pt.zeros(2 * _source_width, 2 * _source_height)

    def _pad(self):

        field = pt.zeros()

    def _convolve(
        self,
        _input: pt.Tensor,
        _xshift: float,
        _yshift: float,
    ):

        patch = self.bp.view()

        return self.bp * _input

    def make_rf(_size: int):
        """
        Bipolar cell RF.

        Args:
            _size (int):
                Size of the receptive field.
        """
        pass
