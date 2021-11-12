# ------------------------------------------------------------------------------
# Global imports
# ------------------------------------------------------------------------------
# *** Type hints *** #
from typing import Optional

# *** PyTorch & numerical libs *** #
import torch as pt


class RGCLayer:

    """
    A group of ON- or OFF-type RGCs.
    """

    def __init__(
        self,
        _width: int,
        _height: int,
        _off: bool = False,
    ):

        # Photoreceptor field.
        # The field itself if _width * _height, but it is
        # padded to allow the fovea to be albe to look at
        # any part of the input image, all the way to the edge.
        self.receptors = pt.zeros(2 * _width, 2 * _height)

        # Comparison operator
        self.comp_op = pt.lt if _off else pt.gt

        # Actual activations
        self.activations = pt.zeros(_width)

        # A uniform baseline used in the element-wise
        # comparison with pt.where
        self.baseline = pt.zeros(_width)

    def _padded_rf(self):

        field = pt.zeros()

    def _activate(self, norm_dev):

        # Get the normalised deviation from the mean.
        # The activation depends on whether the cells are ON or OFF
        norm_dev = pt.where(
            self.comp_op(norm_dev, self.baseline),
            pt.abs(norm_dev),
            self.baseline,
        )

        # Compute the new activations.
        self.activations = pt.tanh(norm_dev)
