# ------------------------------------------------------------------------------
# Global imports
# ------------------------------------------------------------------------------
# *** Type hints *** #
from typing import Optional

# *** PyTorch & numerical libs *** #
import torch as pt


# ------------------------------------------------------------------------------
# Local imports
# ------------------------------------------------------------------------------
from mpath.layer.proto import ProtoLayer
from mpath.layer.ganglion import RGCLayer


class Retina(ProtoLayer):

    """
    A retinal layer aims to emulate the operation of the retina
    with separate ON- and OFF-type RGCs.
    """

    def __init__(
        self,
        _size: int,
        _tau: Optional[pt.Tensor] = None,
        *args,
        **kwargs,
    ):  # Activation is graded.

        # Half of the RGCs are ON-type (activated by positive deviations from the mean input)
        # and the other half are OFF-type (activated by negative deviations from the mean input)

        if _tau is None:
            _tau = pt.full((_size,), 100.0)

        super().__init__(_size, _tau, *args, **kwargs)

        # ON RGCs
        self.on = RGCLayer(_off=False, _size=_size)

        # OFF RGCs
        self.off = RGCLayer(_off=True, _size=_size)

        # The size of the retinal layer is twice the size of the input
        self.activations = pt.zeros(2 * _size)

    def _interleave(self):

        """
        Interleave on/off activations.
        """

        self.activations = pt.reshape(
            pt.stack((self.on.activations, self.off.activations), axis=1),
            (-1,),
        )

        if self.activation_history is not None:
            self.activation_history.append(self.activations.squeeze_().numpy())

    def integrate(
        self,
        _input_signals: pt.Tensor,
    ):

        """
        Integrate raw input signals and trigger action potentials.
        """

        # Integrate input signals.
        self.potentials = _input_signals

        # Get the normalised deviation
        norm_dev = self._norm_deviation()

        # Update the stats
        self._update_potential_stats()

        # Compute the activations of ON and OFF cells
        self.on._activate(norm_dev)
        self.off._activate(norm_dev)

        # Interleave the activations of on and off cells.
        self._interleave()
