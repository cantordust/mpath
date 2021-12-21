# ------------------------------------------------------------------------------
# Global imports
# ------------------------------------------------------------------------------
# *** Type hints *** #
from typing import Optional

# *** NN & numerical libs *** #
import torch as pt
import numpy as np

# ------------------------------------------------------------------------------
# Local imports
# ------------------------------------------------------------------------------
from mpath.layer.proto import TemporalAdaptationLayer


class Dense(TemporalAdaptationLayer):

    """
    A dense layer of neurons with incoming connections.
    """

    def __init__(
        self,
        _size: int,
        _input_size: int,
        _tau: float = None,
        _min_weight: float = -1.0,
        _max_weight: float = 1.0,
        _learn: bool = True,
        _learning_rate: float = 0.05,
        _keep_activations: bool = False,
        _keep_weights: bool = False,
    ):

        """
        A layer consisting of stateful neurons and inbound connections providing innervation signals.
        """

        # Learning toggle.
        self.learn = _learn
        self.learning_rate = _learning_rate
        self.stdp_steps = 0

        self.weight_range = np.array([_min_weight, _max_weight])

        # Weights for connections between the preceding layer and the current layer.
        self.weights = pt.FloatTensor(_size, _input_size).uniform_(
            _min_weight,
            _max_weight,
        )
        self.matmul_op = pt.matmul

        # Weight history
        self.weight_history = [] if _keep_weights else None

        super().__init__(_size, _tau, _keep_activations)

    def _stdp(
        self,
        _input_signals: pt.Tensor,
    ):

        """
        Generalised Hebbian rule suitable for a multi-neuron layer.
        For now, this only works with dense weights.
        """

        # Lower triangular subtensor of the outer product of the layer activations
        lt = pt.tril(pt.outer(self.activations, self.activations))

        # Weight adjustment
        self.weights += self.learning_rate * (
            pt.outer(self.activations, _input_signals) - pt.matmul(lt, self.weights)
        )

        # Normalise each row of weights
        self.weights = pt.nn.functional.normalize(self.weights, dim=1)

        # Reduce the learning rate every 10 steps
        self.stdp_steps += 1
        if self.stdp_steps == 10:
            self.learning_rate *= 0.99
            self.stdp_steps = 0

    def integrate(
        self,
        _input_signals: pt.Tensor,
    ):

        """
        Integrate incoming signals, update input statistics and trigger action potentials.
        """

        # Membrane potential and activity decay.
        self._decay()

        # Compute the input potentials from the input signals scaled by the synaptic weights.
        self.potentials += self.matmul_op(self.weights, _input_signals)

        # Trigger action potentials.
        self._activate()

        # STDP
        if self.learn:
            self._stdp(_input_signals)

        if self.weight_history is not None:
            self.weight_history.append(self.weights.numpy())
