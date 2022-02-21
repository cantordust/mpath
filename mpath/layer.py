# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
from typing import Any
from typing import Set
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional

# --------------------------------------
import numpy as np

# --------------------------------------
import torch as pt
from torch.utils.tensorboard import SummaryWriter

# --------------------------------------
import enum

# --------------------------------------
from mpath.aux.types import Param


class LearningMethod(enum.Enum):
    STDP = enum.auto()
    Backprop = enum.auto()


class Layer:
    """
    A proto-layer containing neurons and associated weights.
    All aspects of neuron dynamics (potentiation, activation, etc.)
    except input integration are handled here.
    Forward propagation is handled separately by each derived class.
    """

    ##[ Public methods ]##

    def __init__(
        self,
        _input_shape: Tuple[int],
        _output_shape: Tuple[int],
        _min_weight: float = -1.0,
        _max_weight: float = 1.0,
        _tau: Union[pt.Tensor, Tuple[float, float]] = (5.0, 50.0),
        _threshold_alpha: Optional[float] = None,
        _learn: bool = True,
        _track: Optional[Set[Param]] = None,
        _sparse: bool = False,
        _learning_method: LearningMethod = LearningMethod.STDP,
        _learning_rate: Optional[float] = None,
        _name: str = "",
    ):

        # ==[ Store init parameters ]==

        self.min_weight = _min_weight
        self.max_weight = _max_weight

        self.input_shape = _input_shape
        self.output_shape = _output_shape

        # Matrix multiplication operation
        # TODO: Implement sparse matrix multiplication (uncomment second line below)
        self.matmul_op = pt.matmul
        # self.matmul_op = pt.matmul if not _sparse else pt.smm

        # Layer name
        self.name = _name

        # Learning switch.
        self.learn = _learn

        # Learning toggle.
        self.learning_method = _learning_method
        self.learning_rate = _learning_rate

        self.stdp_steps = 0

        # ==[ Layer parameters ]==
        # Membrane time constants.
        # This is the only parameter defining the neuron state.
        if isinstance(_tau, tuple):
            if len(_tau) != 2:
                raise ValueError(
                    "Please provide a tuple of exactly two floatint-point numbers."
                )

            tau_min = _tau[0]
            tau_max = _tau[1]
            _tau = pt.reshape(
                pt.linspace(tau_min, tau_max, pt.prod(_output_shape)), _output_shape
            )

        elif isinstance(_tau, pt.Tensor):

            # Sanity check
            if pt.count_nonzero(pt.any(_tau <= 0.0)).item() > 0:
                raise ValueError(
                    "The membrane time constant for each neuron must be greater than 0."
                )

        self.tau = _tau

        # "Forgetting rate" for the membrane potential EMA / EMV of each neuron.
        # Also used for computing the decay of the membrane potential
        # and the activation in the absence of input.
        # Adding 1.0 is a sanity check to ensure that alpha
        # is less than 1.
        self.alpha = 1.0 / (_tau + 1.0)

        # Threshold time constant.
        # The threshold adapts much faster than the membrane potential.
        # TODO: Separate values for the threshold and activation forgetting rates.

        if _threshold_alpha is None:
            # The threshold
            _threshold_alpha = 10 * self.alpha

        self.threshold_alpha = _threshold_alpha
        self.activation_alpha = _threshold_alpha

        # A layer consists of an array of stateful neurons,
        # each with a dynamic membrane potential.
        self.potentials = pt.zeros(_output_shape)

        # Baseline (a tensor of 0s) used for resetting potentials and activations.
        self.baseline = pt.zeros_like(self.potentials)

        # Exponential moving mean and variance for the membrane potentials.
        self.potential_mean = pt.zeros_like(self.potentials)
        self.potential_var = pt.zeros_like(self.potentials)

        # Neuron activations.
        # These decay gradually at the same rate as the membrane potential.
        self.activations = pt.zeros_like(self.potentials)

        self.stdp_steps = 0

        # Weights for connections between the preceding layer and the current layer.
        # TODO: Sparse weights
        self.weights = pt.FloatTensor(
            [np.prod(_output_shape), np.prod(_input_shape)]
        ).uniform_(
            _min_weight,
            _max_weight,
        )

        # Parameter dictionary
        self.params = {
            Param.Input: lambda: None,  # Return the input tensor
            Param.Tau: lambda: self.tau,
            Param.Alpha: lambda: self.alpha,
            Param.ThresholdAlpha: lambda: self.threshold_alpha,
            Param.ActivationAlpha: lambda: self.activation_alpha,
            Param.Potentials: lambda: self.potentials,
            Param.PotentialMean: lambda: self.potential_mean,
            Param.PotentialVar: lambda: self.potential_var,
            Param.Activations: lambda: self.activations,
            Param.Weights: lambda: self.weights,
        }

        # Parameters to keep track of
        self.track = _track if _track is not None else set()

        # Parameter history
        self.history = {t: [] for t in self.track}

    def _get_param(self, _param: Param):

        return self.params[_param]()

    def _update_potential_stats(self):

        """
        Compute the running mean and variance of the membrane potentials.
        """

        diff = self.potentials - self.potential_mean
        inc = self.alpha * diff
        self.potential_mean += inc
        self.potential_var = (1.0 - self.alpha) * (self.potential_var + diff * inc)

    def _norm_deviation(self):

        """
        Normalised deviation from the mean membrane potential for each neuron.
        """

        # Use nan_to_num_() to avoid crashing if the SD is 0.
        return (self.potentials - self.potential_mean) / pt.sqrt(
            self.potential_var
        ).nan_to_num_()

    def _decay(self):

        """
        Membrane repolarisation (membrane potential decay) and activation decay.
        """

        # Exponential decay of the membrane potential and the total activation.
        self.potentials *= pt.exp(-self.alpha)
        self.activations *= pt.exp(-self.activation_alpha)

        # Update the mean potential.
        self._update_potential_stats()

    def _update_history(
        self,
        _step: int,
    ):
        for param in self.track:
            self.history[param].append(self.params[param]())

    def _activate(self):

        """
        The normalised deviation ND is defined by the mean (V_{\mu}) and standard deviation (V_{\sigma})
        values of the membrane potential of the respective neuron:

            ND = \frac{V - V_{\mu}}{V_{\sigma}}

        The activation profile of a neuron is in the shape of a sigmoid (tanh), which approaches 1 asymptotically:

            \rho = \tanh(norm_diff)

        The activation threshold \theta is defined as follows:

            \theta = \exp(- \threshold_alpha * norm_diff)

        Neurons produce action potentials at a (normalised) rate \eta when the potential crosses the activation threshold, i.e., when

            \eta = \rho - \theta > 0
        """

        # Compute the normalised deviation from the mean.
        norm_dev = self._norm_deviation()

        # Update the moving stats for the membrane potentials.
        self._update_potential_stats()

        # Neurons are activated when the membrane potential crosses the activation threshold
        theta = pt.exp(-self.threshold_alpha * norm_dev)

        # Compute the new activations
        rho = pt.tanh(norm_dev) - theta
        activations = pt.greater_equal(rho, self.baseline)
        self.activations = pt.where(activations, rho, self.activations)

        # Reset potentials to 0 for neurons that have produced action potentials.
        # The mean potential will be updated at the next step.
        self.potentials = pt.where(activations, self.baseline, self.potentials)

    ##[ Protected methods ]##

    def _update_history(
        self,
        _step: int,
    ):
        for param in self.track:
            self.history[param].append(self.params[param]())

    def _stdp(
        self,
        _input: pt.Tensor,
    ):

        """
        Generalised Hebbian rule suitable for a multi-neuron layer.
        For now, this only works with dense weights.
        """

        # Lower triangular subtensor of the outer product of the layer activations
        lt = pt.tril(pt.outer(self.activations, self.activations))

        # Weight adjustment
        self.weights += self.learning_rate * (
            pt.outer(self.activations, _input) - self.matmul_op(lt, self.weights)
        )

        # Normalise each row of weights
        self.weights = pt.nn.functional.normalize(self.weights, dim=1)

        # Reduce the learning rate every 10 steps
        self.stdp_steps += 1
        if self.stdp_steps == 10:
            self.learning_rate *= 0.99
            self.stdp_steps = 0

    ##[ Public methods ]##

    def forward(
        self,
        _input: pt.Tensor,
    ):

        """
        Integrate incoming signals, update input statistics and trigger action potentials.
        """

        # Membrane potential and activity decay.
        if self.tau is not None:
            self._decay()

        # Compute the input potentials from the input signals scaled by the synaptic weights.
        self.potentials += self.matmul_op(self.weights, _input)

        # Trigger action potentials.
        self._activate()

        # STDP
        # TODO: Other learning methods.
        # TODO: Learning methods as classes.
        if self.learning_method == LearningMethod.STDP:
            self._stdp(_input)

        # Store all tracked parameters.
        self._update_history()
