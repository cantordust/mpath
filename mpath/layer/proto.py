# ------------------------------------------------------------------------------
# Global imports
# ------------------------------------------------------------------------------
# *** Type hints *** #
from typing import Optional

# *** NN & numerical libs *** #
import torch as pt
import numpy as np


class TemporalAdaptationLayer:
    """
    A proto-layer containing only neurons (no synapses).
    All aspects of neuron dynamics (potentiation, activation, etc.)
    except input integration are handled here.
    Input integration is handled separately by each derived class.
    """

    def __init__(
        self,
        _size: int,
        _tau: Optional[np.ndarray] = None,
        _threshold: bool = False,
        _activation_history: bool = False,
    ):

        # A layer consists of an array of stateful neurons,
        # each with a dynamic membrane potential.
        self.potentials = pt.zeros(_size)

        # Membrane time constants.
        # This is the only parameter defining the neuron state.
        if _tau is None:
            tau_min = 5.0
            tau_max = 30.0
            _tau = pt.Tensor(
                np.reshape(
                    np.linspace(tau_min, tau_max, _size),
                    (_size,),
                ),
            )

        # Membrane potentials.
        self.tau = _tau

        # "Forgetting rates" for the membrane potential EMA / EMV.
        # Also used for computing the decay of the membrane potential
        # and the activation in the absence of input.
        self.alpha = 1.0 / _tau

        # Threshold time constant.
        # The threshold adapts much faster than the membrane potential.
        self.threshold_alpha = 10 * self.alpha if _threshold else None
        self.activation_alpha = self.threshold_alpha if _threshold else None

        # Exponential moving average and variance for the membrane potentials.
        self.potential_avg = pt.zeros_like(self.potentials)
        self.potential_var = pt.zeros_like(self.potentials)

        # Neuron activations.
        # These decay gradually at the same rate as the membrane potential.
        self.activations = pt.zeros_like(self.potentials)

        # Baseline (a tensor of 0s) used for resetting potentials and activations.
        self.baseline = pt.zeros_like(self.potentials)

        # Activation history
        self.activation_history = [] if _activation_history else None

    def _update_potential_stats(self):

        """
        Compute the running mean and variance of the membrane potentials.
        """

        diff = self.potentials - self.potential_avg
        inc = self.alpha * diff
        self.potential_avg += inc
        self.potential_var = (1.0 - self.alpha) * (self.potential_var + diff * inc)

    def _norm_deviation(self):

        """
        Normalised deviation from the mean membrane potential for each neuron.
        """

        # Use nan_to_num_() to ensure that we don't crash if the SD is 0.
        return (self.potentials - self.potential_avg) / pt.sqrt(
            self.potential_var
        ).nan_to_num_()

    def _decay(self):

        """
        Membrane repolarisation (membrane potential decay) and activation decay.
        """

        # Exponential decay of the membrane potential and the total activation.
        self.potentials *= pt.exp(-self.alpha)
        self.activations *= pt.exp(-self.activation_alpha)

        # print(f"==[ Potentials: {self.potentials.shape}")
        # print(f"==[ Activations: {self.activations.shape}")

        # Update the average potential.
        self._update_potential_stats()

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
        # The average potential will be updated at the next step.
        self.potentials = pt.where(activations, self.baseline, self.potentials)

        # Append activations to the history (used for plotting the activations against time)
        if self.activation_history is not None:
            self.activation_history.append(self.activations.squeeze_().numpy())
