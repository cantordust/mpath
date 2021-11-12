# ------------------------------------------------------------------------------
# Global imports
# ------------------------------------------------------------------------------
from typing import List
from typing import Optional

from pathlib import Path

import torch as pt

import numpy as np

from mpath.layer import Retina
from mpath.layer import Dense


class Network:

    """
    A stateful network consisting of a sensor layer and one or more deeper layers.
    """

    def __init__(
        self,
        _layer_sizes: List[int],
        _layer_tau: Optional[List[float]] = None,
        _learn: bool = True,
        _min_weight: float = -1.0,
        _max_weight: float = 1.0,
        _learning_rate: float = 0.05,
        _keep_activations: bool = False,
        _keep_weights: bool = False,
    ):

        """
        Initialise a network with a retinal layer and a number of deeper layers.
        """

        assert _layer_tau is None or len(_layer_tau) == len(
            _layer_sizes
        ), f"==[ Error: You have to provide either all or none of the membrane time constants."

        # Create the retina
        self.retina = Retina(
            _size=_layer_sizes[0],
            _tau=_layer_tau[0] if _layer_tau else None,
            _activation_history=_keep_activations,
        )

        # Create layers.
        # Retinal layers have ON and OFF cells, so they are actually twice the size of the input variables.

        self.layers = []
        input_size = 2 * _layer_sizes[0]

        for idx, layer_size in enumerate(_layer_sizes[1:]):
            self.layers.append(
                Dense(
                    size=layer_size,
                    input_size=input_size,
                    tau=None if _layer_tau is None else _layer_tau[idx],
                    min_weight=_min_weight,
                    max_weight=_max_weight,
                    learn=_learn,
                    learning_rate=_learning_rate,
                    activation_history=_keep_activations,
                    weight_history=_keep_weights,
                )
            )

            input_size = layer_size

        # Indicate if we want the network to learn
        self.learn: bool = _learn

        # Learning rate
        self.learning_rate: float = _learning_rate

        # Input history
        self.input_history: Optional[List] = [] if _keep_activations else None

        # Weight history
        self.weight_history: Optional[pt.Tensor] = (
            _keep_weights if _keep_weights else None
        )

    def integrate(
        self,
        _input_signals: pt.Tensor,
    ):

        """
        Integrate input signals and propagate activations through the network.
        """

        if self.input_history is not None:
            self.input_history.append(_input_signals.squeeze_().numpy())

        # Capture inputs with the retina
        self.retina.integrate(_input_signals)

        _input_signals = self.retina.activations

        # Propagate the inputs through the layers
        for layer in self.layers:
            layer.integrate(_input_signals)
            _input_signals = layer.activations

    def freeze(self):

        """
        Stop learning.
        """

        for layer in self.layers:
            layer.learn = False

    def unfreeze(self):

        """
        Resume learning.
        """

        for layer in self.layers:
            layer.learn = True

    def _params(self):

        """
        Get the network parameters.
        """

        params = {}

        params[f"lr"] = np.array([self.learning_rate])
        params[f"tau_ret"] = self.retina.tau.numpy()
        params[f"size_ret"] = np.array(self.retina.activations.shape)

        for idx, layer in enumerate(self.layers, 1):
            params[f"tau_{idx}"] = layer.tau.numpy()
            params[f"wrange_{idx}"] = layer.weight_range
            params[f"size_{idx}"] = np.array(layer.activations.shape)

        return params

    def _activation_history(self):

        """
        Network activation history.
        """

        if self.input_history is None:
            return None

        activation_history = {}

        activation_history["hist_in"] = np.array(self.input_history)
        activation_history["hist_ret"] = np.array(self.retina.activation_history)

        for idx, layer in enumerate(self.layers, 1):
            activation_history[f"hist_act{idx}"] = np.array(layer.activation_history)

        return activation_history

    def _weight_history(self):

        """
        Network weight history.
        """

        if self.weight_history is None:
            return None

        weight_history = {}

        for idx, layer in enumerate(self.layers, 1):
            weight_history[f"wt_{idx}"] = np.array(layer.weight_history)

        return weight_history

    def save(
        self,
        path: str,
    ):

        print(f"==[ Saving network parameters...")

        # Cretate the path.
        path = Path(path)
        path.mkdir(parents=True)

        params = self._params()

        activation_history = self._activation_history()

        if activation_history is None:
            print(
                f"==[ It seems that the network did not keep its input and activation history."
            )
            print(
                f'==[ Create a network with "_keep_activations = True" and try again.'
            )

        else:
            params.update(activation_history)

        weight_history = self._weight_history()

        if weight_history is None:
            print(f"==[ It seems that the network did not keep its weight history.")
            print(f'==[ Create a network with "_keep_weights = True" and try again.')

        else:
            params.update(weight_history)

        np.savez(path / "params.npz", **params)
