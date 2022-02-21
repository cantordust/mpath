# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
from typing import Any
from typing import List
from typing import Optional

# --------------------------------------
from pathlib import Path

# --------------------------------------
import numpy as np

# --------------------------------------
import torch as pt

# --------------------------------------
from mpath.aux.types import Param
from mpath.layer import Layer

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
pt.set_default_dtype(pt.float32)
pt.set_default_tensor_type(pt.FloatTensor)


class Network:

    """
    A stateful network consisting of a sensor layer and one or more deeper layers.
    """

    ##[ Private methods ]##

    def __init__(
        self,
        _layers: List[Layer],
        _name: str = "Network",
    ):

        # Store the layers
        self.layers = _layers

        # Assign generic names to anonymous layers
        for idx, layer in enumerate(self.layers, 1):
            if layer.name == "":
                layer.name = f"Layer {idx}"

        # Summary writer operations.

    # ==[ Protected methods ]==

    def _get_params(self):

        params = {}
        for idx, layer in enumerate(self.layers, 1):
            params[layer.name] = {idx: getter() for p, getter in layer.params.items()}

        return params

    def _get_history(self):

        history = []

        for idx, layer in enumerate(self.layers, 1):
            history.append(layer.history)

        return history

    ##[ Public methods ]##

    def forward(
        self,
        _input: pt.Tensor,
    ):

        """
        Integrate input signals and propagate activations through the network.
        """
        # Propagate the inputs through the layers
        for layer in self.layers:
            layer.forward(_input)
            _input = layer.activations

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

    def save(
        self,
        path: str,
    ):

        print(f"==[ Saving network...")

        # Cretate the path.
        path = Path(path)
        path.mkdir(parents=True)

        pt.save(
            self._get_params(),
            path / "network.pt",
        )
