#------------------------------------------------------------------------------
# Path
#------------------------------------------------------------------------------
from pathlib import Path

#------------------------------------------------------------------------------
# TensorFlow
#------------------------------------------------------------------------------
import tensorflow as tf

#------------------------------------------------------------------------------
# Numpy
#------------------------------------------------------------------------------
import numpy as np

#------------------------------------------------------------------------------
# MPATH imports
#------------------------------------------------------------------------------
from mpath.layer import Retina
from mpath.layer import Layer

class Network:

    '''
    A stateful network consisting of a sensor layer and one or more deeper layers.
    '''

    def __init__(self,
                 layer_sizes,
                 layer_tau = None,
                 learn = True,
                 min_weight = -1.0,
                 max_weight = 1.0,
                 learning_rate = 0.05,
                 activation_history = False,
                 weight_history = False):

        '''
        Initialise a network with a retinal layer and a number of deeper layers.
        '''

        assert layer_tau is None or len(layer_tau) == len(layer_sizes), f'==[ Error: You have to provide either all or none of the membrane time constants.'

        # Create the retina
        self.retina = Retina(size = layer_sizes[0],
                             tau = layer_tau[0] if layer_tau else None,
                             activation_history = activation_history)


        # Create layers.
        # Retinal layers have ON and OFF cells, so they are actually twice the size of the input variables.

        self.layers = []
        input_size = 2 * layer_sizes[0]

        for idx, layer_size in enumerate(layer_sizes[1:]):
            self.layers.append(Layer(size = layer_size,
                                     input_size = input_size,
                                     tau = None if layer_tau is None else layer_tau[idx],
                                     min_weight = min_weight,
                                     max_weight = max_weight,
                                     learn = learn,
                                     learning_rate = learning_rate,
                                     activation_history = activation_history,
                                     weight_history = weight_history))

            input_size = layer_size

        # Indicate if we want the network to learn
        self.learn = learn

        # Learning rate
        self.learning_rate = learning_rate

        # Input history
        self.input_history = [] if activation_history else None

        # Weight history
        self.weight_history = weight_history if weight_history else None

    def integrate(self,
                  input_signals):

        '''
        Integrate input signals and propagate activations through the network.
        '''

        if self.input_history is not None:
            self.input_history.append(tf.squeeze(input_signals, axis = 1).numpy())

        # Capture inputs with the retina
        self.retina.integrate(input_signals)

        input_signals = self.retina.activations

        # Propagate the inputs through the layers
        for layer in self.layers:
            layer.integrate(input_signals)
            input_signals = layer.activations

    def freeze(self):

        '''
        Stop learning.
        '''

        for layer in self.layers:
            layer.learn = False

    def unfreeze(self):

        '''
        Resume learning.
        '''

        for layer in self.layers:
            layer.learn = True

    def _params(self):

        '''
        Get the network parameters.
        '''

        params = {}

        params[f'lr'] = np.array([self.learning_rate])
        params[f'in_size'] = tf.constant(self.retina.activations).numpy()
        params[f'tau_r'] = self.retina.tau.numpy()

        for idx, layer in enumerate(self.layers, 1):
            params[f'tau{idx}'] = layer.tau.numpy()
            params[f'wrange{idx}'] = layer.weight_range
            params[f'size{idx}'] = tf.constant(layer.activations.shape).numpy()

        return params

    def _activation_history(self):

        '''
        Network activation history.
        '''

        if self.input_history is None:
            return None

        activation_history = {}

        activation_history['in'] = np.array(self.input_history)
        activation_history['ret'] = np.array(self.retina.activation_history)

        for idx, layer in enumerate(self.layers, 1):
            activation_history[f'act{idx}'] = np.array(layer.activation_history)

        return activation_history

    def _weight_history(self):

        '''
        Network weight history.
        '''

        if self.weight_history is None:
            return None

        weight_history = {}

        for idx, layer in enumerate(self.layers, 1):
            weight_history[f'wt{idx}'] = np.array(layer.weight_history)

        return weight_history

    def save(self,
             path):

        print(f'==[ Saving network parameters...')

        # Cretate the path.
        path = Path(path)
        path.mkdir(parents = True)

        params = self._params()

        activation_history = self._activation_history()

        if activation_history is None:
            print(f'==[ It seems that the network did not keep its input and activation history.')
            print(f'==[ Create a network with "activation_history = True" and try again.')

        else:
            params.update(activation_history)

        weight_history = self._weight_history()

        if weight_history is None:
            print(f'==[ It seems that the network did not keep its weight history.')
            print(f'==[ Create a network with "weight_history = True" and try again.')

        else:
            params.update(weight_history)

        np.savez(path / 'params.npz', **params)
