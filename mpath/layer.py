#------------------------------------------------------------------------------
# TensorFlow
#------------------------------------------------------------------------------
import tensorflow as tf

#------------------------------------------------------------------------------
# Numpy
#------------------------------------------------------------------------------
import numpy as np

#------------------------------------------------------------------------------
# Random
#------------------------------------------------------------------------------
import random

#------------------------------------------------------------------------------
# A proto-layer containing only neurons.
# All aspects of neuron dynamics (potentiation, activation, etc.)
# except input integration are handled here.
# Input integration is handled separately by each derived class.
#------------------------------------------------------------------------------

class ProtoLayer:

    '''
    A prototypical layer of neurons.

    This can be used to create sensor layers (which do not have incoming connections)
    or deeper layers with connections to lower layers.
    '''

    def __init__(self,
                 size,
                 tau = None,
                 activation_history = False):

        '''
        Initialise a layer with an array of stateful neurons.
        '''

        # An array of stateful neurons, each with a dynamic membrane potential
        self.potentials = tf.zeros((size, 1))

        # Membrane time constant.
        # This is the only parameter defining the neuron state.
        if tau is None:
            tau_min = 5.0
            tau_max = 30.0
            tau = tf.constant(np.arange(tau_min, tau_max, (tau_max - tau_min) / size ), shape = (size, 1), dtype = tf.float32)

        # Membrane potential
        self.tau = tau

        # "Forgetting rate" for the membrane potential EMA / EMV.
        # Also used for computing the decay of the membrane potential
        # and the activation in the absence of input.
        self.alpha = 1.0 / tau

        # Threshold time constant.
        # The threshold adapts much faster than the membrane potential.
        self.threshold_alpha = 10 * self.alpha
        self.activation_alpha = self.threshold_alpha

        # Exponential moving average and variance for the membrane potentials.
        self.potential_avg = tf.zeros_like(self.potentials)
        self.potential_var = tf.zeros_like(self.potentials)

        # Neuron activations.
        # These decay gradually at the same rate as the membrane potential.
        self.activations = tf.zeros_like(self.potentials)

        # Baseline (a tensor of 0s) used for resetting potentials and activations.
        self.baseline = tf.zeros_like(self.potentials)

        # Activation history
        self.activation_history = [] if activation_history else None

    def _update_potential_stats(self):

        '''
        Compute the running mean and variance of the membrane potentials.
        '''

        diff = self.potentials - self.potential_avg
        inc = self.alpha * diff
        self.potential_avg += inc
        self.potential_var = (1.0 - self.alpha) * (self.potential_var + tf.math.multiply(diff, inc))

    def _norm_deviation(self):

        '''
        Normalised deviation from the mean membrane potential for each neuron.
        '''

        # The SD is in the denominator, so we have to ensure that we don't crash if it is 0.
        # Use tf.math.divide_no_nan
        return tf.math.divide_no_nan(self.potentials - self.potential_avg, tf.math.sqrt(self.potential_var))

    def _decay(self):

        '''
        Membrane repolarisation (membrane potential decay) and activation decay.
        '''

        # Exponential decay of the membrane potential and the total activation.
        self.potentials *= tf.exp(- self.alpha)
        self.activations *= tf.exp(- self.activation_alpha)

        # Update the average potential.
        self._update_potential_stats()

        # print(f'==[ Potentials: {self.potentials}')
        # print(f'==[ Activations: {self.activations}')

    def _activate(self):

        '''
        The normalised deviation ND is defined by the mean (V_{\mu}) and standard deviation (V_{\sigma})
        values of the membrane potential of the respective neuron:

            ND = \frac{V - V_{\mu}}{V_{\sigma}}

        The activation profile of a neuron is in the shape of a sigmoid (tanh), which approaches 1 asymptotically:

            \rho = \tanh(norm_diff)

        The activation threshold \theta is defined as follows:

            \theta = \exp(- \threshold_alpha * norm_diff)

        Neurons produce action potentials at a (normalised) rate \eta when the potential crosses the activation threshold, i.e., when

            \eta = \rho - \theta > 0
        '''

        # Compute the normalised deviation from the mean.
        norm_dev = self._norm_deviation()

        # Update the moving stats for the membrane potentials.
        self._update_potential_stats()

        # Neurons are activated when the membrane potential crosses the activation threshold
        theta = tf.exp( - self.threshold_alpha * norm_dev)

        # Compute the new activations
        rho = tf.tanh(norm_dev) - theta
        activations = tf.greater_equal(rho, self.baseline)
        self.activations = tf.where(activations, rho, self.activations)

        # Reset potentials to 0 for neurons that have produced action potentials.
        # The average potential will be updated at the next step.
        self.potentials = tf.where(activations, self.baseline, self.potentials)

        # Append activations to the history (used for plotting the activations against time)
        if self.activation_history is not None:
            self.activation_history.append(tf.squeeze(self.activations, axis = 1).numpy())

#------------------------------------------------------------------------------
# A layer with neurons and inbound connections.
#------------------------------------------------------------------------------

class Layer(ProtoLayer):

    '''
    A layer with incoming connections.
    '''

    def __init__(self,
                 size,
                 input_size,
                 tau = None,
                 min_weight = -1.0,
                 max_weight = 1.0,
                 learn = True,
                 learning_rate = 0.05,
                 activation_history = False,
                 weight_history = False):

        '''
        A layer consisting of stateful neurons and inbound connections providing innervation signals.
        '''

        # Learning toggle.
        self.learn = learn
        self.learning_rate = learning_rate
        self.stdp_steps = 0

        self.weight_range = np.array([min_weight, max_weight])

        # Weights for connections between the preceding layer and the current layer.
        self.weights = tf.random.uniform((size, input_size), minval = min_weight, maxval = max_weight)
        self.matmul_op = tf.matmul

        # Weight history
        self.weight_history = [] if weight_history else None

        super().__init__(size, tau, activation_history)

    def _stdp(self,
              input_signals):

        '''
        Spike-timing-dependent plasticity (Hebbian learning).

        Generalised rule suitable for a multi-neuron layer.
        For now, this only works with dense weights.
        '''

        def outer(v1, v2):

            '''
            Outer product of two vectors v1 and v2.
            It is assumed that both v1 and v2 are column vectors, so v2 is transposed.
            '''

            return tf.tensordot(v1, tf.transpose(v2), [[1], [0]])

        # Lower triangular subtensor of the outer product of the layer activations
        lt = tf.linalg.LinearOperatorLowerTriangular(outer(self.activations, self.activations))

        # Weight adjustment
        self.weights += self.learning_rate * (outer(self.activations, input_signals) - tf.linalg.matmul(lt, self.weights))

        # Normalise each row of weights
        self.weights, _ = tf.linalg.normalize(self.weights, axis = 1)

        # Reduce the learning rate every 10 steps
        self.stdp_steps += 1
        if self.stdp_steps == 10:
            self.learning_rate *= 0.99
            self.stdp_steps = 0

    def integrate(self,
                  input_signals):

        '''
        Integrate incoming signals, update input statistics and trigger action potentials.
        '''

        # Membrane potential and activity decay.
        self._decay()

        # Compute the input potentials from the input signals scaled by the synaptic weights.
        self.potentials += self.matmul_op(self.weights, input_signals)

        # Trigger action potentials.
        self._activate()

        # STDP
        if self.learn:
            self._stdp(input_signals)

        if self.weight_history is not None:
            self.weight_history.append(self.weights)

#------------------------------------------------------------------------------
# A sensor layer emulating the operation of the mammalian retina.
# TODO: Retinotopic organisation of RGCs.
#------------------------------------------------------------------------------

class Retina(ProtoLayer):

    '''
    This type of layer aims to emulate the operation of the retina
    with separate ON- and OFF-type RGCs.
    '''

    class RGCLayer:

        '''
        A group of ON- or OFF-type RGCs.
        '''

        def __init__(self,
                     size,
                     off = False):

            # Comparison operator
            self.comp_op = tf.math.less if off else tf.math.greater

            self.activations = tf.zeros(shape = (size, 1))

            self.baseline = tf.zeros(shape = (size, 1))

        def _activate(self,
                      norm_dev):

            # Get the deviation from the mean.
            # The activation depends on whether the cells are ON or OFF
            norm_dev = tf.where(self.comp_op(norm_dev, self.baseline), tf.math.abs(norm_dev), self.baseline)

            # Compute the new activations.
            # Activation is graded, there is no threshold.
            self.activations = tf.tanh(norm_dev)

    def __init__(self,
                 size,
                 tau = None,
                 activation_history = False,
                 *args,
                 **kwargs):

        '''
        Initialise a retinal layer.
        '''

        # Half of the RGCs are ON-type (activated by positive deviations from the mean input)
        # and the other half are OFF-type (activated by negative deviations from the mean input)

        kwargs['activation_history'] = activation_history

        if tau is None:
            tau = tf.constant(100.0, shape = (size, 1))

        super().__init__(size, tau, *args, **kwargs)

        self.on = Retina.RGCLayer(off = False,
                                  size = size)

        self.off = Retina.RGCLayer(off = True,
                                   size = size)

        # The size of the activation layer is twice the size of the input
        self.activations = tf.zeros(shape = (2 * size, 1))

    def _interleave(self):

        '''
        Interleave on/off activations.
        '''

        self.activations = tf.reshape(tf.stack([self.on.activations, self.off.activations],
                                               axis = 1),
                                      [-1, tf.shape(self.on.activations)[1]])

        if self.activation_history is not None:
            self.activation_history.append(tf.squeeze(self.activations, axis = 1).numpy())

    def integrate(self,
                  input_signals):

        '''
        Integrate raw input signals and trigger action potentials.
        '''

        # Integrate input signals.
        self.potentials = input_signals

        # Get the normalised deviation
        norm_dev = self._norm_deviation()

        # Update the stats
        self._update_potential_stats()

        # Compute the activations of ON and OFF cells
        self.on._activate(norm_dev)
        self.off._activate(norm_dev)

        # Interleave the activations of on and off cells.
        self._interleave()