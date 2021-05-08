# ------------------------------------------------------------------------------
# TensorFlow
# ------------------------------------------------------------------------------
import tensorflow as tf

# ------------------------------------------------------------------------------
# Numpy
# ------------------------------------------------------------------------------
import numpy as np

# ------------------------------------------------------------------------------
# Math functions
# ------------------------------------------------------------------------------
import math

# ------------------------------------------------------------------------------
# Random
# ------------------------------------------------------------------------------
import random


class Generator:
    def __init__(
        self,
        rows,
        cols=1,
    ):

        """
        Signal generator.

        Produces various signals with the right shape.
        """

        self.rows = rows
        self.cols = cols

    def constant(self, value=1.0):

        """
        Constant input.
        """

        return tf.constant(value, shape=(self.rows, self.cols))

    def noise(self, mean=0.0, sd=1.0):

        """
        Gaussian noise.
        """

        return tf.random.normal(shape=(self.rows, self.cols), mean=mean, stddev=sd)

    def unoise(self, min_val=-1.0, max_val=1.0):

        """
        Uniform noise.
        """

        return tf.random.uniform(
            shape=(self.rows, self.cols), minval=min_val, maxval=max_val
        )

    def flash(self, value=1.0, pos=0, spot_size=4):

        indices = [[i, 0] for i in range(pos, pos + spot_size)]
        values = [value] * len(indices)

        fl = tf.sparse.SparseTensor(indices, values, dense_shape=[self.rows, self.cols])

        return tf.sparse.to_dense(fl)


class GratingGenerator(Generator):
    def __init__(
        self,
        rows,
        cols=1,
        tp=2,  # Temporal period
        ext=1,  # Spatial extent
        gap=1,  # Gap size
    ):

        """
        Grating signal generator.
        """

        assert (
            gap + ext < rows
        ), f"==[ Error: Invalid combination of gap and extent (gap + ext must be less than {rows})"

        super().__init__(rows, cols)

        self.grating = tf.zeros(shape=(rows, cols), dtype=np.float32)

        indices = []
        values = []
        x = 0
        while x < rows:
            if x % (ext + gap) < ext:
                indices.append([x, 0])
                values.append(1.0)
            x += 1

        on_off = tf.sparse.SparseTensor(indices, values, dense_shape=(rows, cols))
        self.grating += tf.sparse.to_dense(on_off)

        self.tp = tp
        self.step = 0

    def roll(self, shift=1, axis=0):

        if self.step >= self.tp:
            self.grating = tf.roll(self.grating, shift, axis)
            self.step = 0

        self.step += 1
        return self.grating