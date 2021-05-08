# ------------------------------------------------------------------------------
# Filesystem
# ------------------------------------------------------------------------------
from pathlib import Path

# ------------------------------------------------------------------------------
# Date and time
# ------------------------------------------------------------------------------
from datetime import datetime

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# ------------------------------------------------------------------------------
# TensorFlow
# ------------------------------------------------------------------------------
import tensorflow as tf

# ------------------------------------------------------------------------------
# Math
# ------------------------------------------------------------------------------
import math

# ------------------------------------------------------------------------------
# Numpy
# ------------------------------------------------------------------------------
import numpy as np

# ------------------------------------------------------------------------------
# Random
# ------------------------------------------------------------------------------
import random

# ------------------------------------------------------------------------------
# OpenCV
# ------------------------------------------------------------------------------
import cv2 as cv

# ------------------------------------------------------------------------------
# Time
# ------------------------------------------------------------------------------
import time

# ------------------------------------------------------------------------------
# MPATH imports
# ------------------------------------------------------------------------------
from mpath.siggen import Generator
from mpath.siggen import GratingGenerator
from mpath.network import Network
from mpath.plot import plot_activations
from mpath.plot import plot_weights

if __name__ == "__main__":

    # Sizes for the network layers.
    # The first one is the size of the retinal layer,
    # so the first element in the array represents
    # only *half* of the neurons in that layer.
    input_size = 30
    layer_sizes = [input_size, 20, 10]

    # Network with activation and weight history
    net = Network(
        layer_sizes=layer_sizes,
        learn=True,
        activation_history=True,
        weight_history=True,
    )

    # Signal generator
    # gen = Generator(input_size)
    gen = GratingGenerator(
        input_size,
        tp=10,
        ext=5,
        gap=5,
    )

    # Run the simulation
    steps = 2000

    print(f"==[ Running simulation for {steps} steps", end="")

    pct = 1
    spot_size = 4

    # Disable learning
    net.freeze()

    for step in range(1, steps + 1):

        if step % (steps / 10) == 0:
            print(f"...{10 * pct}%", end="", flush=True)
            pct += 1

        ###############
        # Uniform noise
        ###############
        # signal = gen.unoise()

        ################
        # Gaussian noise
        ################
        # signal = gen.noise()

        ##############
        # Sine grating
        ##############
        # signal = gen.roll()

        ########################################
        # Gaussian noise added to a sine grating
        ########################################
        # signal = gen.roll()
        # if random.random() < 0.05:
        #     signal += gen.noise(0.0, 3.0)

        ###########################################
        # Gaussian noise added to a constant output
        ###########################################
        # signal = gen.constant(1.0)
        # if random.random() < 0.05:
        #     signal += gen.noise()

        ##################################
        # Flash added to a constant output
        ##################################
        # signal = gen.constant(0.0)
        # if step % 10 == 0:
        #     signal += gen.flash(random.random(),
        #                         math.floor((input_size - spot_size) / 2),
        #                         spot_size)

        ###################################################
        # Constant flashes superimposed onto Gaussian noise
        ###################################################
        signal = gen.noise()
        if step % 10 == 0:
            # signal = gen.constant(2 * random.random())
            signal = gen.constant(3.0)

        ##########################################################
        # Constant flashes superimposed onto a constant background
        ##########################################################
        # signal = gen.constant(0.0)
        # if random.random() < 0.05:
        #     signal = gen.noise(2.0)

        ##########################################################
        # Constant flashes superimposed onto a constant background
        ##########################################################
        # signal = gen.constant(0.0)
        # if random.random() < 0.05:
        #     signal = gen.constant(2.0)

        ###################################################
        # Constant flashes superimposed onto a sine grating
        ###################################################
        # signal = gen.roll(shift=5)
        # if random.random() < 0.05:
        #     signal = gen.constant(5.0)

        #################################################################
        # Steps 0 - 1000: train on a sine grating
        # Steps 1000 - 1500: Freeze training and just integrate noise
        # Steps > 1500: Integrate the same sine grating, without training
        #################################################################
        # if 0 < step < 50 or 1000 < step <= 1500:
        #     signal = gen.noise()

        # elif 50 <= step <= 1000:

        #     signal = gen.roll()
        #     if step == 1000:
        #         # Stop learning
        #         net.freeze()

        # else:
        #     signal = gen.roll()

        #############################################################################
        # Steps 0 - 1000: train on a sine grating
        # Steps 1000 - 1500: Freeze training and just integrate constant illumination
        # Steps > 1500: Integrate the same sine grating, without training
        #############################################################################
        # if step <= 1000:
        #     signal = gen.roll()
        #     if step == 1000:
        #         # Stop learning
        #         net.freeze()

        # elif 1000 < step <= 1500:
        #     signal = gen.constant(1.0)

        # else:
        #     signal = gen.roll()

        #############################################################################
        # Steps 0 - 1000: train on a sine grating with Gaussian noise added at random
        # Steps 1000 - 1500: Freeze training and just integrate Gaussian noise
        # Steps > 1500: Integrate the same sine grating, without training or noise
        #############################################################################
        # signal = gen.roll()

        # if step <= 1000:
        #     signal = gen.roll()
        #     if random.random() < 0.05:
        #         signal += gen.noise(1.0, 3.0)

        #     if step == 1000:
        #         # Stop learning
        #         net.freeze()

        # elif 1000 < step <= 1500:
        #     signal = gen.noise()

        # else:
        #     signal = gen.roll()

        ######################
        # Integrate the signal
        ######################
        net.integrate(signal)

    print()

    path = f"./sim/{timestamp}"

    net.save(path)

    plot_activations(path + "/params.npz", ["in", "ret", "act1", "act2"])
    plot_weights(path + "/params.npz", [0, 1, 2])

# ===============================================

# def test_cv():

#     src = 0
#     cap = cv.VideoCapture(src)

#     try:

#         if not cap.isOpened():
#             print(f"Cannot open camera {src}")
#             return

#         # Read one frame
#         ret, frame = cap.read()
#         h, w, d = frame.shape

#         # Sizes for the network layers.
#         # The first one is the size of the retinal layer,
#         # so the first element in the array represents
#         # only *half* of the neurons in that layer.
#         input_size = 30
#         layer_sizes = [input_size, 20, 10]

#         # Network with activation and weight history
#         net = Network(layer_sizes = layer_sizes,
#                       learn = True,
#                       activation_history = True,
#                       weight_history = True)

#         start = time.perf_counter()

#         frames = 0

#         for step in range(1000):

#             # Capture frame-by-frame
#             ok, frame = cap.read()

#             # if frame is read correctly ret is True
#             if not ok:
#                 print(f"Can't receive frame. Exiting.")
#                 return

#             frames += 1

#             frame = cv.flip(frame, 1)

#             # Our operations on the frame come here
#             grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#             image = tf.transpose(tf.convert_to_tensor(grey,
#                                                       dtype = tf.float32))

#             # if cv.waitKey(1) == ord('q'):
#             #     break

#         timespan = time.perf_counter() - start
#         fps = frames / timespan
#         # print(f'==[ FPS: {fps}')

#     finally:

#         # When everything done, release the capture
#         cap.release()
#         cv.destroyAllWindows()

# if __name__ == '__main__':

#     test_cv()