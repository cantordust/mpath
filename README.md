# MPATH model
This is the code used for producing the simulation results and the plots for the paper _Continuous Learning and Adaptation with Membrane Potential and Activation Threshold Homeostasis_.

## Installation
Install MPATH as a local package

`$ pip install --user mpath`

**Or** if you would like to be able to modify the source:

`$ pip install --user -e .`

If you are in a virtual environment, you can skip the `--user` part. All requirements should be installed automatically

## Running a simulation
Edit `mpath/main.py` as necessary and run it directly:

`$ python3 ./mpath/main.py`

## Plotting
Run

`$ python3 ./mpath/plot.py -p <path-to-numpy-params.npz> -a ret in act1 act2 -w wt1 wt2`

The `-p` switch is used for the path to the NumPy parameters (a `.npz` archive), `-a` is used to indicate the layers whose activations should be plotted, and `-w` is used to indicate the layers whose weights should be plotted.
