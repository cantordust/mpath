#------------------------------------------------------------------------------
# Numpy
#------------------------------------------------------------------------------
import numpy as np

#------------------------------------------------------------------------------
# Matplotlib
#------------------------------------------------------------------------------
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import IndexLocator
from matplotlib.ticker import NullLocator

#------------------------------------------------------------------------------
# Command-line argument parser
#------------------------------------------------------------------------------
import argparse

plt.rcParams.update({
    'font.size': 6,
    'text.usetex': True
    })

fmt = 'pdf'

def add_axis(axes, fig, grid, axis):
    axes.append(fig.add_subplot(grid[axis]))
    return axes[-1], axis + 1

def plot_activations(path,
                     layers,
                     fmt = fmt,
                     steps = [(0, 100),
                              (500, 600),
                              (950, 1050),
                              (1450, 1550),
                              (1500, 1600),
                              (1900, 2000)]):

    # Set up matplotlib
    mpl.use('Qt5Agg')

    arch = np.load(path)

    # Info about requested layers
    req = len(layers) > 0
    requested_layers = set(layers)

    layers = []
    all_layers = []

    for f in arch.files:
        if f.startswith(('in', 'ret', 'act')):
            all_layers.append(f)

            if not req or f in requested_layers:
                layers.append(f)

    layers = set(layers)

    for (begin, end) in steps:

        fig = plt.figure(constrained_layout = True, figsize = (6,4))

        # Relative width and height values for the subplots
        widths = [10, 0.3, 1]
        heights = [1] * len(layers)

        # Gridspace
        gs = fig.add_gridspec(ncols = 3,
                              nrows = len(layers),
                              width_ratios = widths,
                              height_ratios = heights)


        # fig.suptitle(f'Steps {begin} - {end}', fontsize = 14)

        grid = {num: ss for num, ss in enumerate([ss for ss in gs])}

        ax_count = len(widths) * len(heights)

        axes = []
        axis = 0
        for l in all_layers:

            if l in layers:

                if l == 'in':
                    title = 'Input'
                    ylbl = 'Channel'
                    t_src = None

                elif l == 'ret':
                    title = 'Retina'
                    ylbl = r'Retina\\(neuron \#)'
                    t_src = 'tau_r'

                else:
                    title ='Layer ' + l[3:]
                    ylbl = 'Layer ' + l[3:] + r'\\(neuron \#)'
                    t_src = 'tau' + l[3:]

                # Activation plot
                ax, axis = add_axis(axes, fig, grid, axis)
                heatmap = ax.imshow(np.transpose(arch[l][begin:end]),
                                    origin = 'lower',
                                    cmap='viridis',
                                    aspect = 'auto')

                ax.xaxis.set_major_locator(IndexLocator(base = (end - begin) / 10, offset = 0.5))
                ax.yaxis.set_major_locator(IndexLocator(base = arch[l].shape[1] / 10, offset = 0.5))

                ax.set_xticklabels(np.arange(begin, end, (end - begin) / 10, dtype = np.int))

                ylabels = np.arange(1, arch[l].shape[1] + 1, arch[l].shape[1] / 10, dtype = np.int)
                ax.set_yticklabels(ylabels)

                if axis == ax_count - 2:
                    ax.set_xlabel('Step', fontsize = 12)

                # ax.set_title(title, fontsize = 12)
                ax.set_ylabel(ylbl, fontsize = 10)

                # Colour bar
                cax, axis = add_axis(axes, fig, grid, axis)
                cbar = ax.figure.colorbar(heatmap, ax = ax, cax = cax)
                cbar.set_label('Activation', fontsize = 10)

                # Tau
                if l != 'in':
                    tax, axis = add_axis(axes, fig, grid, axis)
                else:
                    # Increment the axis count
                    tax = None
                    axis += 1

                # Tau bar plot
                if tax is not None:

                    tau = np.squeeze(arch[t_src], axis = 1)

                    if l == 'ret':
                        tau_array = np.empty((2 * tau.size,), dtype = tau.dtype)
                        tau_array[0::2] = tau
                        tau_array[1::2] = tau
                        tau = tau_array

                    tau_bars = tax.barh(y = np.arange(1, len(tau) + 1, 1.0, dtype = np.float32),
                                        width = tau,
                                        height = 8 / len(tau) if len(tau) <= 20 else 1.0,
                                        align = 'edge')

                    tax.yaxis.set_major_locator(IndexLocator(base = tau.size / 10, offset = 4 / len(tau)))

                    nlabels = np.arange(1, tau.size + 1, tau.size / 10, dtype = np.int)
                    tax.set_yticklabels(nlabels)

                    if axis == ax_count:
                        tax.set_xlabel(r'$\tau_{m}$', fontsize = 10)

        gs.tight_layout(fig)
    plt.show()

def plot_weights(path,
                 layers,
                 fmt = fmt,
                 steps = [0, 999, 1999]):

    # Set up matplotlib
    mpl.use('Qt5Agg')

    arch = np.load(path)

    # Info about requested layers
    req = len(layers) > 0
    requested_layers = set(layers)

    layers = []
    all_layers = []

    for f in arch.files:
        if f.startswith('wt'):
            all_layers.append(f)

            if not req or f in requested_layers:
                layers.append(f)

    layers = set(layers)

    for step in steps:

        fig = plt.figure(constrained_layout = True, figsize = (6,4))

        widths = [10, 0.5]
        heights = [1] * len(layers)

        # Gridspace
        gs = fig.add_gridspec(ncols = 2,
                              nrows = len(layers),
                              width_ratios = widths,
                              height_ratios = heights)


        fig.suptitle(f'Step {step + 1}', fontsize = 14)

        grid = {num: ss for num, ss in enumerate([ss for ss in gs])}

        axes = []
        axis = 0
        for l in all_layers:

            if l in layers:

                title ='Layer ' + l[2:]

                # Weight plot
                ax, axis = add_axis(axes, fig, grid, axis)
                heatmap = ax.imshow(arch[l][step],
                                    origin = 'lower',
                                    cmap = 'viridis',
                                    aspect = 'auto')

                ax.xaxis.set_major_locator(IndexLocator(base = arch[l][step].shape[1] / 10, offset = 0.5))
                ax.yaxis.set_major_locator(IndexLocator(base = arch[l][step].shape[0] / 10, offset = 0.5))

                xlabels = np.arange(1, arch[l][step].shape[1] + 1, arch[l][step].shape[1] / 10, dtype = np.int)
                ylabels = np.arange(1, arch[l][step].shape[0] + 1, arch[l][step].shape[0] / 10, dtype = np.int)

                ax.set_xticklabels(xlabels)
                ax.set_yticklabels(ylabels)

                xlbl = int(l[2:]) - 1
                xlbl = 'Layer ' + str(xlbl) if xlbl > 0 else 'Retina'

                ax.set_xlabel(xlbl + r'\\(neuron \#)', fontsize = 10)
                ax.set_ylabel('Layer ' + l[2:] + r'\\(neuron \#)', fontsize = 10)

                # # Colour bar
                cax, axis = add_axis(axes, fig, grid, axis)
                cbar = ax.figure.colorbar(heatmap, ax = ax, cax = cax)
                cbar.set_label('Weight', fontsize = 10)

        gs.tight_layout(fig)
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'MPATH plotter')

    parser.add_argument('-p',
                        type = str,
                        required = True,
                        help='Path to NumPy archive.')

    parser.add_argument('-a',
                        nargs = '*',
                        help='Plot activations.')

    parser.add_argument('-w',
                        nargs = '*',
                        help='Plot weights.')

    args = parser.parse_args()

    if args.a is not None:
        plot_activations(path = args.p,
                         layers = args.a)

    if args.w is not None:
        plot_weights(path = args.p,
                     layers = args.w)