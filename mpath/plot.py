# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
from typing import Any
from typing import List
from typing import Dict
from typing import Tuple
from typing import Optional

# --------------------------------------
import torch as pt
import numpy as np
import math

# --------------------------------------
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import axes_size
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import IndexLocator
from matplotlib.ticker import NullLocator

import plotly.offline as py
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import argparse

plt.rcParams.update(
    {
        "font.size": 6,
        "text.usetex": True,
    }
)
# fmt = "pdf"


# def add_axis(
#     _axes,
#     _fig,
#     _grid,
#     _axis,
# ):
#     _axes.append(_fig.add_subplot(_grid[_axis]))
#     return _axes[-1], _axis + 1


# def plot_activations(
#     _path: str,
#     _layers: List[str],
#     _fmt: str = fmt,
#     _steps: List[Tuple[int, int]] = [
#         (0, 100),
#         (500, 600),
#         (950, 1050),
#         (1450, 1550),
#         (1500, 1600),
#         (1900, 2000),
#     ],
# ):

#     # Set up matplotlib
#     mpl.use("Qt5Agg")

#     arch = np.load(_path)

#     # Info about requested layers
#     req = len(_layers) > 0
#     requested_layers = set([f"hist_{l}" for l in _layers])

#     layers_to_plot = []
#     all_layers = []

#     for f in arch.files:
#         if f.startswith("hist_"):
#             all_layers.append(f)

#             if not req or f in requested_layers:
#                 layers_to_plot.append(f)

#     layers_to_plot = set(layers_to_plot)

#     for (begin, end) in _steps:

#         fig = plt.figure(constrained_layout=True)

#         # Relative width and height values for the subplots
#         widths = [15, 0.3, 1]
#         heights = [1] * len(layers_to_plot)

#         # Gridspace
#         gs = fig.add_gridspec(
#             ncols=3,
#             nrows=len(layers_to_plot),
#             width_ratios=widths,
#             height_ratios=heights,
#         )

#         # fig.suptitle(f'Steps {begin} - {end}', fontsize = 14)

#         grid = {num: ss for num, ss in enumerate([ss for ss in gs])}

#         ax_count = len(widths) * len(heights)

#         axes = []
#         axis = 0
#         for l in all_layers:

#             if l in layers_to_plot:

#                 if l == "hist_in":
#                     # Input
#                     title = "Input"
#                     ylbl = "Channel"
#                     t_src = None

#                 elif l == "hist_ret":
#                     # Retinal layer
#                     title = "Retina"
#                     ylbl = r"Retina\\(neuron \#)"
#                     t_src = "tau_ret"

#                 else:
#                     # Activation layer
#                     title = "Layer " + l[8:]
#                     ylbl = "Layer " + l[8:] + r"\\(neuron \#)"
#                     t_src = "tau_" + l[8:]

#                 # Activation plot
#                 ax, axis = add_axis(axes, fig, grid, axis)
#                 heatmap = ax.imshow(
#                     np.transpose(arch[l][begin:end]),
#                     origin="lower",
#                     cmap="turbo",
#                     aspect="auto",
#                 )

#                 ax.xaxis.set_major_locator(
#                     IndexLocator(base=(end - begin) / 10, offset=0.5)
#                 )
#                 ax.yaxis.set_major_locator(
#                     IndexLocator(base=arch[l].shape[1] / 10, offset=0.5)
#                 )

#                 xlabels = np.arange(begin, end, (end - begin) / 10, dtype=np.int)
#                 ax.set_xticklabels(xlabels)

#                 ylabels = np.arange(
#                     1, arch[l].shape[1] + 1, arch[l].shape[1] / 10, dtype=np.int
#                 )
#                 ax.set_yticklabels(ylabels)

#                 if axis == ax_count - 2:
#                     ax.set_xlabel("Step", fontsize=12)

#                 ax.set_title(title, fontsize=12)
#                 ax.set_ylabel(ylbl, fontsize=10)

#                 # Colour bar
#                 cax, axis = add_axis(axes, fig, grid, axis)
#                 cbar = ax.figure.colorbar(
#                     heatmap,
#                     ax=ax,
#                     cax=cax,
#                 )
#                 # cbar.set_title(
#                 #     "Activation",
#                 #     fontsize=8,
#                 # )

#                 # Tau
#                 if l != "hist_in":
#                     tax, axis = add_axis(axes, fig, grid, axis)
#                 else:
#                     # Increment the axis count
#                     tax = None
#                     axis += 1

#                 # Tau bar plot
#                 if tax is not None:

#                     tau = arch[t_src]

#                     if l == "hist_ret":
#                         tau_array = np.empty((2 * tau.size,), dtype=tau.dtype)
#                         tau_array[0::2] = tau
#                         tau_array[1::2] = tau
#                         tau = tau_array

#                     tau_bars = tax.barh(
#                         y=np.arange(1, len(tau) + 1, 1.0, dtype=np.float32),
#                         width=tau,
#                         height=8 / len(tau) if len(tau) <= 20 else 1.0,
#                         align="edge",
#                     )

#                     tax.yaxis.set_major_locator(
#                         IndexLocator(base=tau.size / 10, offset=4 / len(tau))
#                     )

#                     nlabels = np.arange(1, tau.size + 1, tau.size / 10, dtype=np.int)
#                     tax.set_yticklabels(nlabels)

#                     if axis == ax_count:
#                         tax.set_xlabel(r"$\tau_{m}$", fontsize=10)

#         # gs.tight_layout(fig)
#         # gs.update(wspace=0.15, hspace=0.05)
#         # plt.subplots_adjust(wspace=0, hspace=0)
#     plt.show()


# def plot_weights(
#     _path: str,
#     _layers: List[int],
#     _fmt: str = fmt,
#     _steps: List[int] = [0, 999, 1999],
# ):

#     # Set up matplotlib
#     mpl.use("Qt5Agg")

#     arch = np.load(_path)

#     # Info about requested layers
#     req = len(_layers) > 0
#     requested_layers = set([f"wt_{l}" for l in _layers])

#     layers_to_plot = []
#     all_layers = []

#     for f in arch.files:
#         if f.startswith("wt_"):
#             all_layers.append(f)

#             if not req or f in requested_layers:
#                 layers_to_plot.append(f)

#     layers_to_plot = set(layers_to_plot)

#     for step in _steps:

#         fig = plt.figure(constrained_layout=True)

#         widths = [20, 1.0]
#         heights = [1] * len(layers_to_plot)

#         # Gridspace
#         gs = fig.add_gridspec(
#             ncols=2,
#             nrows=len(layers_to_plot),
#             width_ratios=widths,
#             height_ratios=heights,
#         )

#         fig.suptitle(f"Step {step + 1}", fontsize=14)

#         grid = {num: ss for num, ss in enumerate([ss for ss in gs])}

#         axes = []
#         axis = 0
#         for l in all_layers:

#             if l in layers_to_plot:

#                 title = "Layer " + l[2:]

#                 # Weight plot
#                 ax, axis = add_axis(axes, fig, grid, axis)
#                 heatmap = ax.imshow(
#                     arch[l][step], origin="lower", cmap="viridis", aspect="auto"
#                 )

#                 ax.xaxis.set_major_locator(
#                     IndexLocator(base=arch[l][step].shape[1] / 10, offset=0.5)
#                 )
#                 ax.yaxis.set_major_locator(
#                     IndexLocator(base=arch[l][step].shape[0] / 10, offset=0.5)
#                 )

#                 xlabels = np.arange(
#                     1,
#                     arch[l][step].shape[1] + 1,
#                     arch[l][step].shape[1] / 10,
#                     dtype=np.int,
#                 )
#                 ylabels = np.arange(
#                     1,
#                     arch[l][step].shape[0] + 1,
#                     arch[l][step].shape[0] / 10,
#                     dtype=np.int,
#                 )

#                 ax.set_xticklabels(xlabels)
#                 ax.set_yticklabels(ylabels)

#                 xlbl = int(l[3:]) - 1
#                 xlbl = "Layer " + str(xlbl) if xlbl > 0 else "Retina"

#                 ax.set_xlabel(xlbl + r"\\(neuron \#)", fontsize=10)
#                 ax.set_ylabel("Layer " + l[3:] + r"\\(neuron \#)", fontsize=10)

#                 # Colour bar
#                 cax, axis = add_axis(axes, fig, grid, axis)
#                 cbar = ax.figure.colorbar(heatmap, ax=ax, cax=cax)
#                 # cbar.set_label("Weight", fontsize=10)

#     plt.show()


class Plotter:

    defaults = {}

    def __init__(
        self,
        _layers: List[Dict[str, pt.Tensor]],
        _vspace: Optional[float] = None,
        _layout: Optional[Dict[str, Any]] = None,
    ):

        if len(_layers) == 0:
            return

        self.param_sets = []
        self.cols = 1
        self.rows = 0

        self.getters = {
            "bar": self._get_bar_params,
            "heatmap": self._get_heatmap_params,
        }
        self.makers = {
            "bar": self._make_bar_plot,
            "heatmap": self._make_heatmap_plot,
        }

        widths = {}
        titles = []

        # Collect information from the layers
        for row, layer in enumerate(_layers, 1):

            self.rows += 1

            for col, item in enumerate(layer, 1):

                plot_type = item.get("type", None)

                if plot_type is None:
                    raise KeyError("Missing required key 'type'.")

                getter = self.getters.get(plot_type, None)

                if getter is None:
                    raise KeyError(
                        f"Sorry, there is no plot function yet for plots of type '{plot_type}'"
                    )

                params = getter(item)

                if params is None:
                    continue

                # Plot type
                params["type"] = plot_type

                # X label
                params["xlabel"] = item.get("xlabel", None)

                # Y label
                params["ylabel"] = item.get("ylabel", None)

                # Row and column to put this plot in
                params["row"] = row

                plot_col = item.get("col", col)
                self.cols = max(self.cols, plot_col)
                params["col"] = plot_col

                # Column width
                if item.get("width", 0.0) > 0.0:
                    widths[plot_col] = max(item["width"], widths.get(col, 0.0))

                # Additional parameters
                params["meta"] = item.get("meta", dict())

                # Subplot titles
                titles.extend([item.get("title", f"Layer {row}"), None])

                # Append the plot to the list of plots
                self.param_sets.append(params)

        # Compute column widths
        width_sum = sum(widths.values())
        if width_sum > 1.0:
            raise ValueError(
                "Invalid column widths (the sum of all specified widths has to be <= 1.0)."
            )

        even = (1.0 - width_sum) / (self.cols - len(widths))

        column_widths = [
            widths[col] if col in widths else even for col in range(1, self.cols + 1)
        ]

        # Plot parameters
        self.vspace = _vspace if _vspace is not None else 0.3 / self.rows

        # Row height
        self.rh = (1 - (self.rows - 1) * self.vspace) / self.rows

        if _layout is None:
            _layout = {
                "margin": {
                    "l": 0,
                    "r": 0,
                    "b": 40,
                    "t": 20,
                    "pad": 0,
                },
                "font": {
                    "size": 8,
                },
                "xaxis": {
                    "title": {
                        "font": {
                            "size": 10,
                        },
                        "standoff": 50,
                    },
                },
            }

        self.layout = go.Layout(**_layout)

        self.fig = make_subplots(
            rows=self.rows,
            cols=self.cols,
            start_cell="top-left",
            column_widths=column_widths,
            shared_xaxes=True,
            subplot_titles=titles,
            vertical_spacing=self.vspace,
            horizontal_spacing=0.0,
        )

    def _get_bar_params(
        self,
        _item: Dict[str, Any],
    ):

        params = {}

        # Extract the data
        data = _item.get("data", None)

        shape = data.shape
        if len(shape) > 2:
            raise ValueError(
                "Tensor has too many dimensions for a bar plot. Please reduce the dimensions to one or two."
            )

        if len(shape) == 1:
            # One-dimensional tensor.
            # x is just a range from 1 to the tensor length.
            x = data
            y = list(range(1, shape[0] + 1))

        else:
            # Two-dimensional tensor.
            # x is given explicitly
            x = data[:, 1]
            y = data[:, 0]

        # X ticks
        params["x"] = x

        # Y ticks
        params["y"] = y

        return params

    def _make_bar_plot(
        self,
        _params: Dict[str, Any],
    ):

        _params["meta"].update(
            {
                "showlegend": False,
            }
        )

        params = {
            "x": _params["x"],
            "y": _params["y"],
            "row": _params["row"],
            "col": _params["col"],
        }

        params.update(_params["meta"])

        self.fig.add_bar(**params)

        # Update axes

        xtickvals = list(
            np.linspace(0.0, math.ceil(max(_params["x"])), len(_params["x"]) // 3)[1:]
        )
        self.fig.update_xaxes(
            # autorange="reversed",
            # showticklabels=False,
            tickvals=xtickvals,
            tickangle=0,
            row=_params["row"],
            col=_params["col"],
            titlefont={
                "size": 8,
            },
            title={
                "text": _params["xlabel"],
            },
            tickfont={
                "size": 5,
            },
        )

        self.fig.update_yaxes(
            row=_params["row"],
            col=_params["col"],
            showticklabels=False,
            title={
                "text": _params["ylabel"],
            },
        )

    def _get_heatmap_params(
        self,
        _item: Dict[str, Any],
    ):

        params = {}

        # Extract the data
        data = _item.get("data", None)

        if data is None:
            raise ValueError("Please provide a 'data' key.")

        # Extract the range of steps to plot
        steps = _item.get("steps", (0, data.shape[1]))
        data = data[:, steps[0] : steps[1]]

        params["z"] = data

        # X ticks
        params["x"] = list(range(steps[0] + 1, steps[1] + 1))

        # Y ticks
        params["y"] = list(range(1, int(data.shape[0]) + 1))

        # Summarise everything that we need to create the plot.
        return params

    def _make_heatmap_plot(
        self,
        _params: Dict[str, Any],
    ):

        _params["meta"].update(
            {
                "colorbar": {
                    "len": self.rh,
                    "thickness": 10,
                    "yanchor": "bottom",
                    "ypad": 0,
                    "y": (_params["row"] - 1) * (self.rh + self.vspace),
                },
                "colorscale": "Viridis",
            }
        )

        params = {
            "z": _params["z"],
            "x": _params["x"],
            "y": _params["y"],
            "row": _params["row"],
            "col": _params["col"],
        }

        params.update(_params["meta"])

        self.fig.add_heatmap(**params)

        self.fig.update_xaxes(
            tickfont={
                "size": 5,
            },
        )

        self.fig.update_yaxes(
            title_text=_params["ylabel"],
            row=_params["row"],
            col=_params["col"],
        )

    def plot(self):

        for params in self.param_sets:

            plot_maker = self.makers.get(params["type"], None)

            if plot_maker is None:
                raise KeyError(
                    f"Sorry, there is no plot function yet for plots of type '{params['type']}'"
                )

            # Create the actual plot
            plot_maker(params)

            self.fig.update_xaxes(
                title_text=params["xlabel"],
                row=params["row"],
                col=params["col"],
            )

        self.fig.update_layout(self.layout)

        self.fig.update_annotations(
            font_size=10,
        )

        self.fig.write_image("plot.pdf")
        # pio.show(self.fig, renderer="chromium")


if __name__ == "__main__":

    step_ranges = [
        (0, 100),
        (400, 500),
    ]

    for steps in step_ranges:

        data = [
            [
                {
                    "type": "heatmap",
                    "title": "Input",
                    "ylabel": "Channel",
                    "data": pt.rand((100, 500)),
                    "col": 1,
                    "width": 0.95,
                    "steps": steps,
                },
            ],
            [
                {
                    "type": "heatmap",
                    "title": "Retina",
                    "ylabel": "Neuron #",
                    "data": pt.rand((100, 500)),
                    "col": 1,
                    "steps": steps,
                },
            ],
            [
                {
                    "type": "heatmap",
                    "title": "Layer 1",
                    "ylabel": "Neuron #",
                    "data": pt.rand((30, 500)),
                    "col": 1,
                    "steps": steps,
                },
                {
                    "type": "bar",
                    "data": pt.rand((30,)),
                    "col": 2,
                    "meta": {
                        "orientation": "h",
                    },
                },
            ],
            [
                {
                    "type": "heatmap",
                    "title": "Layer 2",
                    "ylabel": "Neuron #",
                    "data": pt.rand((10, 500)),
                    "col": 1,
                    "steps": steps,
                },
                {
                    "type": "bar",
                    "data": pt.rand((10,)),
                    "col": 2,
                    "meta": {
                        "orientation": "h",
                    },
                },
            ],
            [
                {
                    "type": "heatmap",
                    "title": "Output layer",
                    "ylabel": "Neuron #",
                    "xlabel": "Step",
                    "data": pt.rand((10, 500)),
                    "col": 1,
                    "steps": steps,
                },
                {
                    "type": "bar",
                    "data": pt.rand((10,)),
                    "col": 2,
                    "xlabel": r"$\tau_{m}$",
                    "meta": {
                        "orientation": "h",
                    },
                },
            ],
        ]

        plotter = Plotter(data, _vspace=0.05)
        plotter.plot()
