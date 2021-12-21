# ------------------------------------------------------------------------------
# Global imports
# ------------------------------------------------------------------------------
# *** Type hints *** #
from typing import Any
from typing import Optional
from typing import Union

# *** Enumerations *** #
import enum

# *** PyTorch & numerical libs *** #
import torch as pt
import torch.nn.functional as ptf
from torch.distributions.one_hot_categorical import OneHotCategorical

import numpy as np

# *** Plotting *** #
import matplotlib.cm as cm
import matplotlib.pyplot as plt

pt.set_printoptions(linewidth=200)
plt.rcParams["figure.figsize"] = [12, 12]

# *** Images & video *** #
import cv2 as cv


class Source:
    """
    Baisc image operations.
    """

    def __init__(
        self,
        _src: Union[str, cv.VideoCapture],
        _height: Optional[int] = None,
        _width: Optional[int] = None,
        _mode: Optional[enum.Enum] = cv.COLOR_RGB2GRAY,
    ):

        if _height is not None and _width is not None:
            raise ValueError("Please provide the new height *or* width, but not both.")

        self.source = _src
        self.frame = None
        self.op = None
        self.processing = True
        self.mode = _mode

        if isinstance(_src, str):
            self.op = lambda processing, src: (processing, cv.imread(_src))

        elif isinstance(_src, cv.VideoCapture):
            self.op = lambda processing, src: src.read()

        else:
            raise TypeError("Invalid input source.")

        self._get_frame(_probe=True)

        shape = tuple(self.frame.shape)

        # Image dimensions
        self.source_height = shape[0]
        self.source_width = shape[1]

        # NOTE: Perhaps add transparency as well
        self.source_depth = shape[2] if len(self.frame.shape) > 2 else 1

        self.output_height = self.source_height
        self.output_width = self.source_width
        self.output_depth = self.source_depth

        self.resize = None
        if (_width, _height) != (None, None):
            if _height is not None:
                # Fixed height, calculate the width with the same AR
                pct = _height / float(self.source_height)
                _width = int((float(self.source_width) * pct))

            elif _width is not None:
                # Fixed width, calculate the height with the same AR
                pct = _width / float(self.source_width)
                _height = int((float(self.source_height) * pct))

            self.resize = (_width, _height)
            self.output_height = _height
            self.output_width = _width

        # Create a flatmask
        self._make_flatmask()

        # Display some useful info
        print(f"==[ Press ESC to quit.")

    def __del__(self):

        if isinstance(self.source, cv.VideoCapture):
            self.source.release()

        cv.destroyAllWindows()

    def _make_flatmask(self):
        """
        Create a mask that will can be used to obtain a flattened
        version of the original image with colour channel sampling.
        """

        self.flatmask = None

        if self.source_depth == 1:
            # The image is already flat
            return

        print(f"==[ Creating flatmask...")

        probs = pt.zeros((self.output_height, self.output_width, self.output_depth))

        r_prob = 0.475
        g_prob = 0.475
        b_prob = 0.05

        probs[:, :, 0] = r_prob  # R channel
        probs[:, :, 1] = g_prob  # G channel
        probs[:, :, 2] = b_prob  # B channel

        ohc = OneHotCategorical(probs)

        self.flatmask = ohc.sample()

        self.flatmask[:, :, 0] *= r_prob  # R channel
        self.flatmask[:, :, 1] *= g_prob  # G channel
        self.flatmask[:, :, 2] *= b_prob  # B channel

        # print(f"==[ R: {self.flatmask[:,:,0].sum()}")
        # print(f"==[ G: {self.flatmask[:,:,1].sum()}")
        # print(f"==[ B: {self.flatmask[:,:,2].sum()}")

        # Set the depth to 1
        self.output_depth = 1

    def show(
        self,
        *_frames,
    ):

        if _frames is None:
            _frames = [self.frame]

        # Show all images
        # for idx, _img in enumerate(_frames):
        cv.imshow(
            f"Result",
            np.hstack([_img.numpy() for _img in _frames]).astype(np.uint8),
        )

        # Press ESC to quit
        self.processing &= cv.waitKey(10) != 27

    def _get_frame(
        self,
        _center: Optional[pt.Tensor] = None,
        _probe: Optional[bool] = False,
    ):

        self.processing, self.frame = self.op(self.processing, self.source)

        if self.mode is not None:
            self.frame = cv.cvtColor(self.frame, self.mode)

        if _probe:
            self.processing = True
            return

        if self.resize is not None:
            self.frame = cv.resize(self.frame, self.resize, interpolation=cv.INTER_AREA)

        self.frame = pt.from_numpy(self.frame).float()

    def read(
        self,
        center: Optional[pt.Tensor] = None,
    ) -> pt.Tensor:
        """
        Read a frame and flatten it (remove all channel information).
        """

        self._get_frame(center)

        if self.source_depth == 1:
            # The image is already flattened

            return self.frame

        self.frame = self.flatmask * self.frame

        return self.frame.sum(axis=2)

    def scale(
        self,
        _frame: pt.Tensor,
        _min: Optional[float] = 0.0,
        _max: Optional[float] = 255.0,
    ) -> pt.Tensor:
        """
        Min-max normalised version of the frame.
        """

        fmin = float(pt.min(_frame))
        fmax = float(pt.max(_frame))

        return _min + (_max - _min) * (_frame - fmin) / (fmax - fmin)

    def stretch(
        self,
        _frame: Optional[pt.Tensor] = None,
    ) -> pt.Tensor:
        """
        Return a 2D image stretched into a 1D vector.
        """
        if _frame is None:
            _frame = self.frame

        return _frame.t().flatten().reshape(_frame.numel(), 1)

    def fold(
        self,
        _frame: Optional[pt.Tensor] = None,
    ) -> pt.Tensor:
        """
        Return a 2D image from a 1D vector.
        """
        if _frame is None:
            _frame = self.frame
            return _frame

        return _frame.reshape(self.output_width, self.output_height).t()
