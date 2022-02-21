# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
from typing import List
from typing import Dict
from typing import Tuple
from typing import Union
from typing import Optional

# --------------------------------------
import numpy as np

# --------------------------------------
import math

# --------------------------------------
import enum

# --------------------------------------
from dotmap import DotMap

# --------------------------------------
from scipy.sparse import csr_matrix

# --------------------------------------
import torch as pt
import torch.nn.functional as ptf
from torch.distributions.one_hot_categorical import OneHotCategorical

# --------------------------------------
import cv2 as cv

# --------------------------------------
from mpath.aux.types import View


class ReceptorLayer:

    """
    A layer of receptors and horizontal cells.

    This layer applies local normalisation to the input
    by looking at an eccentricity-dependent patch of receptors
    and their activations to normalise the activation of the
    receptor in the centre.
    """

    def __init__(
        self,
        _source: Union[str, cv.VideoCapture],
        _scaled_height: Optional[int] = None,
        _scaled_width: Optional[int] = None,
        _mode: Optional[enum.Enum] = cv.COLOR_RGB2GRAY,
        _hrf_start_size: int = 13,
        _hrf_shrink_rate: int = 2,
        _center_size: int = 1,
        _center_dilation: int = 1,
        _scale: int = 1,
    ):

        # Store the source
        self.source = _source

        # Indicator showing whether the source is a video capture source
        self.is_capture = isinstance(self.source, cv.VideoCapture)

        # Compute source and output dimensions
        if _scaled_height is not None and _scaled_width is not None:
            raise ValueError("Please provide the new height *or* width, but not both.")

        # Output mode
        self.mode = _mode

        # processing: flag indicating whether the source is still being processed
        # frame: a container for individual source frames
        # frame_op: a function used to extract a source frame
        self.processing = True

        if isinstance(_source, str):
            self.frame_op = lambda processing, src: (processing, cv.imread(_source))

        elif isinstance(_source, cv.VideoCapture):
            self.frame_op = lambda processing, src: src.read()

        else:
            raise TypeError("Invalid input source.")

        # Dimensions and resize flag
        (
            self.original,
            self.scaled,
            self.padding,
            self.resize,
        ) = self._compute_dimensions(_scaled_height, _scaled_width)

        # Create a flatmask
        self.flatmask = self._make_flatmask(_mode)

        # Create the receptor field.
        # This is twice the size of the
        # original input in each dimension.
        self.receptor_field = self._make_receptor_field(
            _hrf_start_size,
            _hrf_shrink_rate,
            _center_size,
            _center_dilation,
            _scale,
        )

        # Change the depth of the processed frame.
        if self.flatmask is not None:
            self.scaled.padded.depth = 1

        # Display some useful info
        print(f"==[ Press ESC to quit.")

    def __del__(self):

        if self.is_capture:
            self.source.release()

        cv.destroyAllWindows()

    def _compute_dimensions(
        self,
        _scaled_height: Optional[int] = None,
        _scaled_width: Optional[int] = None,
    ):

        frame = self._get_frame(_probe=True)

        original = DotMap()

        # NOTE: Dimension order is (H,W,D)
        # NOTE: Perhaps use transparency as well?
        original_shape = list(frame.shape)

        if len(frame.shape) == 2:
            original_shape.append(1)

        (original.height, original.width, original.depth) = tuple(original_shape)
        original.span = original.height * original.width * original.depth

        scaled = DotMap(original)

        resize = False
        if bool(_scaled_width) ^ bool(_scaled_height):
            if _scaled_height is not None:
                # Fixed height, calculate the width with the same AR
                pct = _scaled_height / float(original_shape[0])
                _scaled_width = int((float(original_shape[1]) * pct))

            elif _scaled_width is not None:
                # Fixed width, calculate the height with the same AR
                pct = _scaled_width / float(original_shape[1])
                _scaled_height = int((float(original_shape[0]) * pct))

            scaled.height = _scaled_height
            scaled.width = _scaled_width
            scaled.span = scaled.height * scaled.width * scaled.depth
            resize = True

        # Left and right padding
        left_right_padding = scaled.width // 2 + scaled.width % 2

        # Top and bottom padding
        top_bottom_padding = scaled.height // 2 + scaled.height % 2

        scaled.padded.width = scaled.width + 2 * left_right_padding
        scaled.padded.height = scaled.height + 2 * top_bottom_padding
        scaled.padded.span = scaled.padded.width * scaled.padded.height * scaled.depth

        # The frame is padded only at the top and the bottom,
        # but the left and right padding values are used
        # to compute the size of the retinal field below.
        padding = np.array(
            [
                left_right_padding,
                left_right_padding,
                top_bottom_padding,
                top_bottom_padding,
            ]
        )

        print(f"==[ original: {original}")
        print(f"==[ scaled: {scaled}")
        print(f"==[ padding: {padding}")

        return (original, scaled, padding, resize)

    def _get_rf_indices(
        self,
        _background: pt.Tensor,
        _center: np.ndarray,
        _patch: np.ndarray,
        _extent: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):

        # Receptive field span
        extent = (_center - _patch // 2, _center + _patch // 2)

        _background[extent[0][0] : extent[1][0], extent[0][1] : extent[1][1]] = 1

        if _extent is not None:
            _background[
                _extent[0][0] : _extent[1][0], _extent[0][1] : _extent[1][1]
            ] = -1

        indices = (_background > 0).nonzero()

        return (indices, extent)

    def _get_indices_by_rf_size(
        self,
        _hrf_start_size: int,
        _hrf_shrink_rate: int,
        _center_size: int,
        _center_dilation: int,
        _scale: int,
    ):

        """
        Compute the receptive field size for horizontal cells.

        This will be used to construct a local
        illuminance normalisation map.
        """

        # Get the indices for all the pixels that need
        # to be covered by a RF of a certain size
        patches = []

        # Height and width of the receptor tensor
        rf_width = 0
        rf_height = 0
        extent = 0
        last = False

        while True:

            # Number of pixels covered by kernels of the current size
            # in the corresponding dimension
            extent += 2 * _scale * _center_size * (_center_size + _center_dilation)

            if last:
                rf_width = self.scaled.padded.width
                rf_height = self.scaled.padded.height

            else:
                rf_width = min(extent, self.scaled.padded.width)
                rf_height = min(extent, self.scaled.padded.height)

            # print(f"==[ extent: {extent}")
            # print(f"==[ rw: {rw}")
            # print(f"==[ rh: {rh}")

            patches.append((_hrf_start_size, np.array([rf_height, rf_width])))

            _center_size += _center_dilation
            _hrf_start_size -= _hrf_shrink_rate

            # Sanity check for the horizontal cell RF size
            if _hrf_start_size <= 3:
                _hrf_start_size = 3
                last = True

            if (
                rf_width >= self.scaled.padded.width
                and rf_height >= self.scaled.padded.height
            ):
                break

        # print(f"==[ patches: {[(p[0], p[1]) for p in patches]}")

        # Indices of all pixels requiring RF patches of a particular size
        indices = {}

        # Baseline for torch.where()
        baseline = -pt.ones(self.scaled.padded.height, self.scaled.padded.width)

        # Center of the retina
        center = np.array(
            [self.scaled.padded.height // 2, self.scaled.padded.width // 2]
        )

        extent = None

        for hrf_start_size, patch in patches:

            (indices[hrf_start_size], extent) = self._get_rf_indices(
                baseline, center, patch, extent
            )

        # print(f"==[ indices: {indices}")

        return indices

    def _make_receptor_field(
        self,
        _hrf_start_size: int,
        _hrf_shrink_rate: int,
        _center_size: int,
        _center_dilation: int,
        _scale: int,
    ):
        indices_by_rf_size = self._get_indices_by_rf_size(
            _hrf_start_size,
            _hrf_shrink_rate,
            _center_size,
            _center_dilation,
            _scale,
        )

        # Row and column indices of non-zero entries in the sparse tensor
        # and their corresponding values
        sparse = DotMap()
        sparse.rows = []
        sparse.cols = []
        sparse.vals = []

        # Sparse tensor row index
        for (rfsize, indices) in indices_by_rf_size.items():

            halfrf = rfsize // 2
            patch_end = halfrf + 1 if rfsize % 2 == 1 else halfrf

            # Indices of all the patches for the current RF size
            patch_rows = pt.arange(-halfrf, patch_end).repeat(rfsize)[:, None]
            patch_cols = pt.arange(-halfrf, patch_end).repeat_interleave(rfsize)[
                :, None
            ]

            # print(f"==[ prstrip:\n{patch_rows}")
            # print(f"==[ pcstrip:\n{patch_cols}")

            # Indices of all the patches for the current RF size
            patch_indices = pt.cat((patch_rows, patch_cols), axis=1)

            # print(f"==[ patch indices: {patch_indices.shape}")

            # Stretched pixel indices
            pixel_indices = indices[:, None, :].repeat(1, rfsize ** 2, 1)

            # print(f"==[ pixel indices: {pixel_indices.shape}")

            # Indices for each pixel's RF patch in a single 3D tensor.
            patches = pixel_indices + patch_indices[None, :, :].repeat(
                pixel_indices.shape[0], 1, 1
            )

            # Image boundary mask for the `patches` tensor.
            # Ensures that patches don't stretch beyond the image borders.
            # We can afford to do that because we are using a sparse tensor.
            # If we use a dense tensor, we'd have to pad the image beyond
            # the padding necessary for saccades.
            im_row_bound = patches[:, :, 0]
            im_col_bound = patches[:, :, 1]

            mask = (
                im_row_bound.ge(0)
                * im_row_bound.lt(self.scaled.padded.height)
                * im_col_bound.ge(0)
                * im_col_bound.lt(self.scaled.padded.width)
            )

            # Number of effective pixels contributing to the RF.
            # There will be fewer effective pixels in the corners
            # and around the edges of the image than in the central
            # parts, so they will contribute relatively more to
            # the estimated background illumination.
            effective_pixels = (mask * 1).sum(axis=1).tolist()

            # Pixel weights
            for px in effective_pixels:
                sparse.vals.extend([1 / px] * px)

            # Extract the indices for the patch pixels
            # corresponding to the ones set in the mask
            # TODO: Implement sparsified horizontal cells
            masked_patches = patches[mask]
            # masked_patches = self._make_masked_patches(
            #     _source_height, _source_width, patches[mask]
            # )

            # Update the rows and columns
            sparse.cols.extend(
                (
                    masked_patches[:, 1] * self.scaled.padded.height
                    + masked_patches[:, 0]
                ).tolist()
            )

            pixel_indices = pixel_indices[mask]
            sparse.rows.extend(
                (
                    pixel_indices[:, 1] * self.scaled.padded.height
                    + pixel_indices[:, 0]
                ).tolist()
            )

        return pt.sparse_coo_tensor(
            np.array(
                [
                    sparse.rows,
                    sparse.cols,
                ]
            ),
            np.array(sparse.vals),
            size=(
                self.scaled.padded.span,
                self.scaled.padded.span,
            ),
            dtype=pt.float32,
        ).to_sparse_csr()

    def _make_flatmask(
        self,
        _mode: Optional[int] = None,
    ):
        """
        Create a mask that can be used to obtain a flattened
        version of the original image with colour channel sampling.
        """

        if self.original.depth == 1 or _mode is None:
            # The image is already flat
            return

        print(f"==[ Creating flatmask...")

        probs = pt.zeros((self.scaled.height, self.scaled.width, self.scaled.depth))

        r_prob = 0.475
        g_prob = 0.475
        b_prob = 0.05

        probs[:, :, 0] = r_prob  # R channel
        probs[:, :, 1] = g_prob  # G channel
        probs[:, :, 2] = b_prob  # B channel

        ohc = OneHotCategorical(probs)

        return ohc.sample()

        # print(f"==[ R: {self.flatmask[:,:,0].sum()}")
        # print(f"==[ G: {self.flatmask[:,:,1].sum()}")
        # print(f"==[ B: {self.flatmask[:,:,2].sum()}")

        # # Create the actual tensor
        # self.st = pt.sparse_coo_tensor(
        #     np.array(
        #         [
        #             sparse_rows,
        #             sparse_cols,
        #         ]
        #     ),
        #     np.array(sparse_vals),
        #     size=(st_size, st_size),
        # ).to_sparse_csr()

    def _get_frame(
        self,
        _probe: Optional[bool] = False,
    ):
        """
        Extract a frame at a certain offset from the center.
        """

        # Get the next frame by applying self.frame_op to the source
        self.processing, frame = self.frame_op(self.processing, self.source)

        # Apply the mode
        if self.mode is not None:
            frame = cv.cvtColor(frame, self.mode)

        if _probe:
            self.processing = True
            return frame

        if self.resize is not None:
            frame = cv.resize(
                frame,
                (self.scaled.width, self.scaled.height),
                interpolation=cv.INTER_AREA,
            )

        frame = pt.from_numpy(frame).float()

        if self.flatmask is not None:
            frame *= self.flatmask

        return frame

    def _normalise(
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

    def _stretch(
        self,
        _frame: pt.Tensor,
    ) -> pt.Tensor:
        """
        Return a 2D image stretched into a 1D vector.
        """
        if self.scaled.depth > 1:
            # Stretch out
            return _frame.flatten(_frame.transpose(0, 2))

        return _frame.t().flatten()[:, None]

    def _fold(
        self,
        _frame: pt.Tensor,
    ) -> pt.Tensor:
        """
        Fold a 1D vector into a 2D image.
        """

        # print(f"==[ frame shape: {_frame.shape}")

        return _frame.reshape(
            self.scaled.padded.width,
            self.scaled.padded.height,
        ).t()

    def _get_padding(
        self,
        _width_offset: float = 0.0,
        _height_offset: float = 0.0,
    ) -> pt.Tensor:

        """
        Compute the horizontal and vertical offsets
        from the width and height offset values and
        then compute the padding values from the offsets.
        """

        # TODO boundary check

        horizontal_offset_pixels = int(
            np.sign(_width_offset)
            * math.floor(math.fabs(_width_offset) * self.scaled.width)
        )
        vertical_offset_pixels = int(
            np.sign(_height_offset)
            * math.floor(math.fabs(_height_offset) * self.scaled.height)
        )

        padding = self.padding + np.array(
            [
                horizontal_offset_pixels,
                -horizontal_offset_pixels,
                vertical_offset_pixels,
                -vertical_offset_pixels,
            ]
        )

        # print(f"==[ padding: {padding}")

        return tuple(padding.tolist())

    def _local_mean(
        self,
        _frame: pt.Tensor,
        _padding: Optional[Tuple[int, int, int, int]] = None,
    ) -> pt.Tensor:

        # Sanity check for the padding
        if _padding is None:
            _padding = tuple()

        # Pad the frame so that we can shift the FOV
        # without making the frame 'jump'
        _frame = ptf.pad(_frame, _padding)

        return pt.mm(self.receptor_field, self._stretch(_frame))

    # ==[ Public methods ]==

    def read(
        self,
        _width_offset: float = 0.0,
        _height_offset: float = 0.0,
        _views: Optional[List[View]] = None,
        _overlay: bool = True,
    ) -> pt.Tensor:
        """
        Read the input by applying a certain offset:
        - Read the next frame
            - (Optional) flatten the frame (remove all channel information)
        - Get the input padding corresponding to the specified offset
        - Compute the local contrast normalisation
        """

        views = []

        if _views is None:
            _views = {View.Original}

        frame = self._get_frame()

        if View.Original in _views:
            views.append(frame)

        padding = self._get_padding(_width_offset, _height_offset)

        local_mean = self._local_mean(frame, padding)

        local_mean = self._fold(local_mean)[
            padding[2] : -padding[3],
            padding[0] : -padding[1],
        ]

        if View.LocalMean in _views:
            views.append(local_mean)

        # Subtract the mean and normalise
        norm = self._normalise(frame - local_mean)

        if View.Normalised in _views:
            views.append(norm)

        return views

    def show(
        self,
        _views: List[pt.Tensor],
    ):

        if len(_views) == 0:
            return

        # Show all images
        # for idx, _img in enumerate(_frames):
        cv.imshow(
            f"Result",
            np.hstack([view.numpy() for view in _views]).astype(np.uint8),
        )

        # Press ESC to quit
        self.processing &= cv.waitKey(10) != 27
