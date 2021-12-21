# ------------------------------------------------------------------------------
# Global imports
# ------------------------------------------------------------------------------
# *** Type hints *** #
from typing import Optional
from typing import List
from typing import Tuple

# *** PyTorch & numerical libs *** #
import torch as pt
from torch.distributions.one_hot_categorical import OneHotCategorical
import numpy as np

pt.set_printoptions(linewidth=200)

# ------------------------------------------------------------------------------
# Local imports
# ------------------------------------------------------------------------------
from mpath.layer.source import Source


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
        _iw: int,  # Image width
        _ih: int,  # Image height
    ):

        self.iw = _iw
        self.ih = _ih

        # print(f"==[ _iw: {_iw}")
        # print(f"==[ _ih: {_ih}")

        indices_by_rf_size = self._get_indices_by_rf_size(_iw, _ih)

        st_rows = []
        st_cols = []
        st_vals = []

        # Sparse tensor row index
        st_row = 0
        for rfsize, indices in indices_by_rf_size.items():

            halfrf = rfsize // 2

            # Indices of all the patches for the current RF size
            patch_rows = pt.arange(-halfrf, halfrf + 1).repeat(rfsize)[:, None]
            patch_cols = pt.arange(-halfrf, halfrf + 1).repeat_interleave(rfsize)[
                :, None
            ]

            # print(f"==[ prstrip:\n{patch_rows}")
            # print(f"==[ pcstrip:\n{patch_cols}")

            # Indices of all the patches for the current RF size
            patch_indices = pt.cat((patch_rows, patch_cols), axis=1)

            # print(f"==[ pidx: {patch_indices}")

            # Stretched pixel indices
            pixel_indices = indices[:, None, :].repeat(1, rfsize ** 2, 1)

            # Indices for each pixel's RF patch in a single 3D tensor.
            patches = pixel_indices + patch_indices[None, :, :].repeat(
                pixel_indices.shape[0], 1, 1
            )

            # Image boundary mask for the `patches` tensor.
            # Ensures that patches don't stretch beyond the image borders.
            # We can afford to do that because we are using a sparse tensor.
            # If we use a dense tensor, we'd have to pad the image.
            im_row_bound = patches[:, :, 0]
            im_col_bound = patches[:, :, 1]

            mask = (
                im_row_bound.ge(0)
                * im_row_bound.lt(_ih)
                * im_col_bound.ge(0)
                * im_col_bound.lt(_iw)
            )

            # Number of effective pixels contributing to the RF.
            # There will be fewer effective pixels in the corners
            # and around the edges of the image than in the central
            # parts, so they will contribute relatively more to
            # the estimated background illumination.
            effective_pixels = (mask * 1).sum(axis=1).tolist()

            # Pixel weights
            for px in effective_pixels:
                st_vals.extend([1 / px] * px)

            # Extract the indices for the patch pixels
            # corresponding to the ones set in the mask
            masked_patches = patches[mask]

            # if len(pixel_weights) == masked_patches.size(0):
            #     print(
            #         f"==[ Dimensions match: {len(pixel_weights)} vs. {masked_patches.size(0)}"
            #     )
            # else:
            #     print(
            #         f"==[ Dimensions do NOT mismatch: {len(pixel_weights)} vs. {masked_patches.size(0)}"
            #     )

            # Update the rows and columns
            st_cols.extend((masked_patches[:, 1] * _ih + masked_patches[:, 0]).tolist())

            pixel_indices = pixel_indices[mask]
            st_rows.extend((pixel_indices[:, 1] * _ih + pixel_indices[:, 0]).tolist())

        st_size = _iw * _ih

        # Create the actual tensor
        self.st = pt.sparse_coo_tensor(
            [st_rows, st_cols],
            st_vals,
            size=(st_size, st_size),
        ).coalesce()

        # print(f"==[ self.st: {self.st}")

        # ############################################

        # img = cv2.imread(_path, _mode)

        # if _height is not None and _width is not None:
        #     raise ValueError("Please provide the new height *or* width, but not both.")

        # shape = tuple(img.shape)

        # # Image dimensions
        # self.orig_width = shape[0]
        # self.orig_height = shape[1]

        # # NOTE
        # # In what cases would we have more than 3 channels?
        # # Perhaps in the case of transparency...?
        # self.orig_depth = shape[2] if len(img.shape) > 2 else 1

        # self.resize = _height is not None or _width is not None
        # self.height = self.orig_width
        # self.width = self.orig_height
        # self.depth = self.orig_depth

        # if _height is not None:
        #     # Fixed height, calculate the width with the same AR
        #     pct = _height / float(self.orig_height)
        #     _width = int((float(self.orig_width) * pct))

        # elif _width is not None:
        #     # Fixed width, calculate the height with the same AR
        #     pct = _width / float(self.orig_width)
        #     _height = int((float(self.orig_height) * pct))

        # if self.resize:
        #     img = cv2.resize(img, (_width, _height), interpolation=cv2.INTER_AREA)
        #     self.height = _height
        #     self.width = _width

        # # Create a flatmask
        # self._make_flatmask()

        # self.img = pt.from_numpy(img).float()

    def _make_flatmask(self):
        """
        Create a mask that will can be used to obtain a flattened
        version of the original image with colour channel sampling.
        """

        self.flatmask = None

        if self.orig_depth == 1:
            # The image is already flat
            return

        print(f"==[ Creating flatmask...")

        probs = pt.zeros((self.height, self.width, self.depth))

        r_prob = 0.45
        g_prob = 0.45
        b_prob = 0.1

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
        self.depth = 1

    def _get_rf_indices(self, _im_row: int, _im_col: int, _halfrf: int):

        # Start and end of the patch in the sparse tensor
        patch_cstart = max(_im_col - _halfrf, 0)
        patch_cend = min(_im_col + _halfrf, self.iw)

        patch_rstart = max(_im_row - _halfrf, 0)
        patch_rend = min(_im_row + _halfrf, self.ih)

        st_col_idx = []

        # Create a patch
        for patch_col in range(patch_cstart, patch_cend):

            st_col_idx.extend(
                list(self.ih * patch_col + np.arange(patch_rstart, patch_rend))
            )

        st_row = self.ih * _im_col + _im_row
        st_row_idx = [st_row] * len(st_col_idx)

        return (st_row_idx, st_col_idx)

    def _get_indices_by_rf_size(
        self,
        _iw: int,
        _ih: int,
        _scale: int = 1,
    ):

        """
        Compute the receptive field size for horizontal cells.

        This will be used to construct a local
        illuminance normalisation map.
        """

        # Initial RF size
        rfsize = 1

        # Increment for the RF size
        dilation = 4

        # Get the indices for all the pixels that need
        # to be covered by a RF of a certain size
        patches = []

        # Height and width of the receptor tensor
        rw = 0
        rh = 0
        extent = 0

        while True:

            # Number of pixels covered by kernels of the current size
            # in the corresponding dimension
            extent += 2 * _scale * rfsize * (rfsize + dilation)

            rw = min(extent, _iw)
            rh = min(extent, _ih)

            # print(f"==[ extent: {extent}")
            # print(f"==[ rw: {rw}")
            # print(f"==[ rh: {rh}")

            patches.append((rfsize, np.array([rh, rw])))

            rfsize += dilation

            if rw >= _iw and rh >= _ih:
                break

        # print(f"==[ patches: {[(p[0], p[1]) for p in patches]}")

        # Indices of all pixels requiring RF patches of a particular size
        indices = {}

        # Background serving as a baseline for torch.where()
        background = -pt.ones(_ih, _iw)

        # Center of the retina
        center = np.array([_ih // 2, _iw // 2])

        ext = None

        for rfsize, patch in patches:

            (indices[rfsize], ext) = self._get_rf_indices(
                background, center, patch, ext
            )

        # print(f"==[ indices: {indices}")

        return indices

    def _get_rf_indices(
        self,
        _background: pt.Tensor,
        _center: np.ndarray,
        _patch: np.ndarray,
        _ext: Optional[Tuple[np.ndarray]] = None,
    ):

        ext = (_center - _patch // 2, _center + _patch // 2)

        # print(f"==[ ext: {ext}")

        _background[ext[0][0] : ext[1][0], ext[0][1] : ext[1][1]] = 1

        if _ext is not None:
            _background[_ext[0][0] : _ext[1][0], _ext[0][1] : _ext[1][1]] = -1

        indices = (_background > 0).nonzero()

        return (indices, ext)

    def local_norm(self, _frame: pt.Tensor) -> pt.Tensor:

        mean = pt.mm(self.st, _frame)

        return mean
