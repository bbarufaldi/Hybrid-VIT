"""
get_ROIs.py
-----------
Search for candidate lesion insertion positions inside the 3-D breast volume
and extract / export cropped Regions of Interest (ROIs) for downstream tasks
(e.g. training a lesion-present / lesion-absent classifier).

Functions
---------
get_candidate_pos(mask_breast, bdyThick, mask, mask_resolution, stride)
    Back-project the 2-D breast mask to 3-D, then slide a window over the
    mid-plane slice to harvest positions that are fully inside dense breast
    tissue.

crop_and_save_rois(mask_breast, csv_file, train_folder, test_folder,
                   base_name, roi_size, geo)
    Read a CSV of (X, Y, Z) positions, crop fixed-size ROIs from the
    projection image, shuffle, split 50/50 train/test, and save as raw files.

crop_and_save_sliding_rois(binary_mask, original_image, roi_size, output_dir,
                           exam, csv_filename, overlap_percent, geo)
    Slide a window over a binary breast mask, extract all-breast ROIs from
    the original image, shuffle, split 50/50 train/test, save as raw files,
    and record positions in a CSV.

save_raw(data, filename)
    Scale an array to 10-bit range and write it as a big-endian uint16 raw file.
"""

import numpy as np
import math
import os
import csv
import random

import pandas as pd
from skimage import draw
from tifffile import imwrite

# PyDBT projection / reconstruction operators
from functions.projection_operators import backprojectionDDb_cuda, projectionDD
from functions.FBP import FDK as FBP
from parameters.parameterSettings import geometry_settings
from functions.initialConfig import initialConfig

import sys
sys.path.insert(1, 'build')

# Initialise pyDBT shared libraries (no output folder needed here)
libFiles = initialConfig(buildDir='build', createOutFolder=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_candidate_pos(mask_breast, bdyThick, mask, mask_resolution, stride=[100, 100]):
    """
    Find candidate lesion insertion positions inside the dense breast region.

    The 2-D projection-domain breast mask is back-projected to 3-D.  A mid-plane
    slice is extracted and thresholded (> 0.4) to identify voxels that belong to
    dense breast tissue.  A window whose footprint matches the lesion mask size is
    then slid across this slice; positions where every pixel in the window belongs
    to dense tissue are recorded as valid insertion candidates.

    Parameters
    ----------
    mask_breast : np.ndarray, shape (nv, nu)
        2-D projection-domain breast mask (binary or soft mask in [0, 1]).
    bdyThick : float
        Breast body thickness in mm, used to determine the number of
        reconstruction slices.
    mask : np.ndarray, shape (mx, my[, ...])
        Lesion footprint mask used only to infer the spatial extent of the
        insertion window in physical units.
    mask_resolution : list or tuple of float
        Voxel size ``[dx, dy]`` of the lesion mask in mm.
    stride : list of int, optional
        Step size ``[row_step, col_step]`` in reconstruction voxels between
        candidate positions.  Default is ``[100, 100]``.

    Returns
    -------
    positions : tuple of (list, list, list)
        ``(x_pos, y_pos, z_pos)`` — column, row, and slice indices of the
        accepted candidate positions in reconstruction-volume voxel coordinates.
    slice2check_bool : np.ndarray, shape (ny, nx), dtype bool
        Mid-plane binary slice with accepted ROI outlines erased to zero (useful
        for visual debugging).
    geo : geometry_settings
        Geometry object used for the backprojection (Hologic defaults).
    """
    # Density threshold: voxels above this value are considered "dense tissue"
    denseThreshold = 0.4

    # Use Hologic default geometry
    geo = geometry_settings()
    geo.Hologic()

    # Set the geometry dimensions to match the projection-domain breast mask
    geo.nx = mask_breast.shape[1]   # columns
    geo.ny = mask_breast.shape[0]   # rows
    geo.nu = mask_breast.shape[1]
    geo.nv = mask_breast.shape[0]
    geo.nz = np.ceil(bdyThick / geo.dz).astype(int)  # depth from body thickness

    # Keep isotropic in-plane voxel sizes consistent with detector pitch
    geo.dy = geo.du
    geo.dx = geo.dv
    geo.detAngle = 0  # flat-panel detector

    # ------------------------------------------- 3-D back-projection
    # Back-project the 2-D mask to obtain a rough 3-D density volume
    vol = backprojectionDDb_cuda(np.float64(mask_breast), geo, -1, libFiles)

    # Extract the central depth slice (mid-plane) and threshold
    slice2check = vol[..., geo.nz // 2].copy()
    del vol  # free GPU/CPU memory

    slice2check_bool = slice2check > denseThreshold

    # ------------------------------------------- lesion window size
    # Convert the physical lesion mask extent to reconstruction pixels
    mask_pixel_size_x = int(mask.shape[0] * mask_resolution[0] / geo.dx)
    mask_pixel_size_y = int(mask.shape[1] * mask_resolution[1] / geo.dy)

    x_pos, y_pos = [], []
    height, width = slice2check_bool.shape[:2]

    # ------------------------------------------- sliding window search
    # Accept a position only when every pixel in the window is inside dense tissue
    for y in range(0, height - mask_pixel_size_y + 1, stride[0]):
        for x in range(0, width - mask_pixel_size_x + 1, stride[1]):
            roi = slice2check_bool[y:y + mask_pixel_size_y, x:x + mask_pixel_size_x]

            if np.all(roi):
                # Record the lower-right corner as the anchor point
                x_pos.append(int(x + mask_pixel_size_x))
                y_pos.append(int(y + mask_pixel_size_y))

                # Erase the accepted window perimeter from the debug slice
                rr, cc = draw.rectangle_perimeter(
                    (y, x),
                    end=(y + mask_pixel_size_y - 1, x + mask_pixel_size_x - 1),
                    shape=slice2check_bool.shape,
                )
                slice2check_bool[rr, cc] = 0

    # All candidates share the same depth: the central slice
    z_pos = len(x_pos) * [geo.nz // 2]

    return (x_pos, y_pos, z_pos), slice2check_bool, geo


def crop_and_save_rois(mask_breast, csv_file, train_folder, test_folder,
                       base_name, roi_size, geo=None):
    """
    Crop ROIs from a breast projection image and export them as raw files.

    Positions are read from *csv_file*, the corresponding crops are extracted
    from *mask_breast*, shuffled, split 50/50 into train and test sets, and
    saved as 10-bit big-endian raw files.

    Parameters
    ----------
    mask_breast : np.ndarray, shape (nv, nu)
        Source image from which ROIs are cropped.
    csv_file : str
        Path to a CSV file with columns ``X``, ``Y``, ``Z`` (voxel positions).
    train_folder : str
        Output directory for the training split.
    test_folder : str
        Output directory for the test split.
    base_name : str
        Prefix used in output filenames.
    roi_size : list or tuple of float
        Physical ROI dimensions ``[width_mm, height_mm]`` in mm.
    geo : geometry_settings, optional
        DBT geometry used to convert physical mm to pixel counts.  Defaults to
        Hologic geometry when ``None``.
    """
    if geo is None:
        geo = geometry_settings()
        geo.Hologic()

    # Read candidate positions from CSV
    df = pd.read_csv(csv_file)
    x_pos = df['X'].values
    y_pos = df['Y'].values
    z_pos = df['Z'].astype(int).values

    # Ensure output directories exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Shuffle positions and split 50/50
    rois = list(zip(x_pos, y_pos, z_pos))
    random.shuffle(rois)
    split_index = len(rois) // 2
    train_rois = rois[:split_index]
    test_rois  = rois[split_index:]

    def crop_roi(mask, x, y, size):
        """
        Crop a fixed-size ROI from *mask* centred on *(x, y)*, with boundary
        padding when the window extends beyond the image edges.

        Parameters
        ----------
        mask : np.ndarray
            Source projection image.
        x : int
            Right-most column of the ROI anchor.
        y : int
            Vertical centre of the ROI anchor.
        size : list of int
            ``[target_w - 1, target_h - 1]`` in pixels (the +1 offset is added
            internally to match the intended output dimensions).

        Returns
        -------
        roi : np.ndarray
            Cropped (and possibly zero-padded) array of shape
            ``(size[1]+1, size[0]+1)``.
        """
        img_h, img_w = mask.shape[:2]

        # Intended output dimensions (size already includes +1 from caller)
        target_w = size[0] + 1
        target_h = size[1] + 1

        # x is the rightmost pixel; y is the row centre of the window
        x_start = x - target_w + 1
        x_end   = x + 1
        y_start = y - target_h // 2
        y_end   = y_start + target_h

        # Clamp to image boundaries
        roi = mask[max(0, y_start):min(img_h, y_end),
                   max(0, x_start):min(img_w, x_end)]

        # Zero-pad if the window extends outside the image
        if roi.shape[0] != target_h or roi.shape[1] != target_w:
            pad_before_y = max(0, -y_start)
            pad_after_y  = max(0,  y_end - img_h)
            pad_before_x = max(0, -x_start)
            pad_after_x  = max(0,  x_end - img_w)
            roi = np.pad(
                roi,
                ((pad_before_y, pad_after_y), (pad_before_x, pad_after_x)),
                mode='constant',
            )

        return roi

    # Convert physical ROI size (mm) to detector pixels
    size = [math.ceil(roi_size[0] / geo.du), math.ceil(roi_size[1] / geo.dv)]

    # Save training ROIs
    for (x, y, z) in train_rois:
        roi = crop_roi(mask_breast, x, y, size)
        filename = os.path.join(
            train_folder,
            f"{base_name}_{x}_{y}_{z}_{size[0]+1}x{size[1]+1}.raw",
        )
        save_raw(roi, filename)

    # Save test ROIs
    for (x, y, z) in test_rois:
        roi = crop_roi(mask_breast, x, y, size)
        filename = os.path.join(
            test_folder,
            f"{base_name}_{x}_{y}_{z}_{size[0]+1}x{size[1]+1}.raw",
        )
        save_raw(roi, filename)


def save_raw(data, filename):
    """
    Linearly scale an array to the 10-bit range and write it as a raw file.

    The data is normalised to [0, 1023], cast to ``uint16``, byte-swapped to
    big-endian order, and written using :meth:`numpy.ndarray.tofile`.

    Parameters
    ----------
    data : np.ndarray
        Input array of arbitrary floating-point dtype.
    filename : str
        Destination file path (including the ``.raw`` extension).
    """
    min_val = data.min()
    max_val = data.max()

    # Normalise to [0, 1023] (10-bit range)
    roi_10bit = ((data - min_val) / (max_val - min_val) * 1023).astype(np.uint16)

    # Swap to big-endian byte order before writing
    roi_10bit = roi_10bit.byteswap().newbyteorder()
    roi_10bit.tofile(filename)


def crop_and_save_sliding_rois(binary_mask, original_image, roi_size,
                               output_dir, exam, csv_filename,
                               overlap_percent=0, geo=None):
    """
    Extract all-breast ROIs with a sliding window and export as raw files.

    A rectangular window is slid across *binary_mask*.  Positions where the
    entire window contains breast pixels (all ``True``) are accepted.  The
    corresponding crop from *original_image* is saved as a 10-bit raw file, and
    all positions are recorded in *csv_filename*.

    Parameters
    ----------
    binary_mask : np.ndarray, shape (nv, nu), dtype bool
        Binary breast mask; ``True`` indicates breast pixels.
    original_image : np.ndarray, shape (nv, nu)
        Projection intensity image to crop from.
    roi_size : list or tuple of float
        Physical ROI dimensions ``[width_mm, height_mm]`` in mm.
    output_dir : str
        Root output directory.  ``train/`` and ``test/`` sub-directories are
        created automatically.
    exam : str
        Exam identifier prepended to each output filename.
    csv_filename : str
        Path for the CSV file recording accepted ROI positions and their
        train/test assignment.
    overlap_percent : float, optional
        Percentage overlap between adjacent windows in [0, 100).  Default is
        0 (non-overlapping grid).
    geo : geometry_settings, optional
        DBT geometry used to convert physical mm to pixel counts.  Defaults to
        Hologic geometry when ``None``.
    """
    if geo is None:
        geo = geometry_settings()
        geo.Hologic()

    # Convert physical ROI size (mm) to integer pixel counts
    size = [math.ceil(roi_size[0] / geo.du), math.ceil(roi_size[1] / geo.dv)]
    window_height = size[1] + 1
    window_width  = size[0] + 1

    mask_height, mask_width = binary_mask.shape

    # Compute the sliding step based on the requested overlap fraction
    step_y = int(window_height * (1 - overlap_percent / 100))
    step_x = int(window_width  * (1 - overlap_percent / 100))

    rois = []   # list of (roi_image, (y, x)) tuples

    # ------------------------------------------- sliding window extraction
    for y in range(0, mask_height - window_height + 1, step_y):
        for x in range(0, mask_width - window_width + 1, step_x):
            roi_mask = binary_mask[y:y + window_height, x:x + window_width]

            # Accept only windows that are entirely inside the breast
            if np.all(roi_mask == True):
                roi_image = original_image[y:y + window_height, x:x + window_width]

                # Erase the accepted window perimeter for visual debugging
                rr, cc = draw.rectangle_perimeter(
                    (y, x),
                    end=(y + window_height - 1, x + window_width - 1),
                    shape=binary_mask.shape,
                )
                binary_mask[rr, cc] = 0

                rois.append((roi_image, (y, x)))

    # ------------------------------------------- shuffle and 50/50 split
    random.shuffle(rois)
    split_index = len(rois) // 2
    train_rois = rois[:split_index]
    test_rois  = rois[split_index:]

    # Create output sub-directories
    train_dir = os.path.join(output_dir, "train")
    test_dir  = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)

    # Save training ROIs
    for roi_image, (y, x) in train_rois:
        filename = os.path.join(
            train_dir, f"{exam}_{y}_{x}_{size[0]+1}x{size[1]+1}.raw"
        )
        save_raw(roi_image, filename)

    # Save test ROIs
    for roi_image, (y, x) in test_rois:
        filename = os.path.join(
            test_dir, f"{exam}_{y}_{x}_{size[0]+1}x{size[1]+1}.raw"
        )
        save_raw(roi_image, filename)

    # ------------------------------------------- write position CSV
    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["y", "x", "set"])
        for _, (y, x) in train_rois:
            csv_writer.writerow([y, x, "train"])
        for _, (y, x) in test_rois:
            csv_writer.writerow([y, x, "test"])