"""
seg_pectoral.py
---------------
Detect and exclude the pectoral muscle in medio-lateral oblique (MLO) DBT
projection images.

The pectoral muscle appears as a bright triangular region in the upper-inner
corner of MLO projections.  Its boundary is approximated by a straight line
found with the Hough transform, restricted to the angular range expected for a
pectoral edge (25°–45° from vertical).

Functions
---------
segpectoral(I0, cpoints)
    Detect the pectoral muscle line and return a binary mask that excludes the
    muscle region from the breast mask.
"""

import numpy as np

from scipy.ndimage import gaussian_filter

from skimage.morphology import dilation, disk
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks

from .seg_contour import segcontour


def segpectoral(I0, cpoints=None):
    """
    Detect the pectoral muscle boundary and return an exclusion mask.

    The function:

    1. Uses the breast contour points to define a region of interest (ROI)
       in the upper-inner corner of the image where the pectoral muscle
       is expected.
    2. Pre-processes the ROI with morphological dilation and Gaussian smoothing,
       then applies Canny edge detection.
    3. Runs a Hough transform restricted to angles typical for a pectoral edge
       (25°–45°) to find the dominant straight boundary.
    4. Builds a full-image boolean mask that is ``False`` above the detected
       line (pectoral muscle) and ``True`` elsewhere (breast tissue).

    Parameters
    ----------
    I0 : np.ndarray
        Input MLO projection image (2-D, any numeric dtype).
    cpoints : dict, optional
        Breast contour points ``{'x': xs, 'y': ys}`` as returned by
        :func:`segmentation.seg_contour.segcontour`.  If ``None``, contour
        points are computed internally.

    Returns
    -------
    mask : np.ndarray, shape == I0.shape, dtype bool
        ``True`` for pixels that belong to the breast (pectoral excluded).
        If no line is detected the mask is all ``True`` (no region excluded).
    ppoints : dict
        Detected pectoral line parameters and sample points:

        * ``'x'`` : x-coordinates of sample points within the ROI.
        * ``'y'`` : corresponding y-coordinates.
        * ``'A'`` : y-intercept of the fitted line  ``y = A + B·x``.
        * ``'B'`` : slope of the fitted line.

        All values are ``None`` / empty when no line was found.
    """
    # Compute contour points if not supplied by the caller
    if cpoints is None:
        _, cpoints = segcontour(I0)

    M = I0.shape[0]   # image height (number of rows)

    # --------------------------------------------------------- ROI definition
    # The pectoral muscle ROI is the upper portion of the image bounded
    # by the topmost point of the breast contour in the horizontal direction.

    # Index of the contour point with the maximum column (x) coordinate
    i = np.argmax(cpoints['x'])

    # Limit the ROI height to the lower of half the image or the row of that point
    ymax = int(round(min(0.5 * M, cpoints['y'][i])))

    # Find the minimum column among contour points with rows below ymax
    xmax_candidates = cpoints['x'][cpoints['y'] < ymax]
    if len(xmax_candidates) == 0:
        xmax = I0.shape[1] // 2   # fallback: use the image midpoint
    else:
        xmax = int(round(min(xmax_candidates)))

    # Extract the pectoral ROI from the top-left corner
    I0_c = I0[0:ymax, 0:xmax]

    # Fallback if the ROI turns out to be empty
    if I0_c.size == 0:
        I0_c = I0[0:int(0.5 * M), 0:int(0.5 * I0.shape[1])]

    # --------------------------------------------------------- pre-processing
    # Dilate to close gaps in the muscle boundary, then smooth for stable edges
    I0_c = dilation(I0_c, disk(8))
    I0_c = gaussian_filter(I0_c, sigma=1)

    # Zero out the lower-right sub-quadrant to suppress non-pectoral edges
    M_c, N_c = I0_c.shape
    I0_c[int(0.66 * M_c):, int(0.66 * N_c):] = 0

    # Canny edge detection on the preprocessed ROI
    edge_map = canny(I0_c, sigma=2)

    # --------------------------------------------------------- Hough transform
    # Restrict to angles between 25° and 45° (typical pectoral slope in MLO)
    N         = 128
    theta_min = 25 * np.pi / 180
    theta_max = 45 * np.pi / 180
    theta     = np.linspace(theta_min, theta_max, N)

    hspace, angles, distances = hough_line(edge_map, theta=theta)

    # Find the single strongest peak
    accum, angles_peaks, dists_peaks = hough_line_peaks(
        hspace, angles, distances, num_peaks=1
    )

    # If no line is found, return an all-True mask (no pectoral exclusion)
    if len(angles_peaks) == 0:
        mask    = np.ones_like(I0, dtype=bool)
        ppoints = {'x': np.array([]), 'y': np.array([]), 'A': None, 'B': None}
        return mask, ppoints

    # ------------------------------------------------- line parameterisation
    T = angles_peaks[0]
    R = dists_peaks[0]

    sin_T = np.sin(T)
    cos_T = np.cos(T)

    # Guard against numerical instability near sin(T) = 0
    if sin_T == 0:
        sin_T = np.finfo(float).eps

    # Line equation: y = A + B·x  (in image row/column coordinates)
    A_line = R / sin_T           # y-intercept
    B_line = -cos_T / sin_T      # slope

    # --------------------------------------------------------- full-image mask
    # Build a pixel-coordinate grid and exclude the region above the fitted line
    x_grid, y_grid = np.meshgrid(np.arange(I0.shape[1]), np.arange(I0.shape[0]))

    mask = np.ones_like(I0, dtype=bool)
    mask[y_grid < A_line + B_line * x_grid] = False   # above the line → pectoral

    # -------------------------------------------- sample the line within the ROI
    ppoints_x = np.arange(I0_c.shape[1])
    ppoints_y = A_line + B_line * ppoints_x

    # Keep only points that fall within the ROI row bounds
    valid     = (ppoints_y >= 1) & (ppoints_y <= I0_c.shape[0])
    ppoints_x = ppoints_x[valid]
    ppoints_y = ppoints_y[valid]

    ppoints = {'x': ppoints_x, 'y': ppoints_y, 'A': A_line, 'B': B_line}

    return mask, ppoints