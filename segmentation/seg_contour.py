"""
seg_contour.py
--------------
Segment the breast outline from a single DBT projection image.

The algorithm uses histogram analysis to locate the air–tissue intensity
threshold, thresholds the image to a binary mask, and optionally applies
curvature analysis to truncate the contour at the chest wall.

Functions
---------
segcontour(im, ccflag)
    Compute the breast contour mask and return it together with the boundary
    points.
"""

import numpy as np

from scipy.signal.windows import gaussian
from scipy.signal import savgol_filter, convolve
from scipy.interpolate import interp1d

from skimage.measure import label, find_contours
from skimage.draw import polygon2mask


def segcontour(im, ccflag=False):
    """
    Find the breast contour and produce a binary mask.

    The function:

    1. Identifies the range of rows that contain breast signal (by row
       intensity range).
    2. Builds a smoothed intensity histogram of the central region and finds the
       air–tissue threshold as the first drop below 5 % of the histogram peak.
    3. Thresholds the full image, keeps only the largest connected foreground
       component, and extracts its outer contour.
    4. For CC views (``ccflag=True``) the contour is returned directly.
       For MLO views (``ccflag=False``) curvature analysis is performed to trim
       the contour at the chest wall, then a closed polygon mask is generated.

    Parameters
    ----------
    im : np.ndarray
        Input mammography projection image (any numeric dtype).
    ccflag : bool, optional
        If ``True``, return the thresholded mask and raw contour points without
        curvature-based chest-wall trimming (suitable for CC acquisitions).
        Default is ``False`` (MLO mode with curvature analysis).

    Returns
    -------
    mask : np.ndarray, shape == im.shape, dtype bool
        Binary breast mask.
    cpoints : dict
        Contour points as ``{'x': xs, 'y': ys}`` where *xs* are column
        coordinates and *ys* are row coordinates.
    """
    # Curvature threshold — higher values are more strict (fewer cuts)
    C_th = 0.07

    # Work in float64 throughout
    im = np.asarray(im, dtype=np.float64)

    # -------------------------------------------------------------- Step 1
    # Identify the active row range (rows that contain breast signal).
    row_max = np.max(im, axis=1)
    row_min = np.min(im, axis=1)
    row_range = row_max - row_min
    row_range_normalized = row_range / np.max(row_range)

    # Keep only rows with a non-trivial intensity variation
    indices = np.where(row_range_normalized > 0.001)[0]
    if len(indices) > 0:
        C1 = indices[0]
        C2 = indices[-1]
    else:
        # Fallback: use the entire image height
        C1 = 0
        C2 = im.shape[0] - 1

    # Central region used for histogram analysis
    I0 = im[C1:C2, :]

    # -------------------------------------------------------------- Step 2
    # Find the intensity range within the central region
    x_min = np.min(I0)
    x_max = np.max(I0)

    # -------------------------------------------------------------- Step 3
    # Build a 1000-bin histogram of the central-region intensities
    x_bins = np.linspace(x_min, x_max, 1000)
    n, bin_edges = np.histogram(I0.flatten(), bins=x_bins)
    x_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # -------------------------------------------------------------- Step 4
    # Smooth the histogram with a Gaussian window to suppress noise
    gw = 25
    window = gaussian(M=gw, std=gw / 5)
    window = window / np.sum(window)          # normalise to unit sum
    n_smoothed = convolve(n, window, mode='same')
    nmax = np.max(n_smoothed)

    # -------------------------------------------------------------- Step 5
    # The air–tissue boundary is found where the histogram first rises above
    # 5 % of its peak value, then subsequently falls below it again.
    threshold_value = nmax * 0.05

    # Find where the histogram first exceeds the 5 % threshold
    indices_above = np.where(n_smoothed > threshold_value)[0]
    imax = indices_above[0] if len(indices_above) > 0 else 0

    # Find where it subsequently drops back below the threshold
    indices_below = np.where(n_smoothed[imax:] <= threshold_value)[0]
    airloc = (imax + indices_below[0] + 1) if len(indices_below) > 0 else len(n_smoothed) - 1

    # Intensity value corresponding to the air–tissue boundary
    airthresh = x_centers[airloc]

    # -------------------------------------------------------------- Step 6
    # Threshold the full image to create the initial binary mask
    I0_mask = im >= airthresh

    # -------------------------------------------------------------- Step 7
    # Remove artefacts by keeping the largest connected component whose centre
    # of mass is furthest from the image edge (i.e. the breast itself).
    labels = label(I0_mask)
    label_indices = np.unique(labels)
    label_indices = label_indices[label_indices != 0]   # exclude background
    L = len(label_indices)

    if L > 1:
        ccenter = np.zeros(L)
        csize   = np.zeros(L)
        for idx, label_num in enumerate(label_indices):
            positions       = np.where(labels == label_num)
            ccenter[idx]    = np.mean(positions[1])      # column centre of mass
            csize[idx]      = len(positions[1])          # component size

        # Retain only the largest cluster (most pixels = breast region)
        idx_sorted       = np.argsort(-csize)
        selection_label  = label_indices[idx_sorted[0]]
        I0_mask          = labels == selection_label

    # -------------------------------------------------------------- Step 8
    # Extract the outer boundary of the binary mask
    contours = find_contours(I0_mask, level=0.5)
    if contours:
        B = contours[0]   # largest contour
    else:
        # Nothing found — return an empty mask
        mask    = np.zeros(im.shape, dtype=bool)
        cpoints = {'x': np.array([]), 'y': np.array([])}
        return mask, cpoints

    # Contour coordinates are in (row, column) order
    Bmin = np.min(B, axis=0)
    Bmax = np.max(B, axis=0)

    # Strip points that sit exactly on the image border (likely artefacts)
    remov = (
        (B[:, 0] == Bmin[0]) | (B[:, 0] == Bmax[0]) |
        (B[:, 1] == Bmin[1]) | (B[:, 1] == Bmax[1])
    )
    ys = B[~remov, 0]   # row coordinates
    xs = B[~remov, 1]   # column coordinates

    # For CC views, return the raw contour without chest-wall analysis
    if ccflag:
        cpoints = {'x': xs, 'y': ys}
        mask    = I0_mask
        return mask, cpoints

    # -------------------------------------------------------------- Step 9
    # MLO view: curvature analysis to detect and cut at the chest wall.
    N = len(xs)
    s = np.linspace(0, N - 1, 100)   # 100 evenly-spaced parameter values

    # Cubic interpolation of the contour for smooth derivative estimation
    interp_func_x = interp1d(np.arange(N), xs, kind='cubic')
    interp_func_y = interp1d(np.arange(N), ys, kind='cubic')
    xss = interp_func_x(s)
    yss = interp_func_y(s)

    # Savitzky-Golay smoothing preserves peak positions better than a simple filter
    xss = savgol_filter(xss, window_length=11, polyorder=3)
    yss = savgol_filter(yss, window_length=11, polyorder=3)

    # Drop the two endpoint samples (boundary artefacts from the filter)
    xss = xss[2:-2]
    yss = yss[2:-2]

    # First and second derivatives for curvature computation
    dx1 = np.gradient(xss)
    dy1 = np.gradient(yss)
    dx2 = np.gradient(dx1)
    dy2 = np.gradient(dy1)

    # Signed curvature κ = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
    denominator = (dx1 ** 2 + dy1 ** 2) ** 1.5
    denominator[denominator == 0] = np.finfo(float).eps   # avoid divide-by-zero
    k = (dx1 * dy2 - dy1 * dx2) / denominator

    # Point of maximum (negative) curvature — candidate chest-wall cut point
    kmin = np.min(k)
    i    = np.argmin(k)

    # Cut the contour if curvature exceeds threshold AND the point is near the
    # chest-wall side of the image (x < 25 % of max x)
    if (abs(kmin) > C_th) and (xss[i] < 0.25 * np.max(xss)):
        idx_to_remove = int(np.round(s[i]))
        xs = xs[:idx_to_remove]
        ys = ys[:idx_to_remove]

    # Remove two points from each end to avoid filter boundary artefacts
    P = len(xs)
    if P >= 4:
        xs = xs[2:-2]
        ys = ys[2:-2]

    cpoints = {'x': xs, 'y': ys}

    # ------------------------------------------------------------- Step 10
    # Close the contour by anchoring it to the image edge and build the polygon mask
    xs = np.concatenate(([1],            xs, [1]))
    ys = np.concatenate(([np.min(ys)],   ys, [np.max(ys)]))

    polygon = np.vstack((ys, xs)).T
    mask    = polygon2mask(im.shape, polygon)

    return mask, cpoints