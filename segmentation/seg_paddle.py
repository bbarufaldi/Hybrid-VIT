"""
seg_paddle.py
-------------
Detect and segment compression paddle marks in DBT projection images.

The compression paddle leaves bright horizontal bands at the top and/or bottom
of the projection.  These are detected using Canny edge detection + the Hough
transform and then masked out before breast segmentation.

Functions
---------
has_paddle_marks(im, line_threshold, v_mean_threshold)
    Lightweight check — returns ``True`` if the image contains compression-paddle
    artefacts.

segpaddle(im, thickness)
    Detect paddle bands and return a binary mask covering only the breast region
    (i.e. the area *between* the paddle lines).

get_hline(edge_map, n_max)
    Low-level helper: find up to *n_max* nearly-horizontal lines in an edge map
    via the Hough transform.
"""

import numpy as np

from skimage.morphology import dilation, disk
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks

from scipy.ndimage import gaussian_filter


def has_paddle_marks(im, line_threshold=1, v_mean_threshold=150):
    """
    Check whether a projection image contains compression-paddle marks.

    The test uses Canny edge detection followed by a restricted Hough transform
    (near-horizontal lines only).  If at least *line_threshold* lines are
    detected with a mean accumulator value ≥ *v_mean_threshold*, paddle marks
    are assumed to be present.

    Parameters
    ----------
    im : np.ndarray
        Input DBT projection image (2-D, float or uint).
    line_threshold : int, optional
        Minimum number of detected lines to classify as paddle marks.
        Default is 1.
    v_mean_threshold : float, optional
        Minimum mean Hough accumulator value required.  Higher values demand
        stronger evidence of a straight horizontal edge.  Default is 150.

    Returns
    -------
    bool
        ``True`` if paddle marks are detected, ``False`` otherwise.
    """
    # Pre-process: dilate to fill small gaps, then smooth for stable edge detection
    im_dilated  = dilation(im, disk(8))
    im_filtered = gaussian_filter(im_dilated, sigma=1)
    edge_map    = canny(im_filtered, sigma=2)

    # Detect near-horizontal lines in the edge map
    R, v = get_hline(edge_map, n_max=4)
    v_mean = np.mean(v) if len(v) > 0 else 0

    # Both criteria must be satisfied
    return len(R) >= line_threshold and v_mean >= v_mean_threshold


def segpaddle(im, thickness=1):
    """
    Segment the breast region by removing compression-paddle bands.

    Horizontal lines at and inside the top / bottom 20 % of the image are
    interpreted as paddle edges.  The function returns a binary mask that is
    ``True`` between the inner edge of the top paddle band and the inner edge
    of the bottom paddle band.

    Parameters
    ----------
    im : np.ndarray
        Input DBT projection image (2-D, float or uint).
    thickness : int, optional
        Additional pixel margin applied inward from the detected paddle line.
        Default is 1.

    Returns
    -------
    mask : np.ndarray, shape == im.shape, dtype bool
        ``True`` for pixels inside the breast region (between paddle bands),
        ``False`` within the paddle artefact zones.
    v_mean : float
        Mean Hough accumulator value of all detected lines (for diagnostics).
    """
    n_max  = 4    # maximum number of Hough peaks to consider
    off_set = 5   # additional inward offset from the detected line (pixels)
    v_th   = 150  # accumulator threshold; values below this are rejected

    # ----------------------------------------- pre-processing
    im_dilated  = dilation(im, disk(8))
    im_filtered = gaussian_filter(im_dilated, sigma=1)
    edge_map    = canny(im_filtered, sigma=2)
    M, _        = edge_map.shape

    # Remove two-pixel border edges to avoid detecting frame boundaries
    edge_map[:2,  :] = False
    edge_map[-2:, :] = False

    # ----------------------------------------- Hough line detection
    R, v   = get_hline(edge_map, n_max)
    v_mean = np.mean(v) if len(v) > 0 else 0

    # Discard lines below the accumulator threshold
    valid_lines = v >= v_th
    R = R[valid_lines]
    v = v[valid_lines]

    # Keep only lines in the outer 20 % of the image (top or bottom paddle region)
    outer_mask = (R <= 0.2 * M) | (R >= 0.8 * M)
    R = R[outer_mask]
    v = v[outer_mask]

    # ----------------------------------------- compute breast region bounds
    y_upper = 0     # first valid row (below the top paddle)
    y_lower = M     # last valid row (above the bottom paddle)

    for r in R:
        if r > 0.5 * M:
            # Bottom paddle: the upper boundary of the mask is lowered
            y = r - off_set
            y_lower = min(y_lower, y - thickness)
        else:
            # Top paddle: the lower boundary of the mask is raised
            y = r + off_set
            y_upper = max(y_upper, y + thickness)

    # ----------------------------------------- build boolean mask
    mask    = np.zeros_like(im, dtype=bool)
    y_upper = int(np.clip(y_upper, 0, M))
    y_lower = int(np.clip(y_lower, 0, M))
    mask[y_upper:y_lower, :] = True   # breast region between the paddle bands

    return mask, v_mean


def get_hline(edge_map, n_max):
    """
    Detect nearly-horizontal lines via the Hough transform.

    The search is restricted to angles within ±2.5° of 90° (i.e. lines whose
    normal direction is close to horizontal) to target near-horizontal features.

    Parameters
    ----------
    edge_map : np.ndarray, dtype bool
        Binary edge map produced by a Canny or similar detector.
    n_max : int
        Maximum number of peaks to extract from the Hough accumulator.

    Returns
    -------
    R_out : np.ndarray
        y-intercepts of detected lines (in image pixel coordinates).
    v_out : np.ndarray
        Corresponding Hough accumulator values (line strength).
        Both arrays are sorted in descending order of accumulator value and
        truncated to at most *n_max* entries.
    """
    # Restrict the angle search to near-horizontal lines (87.5° – 92.5°)
    N         = 128   # angular quantisation resolution
    theta_min = np.deg2rad(87.5)
    theta_max = np.deg2rad(92.5)
    theta     = np.linspace(theta_min, theta_max, N)

    # Hough transform
    hspace, angles, distances = hough_line(edge_map, theta=theta)

    # Extract up to n_max peaks from the accumulator
    accum, angles_peaks, dists_peaks = hough_line_peaks(
        hspace, angles, distances, num_peaks=n_max
    )

    R_out = []
    v_out = []

    for acc, angle, dist in zip(accum, angles_peaks, dists_peaks):
        sin_theta = np.sin(angle)

        # Skip degenerate lines where sin(θ) ≈ 0 (would cause division by zero)
        if np.abs(sin_theta) < 1e-6:
            continue

        # Convert Hough (ρ, θ) to the y-intercept in image coordinates
        y_intercept = dist / sin_theta
        R_out.append(y_intercept)
        v_out.append(acc)

    R_out = np.array(R_out)
    v_out = np.array(v_out)

    # Sort by descending accumulator value
    sorted_indices = np.argsort(-v_out)
    R_out = R_out[sorted_indices][:n_max]
    v_out = v_out[sorted_indices][:n_max]

    return R_out, v_out