"""
set_mtf.py
----------
Apply a measured system Modulation Transfer Function (MTF) to forward-projected
lesion masks in the frequency domain.

The MTF is stored as a 1-D radially-symmetric callable (e.g. a scipy interpolant
loaded from a ``.npy`` file).  The function builds a 2-D radial-frequency grid,
evaluates the MTF on it, and multiplies each projection's Fourier spectrum
(modulus only) by the resulting filter — preserving the original phase — before
transforming back to the spatial domain.

Functions
---------
apply_mtf_mask_projs(projs_masks, geo, pathMTF)
    Convolve each 2-D projection mask with the system MTF in the frequency
    domain and return the blurred projection stack.
"""

import numpy as np


def apply_mtf_mask_projs(projs_masks, geo, pathMTF):
    """
    Apply the system MTF to every projection in a forward-projected mask stack.

    The MTF is applied as a frequency-domain multiplication: for each projection
    the 2-D FFT is computed on a zero-padded image to reduce circular aliasing,
    the magnitude spectrum is multiplied by the 2-D MTF filter, the original
    phase is preserved, and the result is inverse-transformed and cropped back to
    the original size.

    Parameters
    ----------
    projs_masks : np.ndarray, shape (nv, nu, nProj)
        Stack of forward-projected lesion masks, one slice per projection angle.
    geo : geometry_settings
        DBT geometry object.  ``geo.du`` (detector element pitch in mm) is used
        to compute the Nyquist frequency.
    pathMTF : str
        Path to the ``.npy`` file containing the fitted MTF interpolant.  The
        file must store a dictionary-like object with a single callable value
        (i.e. ``np.load(pathMTF, allow_pickle=True)[()]``).

    Returns
    -------
    projs_masks_mtf : np.ndarray, shape (nv, nu, nProj)
        MTF-blurred projection mask stack, same shape as the input.
    """
    # ----------------------------------------------------------------- MTF load
    # Load the pre-fitted radial MTF function (typically a scipy interpolant)
    f = np.load(pathMTF, allow_pickle=True)[()]

    # Nyquist frequency for the detector pixel pitch (cycles / mm)
    nyquist = 1 / (2 * geo.du)

    # -------------------------------------------------------- 2-D frequency grid
    # Build 1-D frequency vectors for columns and rows.  Each vector runs from
    # the Nyquist frequency down to 0 and then mirrors back up (FFT ordering).
    x = np.linspace(nyquist, 0, projs_masks.shape[1] + 1)
    x = np.hstack((x, x[1:-1][-1::-1]))

    y = np.linspace(nyquist, 0, projs_masks.shape[0] + 1)
    y = np.hstack((y, y[1:-1][-1::-1]))

    # 2-D meshgrid of spatial frequencies
    xx, yy = np.meshgrid(x, y)

    # Radial distance from DC for each frequency-domain sample
    ri = np.sqrt(xx ** 2 + yy ** 2)

    # Clamp radial frequencies that exceed Nyquist (outside the measurable range)
    idx_extra = ri > nyquist
    ri[idx_extra] = nyquist

    # Evaluate the 1-D radially-symmetric MTF on the 2-D frequency grid
    mtf_2d = f(ri)

    # Shift DC to the corners to match the standard FFT output layout
    mtf_2d = np.fft.ifftshift(mtf_2d)

    # Allocate output array
    projs_masks_mtf = np.empty_like(projs_masks)

    # Padding size equals the original image dimensions (doubles each axis)
    pad_i = projs_masks.shape[0]
    pad_j = projs_masks.shape[1]

    # ------------------------------------------------- per-projection filtering
    for z in range(geo.nProj):
        # Zero-pad the projection to reduce circular-convolution artefacts
        projs_mask_pad = np.pad(projs_masks[:, :, z], ((0, pad_i), (0, pad_j)))

        # Forward 2-D FFT
        projs_mask_pad_fft = np.fft.fft2(projs_mask_pad)

        # Decompose into magnitude and phase
        projs_mask_pad_abs   = np.abs(projs_mask_pad_fft)    # amplitude spectrum
        projs_mask_pad_angle = np.angle(projs_mask_pad_fft)  # phase spectrum

        # Apply the MTF by scaling the magnitude spectrum
        projs_mask_pad_abs = projs_mask_pad_abs * mtf_2d

        # Reconstruct the complex spectrum from the filtered magnitude and original phase
        projs_mask_pad_fft = projs_mask_pad_abs * np.exp(1j * projs_mask_pad_angle)

        # Inverse 2-D FFT → back to spatial domain
        projs_mask_pad = np.real(np.fft.ifft2(projs_mask_pad_fft))

        # Crop to the original projection size (undo zero-padding)
        projs_masks_mtf[:, :, z] = projs_mask_pad[:pad_i, :pad_j]

    return projs_masks_mtf