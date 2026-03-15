"""
seg_breast.py
-------------
Orchestrate per-projection breast segmentation from a DICOM projection stack.

For each projection the image is downscaled, log-normalised, optionally
de-padded (compression paddle marks), and then segmented using the contour
(and optionally pectoral) segmentation routines.  The resulting binary mask is
upscaled back to the full projection resolution and returned as a 3-D stack.

Functions
---------
segbreast(dcmFiles, log)
    Main entry point: iterate over DICOM projections and produce a 3-D binary
    breast mask stack.

decompress(file)
    Decode a JPEG 2000–compressed DICOM file to raw pixel data.

flips(ds, img, log)
    Standardise image orientation based on breast laterality.

getmask(ds, low_res, index, log)
    Select and execute the correct segmentation strategy (CC vs. MLO view),
    then return the largest connected component as the breast mask.

imnorm(im, ntype)
    Normalise a projection image using logarithmic or z-score normalisation.
"""

import numpy as np
import pydicom

from skimage import util, transform, measure
from scipy.ndimage import zoom

from pydicom.encaps import _defragment_data
from imagecodecs import jpeg2k_decode
from pydicom.uid import ExplicitVRLittleEndian, JPEG2000Lossless, JPEG2000

from tifffile import imwrite

# Enable multiple pixel-data handlers; GDCM is preferred, others act as fallbacks
pydicom.config.image_handlers = [
    "pydicom.pixel_data_handlers.gdcm_handler",
    "pydicom.pixel_data_handlers.pillow_handler",
    "pydicom.pixel_data_handlers.jpeg_ls_handler",
    "pydicom.pixel_data_handlers.rle_handler",
]

from segmentation.seg_paddle import segpaddle, has_paddle_marks
from segmentation.seg_contour import segcontour
from segmentation.seg_pectoral import segpectoral


def segbreast(dcmFiles, log):
    """
    Segment the breast from a series of DICOM projection files.

    For each projection the pipeline:
    1. Reads and optionally decompresses the DICOM pixel data.
    2. Downscales to 25 % of the original resolution for speed.
    3. Applies log-normalisation followed by orientation standardisation.
    4. Removes compression-paddle artefacts when present.
    5. Computes a binary breast mask (contour + optional pectoral exclusion).
    6. Upscales the mask back to the original resolution.

    Parameters
    ----------
    dcmFiles : list of str or pathlib.Path
        Ordered list of DICOM projection files (``image<N>.dcm`` naming).
    log : logging.Logger
        Logger used to emit warnings (e.g. missing DICOM tags).

    Returns
    -------
    mask_stack : np.ndarray, shape (nv, nu, nProj), dtype uint8
        3-D binary breast mask, one slice per projection angle.
        Breast pixels are 255, background pixels are 0.
    bdyThick : float
        Breast body thickness in mm, read from the ``BodyPartThickness`` tag of
        projection 0.
    """
    # Pre-allocate result list — one slot per projection angle
    stack = len(dcmFiles) * [None]
    bdyThick = None

    for file in dcmFiles:
        # Parse the projection index from the filename (e.g. "image3.dcm" → 3)
        ind = int(str(file).split('/')[-1].split('.')[0].replace('image', ''))
        ds = pydicom.dcmread(file)

        # Decompress JPEG 2000 encoded data if needed
        if (
            ds.file_meta.TransferSyntaxUID == JPEG2000Lossless
            or ds.file_meta.TransferSyntaxUID == JPEG2000
        ):
            ds = decompress(file)

        img = ds.pixel_array
        size = img.shape   # original full-resolution shape (nv, nu)

        # Read breast thickness from the first projection (index 0)
        if ind == 0:
            bdyThick = np.float32(ds.BodyPartThickness)

        # ------------------------------------------- downscale for segmentation
        scaling = [0.25, 0.25]
        thumb = transform.rescale(
            img, scaling,
            mode='reflect',
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.uint16)

        # Log-normalise to suppress X-ray beam non-uniformity
        thumb = imnorm(thumb)

        # Standardise orientation so the nipple always faces right
        thumb = flips(ds, thumb, log)

        # ------------------------------------------- paddle removal
        # If compression-paddle marks are detected, zero out those pixels
        if has_paddle_marks(thumb):
            paddle, _ = segpaddle(thumb)
            thumb[~paddle] = np.min(thumb)  # replace paddle region with background

        # Compute the binary breast mask (view-dependent logic inside getmask)
        mask = getmask(ds, thumb, ind, log)

        # ------------------------------------------- restore orientation
        # Flip back to the original DICOM orientation
        mask = flips(ds, mask, log)

        # ------------------------------------------- upscale to full resolution
        zoom_factors = (size[0] / mask.shape[0], size[1] / mask.shape[1])
        hi_res = zoom(mask, zoom_factors, order=0)  # nearest-neighbour upscale

        # Convert boolean mask to uint8 (255 = breast, 0 = background)
        stack[ind] = util.img_as_ubyte(hi_res / 255).astype(np.uint8)

    return np.stack(stack, axis=-1), bdyThick


def decompress(file):
    """
    Decode a JPEG 2000–compressed DICOM file to raw pixel data.

    Parameters
    ----------
    file : str or pathlib.Path
        Path to the compressed DICOM file.

    Returns
    -------
    ds : pydicom.Dataset
        Dataset with uncompressed pixel data and the transfer syntax updated to
        ``ExplicitVRLittleEndian``.
    """
    ds = pydicom.dcmread(file)

    # Extract the JPEG 2000 bitstream from the DICOM encapsulation
    fragmented_data = _defragment_data(ds.PixelData)

    # Decode the compressed frame
    fragmented_data = jpeg2k_decode(fragmented_data)

    # Store raw bytes and mark the dataset as uncompressed
    ds.PixelData = fragmented_data.tobytes()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    return ds


def flips(ds, img, log):
    """
    Standardise image orientation based on breast laterality.

    For segmentation the image is normalised so that the nipple faces the right
    side of the image.  Right-laterality (R) images are flipped horizontally;
    left-laterality (L) images are additionally flipped vertically.

    Parameters
    ----------
    ds : pydicom.Dataset
        DICOM dataset containing the ``Laterality`` tag.
    img : np.ndarray
        Image to reorient.
    log : logging.Logger
        Logger used to emit a warning if the ``Laterality`` tag is absent.

    Returns
    -------
    img : np.ndarray
        Reoriented image (or the original if the tag is missing).
    """
    if not hasattr(ds, 'Laterality'):
        log.warning(
            'Laterality tag does not exist. '
            'Considering R Laterality for outline segmentation.'
        )
    elif 'L' in ds.Laterality:
        # Left breast: flip both axes so the nipple points right
        img = np.flip(img, axis=0)   # vertical flip
        img = np.flip(img, axis=1)   # horizontal flip
    else:
        # Right breast: only horizontal flip needed
        img = np.flip(img, axis=1)

    return img


def getmask(ds, low_res, index, log):
    """
    Compute the binary breast mask for a single downscaled projection.

    The view position (CC or MLO) is read from the DICOM ``ViewPosition`` tag
    to select the appropriate segmentation strategy:
    - **CC**: breast contour only.
    - **MLO**: breast contour followed by pectoral muscle exclusion.

    After segmentation, only the largest connected component is retained to
    remove stray artefacts.

    Parameters
    ----------
    ds : pydicom.Dataset
        DICOM dataset for the current projection.
    low_res : np.ndarray
        Downscaled, log-normalised, orientation-standardised projection image.
    index : int
        Projection index (used only for optional debug output).
    log : logging.Logger
        Logger for missing-tag warnings.

    Returns
    -------
    breast_mask : np.ndarray, dtype bool
        Binary mask of the breast region (largest connected component).
    """
    if not hasattr(ds, 'ViewPosition'):
        log.warning(
            'View tag does not exist. '
            'Considering CC view for outline segmentation.'
        )
        mask, cpoints = segcontour(low_res, True)

    else:
        if 'CC' in ds.ViewPosition:
            # Cranio-caudal view — contour only
            mask, cpoints = segcontour(low_res, True)

        elif 'MLO' in ds.ViewPosition:
            # Medio-lateral oblique view — contour + pectoral exclusion
            mask, cpoints = segcontour(low_res, False)
            pec, ppoints  = segpectoral(low_res, cpoints)
            mask[~pec]    = np.min(mask)  # zero out the pectoral muscle region

        else:
            log.warning(
                f'View tag is not CC or MLO: {ds.ViewPosition}. '
                'Considering CC view for outline segmentation.'
            )
            mask, cpoints = segcontour(low_res, True)

    # Keep only the largest connected component to remove small artefacts
    label_mask   = measure.label(mask)
    regions      = measure.regionprops(label_mask)
    largest      = max(regions, key=lambda r: r.area)
    breast_mask  = label_mask == largest.label

    return breast_mask


def imnorm(im, ntype='log'):
    """
    Normalise a projection image.

    Parameters
    ----------
    im : np.ndarray
        Input image (typically ``uint16`` pixel values).
    ntype : {'log', 'zsc'}
        Normalisation method:

        * ``'log'`` — logarithmic transform followed by squared deviation from
          the maximum, which maps X-ray attenuation to a positive signal.
        * ``'zsc'`` — zero-mean, unit-variance (z-score) normalisation,
          ignoring NaN values.

    Returns
    -------
    imn : np.ndarray, dtype float64
        Normalised image.

    Raises
    ------
    ValueError
        If *ntype* is not ``'log'`` or ``'zsc'``.
    """
    if ntype == 'log':
        # Ensure all values are strictly positive before taking the logarithm
        if np.min(im) < 1:
            im = im + abs(np.min(im)) + 1
        im_log = np.log(im)
        # Squared deviation from the maximum enhances tissue–air contrast
        imn = np.abs(im_log - np.max(im_log)) ** 2

    elif ntype == 'zsc':
        tvalues = im[~np.isnan(im)]  # exclude NaN pixels before statistics
        t_std   = np.std(tvalues)
        t_mean  = np.mean(tvalues)
        imn     = (im - t_mean) / t_std

    else:
        raise ValueError(f'Unknown normalisation method {ntype.upper()}')

    return imn