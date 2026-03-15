"""
add_masks.py
------------
Composite forward-projected lesion masks onto original DBT projection images and
write the result back as DICOM files.

Functions
---------
add_masks_to_projs(dcmFiles, projs_masks, contrast)
    Multiply each DICOM projection by a normalised, contrast-scaled lesion mask
    and return the resulting image stack.

write_dicom_projs(dcmFiles, projs_masks, path)
    Overwrite the pixel data of a copy of each DICOM file with the provided
    array and save to *path*.

decompress(file)
    Decode a JPEG 2000–compressed DICOM file and return an uncompressed dataset.
"""

import numpy as np
import os

import pydicom
from pydicom.encaps import _defragment_data
from imagecodecs import jpeg2k_decode
from pydicom.uid import ExplicitVRLittleEndian, JPEG2000Lossless, JPEG2000

# Enable multiple pixel-data handlers in priority order so that GDCM is tried
# first, with fallbacks for Pillow, JPEG-LS, and RLE-encoded data.
pydicom.config.image_handlers = [
    "pydicom.pixel_data_handlers.gdcm_handler",
    "pydicom.pixel_data_handlers.pillow_handler",
    "pydicom.pixel_data_handlers.jpeg_ls_handler",
    "pydicom.pixel_data_handlers.rle_handler",
]


def add_masks_to_projs(dcmFiles, projs_masks, contrast):
    """
    Composite a forward-projected lesion mask stack onto DICOM projections.

    The mask for each projection is normalised to [0, 1], then scaled by
    ``contrast`` and inverted so that a mask value of 1 yields a pixel
    attenuation factor of ``(1 - contrast)`` — simulating absorption.

    Parameters
    ----------
    dcmFiles : list of str or pathlib.Path
        Ordered list of DICOM projection files (``image<N>.dcm`` naming).
    projs_masks : np.ndarray, shape (nv, nu, nProj)
        Forward-projected lesion mask stack produced by
        :func:`projection.proj_mask.get_projection_lesion_mask`.
    contrast : float
        Contrast scaling factor in [0, 1].  Higher values yield a stronger
        (darker) lesion signal.

    Returns
    -------
    stack : np.ndarray, shape (nv, nu, nProj), dtype float32
        Projection stack with the lesion signal composited in.
    """
    # Pre-allocate result list with one slot per projection angle
    stack = len(dcmFiles) * [None]

    for file in dcmFiles:
        # Parse the projection index from the filename (e.g. "image3.dcm" → 3)
        ind = int(str(file).split('/')[-1].split('.')[0].replace('image', ''))
        ds = pydicom.dcmread(file)

        # Work on a float copy to avoid integer overflow during scaling
        img = ds.pixel_array.astype('float32').copy()

        # Retrieve and normalise the mask for this projection angle
        tmp_mask = np.abs(projs_masks[:, :, ind])
        tmp_mask = (tmp_mask - tmp_mask.min()) / (tmp_mask.max() - tmp_mask.min())

        # Scale by the requested contrast level (only non-zero pixels are affected)
        tmp_mask[tmp_mask > 0] *= contrast

        # Invert: pixels inside the lesion are attenuated (multiplication < 1)
        tmp_mask = 1 - tmp_mask

        # Composite the mask onto the original projection
        stack[ind] = img[:, :] * tmp_mask

    return np.stack(stack, axis=-1)


def write_dicom_projs(dcmFiles, projs_masks, path):
    """
    Write modified DICOM projections by replacing pixel data in the originals.

    For each input file the pixel data is replaced with the corresponding
    slice from *projs_masks* (cast to uint16) and the updated dataset is
    saved to *path*.  JPEG 2000–compressed files are decoded first.

    Parameters
    ----------
    dcmFiles : list of str or pathlib.Path
        Original DICOM projection files used as metadata templates.
    projs_masks : np.ndarray, shape (nv, nu, nProj), dtype convertible to uint16
        Modified pixel data to write.  Slice index equals the projection index.
    path : str
        Output directory.  Created automatically if it does not exist.
    """
    os.makedirs(path, exist_ok=True)

    for file in dcmFiles:
        # Parse projection index from the filename
        ind = int(str(file).split('/')[-1].split('.')[0].replace('image', ''))
        ds = pydicom.dcmread(file)

        # Decompress JPEG 2000 data if necessary before modifying pixel data
        if (
            ds.file_meta.TransferSyntaxUID == JPEG2000Lossless
            or ds.file_meta.TransferSyntaxUID == JPEG2000
        ):
            ds = decompress(file)

        # Replace pixel data and update the instance number
        ds.PixelData = projs_masks[:, :, ind].astype(np.uint16).tobytes()
        ds[0x0020, 0x0013].value = str(ind)

        ds.save_as(f"{path}/image{ind:02}.dcm")


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
        DICOM dataset with uncompressed pixel data and its transfer syntax
        updated to ``ExplicitVRLittleEndian``.
    """
    ds = pydicom.dcmread(file)

    # Extract the raw JPEG 2000 bitstream from the DICOM encapsulation
    fragmented_data = _defragment_data(ds.PixelData)

    # Decode the JPEG 2000 compressed frame
    fragmented_data = jpeg2k_decode(fragmented_data)

    # Store the raw bytes and mark the dataset as uncompressed
    ds.PixelData = fragmented_data.tobytes()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    return ds