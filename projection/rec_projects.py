"""
rec_projects.py
---------------
Utilities for reconstructing DBT volumes from DICOM projection stacks and for
writing the resulting slice-by-slice reconstructed volume back to individual
DICOM files.

Functions
---------
reconstruct_projs(projPath, pixel_size)
    Load a set of DICOM projections and run Filtered Back-Projection (FBP) to
    produce a 3-D reconstructed volume.

write_dicom_recon(vol, path, pixel_size)
    Persist a float reconstruction volume as a series of 16-bit grayscale DICOM
    files, one file per reconstructed slice.
"""

import numpy as np
import os
import pathlib

import pydicom
from pydicom.dataset import FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian

# PyDBT functions ---------------------------------------------------------
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

def reconstruct_projs(projPath, pixel_size):
    """
    Reconstruct a DBT volume from a folder of DICOM projection files.

    Each DICOM file is expected to be named ``image<N>.dcm`` where ``<N>`` is
    the zero-based projection index.  The ``BodyPartThickness`` DICOM tag from
    projection 0 is used to set the reconstructed volume depth.

    Parameters
    ----------
    projPath : str or pathlib.Path
        Directory that contains the per-angle DICOM projection files.
    pixel_size : list or tuple of float
        Voxel size ``[dx, dy, dz]`` in mm for the reconstructed volume.

    Returns
    -------
    recon : np.ndarray, shape (ny, nx, nz), dtype float64
        Reconstructed 3-D DBT volume.
    """
    # Collect all DICOM files in the projection directory
    files = [str(item) for item in pathlib.Path(projPath).glob("*.dcm")]

    # Pre-allocate a list with one slot per projection angle
    stack = len(files) * [None]
    bdyThick = None

    for file in files:
        # Parse the projection index from the filename (e.g. "image3.dcm" → 3)
        ind = int(str(file).split('/')[-1].split('.')[0].replace('image', ''))
        ds = pydicom.dcmread(file)
        img = ds.pixel_array

        # Read breast thickness from the first projection (index 0)
        if ind == 0:
            bdyThick = np.float32(ds.BodyPartThickness)

        stack[ind] = img

    # Stack projections into a (rows, cols, n_angles) array
    stack = np.stack(stack, dtype=np.float64, axis=-1)

    # ------------------------------------------------------------------ geometry
    # Build the Hologic-style DBT geometry.  Volume dimensions are derived
    # from the detector size and the physical pixel/voxel sizes.
    geo = geometry_settings(
        voxels=[
            np.ceil(stack.shape[1] * 0.14 / pixel_size[1]).astype(int),  # nx
            np.ceil(stack.shape[0] * 0.14 / pixel_size[0]).astype(int),  # ny
            np.ceil(bdyThick / pixel_size[2]).astype(int),               # nz
        ],
        detector_el=[stack.shape[1], stack.shape[0]],
        voxels_size=pixel_size,
        detector_size=[0.14, 0.14],   # detector element pitch (mm)
        source_dist=700,              # source-to-detector distance (mm)
        gap=25,                       # air gap between breast and detector (mm)
        n_proj=15,                    # number of projection angles
        tube_angle=15,                # total angular range (degrees)
        detector_angle=0,             # detector tilt (degrees)
        offset=[0, 0],
    )

    # Run Filtered Back-Projection (Ram-Lak filter, apodisation 0.75)
    recon = FBP(stack, geo, 'BP', 0.75, libFiles)
    return recon


def write_dicom_recon(vol, path, pixel_size):
    """
    Write a 3-D reconstruction volume to individual per-slice DICOM files.

    The volume is transposed and rotated to match the standard display
    orientation, then linearly scaled to the full 16-bit unsigned integer range
    before writing.

    Parameters
    ----------
    vol : np.ndarray
        3-D reconstruction volume as returned by :func:`reconstruct_projs`.
    path : str
        Output directory.  Created automatically if it does not exist.
    pixel_size : list or tuple of float
        Pixel spacing ``[row_spacing, col_spacing]`` in mm, written into the
        DICOM ``PixelSpacing`` tag.

    Notes
    -----
    Each output file is named ``image_NNNN.dcm`` (zero-padded four digits).
    The DICOM SOP class is set to *Multi-frame Single-bit Secondary Capture*
    which is a reasonable generic placeholder for secondary-capture images.
    """
    os.makedirs(path, exist_ok=True)
    base_filename = "image"

    # Reorient the volume: (x, y, z) → (z, y, x) then rotate in-plane
    vol = np.transpose(vol, (2, 1, 0))
    vol = np.rot90(vol, axes=(1, 2), k=1)

    # ---------------------------------------------------------------- scaling
    # Linearly rescale the floating-point volume to the full 16-bit range
    bits_alloc = 16
    bits = 16
    array_min, array_max = vol.min(), vol.max()
    scaled_array = (
        (vol - array_min) / (array_max - array_min) * (2 ** bits - 1)
    ).astype(np.uint16)

    frames, rows, cols = scaled_array.shape

    for i in range(frames):
        # --------------------------------------------------------- file meta
        ds = FileDataset(
            f"{path}/{base_filename}.{i:04d}.dcm", {},
            file_meta=pydicom.Dataset(),
            preamble=b"\0" * 128,
        )

        # Required DICOM file-meta attributes
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = (
            pydicom.uid.MultiFrameSingleBitSecondaryCaptureImageStorage
        )
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.file_meta.ImplementationClassUID = generate_uid()

        # ---------------------------------------------------- SOP / UIDs
        ds.SOPClassUID    = ds.file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID  = generate_uid()
        ds.SeriesInstanceUID = generate_uid()
        ds.FrameOfReferenceUID = generate_uid()

        # ------------------------------------------ essential DICOM fields
        ds.Modality     = "MG"
        ds.PatientName  = "Annonymous"
        ds.PatientID    = "123456"
        ds.StudyID      = "1"
        ds.SeriesNumber = "1"
        ds.InstanceNumber = str(i + 1)

        # ------------------------------------------- pixel-data attributes
        ds.Rows    = rows
        ds.Columns = cols
        ds.SamplesPerPixel = 1

        ds.BitsAllocated = bits_alloc
        ds.BitsStored    = bits
        ds.HighBit       = bits - 1
        ds.PixelRepresentation      = 0          # unsigned integer
        ds.PhotometricInterpretation = "MONOCHROME1"
        ds.PixelSpacing = [pixel_size[0], pixel_size[1]]

        # Write the pixel data for this slice
        ds.PixelData = scaled_array[i].tobytes()

        # Save the individual DICOM file
        dicom_filename = os.path.join(path, f"{base_filename}_{i:04d}.dcm")
        ds.save_as(dicom_filename)