"""
proj_mask.py
------------
Forward-project 3-D lesion volumes into 2-D projection-domain masks using the
pyDBT distance-driven projection operator.

Two strategies are provided:

* :func:`get_projection_lesion_mask` — project a *single* lesion at a specified
  3-D position.  This is the standard path used by the main pipeline.

* :func:`get_projection_lesion_grid` — place the *same* lesion at multiple
  positions simultaneously, then project the combined volume in one pass.
  (Experimental — not yet integrated into the main pipeline.)

The geometry object ``geo`` is temporarily modified to match the lesion volume
dimensions and position, then fully restored before returning.

Functions
---------
get_projection_lesion_mask(roi_3D, geo, position, pixel_size, contrast)
    Forward-project a 3-D lesion mask at a single grid position.

get_projection_lesion_grid(roi_3D, geo, positions, pixel_size, contrast)
    Forward-project a 3-D lesion mask placed at multiple positions (experimental).
"""

import numpy as np

# PyDBT forward/backward projection operators
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

def get_projection_lesion_mask(roi_3D, geo, position, pixel_size, contrast):
    """
    Forward-project a 3-D lesion volume to obtain 2-D projection-domain masks.

    The geometry is temporarily reconfigured to match the lesion volume's
    dimensions and its 3-D position within the breast.  After projection the
    geometry is fully restored to its original state so the object can be
    reused by the caller.

    Parameters
    ----------
    roi_3D : np.ndarray, shape (nx, ny, nz)
        3-D lesion mask in voxel space.  Values are assumed to be in [0, 1].
    geo : geometry_settings
        DBT acquisition geometry (Hologic-style).  Modified locally during
        projection and restored on return.
    position : tuple of int
        Insertion location ``(x, y, z)`` expressed in *volume* voxel indices
        referring to the breast reconstruction grid.
    pixel_size : list or tuple of float
        Lesion voxel size ``[dx, dy, dz]`` in mm.
    contrast : float
        Multiplicative contrast scale applied after normalisation.

    Returns
    -------
    projs_masks : np.ndarray, shape (nv, nu, nProj)
        Normalised forward-projected lesion mask stack.  Values are in [0, 1].
    """
    # --------------------------------------------------------- geometry backup
    # Save the full-volume geometry so we can restore it after projecting the
    # (smaller) lesion sub-volume.
    geo_nx, geo_ny, geo_nz = geo.nx, geo.ny, geo.nz
    geo_dx, geo_dy, geo_dz = geo.dx, geo.dy, geo.dz
    geo_DSD, geo_DSO, geo_DAG = geo.DSD, geo.DSO, geo.DAG

    # ------------------------------------------------- sub-volume positioning
    # Compute the z-axis air-gap offset so the lesion is placed at the correct
    # depth inside the breast (position[2] is in full-volume voxel coordinates).
    vol_z_offset = (position[2] * geo.dz) - (roi_3D.shape[2] // 2 * pixel_size[2])

    # Set lateral offsets so the projected lesion lands at (position[0], position[1])
    geo.x_offset = ((geo.nx - 1) - position[0]) * geo.dx
    geo.y_offset = (position[1] - (geo.ny / 2)) * geo.dy

    # -------------------------------------------- reconfigure geometry for ROI
    geo.nx = roi_3D.shape[0]
    geo.ny = roi_3D.shape[1]
    geo.nz = roi_3D.shape[2]

    geo.dx = pixel_size[0]
    geo.dy = pixel_size[1]
    geo.dz = pixel_size[2]

    # Shift the detector air-gap to account for the lesion's z position
    geo.DAG += vol_z_offset

    # Allocate the output projection stack (detector size × number of angles)
    projs_masks = np.zeros((geo.nv, geo.nu, geo.nProj))

    # ----------------------------------------------------- forward projection
    # Project the 3-D lesion volume through all acquisition angles
    projs_masks_tmp = projectionDD(np.float64(roi_3D), geo, -1, libFiles)

    # Normalise and scale by contrast (skip if the projection is all zeros)
    if projs_masks_tmp.max() != 0:
        projs_masks_tmp = (projs_masks_tmp / projs_masks_tmp.max()) * contrast
        projs_masks += projs_masks_tmp

    # Final normalisation to [0, 1]
    projs_masks = projs_masks / projs_masks.max()

    # --------------------------------------------------- restore full geometry
    geo.nx, geo.ny, geo.nz = geo_nx, geo_ny, geo_nz
    geo.dx, geo.dy, geo.dz = geo_dx, geo_dy, geo_dz
    geo.DSD, geo.DSO, geo.DAG = geo_DSD, geo_DSO, geo_DAG

    return projs_masks


# TODO: Debug / validate this function before integrating into the main pipeline
def get_projection_lesion_grid(roi_3D, geo, positions, pixel_size, contrast):
    """
    Forward-project a 3-D lesion placed at multiple positions in a single pass.

    The lesion ``roi_3D`` is stamped into a full-volume zero buffer at each
    position in *positions*, and the combined volume is then projected once.
    This is more efficient than calling :func:`get_projection_lesion_mask`
    repeatedly when many insertion sites are needed simultaneously.

    .. warning::
        This function is experimental and has not been validated against the
        single-lesion path.  Use with caution.

    Parameters
    ----------
    roi_3D : np.ndarray, shape (sx, sy, sz)
        3-D lesion mask in voxel space.
    geo : geometry_settings
        DBT acquisition geometry.  Modified locally and restored on return.
    positions : list of tuple of int
        List of ``(ix, iy, iz)`` insertion centres in full-volume voxel
        coordinates.  Positions that would place the lesion outside the volume
        boundaries are silently skipped.
    pixel_size : list or tuple of float
        Lesion voxel size ``[dx, dy, dz]`` in mm.
    contrast : float
        Multiplicative contrast scale applied to the normalised projections.

    Returns
    -------
    projs : np.ndarray, shape (nv, nu, nProj)
        Normalised forward-projected mask stack for the combined lesion grid.
    """
    # --------------------------------------------------------- geometry backup
    geo_nx, geo_ny, geo_nz = geo.nx, geo.ny, geo.nz
    geo_dx, geo_dy, geo_dz = geo.dx, geo.dy, geo.dz
    geo_DSD, geo_DSO, geo_DAG = geo.DSD, geo.DSO, geo.DAG

    # Full-volume dimensions before overriding voxel size
    nx_full, ny_full, nz_full = geo.nx, geo.ny, geo.nz

    # Combined "lesion grid" volume — same shape as the breast reconstruction
    vol = np.zeros((ny_full, nx_full, nz_full), dtype=np.float32)

    # Override voxel size to match the lesion resolution
    geo.dx = pixel_size[0]
    geo.dy = pixel_size[1]
    geo.dz = pixel_size[2]

    sx, sy, sz = roi_3D.shape
    half = np.array([sx, sy, sz]) // 2

    # ------------------------------------------- stamp the lesion at each site
    for (ix, iy, iz) in positions:
        # Bounding box of the lesion centred at (ix, iy, iz)
        x0, x1 = int(ix - half[0]), int(ix - half[0]) + sx
        y0, y1 = int(iy - half[1]), int(iy - half[1]) + sy
        z0, z1 = int(iz - half[2]), int(iz - half[2]) + sz

        # Skip positions that would exceed the volume boundaries
        if x0 < 0 or y0 < 0 or z0 < 0 or x1 > nx_full or y1 > ny_full or z1 > nz_full:
            continue

        vol[y0:y1, x0:x1, z0:z1] += roi_3D

    # -------------------------------------------------------- single projection
    projs = projectionDD(vol.astype(np.float64), geo, -1, libFiles)

    # Normalise and scale by contrast
    if projs.max() > 0:
        projs = projs / projs.max() * contrast
        projs = projs / projs.max()

    # --------------------------------------------------- restore full geometry
    geo.nx, geo.ny, geo.nz = geo_nx, geo_ny, geo_nz
    geo.dx, geo.dy, geo.dz = geo_dx, geo_dy, geo_dz
    geo.DSD, geo.DSO, geo.DAG = geo_DSD, geo_DSO, geo_DAG

    return projs