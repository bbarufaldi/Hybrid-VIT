"""
lesion.py
---------
Utilities for loading and augmenting 3-D synthetic lesion models used in the
Hybrid-VIT virtual insertion pipeline.

Lesion models are stored as raw binary files (uint8) inside a ZIP archive.  The
filename encodes the volume dimensions as ``<name>_<nz>x<ny>x<nx>.raw``.

Two augmentation strategies are available:

* **Perlin-noise blending** (:func:`modify_lesion`) — blends a Euclidean
  distance transform of the lesion mask with a 3-D Perlin noise field to
  produce a smooth, randomised attenuation profile.  This is the default path
  in the main pipeline.

* **Cluster simulation** — use the raw normalised volume returned by
  :func:`load_lesion` directly (skip :func:`modify_lesion`).

Functions
---------
load_lesion(zip_file, target_size, vxl)
    Load and spatially resize a lesion volume from a ZIP archive.

perlin_noise_3d(shape, scale, seed)
    Generate a 3-D Perlin noise volume of the given shape.

modify_lesion(lesion3d)
    Blend the distance transform of a lesion mask with Perlin noise to produce
    a physically plausible, continuously varying attenuation field.
"""

import numpy as np
import zipfile

import noise
from scipy import ndimage


def load_lesion(zip_file, target_size, vxl):
    """
    Load a lesion volume from a ZIP archive and resize it to the target size.

    The ZIP is expected to contain exactly one ``.raw`` file whose name encodes
    the original volume dimensions as ``<stem>_<nz>x<ny>x<nx>.raw`` (uint8,
    C-order).  The volume is linearly rescaled to match *target_size* expressed
    in physical units (mm), then normalised to the range ``[0, 1]``.

    Parameters
    ----------
    zip_file : str
        Path to the ``.zip`` archive containing the lesion raw file.
    target_size : list or tuple of float
        Desired physical dimensions ``[x_mm, y_mm, z_mm]`` of the resized
        lesion volume.
    vxl : list or tuple of float
        Voxel size ``[dx, dy, dz]`` in mm/voxel, used to convert *target_size*
        from physical units to a voxel count.

    Returns
    -------
    resized_lesion : np.ndarray, shape (nx_target, ny_target, nz_target), dtype float32
        Normalised lesion volume in the range ``[0, 1]``.
    """
    # Convert physical target size (mm) to an integer voxel count per axis
    target = np.array(target_size) / np.array(vxl)
    target = np.round(target).astype(int)

    # --------------------------------------------------------- load from ZIP
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Locate the .raw file inside the archive
        file = [f for f in zip_ref.namelist() if f.endswith('.raw')][0]

        # Parse the original volume dimensions from the filename
        # Expected format: <anything>_<nz>x<ny>x<nx>.raw
        size = str(file).replace('.raw', '').split('_')[-1].split('x')

        with zip_ref.open(file) as f:
            raw = f.read()
            # Reshape the flat byte buffer into a (nz, ny, nx) uint8 array
            volume = np.frombuffer(bytearray(raw), dtype=np.uint8).reshape(
                int(size[2]), int(size[1]), int(size[0])
            )

    # --------------------------------------------------------- resize
    # Compute per-axis zoom factors to reach the desired voxel count
    scaling_factors = target / np.array(volume.shape)

    # Trilinear interpolation (order=1) preserves smooth lesion boundaries
    resized_lesion = ndimage.zoom(volume.astype(float), scaling_factors, order=1)

    # Normalise intensity to [0, 1]
    resized_lesion = resized_lesion / np.max(resized_lesion)

    return resized_lesion.astype(np.float32)


def perlin_noise_3d(shape, scale=10, seed=None):
    """
    Generate a 3-D volume of Perlin coherent noise.

    Perlin noise produces smooth, spatially correlated random values — suitable
    for simulating heterogeneous tissue attenuation.  The output is normalised
    to ``[0, 1]``.

    Parameters
    ----------
    shape : tuple of int
        Output volume shape ``(nx, ny, nz)``.
    scale : float, optional
        Spatial frequency scale.  Larger values produce lower-frequency
        (smoother) noise patterns.  Default is 10.
    seed : int or None, optional
        Random seed passed to both NumPy and the Perlin noise generator for
        reproducibility.  Default is ``None`` (random).

    Returns
    -------
    noise_volume : np.ndarray, shape == *shape*, dtype float32
        Perlin noise field normalised to ``[0, 1]``.
    """
    if seed is not None:
        np.random.seed(seed)

    noise_volume = np.zeros(shape, dtype=np.float32)

    # Evaluate the Perlin noise function at every voxel
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                noise_volume[x, y, z] = noise.pnoise3(
                    x / scale,
                    y / scale,
                    z / scale,
                    octaves=3,        # number of frequency octaves
                    persistence=0.5,  # amplitude decay per octave
                    lacunarity=2.0,   # frequency growth per octave
                    repeatx=shape[0],
                    repeaty=shape[1],
                    repeatz=shape[2],
                    base=seed,
                )

    # Normalise to [0, 1]
    noise_volume = (
        (noise_volume - noise_volume.min())
        / (noise_volume.max() - noise_volume.min())
    )
    return noise_volume


def modify_lesion(lesion3d):
    """
    Augment a binary lesion mask with a blended distance-transform / Perlin
    noise attenuation profile.

    The function creates a physically plausible continuous attenuation field by
    combining two components:

    * **Euclidean distance transform** — voxels far from the lesion boundary
      receive higher values, ensuring the lesion core is brighter (more
      attenuating) than its edges.
    * **Perlin noise** — spatially smooth random variation is multiplied by the
      original lesion mask and blended in at equal weight, introducing
      heterogeneity within the lesion body.

    The formula is::

        contrast_volume = 0.5 * dist_norm + 0.5 * perlin * lesion3d

    where ``dist_norm`` is the distance transform normalised to ``[0, 1]`` and
    ``perlin`` is the Perlin noise field.  The result is then re-normalised to
    ``[0, 1]``.

    Parameters
    ----------
    lesion3d : np.ndarray, shape (nx, ny, nz), dtype float32
        Binary or soft lesion mask as returned by :func:`load_lesion`.

    Returns
    -------
    contrast_volume : np.ndarray, shape (ny, nx, nz), dtype float32
        Augmented attenuation volume, transposed so the y-axis is first (as
        expected by the pyDBT projection operator).  Values are in ``[0, 1]``.

    Notes
    -----
    The output is transposed ``(1, 0, 2)`` relative to the input to match the
    pyDBT axis convention ``(ny, nx, nz)`` expected by the forward projector.

    To skip augmentation and use the raw lesion volume instead (e.g. for
    cluster simulations), comment out the ``modify_lesion`` call in
    ``projection_insert.py`` and pass the output of :func:`load_lesion` directly.
    """
    # Compute the Euclidean distance from each foreground voxel to the
    # nearest background voxel (boundary proximity measure)
    distance_map = ndimage.distance_transform_edt(lesion3d)

    # Normalise distance transform to [0, 1]
    distance_map_normalized = distance_map / distance_map.max()

    # Generate reproducible Perlin noise matching the lesion volume shape
    perlin = perlin_noise_3d(lesion3d.shape, scale=20, seed=42)

    # Blend: 50 % distance-based core + 50 % noise-modulated lesion body
    contrast_volume = (
        0.5 * distance_map_normalized + 0.5 * perlin * lesion3d
    ).astype(np.float32)

    # Re-normalise the blended volume to [0, 1]
    contrast_volume = (
        (contrast_volume - contrast_volume.min())
        / (contrast_volume.max() - contrast_volume.min())
    )

    # Transpose to (ny, nx, nz) — required axis order for the pyDBT projector
    return np.transpose(contrast_volume, (1, 0, 2))