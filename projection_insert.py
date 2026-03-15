"""
projection_insert.py
====================
Main entry point for performing virtual lesion insertion into Digital Breast
Tomosynthesis (DBT) projections.

This script iterates over a dataset of patient DBT exams and, for each exam:
  1. Segments the breast volume to obtain the 3-D breast mask.
  2. Loads and resizes a synthetic lesion model from a zip archive.
  3. Identifies candidate insertion positions within the breast mask using a
     stride-based search.
  4. Computes the forward-projected lesion mask across all DBT projection angles.
  5. Applies the system Modulation Transfer Function (MTF) to the projected mask.
  6. Adds the blurred lesion signal to the original DBT projections.
  7. Writes the modified projections as DICOM files.

Dependencies
------------
- pyDBT  : DBT geometry and reconstruction utilities (installed separately;
           see https://github.com/LAVI-USP/pyDBT).
- Helper modules in `projection/`, `segmentation/`, `parameters/`, and
  `functions/` are bundled with this repository.

Usage
-----
    python projection_insert.py

Configuration is done directly in the "Configuration" section below.
"""

import numpy as np
import pathlib
import pandas as pd
import logging as log

from tifffile import imwrite

from segmentation.seg_breast import segbreast
from models.lesion import load_lesion, modify_lesion
from projection.get_ROIs import get_candidate_pos
from projection.proj_mask import get_projection_lesion_mask
from projection.set_mtf import apply_mtf_mask_projs
from projection.add_masks import add_masks_to_projs, write_dicom_projs


# ===========================================================================
# Configuration
# ===========================================================================

# --- Paths ------------------------------------------------------------------
PATH_PATIENT_CASES = "./patients"          # Root directory of patient exams
PATH_MTF           = "./parameters/mtf_function_hologic3d_fourier.npy"  # System MTF file

# Lesion model selection (must match a .zip archive under ./models/mass/)
LESION_MODEL = "Perlin_S_001"

PATH_LESIONS  = f"./models/mass/{LESION_MODEL}.zip"
PATH_SAVE_DIR = f"./results/{LESION_MODEL}"

# --- Inclusion / Exclusion lists -------------------------------------------
# Accessions listed in `exclude.csv` are skipped (e.g. failed segmentations).
# Only exams whose accession number appears in `inc` are processed.
PATH_EXCLUDE_CSV = "exclude.csv"
INCLUDE_ACCESSIONS = ["3012874"]  # Replace with full list or load from CSV

# --- Imaging / Insertion Parameters ----------------------------------------
# Voxel resolution of the lesion model [x, y, z] in mm/voxel
RESOLUTION  = [0.1, 0.1, 0.1]

# Target bounding-box size of the inserted lesion [x, y, z] in mm
TARGET_SIZE = [15, 15, 5]

# ROI crop size used for inspection/export [x, y, z] in mm
ROI_SIZE = TARGET_SIZE

# Stride between candidate insertion centres [col, row] in pixels
STRIDE = [200, 200]

# Contrast levels to iterate over (signal amplitude scaling factor)
CONTRAST_LEVELS = [0.1]


# ===========================================================================
# Logging setup
# ===========================================================================

log.basicConfig(
    filename="hybrid-vit.log",
    level=log.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = log.getLogger()


# ===========================================================================
# Main processing loop
# ===========================================================================

if __name__ == "__main__":

    # Load the list of excluded exam identifiers
    df_exc = pd.read_csv(PATH_EXCLUDE_CSV)
    excluded_exams = df_exc["exclude"].astype(str).tolist()

    # Enumerate all patient case directories
    patient_cases = sorted(str(p) for p in pathlib.Path(PATH_PATIENT_CASES).iterdir() if p.is_dir())

    lesion_index = 0

    for contrast in CONTRAST_LEVELS:

        logger.info(f"--- Starting run with contrast = {contrast} ---")

        for patient_case in patient_cases:

            # Each patient may have multiple exam directories
            exams = sorted(
                str(e) for e in pathlib.Path(patient_case).iterdir() if e.is_dir()
            )

            for exam in exams:
                # Build a unique identifier: "accession/exam_folder"
                exam_parts  = pathlib.Path(exam).parts
                current_id  = "/".join(exam_parts[-2:])
                accession   = exam_parts[-2]
                exam_folder = exam_parts[-1]

                # --- Filtering -----------------------------------------------
                # Skip views that are not MLO/CC (e.g. ML, CV), excluded exams,
                # or accessions not in the inclusion list.
                if (
                    exam_folder in excluded_exams
                    or accession not in INCLUDE_ACCESSIONS
                    or "_ML_" in exam_folder
                    or "_CV_" in exam_folder
                ):
                    logger.warning(f"Skipping exam: {current_id}")
                    continue

                base_name = exam_folder
                logger.info(f"Processing exam: {current_id}")

                # Collect all DICOM projection files for this exam
                dicom_files = sorted(
                    str(f) for f in pathlib.Path(exam).glob("*.dcm")
                )

                # --- Step 1: Breast segmentation -----------------------------
                try:
                    breast_mask, breast_thickness = segbreast(dicom_files, logger)
                    logger.info(f"Breast mask shape: {breast_mask.shape}")
                except Exception as exc:
                    logger.warning(f"Segmentation failed for {current_id}: {exc}")
                    continue

                # --- Step 2: Load and prepare the lesion model ---------------
                lesion = load_lesion(PATH_LESIONS, TARGET_SIZE, RESOLUTION)
                lesion = modify_lesion(lesion) # Perlin noise - comment this for cluster simulation

                logger.info(
                    f"Lesion | resolution: {RESOLUTION}  "
                    f"target size: {TARGET_SIZE}  "
                    f"actual shape: {lesion.shape}"
                )

                # Save the raw lesion volume for visual verification
                #imwrite(f"{base_name}_lesion.tif", np.transpose(lesion, (2, 1, 0)))

                # --- Step 3: Find candidate insertion positions --------------
                (x_pos, y_pos, z_pos), _, geo = get_candidate_pos(
                    breast_mask, breast_thickness, lesion, RESOLUTION, STRIDE
                )

                # --- Step 4: Build the 3-D projection mask for each position -
                # Accumulate contributions from all candidate positions into a
                # single projection-domain mask of shape (nv, nu, nProj).
                projs_masks = np.zeros((geo.nv, geo.nu, geo.nProj))
                for i, (xp, yp, zp) in enumerate(zip(x_pos, y_pos, z_pos)):
                    projs_masks += get_projection_lesion_mask(
                        lesion, geo, [xp, yp, zp], RESOLUTION, contrast
                    )
                    logger.info(f"Projected ROI {i + 1}/{len(x_pos)}.")

                #imwrite(f"{base_name}_projs_masks.tif",np.transpose(projs_masks, (2, 1, 0)))

                # --- Step 5: Apply system MTF to the lesion mask -------------
                logger.info("Applying MTF to lesion projection mask...")
                projs_mtf = apply_mtf_mask_projs(projs_masks, geo, PATH_MTF)
                #imwrite(f"{base_name}_projs_mtf.tif", np.transpose(projs_mtf, (2, 1, 0)))

                # --- Step 6 & 7: Composite with original data and save -------
                logger.info("Adding lesion signal to original DBT projections...")
                projs = add_masks_to_projs(dicom_files, projs_mtf, contrast)

                proj_out_path = f"{PATH_SAVE_DIR}/{current_id}/{contrast}"
                logger.info(f"Writing DICOM projections to: {proj_out_path}")
                write_dicom_projs(dicom_files, projs, proj_out_path)

                logger.info(
                    f"Done | exam: {current_id}  "
                    f"model: {LESION_MODEL}  "
                    f"contrast: {contrast}"
                )

        lesion_index += 1