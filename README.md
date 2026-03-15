# Hybrid-VIT — Virtual Lesion Insertion for DBT

**Hybrid-VIT** is a Python-based pipeline for inserting synthetic lesions into
Digital Breast Tomosynthesis (DBT) acquisitions. It generates realistic,
physics-informed lesion signals by forward-projecting 3-D lesion models across
all DBT projection angles, applying the system Modulation Transfer Function
(MTF), and compositing the result onto original DICOM projections.

The toolbox is intended for use in virtual clinical trials and reader-study
experiments where large datasets of ground-truth positive DBT cases are required.

---

## Requirements

| Requirement | Details |
|---|---|
| Hardware | NVIDIA GPU with CUDA support |
| Container | Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |
| Python | 3.10 (supplied via Docker image) |
| External dependency | pyDBT — see [installation instructions](https://github.com/LAVI-USP/pyDBT) |

---

## Repository Structure

```
Hybrid-VIT/
├── projection_insert.py      # Main entry point — run this script
├── Dockerfile                # Reproducible GPU-enabled environment
├── requirements.txt          # Python package dependencies
├── projection/               # DBT forward-projection helpers
│   ├── get_ROIs.py           # Candidate insertion-position search & ROI export
│   ├── proj_mask.py          # 3-D → 2-D projection of lesion mask
│   ├── set_mtf.py            # MTF blurring of projection-domain mask
│   ├── add_masks.py          # Signal compositing & DICOM export
│   └── rec_projects.py       # FBP reconstruction & slice-by-slice DICOM output
├── segmentation/             # 3-D breast segmentation from DICOM projections
│   ├── seg_breast.py         # Top-level segmentation orchestrator
│   ├── seg_contour.py        # Histogram-based breast contour detection
│   ├── seg_paddle.py         # Compression paddle marks detection & removal
│   └── seg_pectoral.py       # Pectoral muscle detection (MLO view)
├── parameters/               # Scanner geometry and acquisition settings
└── functions/                # General-purpose image-processing utilities
```

> **Note:** The `build/` folder (pyDBT compiled binaries) is not included.
> Follow the directions at <https://github.com/LAVI-USP/pyDBT> to build and
> install pyDBT separately before running the pipeline.

---

## Installation & Deployment

### 1 — Build the Docker image

```bash
docker build -t hybrid-vit .
```

### 2 — Run the container

Mount your data directory and execute the insertion pipeline:

```bash
docker run --gpus all --rm \
    -v /path/to/your/data:/app/patients \
    -v /path/to/results:/app/results \
    hybrid-vit \
    python3 projection_insert.py
```

---

## Configuration

All parameters are defined at the top of `projection_insert.py` under the
**Configuration** section. No command-line arguments are required.

| Parameter | Description | Default |
|---|---|---|
| `PATH_PATIENT_CASES` | Root directory containing per-patient sub-folders of DICOM exams | `./patients` |
| `PATH_MTF` | Path to the system MTF `.npy` file | `./parameters/mtf_function_hologic3d_fourier.npy` |
| `LESION_MODEL` | Name of the lesion model archive (without `.zip`) inside `./models/mass/` | `Perlin_S_001` |
| `PATH_SAVE_DIR` | Output directory for modified DICOM projections | `./results/<model>` |
| `PATH_EXCLUDE_CSV` | CSV with an `exclude` column listing exam IDs to skip | `exclude.csv` |
| `INCLUDE_ACCESSIONS` | List of accession numbers to process | `["3012874"]` |
| `RESOLUTION` | Lesion voxel resolution `[x, y, z]` in mm | `[0.1, 0.1, 0.1]` |
| `TARGET_SIZE` | Lesion bounding-box size `[x, y, z]` in mm | `[15, 15, 5]` |
| `ROI_SIZE` | ROI crop size for export `[x, y, z]` in mm | same as `TARGET_SIZE` |
| `STRIDE` | Spacing between candidate insertion centres `[col, row]` in pixels | `[200, 200]` |
| `CONTRAST_LEVELS` | List of contrast scaling factors to iterate over | `[0.01]` |

---

## Pipeline Overview

```
DICOM projections
       │
       ▼
 Breast segmentation  ──► 3-D breast mask + thickness
       │
       ▼
 Lesion model loading ──► resize / augment lesion volume
       │
       ▼
 Candidate position search (stride-based, within mask)
       │
       ▼
 Forward projection of lesion mask (all angles)
       │
       ▼
 MTF blurring of projection-domain mask
       │
       ▼
 Signal compositing on original projections
       │
       ▼
 DICOM output → results/<model>/<accession>/<exam>/<contrast>/
```

---

## Module Reference

### `projection/` — Forward-Projection Helpers

#### `get_ROIs.py`

Searches for valid lesion insertion positions inside the breast volume and
exports cropped regions of interest (ROIs) for downstream classifier training.

| Function | Description |
|---|---|
| `get_candidate_pos(mask_breast, bdyThick, mask, mask_resolution, stride)` | Back-projects the 2-D breast mask to 3-D, extracts the mid-plane slice, and slides a window over it to find positions where the entire window falls inside dense breast tissue. Returns `(x_pos, y_pos, z_pos)` lists, a debug slice, and the geometry object. |
| `crop_and_save_rois(mask_breast, csv_file, train_folder, test_folder, base_name, roi_size, geo)` | Reads candidate positions from a CSV, crops fixed-size patches from the projection image, shuffles them, splits 50/50 into train/test sets, and saves each as a 10-bit big-endian raw file. |
| `crop_and_save_sliding_rois(binary_mask, original_image, roi_size, output_dir, exam, csv_filename, overlap_percent, geo)` | Slides a rectangular window over a binary breast mask, extracts all-breast ROIs from the original projection, saves them as raw files, and writes a CSV recording each position and its train/test assignment. |
| `save_raw(data, filename)` | Normalises an array to the 10-bit range `[0, 1023]`, byte-swaps to big-endian, and writes it as a `.raw` file. |

#### `proj_mask.py`

Forward-projects 3-D lesion volumes to 2-D projection-domain masks using the
pyDBT distance-driven projection operator.

| Function | Description |
|---|---|
| `get_projection_lesion_mask(roi_3D, geo, position, pixel_size, contrast)` | Projects a single 3-D lesion volume at the specified `(x, y, z)` voxel position through all acquisition angles. Temporarily reconfigures the geometry to match the lesion sub-volume, then restores it. Returns a `(nv, nu, nProj)` normalised mask stack. |
| `get_projection_lesion_grid(roi_3D, geo, positions, pixel_size, contrast)` | *(Experimental)* Stamps the same lesion at multiple positions in a full-resolution zero buffer and projects the combined volume in a single pass. More efficient than repeated single-lesion calls when many simultaneous insertions are needed. |

#### `set_mtf.py`

Applies the measured system MTF to the forward-projected lesion masks in the
frequency domain, simulating the spatial resolution of the X-ray detector.

| Function | Description |
|---|---|
| `apply_mtf_mask_projs(projs_masks, geo, pathMTF)` | Loads a pre-fitted radial MTF callable from a `.npy` file. For each projection the mask is zero-padded, Fourier-transformed, its magnitude spectrum is multiplied by the 2-D MTF filter (phase is preserved), and the result is inverse-transformed and cropped back to the original size. |

#### `add_masks.py`

Composites the MTF-blurred lesion masks onto the original DICOM projections and
writes the result back to disk.

| Function | Description |
|---|---|
| `add_masks_to_projs(dcmFiles, projs_masks, contrast)` | For each DICOM file reads the pixel array, normalises and contrast-scales the corresponding mask slice, inverts it to produce an attenuation map, and multiplies it into the projection image. Returns the full modified stack. |
| `write_dicom_projs(dcmFiles, projs_masks, path)` | Replaces the pixel data in copies of the original DICOM files with the provided array (cast to `uint16`) and saves them to the output directory. Decompresses JPEG 2000 files automatically when required. |
| `decompress(file)` | Decodes a JPEG 2000–compressed DICOM file using `imagecodecs` and returns an uncompressed pydicom dataset. |

#### `rec_projects.py`

Reconstructs a 3-D DBT volume from a DICOM projection stack using Filtered
Back-Projection (FBP), and provides a complementary function to write the
result back as DICOM slices.

| Function | Description |
|---|---|
| `reconstruct_projs(projPath, pixel_size)` | Loads all `image<N>.dcm` files from `projPath`, stacks them into a `(nv, nu, nProj)` array, builds a Hologic-style DBT geometry, and runs FBP. Returns the reconstructed `(ny, nx, nz)` float64 volume. |
| `write_dicom_recon(vol, path, pixel_size)` | Transposes and rotates the volume to the standard display orientation, linearly rescales to 16-bit unsigned integer, and saves one DICOM file per reconstructed slice. |

---

### `segmentation/` — Breast Segmentation

#### `seg_breast.py`

Top-level orchestrator that iterates over all DICOM projections and returns a
3-D binary breast mask stack.

| Function | Description |
|---|---|
| `segbreast(dcmFiles, log)` | For each projection: reads and optionally decompresses the DICOM data; downscales to 25 %; log-normalises; standardises orientation; removes paddle artefacts when present; computes the binary mask (CC or MLO strategy); upscales back to full resolution. Returns an `(nv, nu, nProj)` uint8 mask stack and the breast body thickness. |
| `decompress(file)` | Decodes a JPEG 2000–compressed DICOM file and returns an uncompressed dataset. |
| `flips(ds, img, log)` | Re-orients the image so the nipple faces right, based on the DICOM `Laterality` tag. |
| `getmask(ds, low_res, index, log)` | Selects the CC or MLO segmentation strategy from the DICOM `ViewPosition` tag, runs contour (+ optional pectoral) segmentation, and returns only the largest connected component. |
| `imnorm(im, ntype)` | Normalises a projection image using logarithmic (`'log'`) or z-score (`'zsc'`) normalisation. |

#### `seg_contour.py`

Detects the breast outline from a single downscaled projection using histogram
thresholding and optional curvature analysis.

| Function | Description |
|---|---|
| `segcontour(im, ccflag)` | Analyses the row intensity range to find the active image area, builds a smoothed histogram to locate the air–tissue threshold, thresholds and keeps the largest connected component, then extracts the contour. For CC views (`ccflag=True`) the raw mask is returned; for MLO views curvature analysis trims the contour at the chest wall and a closed polygon mask is produced. |

#### `seg_paddle.py`

Detects and removes compression-paddle artefacts from DBT projections using
Hough-based line detection.

| Function | Description |
|---|---|
| `has_paddle_marks(im, line_threshold, v_mean_threshold)` | Lightweight check: applies Canny edge detection and the Hough transform; returns `True` if strong near-horizontal lines are present in the outer 20 % of the image. |
| `segpaddle(im, thickness)` | Full segmentation: detects top and bottom paddle bands and returns a boolean mask covering only the breast region between them. |
| `get_hline(edge_map, n_max)` | Low-level helper: runs the Hough transform restricted to angles within ±2.5° of 90° and returns the y-intercepts and accumulator values of the top detected lines. |

#### `seg_pectoral.py`

Identifies and masks out the pectoral muscle in MLO projections using a Hough
line fit restricted to the expected pectoral angle range.

| Function | Description |
|---|---|
| `segpectoral(I0, cpoints)` | Uses the breast contour to focus on the upper-inner ROI, applies Canny edge detection, runs the Hough transform (25°–45°) to find the dominant pectoral boundary line, and returns a boolean exclusion mask together with the line parameters. Returns an all-`True` mask (no exclusion) if no line is detected. |

---

## Output

For each processed exam the script produces:

| File | Description |
|---|---|
| `<exam>_lesion.tif` | Raw 3-D lesion volume (for visual inspection) |
| `<exam>_projs_masks.tif` | Forward-projected lesion masks |
| `<exam>_projs_mtf.tif` | MTF-blurred projection masks |
| `results/<model>/…/*.dcm` | Modified DICOM projections with lesion inserted |

A detailed execution log is written to `hybrid-vit.log`.

---

## Citation

If you use this code in your research, please cite the relevant publication
*(reference TBD)*.

---

## License

*(License TBD)*
