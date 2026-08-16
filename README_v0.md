# ROVER-dMRI_Philips_Share

ROVER-dMRI reconstruction code (Philips phantom). A TCNN (tiny-cuda-nn hash-grid)
implicit neural representation reconstructs an isotropic high-resolution volume
from multi-orientation thick-slice acquisitions, with an RF slice-profile PSF
model along the through-plane direction.

## Pipeline overview

Run the steps in this order:

1. **MATLAB preprocessing — Step 1** (`Preprocessing_code_matlab/Step_1_preprocess.m`)
2. **MATLAB preprocessing — Step 2** (`Preprocessing_code_matlab/Step_2_generate_npy.m`)
3. **Python training** (`rover_b0_hash_philps_v3_tcnn_relu_charb_tv_inputdir.py`)
4. **Python inference / test** (`rover_b0_hash_philps_v3_tcnn_relu_charb_tv_test.py`)

Steps 1–2 turn the raw NIfTI acquisitions into the `.npy` image + affine files
that the Python code reads; steps 3–4 train the model and reconstruct the volume.

---

## 1. Preprocessing (MATLAB)

Run in MATLAB, **Step 1 first, then Step 2**.

**Dependencies**

- FreeSurfer MATLAB tools on the path (`MRIread`, `MRIwrite`)
- [npy-matlab](https://github.com/kwikteam/npy-matlab) on the path (`writeNPY`) — used by Step 2

### Step 1 — `Step_1_preprocess.m`

Reads every `*.nii` in the input folder, keeps a single 3D volume per file
(drops the extra frame), fixes the NIfTI header to describe a single-frame 3D
volume, and writes the cleaned NIfTIs to the output folder.

```matlab
run('Preprocessing_code_matlab/Step_1_preprocess.m')
```

    BB note: skip this step, apply only step 2

### Step 2 — `Step_2_generate_npy.m`

Reads the per-orientation NIfTIs (`Dicom_DWI_rot1..12_*.nii`) and writes, for
each of the 12 views, into `preprocess_rover_nii/`:

- `imgs_nii_<i>.npy` — the image volume
- `Affine_nii_<i>.npy` — the `vox2ras` (affine) matrix

```matlab
run('Preprocessing_code_matlab/Step_2_generate_npy.m')
```

These `imgs_nii_*.npy` and `Affine_nii_*.npy` files are exactly what the Python
scripts load.

> **Edit the paths first.** Both `.m` files have hard-coded `baseDir`/`base_path`
> under `/scratch/home/ql087/data_bwh/...`, and Step 2 `addpath`s a specific
> `npy-matlab` location. Point these at your own data and toolbox locations.

---

## 2. Python environment

Activate the existing prepared ROVER environment:

```bash
cd /cluster/berkin/berkin/Matlab_Code_New/PULSEQ/ROVER-dMRI_Philips_Share-main
source /autofs/cluster/berkin/yuting/miniforge3/etc/profile.d/conda.sh
conda activate "$PWD/.conda/envs/rover"
```

See **[ENVIRONMENT.md](ENVIRONMENT.md)** for the full setup (GPU/CUDA
requirements, package list, `tiny-cuda-nn` install). Quick version:

```bash
conda create -n rover python=3.12 -y && conda activate rover
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

An **NVIDIA GPU with CUDA is required** (the model uses `tiny-cuda-nn` +
mixed precision; it will not run on CPU).

> Run the Python scripts **from this folder** so local imports (`utils`,
> `util_args_rover_b0_v9_lr1e4`, `fda/…`) and `configs/data_v9.yaml` resolve.
> Supply the MATLAB output directory to the training script with `--input_dir`.
> The script finds the image and affine files in that directory and detects the
> image dimensions automatically.

---

## 3. Training — `rover_b0_hash_philps_v3_tcnn_relu_charb_tv_inputdir_v0.py`

The input directory must contain matching files for each view:

- `imgs_nii_<i>.npy` — image volumes
- `Affine_nii_<i>.npy` — affine matrices

The script naturally orders the numbered files, determines the image size, and
reads `spa_res` from the X-axis norm of `Affine_nii_1.npy`. All selected image
views must be 3D, have the same shape, and have consistent in-plane spacing.

```bash
python rover_b0_hash_philps_v3_tcnn_relu_charb_tv_inputdir_v0.py \
    --input_dir "/path/to/preprocess_rover_nii" \
    --num_depth_layers 8
```

Runs with the defaults in `configs/data_v9.yaml` and
`util_args_rover_b0_v9_lr1e4.py` (8000 epochs, 12 views, batch 100000).
Checkpoints are written every 2000 iters to
`output/outputs/<model_name>/checkpoints/model_XXXXXX.pt`.

Useful options:

| Flag                 | Default                           | Meaning                                                                                                                   |
| -------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `--input_dir`      | required                          | Directory containing the`imgs_nii_<i>.npy` and `Affine_nii_<i>.npy` files                                             |
| `--num_depth_layers` | `8`                             | Number of reconstructed depth layers                                                                                       |
| `--weight_mode`    | `sliceprofile`                  | Through-plane PSF weighting:`sliceprofile`, `equal`, or `gaussian`                                                  |
| `--profile_var`    | `Mxy_sinc`                      | Which profile in the`.mat` to use: `Mxy_sinc` (excitation-only), `Mse_sinc` (spin-echo), `Mref_sinc` (refocusing) |
| `--rf_profile_mat` | `rf_slice_profile_mse_sinc.mat` | Slice-profile file (only for`sliceprofile` mode)                                                                        |
| `--charb_epsilon`  | `0.1`                           | Charbonnier loss epsilon                                                                                                  |
| `--config`         | `configs/data_v9.yaml`          | Model/output config                                                                                                       |
| `--output_path`    | `output`                        | Output root                                                                                                               |

Example (uniform weighting instead of slice profile):

```bash
python rover_b0_hash_philps_v3_tcnn_relu_charb_tv_inputdir_v0.py \
    --input_dir "/path/to/preprocess_rover_nii" \
    --weight_mode equal
```

The existing non-`_v0` scripts are retained unchanged for legacy runs.

---

## 4. Inference / test — `rover_b0_hash_philps_v3_tcnn_relu_charb_tv_test_inputdir_v0.py`

Loads a trained checkpoint and writes the reconstructed volume as NIfTI. For
the current `Mxy_sinc` slice-profile run, pass the checkpoint file explicitly:

```bash
python rover_b0_hash_philps_v3_tcnn_relu_charb_tv_test_inputdir_v0.py \
  --input_dir "/autofs/space/daisy_002/users/berkin/2026_08_07_bay5_rover_msepi_750um_invivo/rover_msepi_2026.08.07-09_42_17-DST-1.3.12.2.1107.5.99.3_19900101/nii/preprocess_rover_nii" \
  --num_depth_layers 8 \
  --output_path output_v0 \
  --checkpoint_path "/autofs/cluster/berkin/berkin/Matlab_Code_New/PULSEQ/ROVER-dMRI_Philips_Share-main/output_v0/outputs/rover_philips_tcnn_relu_tv_1e-5_profile1_b1p5_L8_T25/rover_tcnn_relu_psf_charb_hashenc_lev8_r192_d2_log25_nl_2_br16_bs100000_lambda1e-5_wsliceprofile_Mxy_sinc_eps0p1/checkpoints/model_008000.pt"
```

When `--checkpoint_path` is supplied, the script obtains the run directory from
that path, so `--checkpoint_iter 8000` is unnecessary. The reconstructed NIfTI
is saved under the same run directory at:

```text
output_v0/outputs/<model_name>/test-output/
```

> **Current slice-profile folder-name limitation:** `--checkpoint_iter 8000`
> alone does not resolve this run. The test script searches for a folder tagged
> `wsliceprofile`, but training created the folder with the more specific tag
> `wsliceprofile_Mxy_sinc`. Use the explicit `--checkpoint_path` command above.

Useful supported options include `--input_dir`, `--num_depth_layers`,
`--output_path`, `--checkpoint_path`, `--checkpoint_iter`, `--weight_mode`,
`--rf_profile_mat`, `--charb_epsilon`, `--config`, and `--z_spacing_mm`.
`--z_spacing_mm` controls the through-plane voxel size written to the NIfTI
header and defaults to the affine-derived `spa_res`.

---

## Files

| Path                                                       | Purpose                                                 |
| ---------------------------------------------------------- | ------------------------------------------------------- |
| `Preprocessing_code_matlab/Step_1_preprocess.m`          | Step 1: clean NIfTIs to single-frame 3D                 |
| `Preprocessing_code_matlab/Step_2_generate_npy.m`        | Step 2: export`imgs_nii_*.npy` + `Affine_nii_*.npy` |
| `rover_b0_hash_philps_v3_tcnn_relu_charb_tv_inputdir_v0.py` | **Automatic-geometry training (`--input_dir`)** |
| `rover_b0_hash_philps_v3_tcnn_relu_charb_tv_test_inputdir_v0.py` | **Automatic-geometry inference / test** |
| `rover_b0_hash_philps_v3_tcnn_relu_charb_tv_inputdir.py` | Legacy input-directory training (unchanged) |
| `rover_b0_hash_philps_v3_tcnn_relu_charb_tv.py`          | Legacy training with fixed/default input paths          |
| `rover_b0_hash_philps_v3_tcnn_relu_charb_tv_test.py`     | **Inference / test**                              |
| `util_args_rover_b0_v9_lr1e4.py`                         | CLI args / data paths (`--img_path`, sizes, epochs)   |
| `configs/data_v9.yaml`                                   | Model + output config                                   |
| `utils.py`, `fda/…`                                   | Helpers (config load, normalization, dataset object)    |
| `rf_slice_profile_mse_sinc.mat`                          | Simulated RF slice profile used by`sliceprofile` mode |
| `ENVIRONMENT.md`, `requirements.txt`                   | Python environment                                      |
