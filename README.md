# ROVER-dMRI

ROVER-dMRI reconstruction code. A TCNN (tiny-cuda-nn hash-grid) implicit
neural representation reconstructs an isotropic high-resolution volume from
multi-orientation thick-slice acquisitions, with an RF slice-profile PSF
model along the through-plane direction.

Two dataset configurations are provided:

| | Primary: Cima | Legacy: Philips phantom |
|---|---|---|
| Slice profile | TBWP=4 sinc (`rf_profile_tbwp4.mat`) | TBWP=4 sinc, spin-echo variant (`rf_slice_profile_mse_sinc.mat`) |
| Args | `util_args_rover_b0_v18_lr1e4.py` | `util_args_rover_b0_v9_lr1e4.py` |
| Config | `configs/qiang_data_v10.yaml` | `configs/data_v9.yaml` |
| Training script | `rover_b0_hash_cima_v0_tcnn_relu_charb_tv_tbwp4.py` | `rover_b0_hash_philps_v3_tcnn_relu_charb_tv.py` |
| Test script | `rover_b0_hash_cima_v0_tcnn_relu_charb_tv_tbwp4_test.py` | `rover_b0_hash_philps_v3_tcnn_relu_charb_tv_test.py` |
| Views / image size | 12 views, 310x310x28 | 12 views, 224x224x15 |

## Pipeline overview

Run the steps in this order:

1. **MATLAB preprocessing — Step 1** (`Preprocessing_code_matlab/Step_1_preprocess.m`)
2. **MATLAB preprocessing — Step 2** (`Preprocessing_code_matlab/Step_2_generate_npy.m`)
3. **Python training** (`rover_b0_hash_cima_v0_tcnn_relu_charb_tv_tbwp4.py`)
4. **Python inference / test** (`rover_b0_hash_cima_v0_tcnn_relu_charb_tv_tbwp4_test.py`)

Steps 1–2 turn the raw NIfTI acquisitions into the `.npy` image + affine files
that the Python code reads; steps 3–4 train the model and reconstruct the volume.

---

## 0. Example / test dataset

The example acquisitions used by the default paths in the scripts (12-view
thick-slice NIfTI volumes plus the per-view affine matrices they preprocess
to) will be released as an open dataset on
**[OpenNeuro](https://openneuro.org/)**.

> **TODO:** add the OpenNeuro accession link/DOI here once the dataset is
> published.

Until then, `--img_path` (in `util_args_rover_b0_v18_lr1e4.py`) and the
`Affine_nii_*.npy` paths hard-coded in the training/test scripts point at a
local path (`/scratch/home/ql087/data_bwh/Cima_data/preprocess_rover_nii/`).
Point these at wherever you place your own copy of the data (or the
downloaded OpenNeuro dataset once available).

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
> under `/scratch/home/ql087/data_bwh/...`. Point these at your own data and
> toolbox locations.

---

## 2. Python environment

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
> `util_args_rover_b0_v18_lr1e4`, `fda/…`) and `configs/qiang_data_v10.yaml`
> resolve. The input `.npy` path is set by `--img_path` in
> `util_args_rover_b0_v18_lr1e4.py` (default points to
> `.../preprocess_rover_nii/imgs_nii_*.npy`); the per-view `Affine_nii_*.npy`
> paths are set inside the scripts. Edit these to your data.

---

## 3. Training — `rover_b0_hash_cima_v0_tcnn_relu_charb_tv_tbwp4.py`

```bash
python rover_b0_hash_cima_v0_tcnn_relu_charb_tv_tbwp4.py
```

Runs with the defaults in `configs/qiang_data_v10.yaml` and
`util_args_rover_b0_v18_lr1e4.py` (8000 epochs, 12 views, batch 100000, image
size 310x310x28). Checkpoints are written every `--image_save_iter` (default
4000) iterations to `output/outputs/<model_name>/checkpoints/model_XXXXXX.pt`.

Useful options:

| Flag | Default | Meaning |
|------|---------|---------|
| `--weight_mode` | `sliceprofile` | Through-plane PSF weighting: `sliceprofile`, `equal`, or `gaussian` |
| `--rf_profile_mat` | `rf_profile_tbwp4.mat` | Slice-profile file (`z_mm`, `Mse_sinc`), used only for `sliceprofile` mode |
| `--charb_epsilon` | `0.1` | Charbonnier loss epsilon |
| `--tv_fd_eps` | `1e-3` | Finite-difference step for the random TV term |
| `--tv_num_samples` | `0` (= `batch_size // 2`) | Number of random coords for the TV term |
| `--config` | `configs/qiang_data_v10.yaml` | Model/output config |
| `--output_path` | `output` | Output root |

Example (uniform weighting instead of slice profile):

```bash
python rover_b0_hash_cima_v0_tcnn_relu_charb_tv_tbwp4.py --weight_mode equal
```

---

## 4. Inference / test — `rover_b0_hash_cima_v0_tcnn_relu_charb_tv_tbwp4_test.py`

Loads a trained checkpoint and writes the reconstructed volume as NIfTI to
`output/outputs/<model_name>/test-output/`. The `weight_mode` and
`charb_epsilon` must match the training run so the model folder name resolves.

```bash
# by iteration (finds model_008000.pt in the matching run folder)
python rover_b0_hash_cima_v0_tcnn_relu_charb_tv_tbwp4_test.py --checkpoint_iter 8000

# or point directly at a checkpoint file
python rover_b0_hash_cima_v0_tcnn_relu_charb_tv_tbwp4_test.py \
    --checkpoint_path output/outputs/<model_name>/checkpoints/model_008000.pt
```

Useful options: `--checkpoint_iter`, `--checkpoint_path`, `--weight_mode`,
`--charb_epsilon`, `--z_spacing_mm` (through-plane voxel size written to the
NIfTI header, default = in-plane x resolution).

---

## Legacy: Philips phantom pipeline

The original Philips-phantom scripts are kept for reference:

```bash
python rover_b0_hash_philps_v3_tcnn_relu_charb_tv.py
python rover_b0_hash_philps_v3_tcnn_relu_charb_tv_test.py --checkpoint_iter 8000
```

These use `util_args_rover_b0_v9_lr1e4.py`, `configs/data_v9.yaml`, and
`rf_slice_profile_mse_sinc.mat` (image size 224x224x15), and additionally
support `--profile_var` to pick which profile in the `.mat` to use
(`Mxy_sinc`, `Mse_sinc`, `Mref_sinc`).

---

## Files

| Path | Purpose |
|------|---------|
| `Preprocessing_code_matlab/Step_1_preprocess.m` | Step 1: clean NIfTIs to single-frame 3D |
| `Preprocessing_code_matlab/Step_2_generate_npy.m` | Step 2: export `imgs_nii_*.npy` + `Affine_nii_*.npy` |
| `Preprocessing_code_matlab/rf_pulse_simulation_QL_v3.m` | RF slice-profile simulation |
| `rover_b0_hash_cima_v0_tcnn_relu_charb_tv_tbwp4.py` | **Training (primary, Cima)** |
| `rover_b0_hash_cima_v0_tcnn_relu_charb_tv_tbwp4_test.py` | **Inference / test (primary, Cima)** |
| `util_args_rover_b0_v18_lr1e4.py` | CLI args / data paths for the Cima pipeline |
| `configs/qiang_data_v10.yaml` | Model + output config for the Cima pipeline |
| `rf_profile_tbwp4.mat`, `rf_profile_tbwp4.png` | TBWP=4 RF slice profile used by `sliceprofile` mode (Cima) |
| `rover_b0_hash_philps_v3_tcnn_relu_charb_tv.py` | Training (legacy, Philips phantom) |
| `rover_b0_hash_philps_v3_tcnn_relu_charb_tv_test.py` | Inference / test (legacy, Philips phantom) |
| `util_args_rover_b0_v9_lr1e4.py` | CLI args / data paths for the Philips phantom pipeline |
| `configs/data_v9.yaml` | Model + output config for the Philips phantom pipeline |
| `rf_slice_profile_mse_sinc.mat` | Simulated RF slice profile used by the Philips phantom pipeline |
| `utils.py`, `fda/…` | Helpers (config load, normalization, dataset object) |
| `ENVIRONMENT.md`, `requirements.txt` | Python environment |
