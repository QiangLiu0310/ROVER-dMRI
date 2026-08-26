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

## Repository layout

This repo covers two separate stages:

1. **[Sequence and Reconstruction](#part-1-sequence-and-reconstruction)** —
   the Pulseq acquisition sequence and the raw k-space to NIfTI
   reconstruction (DPG + BUDA/S-LORAKS) that runs on the scanner data.
2. **[ROVER-dMRI super-resolution reconstruction](#part-2-rover-dmri-super-resolution-reconstruction)** —
   MATLAB preprocessing plus the TCNN training/inference that turns the
   per-view NIfTIs into an isotropic high-resolution volume.

> **Note:** these two parts are not a verified, matched end-to-end pipeline.
> The example/test dataset used in Part 2
> (`/scratch/home/ql087/data_bwh/Cima_data/preprocess_rover_nii/`) was **not**
> produced by running Part 1's reconstruction code on raw scanner data from
> the same session — they are independent deliverables. If you run the full
> chain yourself (sequence → scanner → recon → preprocessing → training),
> verify the intermediate NIfTIs match what Part 2 expects before trusting
> the result.

---

# Part 1: Sequence and Reconstruction

## 1. Acquisition — Pulseq sequence (MATLAB)

`sequence/` vendors the [Pulseq](https://pulseq.github.io/) v1.5.1 MATLAB
toolbox (`sequence/matlab/+mr/`) plus the ROVER-dMRI sequence built on top of
it, originally from
[Pulseq_V151_ROVER_dMRI](https://github.com/QiangLiu0310/Pulseq_V151_ROVER_dMRI).

- `sequence/matlab/demoSeq/rover_dmri/singleband/cimax/ep2d_6mm_R6_3shots_slc28_dir25_QL_v4.m` —
  writes the `.seq` file for the multi-shot EPI diffusion acquisition (3-shot,
  R=6/shot, b0 + diffusion, run at 12 rotated in-plane orientations to give
  the 12 views the reconstruction expects).
- `sequence/matlab/demoSeq/write2DGRE_QL_v3.m` — 2D GRE calibration scan
  (coil sensitivity maps for the parallel-imaging recon below).
- `sequence/diffusion_table/` — diffusion gradient tables (`.dvs`/`.txt`)
  referenced by the sequence script.

Run the sequence script in MATLAB with `sequence/matlab` (incl. `+mr/`) on
the path; it writes a `.seq` file to load on the scanner. Edit the
`seq_file` name and imaging parameters at the top of the script for your setup.

## 2. Reconstruction — raw k-space to NIfTI (MATLAB)

`recon/` turns the raw scanner data from the 12 rotated-view acquisitions
into per-view NIfTI volumes, in the same layout Step 1 of the Part 2
preprocessing below expects (but see the note above — this has not been
verified against the specific example dataset shipped with Part 2).

| Script | Purpose |
|---|---|
| `recon/rot/script_read_epi_rot_R6_0p5mm_v3.m` | Reads Siemens raw data (`.dat`) for each of the 12 rotations |
| `recon/DPG_sharecode/DPG_recon_QL_v3_0p5_ms_3.m` | DPG (dynamic phase/ghost) correction, per shot |
| `recon/BUDA_SLORAKS/RUN_S_LORAKS_QL_v1.m`, `recon/BUDA_SLORAKS/codes/RUN_BUDA_S_LORAKS.m` | BUDA + S-LORAKS parallel-imaging reconstruction |

**External dependencies not vendored here** — install separately and add to
your MATLAB path:
- [`mapVBVD`](https://github.com/CIC-methods/FID-A) (or equivalent) to read
  Siemens raw `.dat` files, used by `recon/rot/script_read_epi_rot_R6_0p5mm_v3.m`
- GE `orchestra.matlab` SDK, used by `recon/DPG_sharecode/DPG_recon_QL_v3_0p5_ms_3.m`
- `BUDA_LORAKS_UY` (BUDA/LORAKS toolbox), `addpath`-ed by `RUN_BUDA_S_LORAKS.m`

> **Edit the paths first.** These scripts have hard-coded `save_path`/`data_path`
> values under `/scratch/home/ql087/...` and `/rfanfs/...`. Point them at your
> own raw data and output locations.

---

# Part 2: ROVER-dMRI super-resolution reconstruction

## 0. Example / test dataset

The example acquisitions used by the default paths in the scripts (12-view
thick-slice NIfTI volumes plus the per-view affine matrices they preprocess
to) will be released as an open dataset on
**[OpenNeuro](https://openneuro.org/)**.

> **TODO:** add the OpenNeuro accession link/DOI here once the dataset is
> published.

> **Note:** this dataset is not the direct output of the Part 1 reconstruction
> code above — see the note at the top of this README.

Until then, `--img_path` (in `util_args_rover_b0_v18_lr1e4.py`) and the
`Affine_nii_*.npy` paths hard-coded in the training/test scripts point at a
local path (`/scratch/home/ql087/data_bwh/Cima_data/preprocess_rover_nii/`).
Point these at wherever you place your own copy of the data (or the
downloaded OpenNeuro dataset once available).

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
| `sequence/matlab/demoSeq/rover_dmri/singleband/cimax/ep2d_6mm_R6_3shots_slc28_dir25_QL_v4.m` | Pulseq sequence: 3-shot R=6 EPI diffusion acquisition |
| `sequence/matlab/demoSeq/write2DGRE_QL_v3.m` | Pulseq sequence: 2D GRE calibration scan |
| `sequence/matlab/+mr/` | Vendored Pulseq v1.5.1 MATLAB toolbox (sequence dependency) |
| `sequence/diffusion_table/` | Diffusion gradient tables used by the sequence |
| `recon/rot/script_read_epi_rot_R6_0p5mm_v3.m` | Read raw Siemens data for the 12 rotated views |
| `recon/DPG_sharecode/` | DPG ghost/phase correction |
| `recon/BUDA_SLORAKS/` | BUDA + S-LORAKS parallel-imaging reconstruction |
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
