# Python environment

This code trains a TCNN (tiny-cuda-nn hash-grid) INR for ROVER MRI B0
super-resolution reconstruction. It **requires an NVIDIA GPU with CUDA** — the
model uses `tiny-cuda-nn`'s `FullyFusedMLP`/`HashGrid` and mixed-precision
(`torch.cuda.amp`), so it will not run on CPU.

## System requirements

- NVIDIA GPU (the scripts use `CUDA_VISIBLE_DEVICES=0`)
- CUDA toolkit + a C++/CUDA compiler (`nvcc`, gcc) — needed to build `tiny-cuda-nn`
- Python 3.9–3.12 (developed/tested on 3.12)

## Python packages

Direct third-party imports used by the shared scripts:

| Package        | Used for                                             | Imported in |
|----------------|------------------------------------------------------|-------------|
| `torch`        | model, training, AMP, TensorBoard writer             | both scripts, `data_objects_bumonkey.py` |
| `tinycudann`   | hash-grid encoding + fully-fused MLP (`tiny-cuda-nn`)| both scripts |
| `numpy`        | arrays / geometry                                    | everywhere  |
| `scipy`        | `scipy.io.loadmat` (loads the RF slice profile)      | training    |
| `nibabel`      | write reconstructed volume as NIfTI                  | test        |
| `matplotlib`   | training-loss plot (`Agg` backend)                   | training    |
| `PyYAML`       | read `configs/data_v9.yaml` (`import yaml`)          | `utils.py`  |
| `tensorboard`  | backs `torch.utils.tensorboard.SummaryWriter`        | training    |

Everything else (`os`, `sys`, `math`, `argparse`, `glob`, `re`, `datetime`) is
in the Python standard library.

> The MATLAB file `rf_pulse_simulation_QL_v2.m` only *generates*
> `rf_slice_profile_mse_sinc.mat` (already included). You do **not** need MATLAB
> to run the Python code. Regenerating the profile needs MATLAB + the Pulseq and
> `rf_tools` toolboxes (not bundled here).

## Setup

`tiny-cuda-nn` must be built against your installed PyTorch/CUDA, so install
PyTorch **first** (matching your CUDA version), then the rest, then tiny-cuda-nn.

```bash
# 1) create env
conda create -n rover python=3.12 -y
conda activate rover

# 2) PyTorch matching your CUDA (see https://pytorch.org — example: CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3) remaining Python deps
pip install -r requirements.txt

# 4) tiny-cuda-nn PyTorch bindings (compiles CUDA — needs nvcc + gcc in PATH)
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

## Notes / gotchas

- **`tinycudann` is the hard part.** It compiles CUDA kernels at install time and
  must match your `torch` + CUDA toolkit versions. If import fails, rebuild it
  against the exact torch you installed. See the NVlabs/tiny-cuda-nn README.
- **NumPy 2.x:** works with recent PyTorch, but older torch wheels were built
  against NumPy 1.x. If you hit a NumPy-ABI error, pin `numpy<2`.
- **Data paths are hard-coded** in `util_args_rover_b0_v9_lr1e4.py` (`--img_path`)
  and inside the scripts (the `Affine_nii_*.npy` paths under
  `/scratch/home/ql087/data_bwh/...`). Edit these to point at your own data.
- Run from this folder so the local imports (`utils`, `util_args_...`, `fda/…`)
  and `configs/data_v9.yaml` resolve.
