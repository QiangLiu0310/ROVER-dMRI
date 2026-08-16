#!/usr/bin/env python
# coding: utf-8
"""
Test / inference script for the ROVER B0 TCNN model trained with the matching
automatic-geometry input-directory training script.

Geometry: configurable depth layers and spa_res read from Affine_nii_1.npy.
weight_mode: gaussian, equal, or sliceprofile (must match training).
Training sliceprofile uses rf_slice_profile_mse_sinc.mat (z_mm, Mse_sinc).
Config: configs/data_v9.yaml. Args: util_args_rover_b0_v9_lr1e4.

The reconstructed volume interleaves `num_depth_layers` sub-slices per acquired
slice, so the through-plane voxel spacing defaults to the affine-derived
spa_res. The saved NIfTI z-axis voxel size can be overridden with
--z_spacing_mm.
"""

import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
print("Project root:", project_root)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
import nibabel as nib

import sys
sys.path.append("fda")
from data_objects.data_objects_bumonkey import ObservationPoints3D
from utils import get_config, normalization
from util_args_rover_b0_v9_lr1e4 import get_args

import glob
import time
import re
import datetime

import tinycudann as tcnn


def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def positive_int(value):
    """Argparse type for strictly positive integers."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def load_and_validate_affines(affine_paths):
    """Load 4x4 affines and require consistent, finite in-plane spacing."""
    affines = []
    spacings = []
    for path in affine_paths:
        affine = np.asarray(np.load(path), dtype=np.float64)
        if affine.shape != (4, 4):
            raise ValueError(
                f"Affine file must have shape (4, 4): {path} has {affine.shape}."
            )
        if not np.all(np.isfinite(affine)):
            raise ValueError(f"Affine file contains non-finite values: {path}")

        voxel_sizes = np.linalg.norm(affine[:3, :3], axis=0)
        if not np.all(np.isfinite(voxel_sizes)) or np.any(voxel_sizes <= 0):
            raise ValueError(
                f"Affine file has invalid voxel spacing {voxel_sizes}: {path}"
            )
        affines.append(affine)
        spacings.append(voxel_sizes)

    spacings = np.asarray(spacings)
    if not np.allclose(spacings[:, :2], spacings[0, :2], rtol=1e-5, atol=1e-6):
        details = ", ".join(
            f"{os.path.basename(path)}=({spacing[0]:.6f}, {spacing[1]:.6f}) mm"
            for path, spacing in zip(affine_paths, spacings)
        )
        raise ValueError(f"Input views have inconsistent in-plane spacing: {details}")

    return affines, spacings


def fix_affine_z_spacing(reference_affine, target_z_spacing_mm):
    """Keep orientation and origin; set z column length to target_z_spacing_mm."""
    if not np.isfinite(target_z_spacing_mm) or target_z_spacing_mm <= 0:
        raise ValueError("Target z-spacing must be a finite positive number.")
    aff = np.array(reference_affine, dtype=np.float64, copy=True)
    z_vec = aff[:3, 2]
    z_norm = np.linalg.norm(z_vec)
    if z_norm <= 0:
        raise ValueError("Reference affine z-axis vector norm is zero; cannot set z-spacing.")
    aff[:3, 2] = (z_vec / z_norm) * float(target_z_spacing_mm)
    return aff


log_filename = f"rover_b0_hash_philps_test_tcnn_relu_charb_tv_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.out"


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger(log_filename)

print(f"=== ROVER B0 Philps TCNN (Charbonnier+TV) test start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
print(f"Log file: {log_filename}")
print("=" * 80)


class ROVER_TCNN(torch.nn.Module):
    """Same TCNN architecture as rover_b0_hash_philps_v3_tcnn_relu_charb_tv.py."""

    def __init__(self, in_dim=3, out_dim=1, L=16, F=2, T=19, N_min=16, N_max=2048):
        super().__init__()
        b = math.exp((math.log(N_max) - math.log(N_min)) / (L - 1))

        self.encoding = tcnn.Encoding(
            n_input_dims=in_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": L,
                "n_features_per_level": F,
                "log2_hashmap_size": T,
                "base_resolution": N_min,
                "per_level_scale": b,
                "interpolation": "Smoothstep",
            },
        )

        self.mlp = tcnn.Network(
            n_input_dims=self.encoding.n_output_dims + in_dim,
            n_output_dims=out_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 3,
            },
        )

    def forward(self, x):
        original_shape = x.shape
        x_flat = x.view(-1, original_shape[-1]).float()
        encoded = self.encoding(x_flat)
        mlp_in = torch.cat([encoded, x_flat], dim=-1)
        out_flat = self.mlp(mlp_in)
        return out_flat.view(*original_shape[:-1], -1)


start_time = time.time()


# -------------------------------------------------------------------------
# STEP 2: args / config (match Philps training defaults)
# -------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/data_v9.yaml',
                    help='Path to the config file (must match training).')
parser.add_argument('--output_path', type=str, default='output',
                    help="outputs path")
parser.add_argument('--input_dir', type=str, required=True,
                    help='Directory containing imgs_nii_<i>.npy and Affine_nii_<i>.npy files.')
parser.add_argument('--num_depth_layers', type=positive_int, default=8,
                    help='Number of reconstructed depth layers. Default: 8.')
parser.add_argument('--checkpoint_iter', type=int, default=None,
                    help="Checkpoint iteration to load")
parser.add_argument('--weight_mode', type=str, default='sliceprofile',
                    choices=['gaussian', 'equal', 'sliceprofile'],
                    help="Depth-layer PSF weighting mode used during training.")
parser.add_argument('--rf_profile_mat', type=str,
                    default=os.path.join(project_root, 'rf_slice_profile_mse_sinc.mat'),
                    help="Slice profile .mat used during training (for logging; must match training when weight_mode='sliceprofile').")
parser.add_argument('--charb_epsilon', type=float, default=1e-1,
                    help="Charbonnier epsilon used during training (must match). Default 1e-1.")
parser.add_argument('--checkpoint_path', type=str, default=None,
                    help="Optional: full path to a .pt checkpoint. If set, load this file and save under its run folder.")
parser.add_argument('--z_spacing_mm', type=float, default=None,
                    help="NIfTI z-axis voxel size in mm. Default: affine-derived spa_res.")
opts, _ = parser.parse_known_args()
args = get_args(cmd=False)
config = get_config(opts.config)

nv = args.view_num
input_dir = os.path.abspath(os.path.expanduser(opts.input_dir))
if not os.path.isdir(input_dir):
    raise NotADirectoryError(f"Input directory does not exist or is not a directory: {input_dir}")

img_path = os.path.join(input_dir, 'imgs_nii_*.npy')
files_path = sorted(glob.glob(img_path), key=natural_sort_key)
if len(files_path) < nv:
    raise FileNotFoundError(
        f"Expected at least {nv} image files matching {img_path}, but found {len(files_path)}."
    )

image_shapes = [np.load(path, mmap_mode='r').shape for path in files_path[:nv]]
if any(len(shape) != 3 for shape in image_shapes):
    raise ValueError(
        "All input images must be 3D. Found shapes: "
        + ", ".join(f"{os.path.basename(path)}={shape}" for path, shape in zip(files_path[:nv], image_shapes))
    )
if any(shape != image_shapes[0] for shape in image_shapes[1:]):
    raise ValueError(
        "All input images must have the same shape. Found shapes: "
        + ", ".join(f"{os.path.basename(path)}={shape}" for path, shape in zip(files_path[:nv], image_shapes))
    )

# Training transposes images from [ny, nx, nk] to [nx, ny, nk].
ny, nx, nk = image_shapes[0]
print(f"Inferred image size after training transpose: nx={nx}, ny={ny}, nk={nk}")

affine_paths = [
    os.path.join(input_dir, f'Affine_nii_{view_num + 1}.npy')
    for view_num in range(nv)
]
missing_affine_paths = [path for path in affine_paths if not os.path.isfile(path)]
if missing_affine_paths:
    raise FileNotFoundError(
        "Missing affine file(s): " + ", ".join(missing_affine_paths)
    )

# -------------------------------------------------------------------------
# STEP 1: spatial resolution and depth layers (same as Philps training)
# -------------------------------------------------------------------------
print("=== ROVER B0 Philps spatial resolution ===")
affines, affine_spacings = load_and_validate_affines(affine_paths)
reference_affine = affines[0]
x_voxel_size, y_voxel_size, z_voxel_size = affine_spacings[0]
spa_res = float(x_voxel_size)
num_depth_layers = opts.num_depth_layers

print(f"Voxel sizes from the view-0 Affine matrix:")
print(f"  X voxel size: {x_voxel_size:.6f} mm")
print(f"  Y voxel size: {y_voxel_size:.6f} mm")
print(f"  Z voxel size: {z_voxel_size:.6f} mm")
print(f"Reference Affine matrix:\n{reference_affine}")

depth_ratio = z_voxel_size / spa_res
print(f"Depth ratio: {z_voxel_size:.6f} / {spa_res:.6f} = {depth_ratio:.6f}")
print(f"Number of depth layers (command-line input): {num_depth_layers}")
print(f"Through-plane spa_res (from affine X-axis): {spa_res:.6f} mm")

z_spacing_mm = spa_res if opts.z_spacing_mm is None else opts.z_spacing_mm
output_affine = fix_affine_z_spacing(reference_affine, target_z_spacing_mm=z_spacing_mm)
orig_z_spacing = np.linalg.norm(reference_affine[:3, 2])
out_z_spacing = np.linalg.norm(output_affine[:3, 2])
print(f"NIfTI header z-spacing: {orig_z_spacing:.6f} mm -> {out_z_spacing:.6f} mm (target {z_spacing_mm} mm)")


def _epsilon_suffix(eps):
    """Format epsilon for folder names (must match training script)."""
    s = f"{eps:.6g}".replace(".", "p").replace("-", "m")
    return s


print(f"PSF weight_mode: {opts.weight_mode}, charb_epsilon: {opts.charb_epsilon}")
if opts.weight_mode == 'sliceprofile':
    print(f"Training slice profile: {opts.rf_profile_mat}")

if opts.checkpoint_iter is not None:
    config['iter'] = opts.checkpoint_iter
    print(f"Using user-specified iteration: {opts.checkpoint_iter}")
else:
    print(f"Using iteration from config: {config['iter']}")

print("=== TEST STEP 1: Initialize ROVER B0 Philps TCNN test parameters ===")
print(f"Image size: nx={nx}, ny={ny}, nk={nk}")
print(f"Number of views: nv={nv}")

print(f"\n--- Load image data ---")
print(f"Image path pattern: {img_path}")
batch_size = args.batch_size
print(f"Batch size: batch_size={batch_size}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

print("File list after natural sort:")
print(f"Number of image files found: {len(files_path)}")
print(f"Image file list: {files_path}")

print(f"\n--- Create coordinate grid ---")
X, Y, Z = np.mgrid[0:nx:1, 0:ny:1, 0:nk:1]
x, y, z = X.ravel(), Y.ravel(), Z.ravel()
N = nx * ny * nk
print(f"Coordinate grid: X.shape={X.shape}, Y.shape={Y.shape}, Z.shape={Z.shape}")
print(f"Coordinate vectors: x.shape={x.shape}, y.shape={y.shape}, z.shape={z.shape}")
print(f"Total pixels: N = nx*ny*nk = {nx}*{ny}*{nk} = {N}")

print(f"\n--- Initialize {num_depth_layers}-layer coordinate arrays ---")
coods = [np.zeros((N * nv, 3)) for _ in range(num_depth_layers)]
print(f"coodsarray shape: {[c.shape for c in coods]}")
print("=== TEST STEP 1 done ===\n")


def trans(x, y, z, Affine):
    P = np.array([x, y, z, [1] * x.size])
    x_, y_, z_, _ = np.dot(Affine, P)
    base = np.stack([x_, y_, z_], axis=-1).reshape(-1, 3)

    axis_vec = Affine[0:3, 2]
    norm_axis = axis_vec / np.linalg.norm(axis_vec)
    center_offset = (num_depth_layers - 1) / 2
    offsets = [i - center_offset for i in range(num_depth_layers)]
    result = [base + spa_res * offset * norm_axis for offset in offsets]

    print(f"Depth offsets: {offsets}")
    print(f"Z axis vector norm: {np.linalg.norm(axis_vec)}")
    return result


print(f"=== TEST STEP 2: Coordinate transform and{num_depth_layers}-layer generation ===")
print(f"Spatial resolution: spa_res = {spa_res}")
print(f"Coordinate grid: X.shape={X.shape}, Y.shape={Y.shape}, Z.shape={Z.shape}")
print(f"Coordinate vectors: x.shape={x.shape}, y.shape={y.shape}, z.shape={z.shape}")

center_offset = (num_depth_layers - 1) / 2
offsets = [i - center_offset for i in range(num_depth_layers)]
offset_distances = [offset * spa_res for offset in offsets]
print(f"{num_depth_layers}-layer depth offsets: {offsets} * {spa_res} = {[f'{d:.6f}' for d in offset_distances]}")

for view_num in range(nv):
    print(f"\n--- Processing view {view_num}/{nv}  coordinate transform ---")
    affine_path = affine_paths[view_num]
    print(f"Loading Affine matrix: {affine_path}")
    Affine = np.load(affine_path)
    print(f"Affine matrix shape: {Affine.shape}")

    results = trans(x, y, z, Affine)
    print(f"Coordinate transform done, obtained{num_depth_layers}-layer depth coordinates")

    for i in range(num_depth_layers):
        start_idx = view_num * N
        end_idx = (view_num + 1) * N
        coods[i][start_idx:end_idx, :] = results[i]
        print(f"  depth layer {i+1}: store into coods[{i}][{start_idx}:{end_idx}, :]")

    print(f"View {view_num} coordinate processing done")
print("=== TEST STEP 2 done ===\n")


print("=== TEST STEP 3: Coordinate normalization ===")
coords = coods
all_coords = np.concatenate(coords, axis=0)

axi_x_min = np.min(all_coords[:, 0])
axi_x_max = np.max(all_coords[:, 0])
axi_y_min = np.min(all_coords[:, 1])
axi_y_max = np.max(all_coords[:, 1])
axi_z_min = np.min(all_coords[:, 2])
axi_z_max = np.max(all_coords[:, 2])

print(f"X range: [{axi_x_min:.6f}, {axi_x_max:.6f}]")
print(f"Y range: [{axi_y_min:.6f}, {axi_y_max:.6f}]")
print(f"Z range: [{axi_z_min:.6f}, {axi_z_max:.6f}]")

for i in range(len(coords)):
    print(f"\n--- Normalizing depth layer {i+1}/{len(coords)} ---")
    print(f"Before norm: coords[{i}].shape: {coords[i].shape}")
    print(f"Before norm: coords[{i}] value range: X[{coords[i][:,0].min():.6f}, {coords[i][:,0].max():.6f}], "
          f"Y[{coords[i][:,1].min():.6f}, {coords[i][:,1].max():.6f}], "
          f"Z[{coords[i][:,2].min():.6f}, {coords[i][:,2].max():.6f}]")

    coords[i] = normalization(coords[i], axi_x_min, axi_x_max, axi_y_min, axi_y_max, axi_z_min, axi_z_max)

    print(f"After norm: coords[{i}].shape: {coords[i].shape}")
    print(f"After norm: coords[{i}] value range: X[{coords[i][:,0].min():.6f}, {coords[i][:,0].max():.6f}], "
          f"Y[{coords[i][:,1].min():.6f}, {coords[i][:,1].max():.6f}], "
          f"Z[{coords[i][:,2].min():.6f}, {coords[i][:,2].max():.6f}]")

coods = coords
print("=== STEP 3 done ===\n")


print("=== TEST STEP 4: Create data loader and model ===")
print(f"Create ObservationPoints3D object...")
for i in range(num_depth_layers):
    print(f"  - cood{i+1}.shape: {coods[i].shape}")
print(f"  - batch_size: {batch_size}")

Y_flat = np.zeros([N * nv, 1])  # dummy, not used for inference but required by constructor
O = ObservationPoints3D(*coods, Y_flat, batch_size)
print(f"ObservationPoints3Dobject created")

dataloader = DataLoader(O, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)
print(f"DataLoaderCreate done: batch_size=1, shuffle=True")

print(f"\n--- Set up output directory ---")
output_folder = config['output_folder']
print(f"Output folder from config: {output_folder}")

_charb_eps_suffix = _epsilon_suffix(opts.charb_epsilon)
model_name = os.path.join(
    output_folder
    + '/rover_tcnn_relu_psf_charb_{}_lev{}_r{}_d{}_log{}_nl_{}_br{}_bs{}_lambda{}_w{}_eps{}'.format(
        config['experiment_name'],
        config['n_levels'],
        config['r'],
        config['depth'],
        config['log2_hashmap_size'],
        config['n_features_per_level'],
        config['base_resolution'],
        args.batch_size,
        config['lambda_c'],
        opts.weight_mode,
        _charb_eps_suffix,
    )
)

print(f"Model name: {model_name}")

use_direct_checkpoint_path = opts.checkpoint_path is not None
if use_direct_checkpoint_path:
    model_path = os.path.abspath(opts.checkpoint_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint file not found: {model_path}")
    checkpoint_directory = os.path.dirname(model_path)
    output_directory = os.path.dirname(checkpoint_directory)
    image_directory = os.path.join(output_directory, "images")
    log_directory = os.path.join(output_directory, "logs")
    print(f"Using specified checkpoint path: {model_path}")
    print(f"Output directory: {output_directory}")
else:
    checkpoint_directory = os.path.join(opts.output_path, "outputs", model_name, "checkpoints")
    if not os.path.exists(checkpoint_directory):
        model_name_legacy = model_name.replace("_eps" + _charb_eps_suffix, "")
        checkpoint_directory_legacy = os.path.join(opts.output_path, "outputs", model_name_legacy, "checkpoints")
        if os.path.exists(checkpoint_directory_legacy):
            checkpoint_directory = checkpoint_directory_legacy
            model_name = model_name_legacy
            print(f"Using legacy path (no _eps suffix): {checkpoint_directory}")
        else:
            raise FileNotFoundError(
                f"Checkpoint directory does not exist: {checkpoint_directory}\n"
                f"Tried fallback: {checkpoint_directory_legacy}\n"
                f"Or pass --checkpoint_path /full/path/to/model_XXXXXX.pt to load a specific file."
            )
    image_directory = os.path.join(opts.output_path, "outputs", model_name, "images")
    log_directory = os.path.join(opts.output_path, "outputs", model_name, "logs")
    output_directory = os.path.join(opts.output_path, "outputs", model_name)
    model_path = None

print(f"Using checkpoint directory: {checkpoint_directory}")
print(f"Image directory: {image_directory}")
print(f"Log directory: {log_directory}")
print("=== TEST STEP 4 done ===\n")


print(f"\n--- Create  TCNN model ---")
print(f"Model config: {config}")
print(f"Model args: {args}")

L = config['n_levels']
F = config['n_features_per_level']
T = config['log2_hashmap_size']
N_min = config['base_resolution']
b = args.per_level_scale
N_max = int(N_min * (b ** (L - 1)))

print(f"TCNN HashGrid params -> L={L}, F={F}, T={T}, N_min={N_min}, N_max={N_max}, b={b:.4f}")

field_model = ROVER_TCNN(in_dim=3, out_dim=1, L=L, F=F, T=T, N_min=N_min, N_max=N_max)
field_model = field_model.to(device)
field_model.eval()
print(f"ROVER_TCNN model created and moved to device: {device}")
print("=== TEST STEP 4b done ===\n")


print("=== TEST STEP 5: Load pretrained TCNN model ===")

if use_direct_checkpoint_path:
    print(f"Preparing to load checkpoint: {model_path}")
else:
    if not os.path.exists(checkpoint_directory):
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_directory}")
    checkpoints = sorted([f for f in os.listdir(checkpoint_directory) if f.endswith(".pt")])
    if not checkpoints:
        raise FileNotFoundError(f"No .pt files found in directory {checkpoint_directory} !")
    print("Available checkpoint files:")
    for ckpt in checkpoints:
        print("   -", ckpt)
    if opts.checkpoint_iter is not None:
        target_name = f"model_{opts.checkpoint_iter:06d}.pt"
        if target_name in checkpoints:
            ckpt_name = target_name
        else:
            raise FileNotFoundError(f"Specified checkpoint {target_name} not found in {checkpoint_directory}")
    else:
        ckpt_name = checkpoints[-1]
    model_path = os.path.join(checkpoint_directory, ckpt_name)
    print(f"Preparing to load checkpoint: {model_path}")

state_dict = torch.load(model_path, map_location=device)
print(f"checkpoint contains keys: {list(state_dict.keys())}")

field_model.load_state_dict(state_dict['net'])
print(f"Pretrained TCNN model loaded successfully: {model_path}")
print("=== TEST STEP 5 done ===\n")


print("=== TEST STEP 6: Start inference (per-depth-layer reconstruction) ===")
print(f"Fetch full dataset...")
coordmap, data = dataloader.dataset.getfulldata()
print(f"coordmap contains keys: {list(coordmap.keys())}")

print(f"\n--- Initialize {num_depth_layers}-layer result arrays ---")
f_list = [np.zeros((nx * ny * nk, 1)) for _ in range(num_depth_layers)]
for i, f in enumerate(f_list):
    print(f"  f_list[{i}].shape: {f.shape}")

print(f"\n--- Per-slice inference ---")
print(f"Total slices: nk = {nk}")
print(f"Pixels per slice: nx * ny = {nx} * {ny} = {nx * ny}")

for slc in range(nk):
    print(f"\n--- Processing slice {slc+1}/{nk} ---")
    for i in range(num_depth_layers):
        key = f"coord{i+1}"
        coords = coordmap[key][slc * nx * ny : (slc + 1) * nx * ny, :]
        print(f"  depth layer {i+1}: coords.shape = {coords.shape}")

        coords_t = coords.to(device)
        with torch.no_grad():
            f_hat = field_model(coords_t).cpu().numpy()
        print(f"    Inference done, result shape: {f_hat.shape}")

        start_idx = slc * nx * ny
        end_idx = (slc + 1) * nx * ny
        f_list[i][start_idx:end_idx] = f_hat
        print(f"    store into f_list[{i}][{start_idx}:{end_idx}]")

    if (slc + 1) % 10 == 0 or slc == nk - 1:
        print(f"  slice {slc+1}/{nk}  processing done")
print("=== TEST STEP 6 done ===\n")


print("=== TEST STEP 7: Reshape and stack results ===")
print(f"Reshape {num_depth_layers}-layer result arrays...")
f_list = [f.reshape((nx, ny, nk)) for f in f_list]
for i, f in enumerate(f_list):
    print(f"  f_list[{i}].shape after reshape: {f.shape}")

print(f"\n--- Stack into final result ---")
print(f"Final result shape: (nx, ny, {num_depth_layers} * nk) = ({nx}, {ny}, {num_depth_layers * nk})")
result = np.zeros((nx, ny, num_depth_layers * nk))
print(f"Initialize result array: shape = {result.shape}")

for i in range(num_depth_layers):
    print(f"\n--- Processing depth layer {i+1}/{num_depth_layers} ---")
    for slc in range(nk):
        target_idx = num_depth_layers * slc + i
        print(f"  slice {slc+1}: f_list[{i}][:, :, {slc}] store into result[:, :, {target_idx}]")
        result[:, :, target_idx] = f_list[i][:, :, slc]

print(f"\nFinal result array shape: {result.shape}")
print(f"Final result value range: [{result.min():.6f}, {result.max():.6f}]")
print("=== TEST STEP 7 done ===\n")


print(f"\n=== Array shape analysis ===")
print(f"Original image size: nx={nx}, ny={ny}, nk={nk}")
print(f"f_list[0].shape: {f_list[0].shape} (single depth-layer reconstruction)")
print(f"result.shape: {result.shape} (stacked result of all depth layers)")

print(f"\n--- Preparing to save as NIfTI ---")
print(f"Using output affine (z-spacing={out_z_spacing:.6f} mm):\n{output_affine}")
recon = nib.Nifti1Image(result, affine=output_affine)
# Ensure the NIfTI header pixdim along z reports the correct slice thickness.
recon.header.set_zooms((
    float(np.linalg.norm(output_affine[:3, 0])),
    float(np.linalg.norm(output_affine[:3, 1])),
    float(out_z_spacing),
))
print(f"NIfTI header zooms (pixdim): {recon.header.get_zooms()}")

print("=== TEST STEP 8: Save results ===")
test_output_dir = os.path.join(output_directory, 'test-output')
if not os.path.exists(test_output_dir):
    os.makedirs(test_output_dir)
    print(f"Created test output directory: {test_output_dir}")
else:
    print(f"Test output directory already exists: {test_output_dir}")

if use_direct_checkpoint_path:
    ckpt_basename = os.path.splitext(os.path.basename(model_path))[0]
    output_filename = f'rover_b0_hash_philps_tcnn_relu_charb_tv_{ckpt_basename}_z{z_spacing_mm:.5f}.nii'
else:
    output_filename = (
        f'rover_b0_hash_philps_tcnn_relu_psf_charb_{config["r"]}w_{config["depth"]}d_'
        f'{config["n_levels"]}lev_{config["n_features_per_level"]}nplev_'
        f'{config["log2_hashmap_size"]}hasize_{config["base_resolution"]}bs_'
        f'{config["lambda_c"]}lambda_{config["iter"]}iter_w{opts.weight_mode}_eps{_charb_eps_suffix}_'
        f'z{z_spacing_mm:.5f}_tv.nii'
    )
output_path = os.path.join(test_output_dir, output_filename)

print(f"Output filename: {output_filename}")
print(f"Full output path: {output_path}")

print(f"\nStart saving NIfTI file...")
nib.save(recon, output_path)
print(f"Result saved successfully: {output_path}")

print(f"\n=== Test done ===")
end_time = time.time()
execution_time = end_time - start_time
print("Total time: ", execution_time, "seconds")
print("=== TEST STEP 8 done ===\n")
