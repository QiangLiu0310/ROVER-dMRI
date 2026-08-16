#!/usr/bin/env python
# coding: utf-8
"""
ROVER B0 TCNN+ReLU+PSF+Charbonnier+TV with automatic input geometry.

The in-plane depth spacing is read from Affine_nii_1.npy, and the number of
depth layers is configurable with --num_depth_layers.
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
from scipy.io import loadmat
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

import sys
sys.path.append("fda")
from data_objects.data_objects_bumonkey import ObservationPoints3D
from utils import get_config, normalization, prepare_sub_folder
from util_args_rover_b0_v9_lr1e4 import get_args

import glob
import re
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


log_filename = f"rover_b0_training_tcnn_relu_psf_charb_tv_tbwp4_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.out"


class Logger:
    """Redirect stdout to both terminal and log file."""
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

print(f"=== ROVER B0 TCNN+ReLU+PSF+Charbonnier+TV (TBWP4 slice profile option) training start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
print(f"Log file: {log_filename}")
print("=" * 80)


# ---------------------------------------------------------------------------
# TCNN model: ROVER_TCNN
# ---------------------------------------------------------------------------
class ROVER_TCNN(torch.nn.Module):
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


# ---------------------------------------------------------------------------
# Robust loss and PSF weight helpers
# ---------------------------------------------------------------------------
def charbonnier_loss(pred, target, epsilon=1e-3):
    """Robust Charbonnier loss to suppress Poisson hash-collision noise."""
    return torch.mean(torch.sqrt((pred - target) ** 2 + epsilon ** 2))


def tv_loss_random_fd(field_model, num_samples, device, eps=1e-3):
    """
    TCNN-safe TV approximation on random coords in [0,1]^3 using forward finite differences.

    This avoids a fixed lattice and only uses forward passes:
        TV ~= mean(|f(x + eps * e_d) - f(x)| / eps) over random x and axes d.
    Because the loss depends only on model outputs, it requires a single backward pass.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}")
    if not (0.0 < eps < 1.0):
        raise ValueError(f"eps must be in (0, 1), got {eps}")

    # Sample from [0, 1 - eps] so x + eps stays inside the normalized domain.
    coords = torch.rand(num_samples, 3, device=device, dtype=torch.float32) * (1.0 - eps)
    pred0 = field_model(coords).squeeze(-1)

    tv = 0.0
    for axis in range(coords.shape[-1]):
        coords_shift = coords.clone()
        coords_shift[:, axis] = coords_shift[:, axis] + eps
        pred1 = field_model(coords_shift).squeeze(-1)
        tv = tv + torch.abs(pred1 - pred0).mean() / eps

    return tv


def get_gaussian_weights(M, device):
    """
    Generates a 1D Gaussian kernel for the thick-slice PSF.
    sigma = M/4 so ~95% of signal falls within the slice extent.
    Weights are normalised to sum to 1.
    """
    sigma = M / 4.0
    grid = torch.arange(M, dtype=torch.float32, device=device) - (M - 1) / 2.0
    weights = torch.exp(-0.5 * (grid / sigma) ** 2)
    return weights / weights.sum()  # sum == 1


def get_equal_weights(M, device):
    """
    Generates uniform weights across all depth layers.
    Each weight = 1/M, so weights sum to 1.
    """
    return torch.ones(M, dtype=torch.float32, device=device) / M  # sum == 1


def get_psf_weights_from_slice_profile(rf_profile_path, num_depth_layers, spa_res, device,
                                       profile_var='Mxy_sinc'):
    """
    Load slice profile from .mat (z_mm, <profile_var>), sample at the M depth-layer
    positions (spa_res * (i - (M-1)/2) mm), then normalize to sum 1.

    profile_var selects which simulated profile to use:
        'Mxy_sinc' - excitation-only profile (90 deg), widest/flattest
        'Mse_sinc' - spin-echo product (excitation * refocusing efficiency)
        'Mref_sinc'- refocusing efficiency only
    Returns Tensor of shape [num_depth_layers], sum=1, on device.
    """
    data = loadmat(rf_profile_path)
    z_mm = np.asarray(data["z_mm"]).ravel()
    if profile_var not in data:
        raise KeyError(
            f"Profile variable '{profile_var}' not found in {rf_profile_path}. "
            f"Available: {[k for k in data if not k.startswith('__')]}")
    profile = np.asarray(data[profile_var])
    if profile.ndim > 1:
        profile = profile[:, 0].ravel()
    else:
        profile = profile.ravel()

    center_offset = (num_depth_layers - 1) / 2.0
    z_layer_mm = spa_res * (np.arange(num_depth_layers, dtype=np.float64) - center_offset)
    profile_at_layers = np.interp(z_layer_mm, z_mm, profile)
    profile_at_layers = np.maximum(profile_at_layers, 0.0)
    total = profile_at_layers.sum()
    if total <= 0:
        raise ValueError(f"Slice profile sum is non-positive after interpolation. Check z_mm and {profile_var}.")
    weights = profile_at_layers / total
    weights_t = torch.from_numpy(weights.astype(np.float32)).to(device)
    assert torch.isclose(weights_t.sum(), torch.ones(1, device=device), atol=1e-5), \
        f"PSF weights do not sum to 1! Got {weights_t.sum().item():.8f}"

    print(f"PSF weighting mode : SLICE PROFILE (from {rf_profile_path}: z_mm, {profile_var})")
    print(f"Number of layers   : {num_depth_layers}")
    print(f"spa_res (mm)       : {spa_res}")
    print(f"Layer positions (mm): {z_layer_mm.tolist()}")
    print(f"Weights            : {weights_t.cpu().numpy()}")
    print(f"Sum of weights     : {weights_t.sum().item():.8f}  (must be 1.0)")
    return weights_t


def get_psf_weights(M, device, weight_mode='gaussian'):
    """
    Returns normalised PSF weights for depth-layer integration.

    Args:
        M           : number of depth layers
        device      : torch device
        weight_mode : 'gaussian'  – Gaussian PSF weighting (sum=1)
                      'equal'     – uniform weighting (sum=1)

    Returns:
        Tensor of shape [M], guaranteed to sum to 1.
    """
    if weight_mode == 'gaussian':
        weights = get_gaussian_weights(M, device)
    elif weight_mode == 'equal':
        weights = get_equal_weights(M, device)
    else:
        raise ValueError(f"Unknown weight_mode '{weight_mode}'. Choose 'gaussian' or 'equal'.")

    # Verify normalisation (should always pass, but guards against edge cases)
    assert torch.isclose(weights.sum(), torch.ones(1, device=device), atol=1e-5), \
        f"PSF weights do not sum to 1! Got {weights.sum().item():.8f}"

    print(f"PSF weighting mode : {weight_mode.upper()}")
    print(f"Number of layers   : {M}")
    print(f"Weights            : {weights.cpu().numpy()}")
    print(f"Sum of weights     : {weights.sum().item():.8f}  (must be 1.0)")
    return weights


# ---------------------------------------------------------------------------
# Config / args (same as rover_b0_hash_prisma_v3_tcnn_relu_charb_tv.py + sliceprofile)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/data_v9.yaml',
                    help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='output',
                    help='Outputs path.')
parser.add_argument('--input_dir', type=str, required=True,
                    help='Directory containing imgs_nii_<i>.npy and Affine_nii_<i>.npy files.')
parser.add_argument('--num_depth_layers', type=positive_int, default=8,
                    help='Number of reconstructed depth layers. Default: 8.')
parser.add_argument('--weight_mode', type=str, default='sliceprofile',
                    choices=['gaussian', 'equal', 'sliceprofile'],
                    help="Depth-layer PSF weighting: 'gaussian', 'equal', or 'sliceprofile' (from .mat).")
parser.add_argument('--charb_epsilon', type=float, default=1e-1,
                    help="Charbonnier loss epsilon. Larger = more L2-like (e.g. 0.1). Default 1e-1.")
parser.add_argument('--tv_fd_eps', type=float, default=1e-3,
                    help="Finite-difference step for random TV on normalized coords. Default 1e-3.")
parser.add_argument('--tv_num_samples', type=int, default=0,
                    help="Number of random coords for TV. 0 uses batch_size//2 to mirror the v9 random-TV sample count.")
parser.add_argument('--rf_profile_mat', type=str,
                    default=os.path.join(project_root, 'rf_slice_profile_mse_sinc.mat'),
                    help="Path to .mat with z_mm and the profile variable (used when weight_mode='sliceprofile'). Default: rf_slice_profile_mse_sinc.mat.")
parser.add_argument('--profile_var', type=str, default='Mxy_sinc',
                    choices=['Mxy_sinc', 'Mse_sinc', 'Mref_sinc'],
                    help="Which simulated profile to load when weight_mode='sliceprofile': "
                         "'Mxy_sinc' = excitation-only (widest, default), "
                         "'Mse_sinc' = spin-echo product, 'Mref_sinc' = refocusing only.")
opts, _ = parser.parse_known_args()


def _epsilon_suffix(eps):
    """Format epsilon for use in folder names (e.g. 0.1 -> '0p1', 1e-3 -> '0p001')."""
    s = f"{eps:.6g}".replace(".", "p").replace("-", "m")
    return s
args = get_args(cmd=False)
config = get_config(opts.config)

max_iter = args.num_epochs

nv = args.view_num

input_dir = os.path.abspath(os.path.expanduser(opts.input_dir))
if not os.path.isdir(input_dir):
    raise NotADirectoryError(f"Input directory does not exist or is not a directory: {input_dir}")

img_path = os.path.join(input_dir, 'imgs_nii_*.npy')
print('Load image: {}'.format(img_path))
batch_size = args.batch_size

hyper_params = {"lambda_c": config['lambda_c']}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

files_path = sorted(glob.glob(img_path), key=natural_sort_key)
print(files_path)
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

# Images are transposed from [ny, nx, nk] to [nx, ny, nk] when loaded below.
ny, nx, nk = image_shapes[0]
print(f"Inferred image size after transpose: nx={nx}, ny={ny}, nk={nk}")

affine_paths = [
    os.path.join(input_dir, f'Affine_nii_{view_num + 1}.npy')
    for view_num in range(nv)
]
missing_affine_paths = [path for path in affine_paths if not os.path.isfile(path)]
if missing_affine_paths:
    raise FileNotFoundError(
        "Missing affine file(s): " + ", ".join(missing_affine_paths)
    )

print("=== ROVER B0 ===")
affines, affine_spacings = load_and_validate_affines(affine_paths)
reference_affine = affines[0]
x_voxel_size, y_voxel_size, z_voxel_size = affine_spacings[0]
spa_res = float(x_voxel_size)
num_depth_layers = opts.num_depth_layers
depth_ratio = z_voxel_size / spa_res

print(f"Voxel sizes from the view-0 Affine matrix:")
print(f"  X voxel size: {x_voxel_size:.6f} mm")
print(f"  Y voxel size: {y_voxel_size:.6f} mm")
print(f"  Z voxel size: {z_voxel_size:.6f} mm")
print(f"Reference Affine matrix:\n{reference_affine}")
print(f"Depth ratio: {z_voxel_size:.6f} / {spa_res:.6f} = {depth_ratio:.6f}")
print(f"Number of depth layers (command-line input): {num_depth_layers}")
print(f"Spatial resolution (from affine X-axis): {spa_res:.6f} mm")

X, Y, Z = np.mgrid[0:nx:1, 0:ny:1, 0:nk:1]
x, y, z = X.ravel(), Y.ravel(), Z.ravel()
N = nx * ny * nk
Y_flat = np.zeros([N * nv, 1])

print("=== STEP 1: Load ROVER B0 image data ===")
print(f"Image path pattern: {img_path}")
print(f"Number of image files found: {len(files_path)}")
print(f"Image file list: {files_path}")
print(f"Image size: nx={nx}, ny={ny}, nk={nk}")
print(f"Number of views: nv={nv}")
print(f"Total pixels: N = nx*ny*nk = {nx}*{ny}*{nk} = {N}")
print(f"Y_flat array shape: [{N * nv}, 1] = [{N * nv}, 1]")

for view_num in range(nv):
    print(f"\n--- Processing view {view_num}/{nv} ---")
    print(f"Loading file: {files_path[view_num]}")
    Ydata = np.load(files_path[view_num])
    print(f"Original Ydata shape: {Ydata.shape}")
    Ydata = np.transpose(Ydata, [1, 0, 2])
    print(f"Transposed Ydata shape: {Ydata.shape}")
    print(f"Ydata value range: [{Ydata.min():.6f}, {Ydata.max():.6f}]")

    start_idx = view_num * N
    end_idx = (view_num + 1) * N
    print(f"store into Y_flat[{start_idx}:{end_idx}, :]")
    Y_flat[start_idx:end_idx, :] = Ydata.reshape(-1)[:, np.newaxis]
    print(f"View {view_num} processing done")

print(f"\nY_flat value range before normalization: [{Y_flat.min():.6f}, {Y_flat.max():.6f}]")
Y_flat_min = Y_flat.min()
Y_flat_max = Y_flat.max()
Y_flat = (Y_flat - Y_flat_min) / (Y_flat_max - Y_flat_min)
print(f"Y_flat value range after min-max normalization: [{Y_flat.min():.6f}, {Y_flat.max():.6f}]")
print("=== STEP 1 done ===\n")

coods = [np.zeros((N * nv, 3)) for _ in range(num_depth_layers)]


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


print(f"=== STEP 2: Generate {num_depth_layers}-layer depth coordinates ===")
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
    print(f"Affine matrix:\n{Affine}")

    results = trans(x, y, z, Affine)
    print(f"Coordinate transform done, obtained{num_depth_layers}-layer depth coordinates")

    for i in range(num_depth_layers):
        start_idx = view_num * N
        end_idx = (view_num + 1) * N
        coods[i][start_idx:end_idx, :] = results[i]
        print(f"  depth layer {i + 1}: store into coods[{i}][{start_idx}:{end_idx}, :]")

    print(f"View {view_num} coordinate processing done")

coords = coods

all_coords = np.concatenate(coords, axis=0)

axi_x_min = np.min(all_coords[:, 0])
axi_x_max = np.max(all_coords[:, 0])
axi_y_min = np.min(all_coords[:, 1])
axi_y_max = np.max(all_coords[:, 1])
axi_z_min = np.min(all_coords[:, 2])
axi_z_max = np.max(all_coords[:, 2])

print("=== STEP 3: Coordinate normalization ===")
print(f"Shape after concatenating all coordinates: all_coords.shape = {all_coords.shape}")
print(f"X range: [{axi_x_min:.6f}, {axi_x_max:.6f}]")
print(f"Y range: [{axi_y_min:.6f}, {axi_y_max:.6f}]")
print(f"Z range: [{axi_z_min:.6f}, {axi_z_max:.6f}]")

for i in range(len(coords)):
    print(f"\n--- Normalizing depth layer {i + 1}/{len(coords)} ---")
    print(f"Before norm: coords[{i}].shape: {coords[i].shape}")
    print(f"Before norm: coords[{i}] value range: X[{coords[i][:, 0].min():.6f}, {coords[i][:, 0].max():.6f}], "
          f"Y[{coords[i][:, 1].min():.6f}, {coords[i][:, 1].max():.6f}], "
          f"Z[{coords[i][:, 2].min():.6f}, {coords[i][:, 2].max():.6f}]")

    coords[i] = normalization(coords[i], axi_x_min, axi_x_max, axi_y_min, axi_y_max, axi_z_min, axi_z_max)

    print(f"After norm: coords[{i}].shape: {coords[i].shape}")
    print(f"After norm: coords[{i}] value range: X[{coords[i][:, 0].min():.6f}, {coords[i][:, 0].max():.6f}], "
          f"Y[{coords[i][:, 1].min():.6f}, {coords[i][:, 1].max():.6f}], "
          f"Z[{coords[i][:, 2].min():.6f}, {coords[i][:, 2].max():.6f}]")

coods = coords
print("=== STEP 3 done ===\n")


print("=== STEP 4: Create data loader and model ===")
print(f"Create ObservationPoints3D object...")
for i in range(num_depth_layers):
    print(f"  - cood{i + 1}.shape: {coods[i].shape}")
print(f"  - Y_flat.shape: {Y_flat.shape}")
print(f"  - batch_size: {batch_size}")

O = ObservationPoints3D(*coods, Y_flat, batch_size)
print(f"ObservationPoints3Dobject created")

dataloader = DataLoader(O, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)
print(f"DataLoaderCreate done: batch_size=1, shuffle=True")

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

field_model = ROVER_TCNN(in_dim=3, out_dim=1, L=L, F=F, T=T, N_min=N_min, N_max=N_max).to(device)
print(f"ROVER_TCNN model created")
print(f"Number of model parameters: {sum(p.numel() for p in field_model.parameters())}")

print(f"\n--- Create optimizer (Adam) + AMP scaler ---")
optim = torch.optim.Adam(field_model.parameters(), lr=args.learning_rate)
scaler = GradScaler()
print(f"Adam optimizer created, learning rate: {args.learning_rate}")
print("=== STEP 4 done ===\n")

print("=== STEP 5: Set up output directory ===")
output_folder = config['output_folder']
print(f"Output folder from config: {output_folder}")

# Include weight_mode and Charbonnier epsilon in the model name so runs are distinguishable
_charb_eps_suffix = _epsilon_suffix(opts.charb_epsilon)
# For sliceprofile mode, tag which profile variable was used so runs don't collide
_wmode_tag = opts.weight_mode
if opts.weight_mode == 'sliceprofile':
    _wmode_tag = 'sliceprofile_' + opts.profile_var
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
        _wmode_tag,                # e.g. 'sliceprofile_Mxy_sinc', 'equal', 'gaussian'
        _charb_eps_suffix,         # e.g. 0p1 for 0.1, 0p001 for 1e-3
    )
)

print(f"Model name: {model_name}")

output_directory = os.path.join(opts.output_path + "/outputs", model_name)
print(f"Output directory path: {output_directory}")

print(f"Creating subdirectories...")
checkpoint_directory, image_directory, log_directory = prepare_sub_folder(output_directory)

print('=== Output directory setup done ===')
print('Output directory: {}'.format(output_directory))
print('Checkpoint directory: {}'.format(checkpoint_directory))
print('Image directory: {}'.format(image_directory))
print('Log directory: {}'.format(log_directory))
print("=== STEP 5 done ===\n")

print("=== STEP 6: Start TCNN+PSF+Charbonnier+TV training ===")
writer = SummaryWriter(log_directory)

lambda_c = float(hyper_params["lambda_c"])
tv_num_samples = opts.tv_num_samples if opts.tv_num_samples > 0 else max(1, batch_size // 2)

print(f"\n--- Training parameters ---")
print(f"Max iterations: max_iter = {max_iter}")
print(f"Hyperparameters: lambda_c = {lambda_c} (TV weight; 0 = TV disabled)")
print(f"Charbonnier epsilon = {opts.charb_epsilon} (larger => more L2-like)")
print(f"TV prior: random finite-difference on {tv_num_samples} coords, eps={opts.tv_fd_eps}")
print(f"Device: {device}")
print(f"DataLoader length: {len(dataloader)}")

coord_keys = [f"coord{i + 1}" for i in range(num_depth_layers)]

# *** Build PSF weights: gaussian / equal / sliceprofile (from rf_slice_profile_mse_sinc.mat) ***
print(f"\n--- PSF weights init (weight_mode='{opts.weight_mode}') ---")
if opts.weight_mode == 'sliceprofile':
    psf_w = get_psf_weights_from_slice_profile(opts.rf_profile_mat, num_depth_layers, spa_res, device,
                                               profile_var=opts.profile_var)
    # Print per-layer weights for visibility
    print("\n--- PSF weights (used in training) ---")
    w_np = psf_w.cpu().numpy()
    for i in range(num_depth_layers):
        z_mm_i = spa_res * (i - (num_depth_layers - 1) / 2.0)
        print(f"  layer {i+1:2d}  z = {z_mm_i:+.2f} mm   weight = {w_np[i]:.6f}")
    print(f"  Sum = {w_np.sum():.8f}")
    print("---\n")
else:
    psf_w = get_psf_weights(num_depth_layers, device, weight_mode=opts.weight_mode)  # shape [M], sum=1

loss_log_interval = 10  # log loss every N steps for final plot (same as resume script)
steps_log, loss_log = [], []

total_steps = 0
print("\nStart training TCNN+PSF+Charbonnier+TV...")
print("=" * 50)

for epoch in range(max_iter):
    for step, (model_input, data) in enumerate(dataloader):
        model_input = {key: value.to(device) for key, value in model_input.items()}
        data = {key: value.to(device) for key, value in data.items()}

        optim.zero_grad(set_to_none=True)

        with autocast():
            # Forward through all depth planes
            preds = []
            for idx in range(num_depth_layers):
                key = coord_keys[idx]
                out = field_model(model_input[key])  # [1, N, 1] or [N, 1]
                preds.append(out.squeeze(0))          # [N, 1]

            # Stack to [M, N, 1] then apply PSF weights (sum=1) along depth axis
            preds_stacked = torch.stack(preds, dim=0)          # [M, N, 1]
            w = psf_w.view(num_depth_layers, 1, 1)             # [M, 1, 1]
            combined_out = torch.sum(preds_stacked * w, dim=0) # [N, 1]

            yvals = data["yvals"]
            loss_charb = charbonnier_loss(combined_out, yvals, epsilon=opts.charb_epsilon)

            # TCNN-safe TV approximation on random coords via forward finite differences.
            if lambda_c != 0:
                tv_loss = tv_loss_random_fd(field_model, tv_num_samples, device, eps=opts.tv_fd_eps)
                loss = loss_charb + lambda_c * tv_loss
            else:
                loss = loss_charb

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        if (epoch + 1) % args.image_save_iter == 0 and step == 0:
            pt_name = os.path.join(checkpoint_directory, f"model_{epoch+1:06d}.pt")
            torch.save(
                {
                    "net": field_model.state_dict(),
                    "opt": optim.state_dict(),
                },
                pt_name,
            )

        if epoch == 0 or (epoch + 1) % 500 == 0 and step == 0:
            writer.add_scalar('train_loss', loss, epoch + 1)
            if lambda_c != 0:
                writer.add_scalar('train_loss_charb', loss_charb.item(), epoch + 1)
                writer.add_scalar('train_loss_tv', tv_loss.item(), epoch + 1)

        if total_steps % 100 == 0:
            if lambda_c != 0:
                print(f"[epoch {epoch+1}/{max_iter}, step {step}] total_steps={total_steps}, loss={loss.item():.6e} (charb={loss_charb.item():.6e}, tv={tv_loss.item():.6e})")
            else:
                print(f"[epoch {epoch+1}/{max_iter}, step {step}] total_steps={total_steps}, loss={loss.item():.6e}")

        total_steps += 1
        if total_steps % loss_log_interval == 0:
            steps_log.append(total_steps)
            loss_log.append(loss.item())

if steps_log and loss_log:
    plt.figure()
    plt.plot(steps_log, loss_log, 'b-')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training loss (Charbonnier+TV)')
    plt.grid(True)
    loss_plot_path = os.path.join(checkpoint_directory, 'loss_plot.png')
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()
    print(f"Saved loss plot: {loss_plot_path}")

print("=" * 50)
print("TCNN+PSF+Charbonnier+TV training done!")
print("=== STEP 6 done ===\n")
