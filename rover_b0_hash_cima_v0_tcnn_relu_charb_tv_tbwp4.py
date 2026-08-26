#!/usr/bin/env python
# coding: utf-8
"""
ROVER B0 TCNN+ReLU+PSF+Charbonnier+TV with optional TBWP4 slice profile.

Same parameters and settings as rover_b0_hash_prisma_v3_tcnn_relu_charb_tv.py
(num_depth_layers=13, spa_res=0.78125, Prisma paths, config qiang_data_v11).
Adds weight_mode='sliceprofile' to load depth weights from rf_profile_tbwp4.mat
(z_mm, Mse_sinc). Profile figure: rf_profile_tbwp4.png.
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
from util_args_rover_b0_v18_lr1e4 import get_args

import glob
import re
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tinycudann as tcnn


def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


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


def get_psf_weights_from_slice_profile(rf_profile_path, num_depth_layers, spa_res, device):
    """
    Load slice profile from .mat (z_mm, Mse_sinc), sample at the M depth-layer
    positions (spa_res * (i - (M-1)/2) mm), then normalize to sum 1.
    Returns Tensor of shape [num_depth_layers], sum=1, on device.
    """
    data = loadmat(rf_profile_path)
    z_mm = np.asarray(data["z_mm"]).ravel()
    Mse_sinc = np.asarray(data["Mse_sinc"])
    if Mse_sinc.ndim > 1:
        Mse_sinc = Mse_sinc[:, 0].ravel()
    else:
        Mse_sinc = Mse_sinc.ravel()

    center_offset = (num_depth_layers - 1) / 2.0
    z_layer_mm = spa_res * (np.arange(num_depth_layers, dtype=np.float64) - center_offset)
    profile_at_layers = np.interp(z_layer_mm, z_mm, Mse_sinc)
    profile_at_layers = np.maximum(profile_at_layers, 0.0)
    total = profile_at_layers.sum()
    if total <= 0:
        raise ValueError("Slice profile sum is non-positive after interpolation. Check z_mm and Mse_sinc.")
    weights = profile_at_layers / total
    weights_t = torch.from_numpy(weights.astype(np.float32)).to(device)
    assert torch.isclose(weights_t.sum(), torch.ones(1, device=device), atol=1e-5), \
        f"PSF weights do not sum to 1! Got {weights_t.sum().item():.8f}"

    print(f"PSF weighting mode : SLICE PROFILE (from {rf_profile_path}: z_mm, Mse_sinc)")
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
parser.add_argument('--config', type=str, default='configs/qiang_data_v10.yaml',
                    help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='output',
                    help='Outputs path.')
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
                    default=os.path.join(project_root, 'rf_profile_tbwp4.mat'),
                    help="Path to .mat with z_mm and Mse_sinc (used when weight_mode='sliceprofile'). Default: rf_profile_tbwp4.mat.")
opts = parser.parse_args()


def _epsilon_suffix(eps):
    """Format epsilon for use in folder names (e.g. 0.1 -> '0p1', 1e-3 -> '0p001')."""
    s = f"{eps:.6g}".replace(".", "p").replace("-", "m")
    return s
args = get_args()
config = get_config(opts.config)

max_iter = args.num_epochs

nx, ny, nk = args.img_size
nv = args.view_num

print('Load image: {}'.format(args.img_path))
img_path = args.img_path
batch_size = args.batch_size

hyper_params = {"lambda_c": config['lambda_c']}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

files_path = sorted(glob.glob(img_path), key=natural_sort_key)
print(files_path)

print("=== 自动获取ROVER B0空间分辨率和深度坐标数量 ===")
affine_path = '/scratch/home/ql087/data_bwh/Cima_data/preprocess_rover_nii/Affine_nii_1.npy'
if os.path.exists(affine_path):
    reference_affine = np.load(affine_path)
    x_voxel_size = np.sqrt(np.sum(reference_affine[:3, 0] ** 2))
    y_voxel_size = np.sqrt(np.sum(reference_affine[:3, 1] ** 2))
    z_voxel_size = np.sqrt(np.sum(reference_affine[:3, 2] ** 2))

    print(f"从视角0的Affine矩阵获取体素尺寸:")
    print(f"  X方向体素尺寸: {x_voxel_size:.6f} mm")
    print(f"  Y方向体素尺寸: {y_voxel_size:.6f} mm")
    print(f"  Z方向体素尺寸: {z_voxel_size:.6f} mm")
    print(f"参考Affine矩阵:\n{reference_affine}")

    depth_ratio = z_voxel_size / x_voxel_size
    num_depth_layers = 8
    spa_res = 0.7484

    print(f"深度比例计算: {z_voxel_size:.6f} / {x_voxel_size:.6f} = {depth_ratio:.6f}")
    print(f"深度坐标数量: ceil({depth_ratio:.6f}) = {num_depth_layers}")
    print(f"空间分辨率: {spa_res:.6f} mm")
else:
    spa_res = 0.7484
    num_depth_layers = 8
    print(f"无法加载参考Affine矩阵，使用默认值:")
    print(f"  空间分辨率: {spa_res:.6f} mm")
    print(f"  深度坐标数量: {num_depth_layers}")

X, Y, Z = np.mgrid[0:nx:1, 0:ny:1, 0:nk:1]
x, y, z = X.ravel(), Y.ravel(), Z.ravel()
N = nx * ny * nk
Y_flat = np.zeros([N * nv, 1])

print("=== STEP 1: 加载ROVER B0图像数据 ===")
print(f"图像路径模式: {img_path}")
print(f"找到的图像文件数量: {len(files_path)}")
print(f"图像文件列表: {files_path}")
print(f"图像尺寸: nx={nx}, ny={ny}, nk={nk}")
print(f"视角数量: nv={nv}")
print(f"总像素数: N = nx*ny*nk = {nx}*{ny}*{nk} = {N}")
print(f"Y_flat数组形状: [{N * nv}, 1] = [{N * nv}, 1]")

for view_num in range(nv):
    print(f"\n--- 处理视角 {view_num}/{nv} ---")
    print(f"加载文件: {files_path[view_num]}")
    Ydata = np.load(files_path[view_num])
    print(f"原始Ydata形状: {Ydata.shape}")
    Ydata = np.transpose(Ydata, [1, 0, 2])
    print(f"转置后Ydata形状: {Ydata.shape}")
    print(f"Ydata数据范围: [{Ydata.min():.6f}, {Ydata.max():.6f}]")

    start_idx = view_num * N
    end_idx = (view_num + 1) * N
    print(f"存储到Y_flat[{start_idx}:{end_idx}, :]")
    Y_flat[start_idx:end_idx, :] = Ydata.reshape(-1)[:, np.newaxis]
    print(f"视角{view_num}处理完成")

print(f"\nY_flat归一化前数据范围: [{Y_flat.min():.6f}, {Y_flat.max():.6f}]")
Y_flat_min = Y_flat.min()
Y_flat_max = Y_flat.max()
Y_flat = (Y_flat - Y_flat_min) / (Y_flat_max - Y_flat_min)
print(f"Y_flat线性拉伸归一化后数据范围: [{Y_flat.min():.6f}, {Y_flat.max():.6f}]")
print("=== STEP 1 完成 ===\n")

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

    print(f"深度偏移: {offsets}")
    print(f"Z轴向量长度: {np.linalg.norm(axis_vec)}")
    return result


print(f"=== STEP 2: 生成{num_depth_layers}层深度坐标 ===")
print(f"空间分辨率: spa_res = {spa_res}")
print(f"坐标网格: X.shape={X.shape}, Y.shape={Y.shape}, Z.shape={Z.shape}")
print(f"坐标向量: x.shape={x.shape}, y.shape={y.shape}, z.shape={z.shape}")

center_offset = (num_depth_layers - 1) / 2
offsets = [i - center_offset for i in range(num_depth_layers)]
offset_distances = [offset * spa_res for offset in offsets]
print(f"{num_depth_layers}层深度偏移: {offsets} * {spa_res} = {[f'{d:.6f}' for d in offset_distances]}")

for view_num in range(nv):
    print(f"\n--- 处理视角 {view_num}/{nv} 的坐标变换 ---")
    affine_path = f'/scratch/home/ql087/data_bwh/Cima_data/preprocess_rover_nii/Affine_nii_{view_num + 1}.npy'
    print(f"加载Affine矩阵: {affine_path}")
    Affine = np.load(affine_path)
    print(f"Affine矩阵形状: {Affine.shape}")
    print(f"Affine矩阵:\n{Affine}")

    results = trans(x, y, z, Affine)
    print(f"坐标变换完成，得到{num_depth_layers}层深度坐标")

    for i in range(num_depth_layers):
        start_idx = view_num * N
        end_idx = (view_num + 1) * N
        coods[i][start_idx:end_idx, :] = results[i]
        print(f"  深度层{i + 1}: 存储到coods[{i}][{start_idx}:{end_idx}, :]")

    print(f"视角{view_num}坐标处理完成")

coords = coods

all_coords = np.concatenate(coords, axis=0)

axi_x_min = np.min(all_coords[:, 0])
axi_x_max = np.max(all_coords[:, 0])
axi_y_min = np.min(all_coords[:, 1])
axi_y_max = np.max(all_coords[:, 1])
axi_z_min = np.min(all_coords[:, 2])
axi_z_max = np.max(all_coords[:, 2])

print("=== STEP 3: 坐标归一化 ===")
print(f"合并所有坐标后的形状: all_coords.shape = {all_coords.shape}")
print(f"X轴范围: [{axi_x_min:.6f}, {axi_x_max:.6f}]")
print(f"Y轴范围: [{axi_y_min:.6f}, {axi_y_max:.6f}]")
print(f"Z轴范围: [{axi_z_min:.6f}, {axi_z_max:.6f}]")

for i in range(len(coords)):
    print(f"\n--- 归一化深度层 {i + 1}/{len(coords)} ---")
    print(f"归一化前 coords[{i}].shape: {coords[i].shape}")
    print(f"归一化前 coords[{i}] 数据范围: X[{coords[i][:, 0].min():.6f}, {coords[i][:, 0].max():.6f}], "
          f"Y[{coords[i][:, 1].min():.6f}, {coords[i][:, 1].max():.6f}], "
          f"Z[{coords[i][:, 2].min():.6f}, {coords[i][:, 2].max():.6f}]")

    coords[i] = normalization(coords[i], axi_x_min, axi_x_max, axi_y_min, axi_y_max, axi_z_min, axi_z_max)

    print(f"归一化后 coords[{i}].shape: {coords[i].shape}")
    print(f"归一化后 coords[{i}] 数据范围: X[{coords[i][:, 0].min():.6f}, {coords[i][:, 0].max():.6f}], "
          f"Y[{coords[i][:, 1].min():.6f}, {coords[i][:, 1].max():.6f}], "
          f"Z[{coords[i][:, 2].min():.6f}, {coords[i][:, 2].max():.6f}]")

coods = coords
print("=== STEP 3 完成 ===\n")


print("=== STEP 4: 创建数据加载器和模型 ===")
print(f"创建ObservationPoints3D对象...")
for i in range(num_depth_layers):
    print(f"  - cood{i + 1}.shape: {coods[i].shape}")
print(f"  - Y_flat.shape: {Y_flat.shape}")
print(f"  - batch_size: {batch_size}")

O = ObservationPoints3D(*coods, Y_flat, batch_size)
print(f"ObservationPoints3D对象创建完成")

dataloader = DataLoader(O, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)
print(f"DataLoader创建完成: batch_size=1, shuffle=True")

print(f"\n--- 创建 TCNN 模型 ---")
print(f"模型配置: {config}")
print(f"模型参数: {args}")

L = config['n_levels']
F = config['n_features_per_level']
T = config['log2_hashmap_size']
N_min = config['base_resolution']
b = args.per_level_scale
N_max = int(N_min * (b ** (L - 1)))

print(f"TCNN HashGrid params -> L={L}, F={F}, T={T}, N_min={N_min}, N_max={N_max}, b={b:.4f}")

field_model = ROVER_TCNN(in_dim=3, out_dim=1, L=L, F=F, T=T, N_min=N_min, N_max=N_max).to(device)
print(f"ROVER_TCNN模型创建完成")
print(f"模型参数数量: {sum(p.numel() for p in field_model.parameters())}")

print(f"\n--- 创建优化器 (Adam) + AMP scaler ---")
optim = torch.optim.Adam(field_model.parameters(), lr=args.learning_rate)
scaler = GradScaler()
print(f"Adam优化器创建完成，学习率: {args.learning_rate}")
print("=== STEP 4 完成 ===\n")

print("=== STEP 5: 设置输出目录 ===")
output_folder = config['output_folder']
print(f"配置中的输出文件夹: {output_folder}")

# Include weight_mode and Charbonnier epsilon in the model name so runs are distinguishable
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
        opts.weight_mode,          # 'gaussian', 'equal', or 'sliceprofile'
        _charb_eps_suffix,         # e.g. 0p1 for 0.1, 0p001 for 1e-3
    )
)

print(f"模型名称: {model_name}")

output_directory = os.path.join(opts.output_path + "/outputs", model_name)
print(f"输出目录路径: {output_directory}")

print(f"创建子目录...")
checkpoint_directory, image_directory, log_directory = prepare_sub_folder(output_directory)

print('=== 输出目录设置完成 ===')
print('Output directory: {}'.format(output_directory))
print('Checkpoint directory: {}'.format(checkpoint_directory))
print('Image directory: {}'.format(image_directory))
print('Log directory: {}'.format(log_directory))
print("=== STEP 5 完成 ===\n")

print("=== STEP 6: 开始 TCNN+PSF+Charbonnier+TV 训练 ===")
writer = SummaryWriter(log_directory)

lambda_c = float(hyper_params["lambda_c"])
tv_num_samples = opts.tv_num_samples if opts.tv_num_samples > 0 else max(1, batch_size // 2)

print(f"\n--- 训练参数 ---")
print(f"最大迭代次数: max_iter = {max_iter}")
print(f"超参数: lambda_c = {lambda_c} (TV weight; 0 = TV disabled)")
print(f"Charbonnier epsilon = {opts.charb_epsilon} (larger => more L2-like)")
print(f"TV prior: random finite-difference on {tv_num_samples} coords, eps={opts.tv_fd_eps}")
print(f"设备: {device}")
print(f"数据加载器长度: {len(dataloader)}")

coord_keys = [f"coord{i + 1}" for i in range(num_depth_layers)]

# *** Build PSF weights: gaussian / equal / sliceprofile (from rf_profile_tbwp4.mat) ***
print(f"\n--- PSF权重初始化 (weight_mode='{opts.weight_mode}') ---")
if opts.weight_mode == 'sliceprofile':
    psf_w = get_psf_weights_from_slice_profile(opts.rf_profile_mat, num_depth_layers, spa_res, device)
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
print("\n开始训练 TCNN+PSF+Charbonnier+TV...")
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
print("TCNN+PSF+Charbonnier+TV 训练完成!")
print("=== STEP 6 完成 ===\n")
