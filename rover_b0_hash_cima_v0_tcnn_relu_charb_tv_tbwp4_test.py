#!/usr/bin/env python
# coding: utf-8
"""
Test / inference script for the ROVER B0 TCNN model trained with
`rover_b0_hash_cima_v0_tcnn_relu_charb_tv_tbwp4.py`.

Geometry: 8 depth layers, spa_res 0.7484 mm (through-plane), Cima paths.
weight_mode: gaussian, equal, or sliceprofile (must match training).
Training sliceprofile uses rf_profile_tbwp4.mat (z_mm, Mse_sinc).
Config: configs/qiang_data_v10.yaml. Args: util_args_rover_b0_v18_lr1e4.

The reconstructed volume interleaves `num_depth_layers` sub-slices per acquired
slice, so the through-plane voxel spacing of the output equals spa_res (0.7484 mm),
which matches the in-plane x/y resolution. The saved NIfTI z-axis voxel size is
set accordingly (--z_spacing_mm, default = spa_res / x resolution).
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
from util_args_rover_b0_v18_lr1e4 import get_args

import glob
import time
import re
import datetime

import tinycudann as tcnn


def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def fix_affine_z_spacing(reference_affine, target_z_spacing_mm):
    """Keep orientation and origin; set z column length to target_z_spacing_mm."""
    aff = np.array(reference_affine, dtype=np.float64, copy=True)
    z_vec = aff[:3, 2]
    z_norm = np.linalg.norm(z_vec)
    if z_norm <= 0:
        raise ValueError("Reference affine z-axis vector norm is zero; cannot set z-spacing.")
    aff[:3, 2] = (z_vec / z_norm) * float(target_z_spacing_mm)
    return aff


log_filename = f"rover_b0_hash_cima_test_tcnn_relu_charb_tv_tbwp4_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.out"


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

print(f"=== ROVER B0 Cima TCNN (Charbonnier+TV+TBWP4) test start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
print(f"Log file: {log_filename}")
print("=" * 80)


class ROVER_TCNN(torch.nn.Module):
    """Same TCNN architecture as rover_b0_hash_cima_v0_tcnn_relu_charb_tv_tbwp4.py."""

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


# -------------------------------------------------------------------------
# STEP 1: spatial resolution and depth layers (same as Cima training)
# -------------------------------------------------------------------------
print("=== ROVER B0 Cima spatial resolution ===")
affine_path = '/scratch/home/ql087/data_bwh/Cima_data/preprocess_rover_nii/Affine_nii_1.npy'
reference_affine = None
x_voxel_size = None
y_voxel_size = None
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
    print(f"深度坐标数量: {num_depth_layers}")
    print(f"Through-plane spa_res: {spa_res:.6f} mm")
else:
    spa_res = 0.7484
    num_depth_layers = 8
    print(f"无法加载参考Affine矩阵，使用默认值:")
    print(f"  空间分辨率: {spa_res:.6f} mm")
    print(f"  深度坐标数量: {num_depth_layers}")

start_time = time.time()


# -------------------------------------------------------------------------
# STEP 2: args / config (match Cima training defaults)
# -------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/qiang_data_v10.yaml',
                    help='Path to the config file (must match training).')
parser.add_argument('--output_path', type=str, default='output',
                    help="outputs path")
parser.add_argument('--checkpoint_iter', type=int, default=None,
                    help="指定要加载的checkpoint迭代次数")
parser.add_argument('--weight_mode', type=str, default='sliceprofile',
                    choices=['gaussian', 'equal', 'sliceprofile'],
                    help="Depth-layer PSF weighting mode used during training.")
parser.add_argument('--rf_profile_mat', type=str,
                    default=os.path.join(project_root, 'rf_profile_tbwp4.mat'),
                    help="Slice profile .mat used during training (for logging; must match training when weight_mode='sliceprofile').")
parser.add_argument('--charb_epsilon', type=float, default=1e-1,
                    help="Charbonnier epsilon used during training (must match). Default 1e-1.")
parser.add_argument('--checkpoint_path', type=str, default=None,
                    help="Optional: full path to a .pt checkpoint. If set, load this file and save under its run folder.")
parser.add_argument('--z_spacing_mm', type=float, default=None,
                    help="NIfTI z-axis voxel size in mm. Default: in-plane x resolution (= spa_res), so z matches x/y.")
opts = parser.parse_args()
args = get_args(cmd=False)
config = get_config(opts.config)

if reference_affine is None:
    raise FileNotFoundError(
        f"Reference affine not found: {affine_path}\n"
        "Cannot save NIfTI with corrected z-spacing without the reference affine."
    )

# Default z thickness to in-plane x/y resolution (same as spa_res in this Cima setup).
if opts.z_spacing_mm is None:
    opts.z_spacing_mm = float(x_voxel_size) if x_voxel_size is not None else float(spa_res)

output_affine = fix_affine_z_spacing(reference_affine, target_z_spacing_mm=opts.z_spacing_mm)
orig_z_spacing = np.linalg.norm(reference_affine[:3, 2])
out_z_spacing = np.linalg.norm(output_affine[:3, 2])
out_x_spacing = np.linalg.norm(output_affine[:3, 0])
out_y_spacing = np.linalg.norm(output_affine[:3, 1])
print(f"NIfTI header z-spacing: {orig_z_spacing:.6f} mm -> {out_z_spacing:.6f} mm "
      f"(target {opts.z_spacing_mm} mm; x={out_x_spacing:.6f}, y={out_y_spacing:.6f})")
print(f"Isotropic check (z vs x): |z-x| = {abs(out_z_spacing - out_x_spacing):.6e} mm")


def _epsilon_suffix(eps):
    """Format epsilon for folder names (must match training script)."""
    s = f"{eps:.6g}".replace(".", "p").replace("-", "m")
    return s


print(f"PSF weight_mode: {opts.weight_mode}, charb_epsilon: {opts.charb_epsilon}")
if opts.weight_mode == 'sliceprofile':
    print(f"Training slice profile: {opts.rf_profile_mat}")

if opts.checkpoint_iter is not None:
    config['iter'] = opts.checkpoint_iter
    print(f"使用用户指定的迭代次数: {opts.checkpoint_iter}")
else:
    print(f"使用配置文件中的迭代次数: {config['iter']}")

nx, ny, nk = args.img_size
nv = args.view_num

print("=== TEST STEP 1: 初始化ROVER B0 Cima TCNN测试参数 ===")
print(f"图像尺寸: nx={nx}, ny={ny}, nk={nk}")
print(f"视角数量: nv={nv}")

print(f"\n--- 加载图像数据 ---")
print(f"图像路径模式: {args.img_path}")
img_path = args.img_path
batch_size = args.batch_size
print(f"批次大小: batch_size={batch_size}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

files_path = sorted(glob.glob(img_path), key=natural_sort_key)
print("自然排序后的文件列表:")
print(f"找到的图像文件数量: {len(files_path)}")
print(f"图像文件列表: {files_path}")

print(f"\n--- 创建坐标网格 ---")
X, Y, Z = np.mgrid[0:nx:1, 0:ny:1, 0:nk:1]
x, y, z = X.ravel(), Y.ravel(), Z.ravel()
N = nx * ny * nk
print(f"坐标网格: X.shape={X.shape}, Y.shape={Y.shape}, Z.shape={Z.shape}")
print(f"坐标向量: x.shape={x.shape}, y.shape={y.shape}, z.shape={z.shape}")
print(f"总像素数: N = nx*ny*nk = {nx}*{ny}*{nk} = {N}")

print(f"\n--- 初始化{num_depth_layers}层深度坐标数组 ---")
coods = [np.zeros((N * nv, 3)) for _ in range(num_depth_layers)]
print(f"coods数组形状: {[c.shape for c in coods]}")
print("=== TEST STEP 1 完成 ===\n")


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


print(f"=== TEST STEP 2: 坐标变换和{num_depth_layers}层深度生成 ===")
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

    results = trans(x, y, z, Affine)
    print(f"坐标变换完成，得到{num_depth_layers}层深度坐标")

    for i in range(num_depth_layers):
        start_idx = view_num * N
        end_idx = (view_num + 1) * N
        coods[i][start_idx:end_idx, :] = results[i]
        print(f"  深度层{i+1}: 存储到coods[{i}][{start_idx}:{end_idx}, :]")

    print(f"视角{view_num}坐标处理完成")
print("=== TEST STEP 2 完成 ===\n")


print("=== TEST STEP 3: 坐标归一化 ===")
coords = coods
all_coords = np.concatenate(coords, axis=0)

axi_x_min = np.min(all_coords[:, 0])
axi_x_max = np.max(all_coords[:, 0])
axi_y_min = np.min(all_coords[:, 1])
axi_y_max = np.max(all_coords[:, 1])
axi_z_min = np.min(all_coords[:, 2])
axi_z_max = np.max(all_coords[:, 2])

print(f"X轴范围: [{axi_x_min:.6f}, {axi_x_max:.6f}]")
print(f"Y轴范围: [{axi_y_min:.6f}, {axi_y_max:.6f}]")
print(f"Z轴范围: [{axi_z_min:.6f}, {axi_z_max:.6f}]")

for i in range(len(coords)):
    print(f"\n--- 归一化深度层 {i+1}/{len(coords)} ---")
    print(f"归一化前 coords[{i}].shape: {coords[i].shape}")
    print(f"归一化前 coords[{i}] 数据范围: X[{coords[i][:,0].min():.6f}, {coords[i][:,0].max():.6f}], "
          f"Y[{coords[i][:,1].min():.6f}, {coords[i][:,1].max():.6f}], "
          f"Z[{coords[i][:,2].min():.6f}, {coords[i][:,2].max():.6f}]")

    coords[i] = normalization(coords[i], axi_x_min, axi_x_max, axi_y_min, axi_y_max, axi_z_min, axi_z_max)

    print(f"归一化后 coords[{i}].shape: {coords[i].shape}")
    print(f"归一化后 coords[{i}] 数据范围: X[{coords[i][:,0].min():.6f}, {coords[i][:,0].max():.6f}], "
          f"Y[{coords[i][:,1].min():.6f}, {coords[i][:,1].max():.6f}], "
          f"Z[{coords[i][:,2].min():.6f}, {coords[i][:,2].max():.6f}]")

coods = coords
print("=== STEP 3 完成 ===\n")


print("=== TEST STEP 4: 创建数据加载器和模型 ===")
print(f"创建ObservationPoints3D对象...")
for i in range(num_depth_layers):
    print(f"  - cood{i+1}.shape: {coods[i].shape}")
print(f"  - batch_size: {batch_size}")

Y_flat = np.zeros([N * nv, 1])  # dummy, not used for inference but required by constructor
O = ObservationPoints3D(*coods, Y_flat, batch_size)
print(f"ObservationPoints3D对象创建完成")

dataloader = DataLoader(O, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)
print(f"DataLoader创建完成: batch_size=1, shuffle=True")

print(f"\n--- 设置输出目录 ---")
output_folder = config['output_folder']
print(f"配置中的输出文件夹: {output_folder}")

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

print(f"模型名称: {model_name}")

use_direct_checkpoint_path = opts.checkpoint_path is not None
if use_direct_checkpoint_path:
    model_path = os.path.abspath(opts.checkpoint_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint file not found: {model_path}")
    checkpoint_directory = os.path.dirname(model_path)
    output_directory = os.path.dirname(checkpoint_directory)
    image_directory = os.path.join(output_directory, "images")
    log_directory = os.path.join(output_directory, "logs")
    print(f"使用指定checkpoint路径: {model_path}")
    print(f"输出目录: {output_directory}")
else:
    checkpoint_directory = os.path.join(opts.output_path, "outputs", model_name, "checkpoints")
    if not os.path.exists(checkpoint_directory):
        model_name_legacy = model_name.replace("_eps" + _charb_eps_suffix, "")
        checkpoint_directory_legacy = os.path.join(opts.output_path, "outputs", model_name_legacy, "checkpoints")
        if os.path.exists(checkpoint_directory_legacy):
            checkpoint_directory = checkpoint_directory_legacy
            model_name = model_name_legacy
            print(f"使用旧路径 (无 _eps 后缀): {checkpoint_directory}")
        else:
            raise FileNotFoundError(
                f"Checkpoint目录不存在: {checkpoint_directory}\n"
                f"Tried fallback: {checkpoint_directory_legacy}\n"
                f"Or pass --checkpoint_path /full/path/to/model_XXXXXX.pt to load a specific file."
            )
    image_directory = os.path.join(opts.output_path, "outputs", model_name, "images")
    log_directory = os.path.join(opts.output_path, "outputs", model_name, "logs")
    output_directory = os.path.join(opts.output_path, "outputs", model_name)
    model_path = None

print(f"使用checkpoint目录: {checkpoint_directory}")
print(f"图像目录: {image_directory}")
print(f"日志目录: {log_directory}")
print("=== TEST STEP 4 完成 ===\n")


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

field_model = ROVER_TCNN(in_dim=3, out_dim=1, L=L, F=F, T=T, N_min=N_min, N_max=N_max)
field_model = field_model.to(device)
field_model.eval()
print(f"ROVER_TCNN模型创建完成并移动到设备: {device}")
print("=== TEST STEP 4b 完成 ===\n")


print("=== TEST STEP 5: 加载预训练TCNN模型 ===")

if use_direct_checkpoint_path:
    print(f"准备加载checkpoint: {model_path}")
else:
    if not os.path.exists(checkpoint_directory):
        raise FileNotFoundError(f"Checkpoint目录不存在: {checkpoint_directory}")
    checkpoints = sorted([f for f in os.listdir(checkpoint_directory) if f.endswith(".pt")])
    if not checkpoints:
        raise FileNotFoundError(f"在目录 {checkpoint_directory} 下没有找到任何 .pt 文件！")
    print("可用的checkpoint文件:")
    for ckpt in checkpoints:
        print("   -", ckpt)
    if opts.checkpoint_iter is not None:
        target_name = f"model_{opts.checkpoint_iter:06d}.pt"
        if target_name in checkpoints:
            ckpt_name = target_name
        else:
            raise FileNotFoundError(f"指定的checkpoint {target_name} 不存在于 {checkpoint_directory}")
    else:
        ckpt_name = checkpoints[-1]
    model_path = os.path.join(checkpoint_directory, ckpt_name)
    print(f"准备加载checkpoint: {model_path}")

state_dict = torch.load(model_path, map_location=device)
print(f"checkpoint包含键: {list(state_dict.keys())}")

field_model.load_state_dict(state_dict['net'])
print(f"预训练TCNN模型加载成功: {model_path}")
print("=== TEST STEP 5 完成 ===\n")


print("=== TEST STEP 6: 开始推理（逐深度层重建） ===")
print(f"获取完整数据集...")
coordmap, data = dataloader.dataset.getfulldata()
print(f"coordmap包含的键: {list(coordmap.keys())}")

print(f"\n--- 初始化{num_depth_layers}层深度结果数组 ---")
f_list = [np.zeros((nx * ny * nk, 1)) for _ in range(num_depth_layers)]
for i, f in enumerate(f_list):
    print(f"  f_list[{i}].shape: {f.shape}")

print(f"\n--- 逐切片进行推理 ---")
print(f"总切片数: nk = {nk}")
print(f"每切片像素数: nx * ny = {nx} * {ny} = {nx * ny}")

for slc in range(nk):
    print(f"\n--- 处理切片 {slc+1}/{nk} ---")
    for i in range(num_depth_layers):
        key = f"coord{i+1}"
        coords = coordmap[key][slc * nx * ny : (slc + 1) * nx * ny, :]
        print(f"  深度层{i+1}: coords.shape = {coords.shape}")

        coords_t = coords.to(device)
        with torch.no_grad():
            f_hat = field_model(coords_t).cpu().numpy()
        print(f"    推理完成，结果形状: {f_hat.shape}")

        start_idx = slc * nx * ny
        end_idx = (slc + 1) * nx * ny
        f_list[i][start_idx:end_idx] = f_hat
        print(f"    存储到f_list[{i}][{start_idx}:{end_idx}]")

    if (slc + 1) % 10 == 0 or slc == nk - 1:
        print(f"  切片{slc+1}/{nk} 处理完成")
print("=== TEST STEP 6 完成 ===\n")


print("=== TEST STEP 7: 结果重塑和堆叠 ===")
print(f"重塑{num_depth_layers}层深度结果数组...")
f_list = [f.reshape((nx, ny, nk)) for f in f_list]
for i, f in enumerate(f_list):
    print(f"  f_list[{i}].reshape后形状: {f.shape}")

print(f"\n--- 堆叠为最终结果 ---")
print(f"最终结果形状: (nx, ny, {num_depth_layers} * nk) = ({nx}, {ny}, {num_depth_layers * nk})")
result = np.zeros((nx, ny, num_depth_layers * nk))
print(f"初始化result数组: shape = {result.shape}")

for i in range(num_depth_layers):
    print(f"\n--- 处理深度层 {i+1}/{num_depth_layers} ---")
    for slc in range(nk):
        target_idx = num_depth_layers * slc + i
        print(f"  切片{slc+1}: 将f_list[{i}][:, :, {slc}] 存储到 result[:, :, {target_idx}]")
        result[:, :, target_idx] = f_list[i][:, :, slc]

print(f"\n最终结果数组形状: {result.shape}")
print(f"最终结果数据范围: [{result.min():.6f}, {result.max():.6f}]")
print("=== TEST STEP 7 完成 ===\n")


print(f"\n=== 数组Shape分析 ===")
print(f"原始图像尺寸: nx={nx}, ny={ny}, nk={nk}")
print(f"f_list[0].shape: {f_list[0].shape} (单个深度层的重建结果)")
print(f"result.shape: {result.shape} (所有深度层的堆叠结果)")

print(f"\n--- 准备保存为NIfTI格式 ---")
print(f"使用输出affine (z-spacing={out_z_spacing:.6f} mm, matched to x/y):\n{output_affine}")
recon = nib.Nifti1Image(result, affine=output_affine)
# Ensure the NIfTI header pixdim along z reports the same thickness as x/y.
recon.header.set_zooms((
    float(out_x_spacing),
    float(out_y_spacing),
    float(out_z_spacing),
))
print(f"NIfTI header zooms (pixdim): {recon.header.get_zooms()}")

print("=== TEST STEP 8: 保存结果 ===")
test_output_dir = os.path.join(output_directory, 'test-output')
if not os.path.exists(test_output_dir):
    os.makedirs(test_output_dir)
    print(f"创建测试输出目录: {test_output_dir}")
else:
    print(f"测试输出目录已存在: {test_output_dir}")

if use_direct_checkpoint_path:
    ckpt_basename = os.path.splitext(os.path.basename(model_path))[0]
    output_filename = f'rover_b0_hash_cima_tcnn_relu_charb_tv_tbwp4_{ckpt_basename}_z{opts.z_spacing_mm:.5f}.nii'
else:
    output_filename = (
        f'rover_b0_hash_cima_tcnn_relu_psf_charb_{config["r"]}w_{config["depth"]}d_'
        f'{config["n_levels"]}lev_{config["n_features_per_level"]}nplev_'
        f'{config["log2_hashmap_size"]}hasize_{config["base_resolution"]}bs_'
        f'{config["lambda_c"]}lambda_{config["iter"]}iter_w{opts.weight_mode}_eps{_charb_eps_suffix}_'
        f'z{opts.z_spacing_mm:.5f}_tv.nii'
    )
output_path = os.path.join(test_output_dir, output_filename)

print(f"输出文件名: {output_filename}")
print(f"完整输出路径: {output_path}")

print(f"\n开始保存NIfTI文件...")
nib.save(recon, output_path)
print(f"结果保存成功: {output_path}")

print(f"\n=== 测试完成 ===")
end_time = time.time()
execution_time = end_time - start_time
print("总耗时：", execution_time, "seconds")
print("=== TEST STEP 8 完成 ===\n")
