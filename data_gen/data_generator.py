#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSARTree 数据生成器 (增强版)
用于生成各种分布和规模的 RTree 测试数据
支持：均匀分布、正态分布、偏态分布、多峰分布、时空数据等

GSARTree Data Generator (Enhanced Version)
Generates RTree test data with various distributions and scales
Supports: uniform, normal, skewed, multi-modal, spatio-temporal distributions, etc.
"""

import sys
import os
import random
import matplotlib
matplotlib.use('Agg')  # 非交互式后端 / Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skewnorm, truncnorm, gaussian_kde
from tqdm import tqdm
from datetime import datetime

# ============================================================================
# 配置常量 / Configuration Constants
# ============================================================================

# 数据规模映射表：键为规模标识，值为矩形数量
# Data size mapping: key is size identifier, value is number of rectangles
DATASIZE_TYPE = {
    "H1": 100, "2p5": 25000,
    "K5": 5000, "W1": 10000, "W2": 20000, "W3": 30000, "W4": 40000, "W5": 50000,
    "TW1": 100000, "TW2": 200000, "TW5": 500000,
    "HW1": 1000000, "HW5": 5000000, "KW": 10000000, "KW2": 20000000, "KW5": 50000000,
    "WW1": 100000000
}

# 分布类型映射表：键为分布名称，值为内部标识ID
# Distribution type mapping: key is distribution name, value is internal ID
DATADISTRIBUTION_TYPE = {
    # 基础分布 / Basic Distributions
    "NORMAL": 1, 
    "UNIFORM": 2, 
    "SKEW-NOR0": 3, "SKEW-NOR2": 4, "SKEW-NOR4": 5, "SKEW-NOR8": 6,
    # 多峰分布 / Multi-modal Distributions
    "BIMODAL": 7,      # 双峰分布 / Bimodal
    "MULTI-MODAL": 8,  # 多峰分布 / Multi-modal
    "CLUSTERED": 9,    # 聚类分布 / Clustered
    # 复杂分布 / Complex Distributions
    "CORRELATED": 10,  # 相关分布 / Correlated
    "ANTI-CORR": 11,   # 反相关分布 / Anti-correlated
    "GAUSSIAN-MIX": 12,# 高斯混合模型 / Gaussian Mixture
    # 时空数据 / Spatio-temporal Data
    "SPATIAL-TEMPORAL": 13,  # 时空数据 / Spatial-temporal
    "CITY-LIKE": 14,         # 城市网格状分布 / City-like grid
    "HOTSPOTS": 15,          # 热点区域分布 / Hotspots
}

# 默认配置参数
# Default configuration parameters
DEFAULT_CONFIG = {
    'x_min': 0,           # X轴最小值 / Minimum X coordinate
    'x_max': 100000,      # X轴最大值 / Maximum X coordinate
    'y_min': 0,           # Y轴最小值 / Minimum Y coordinate
    'y_max': 100000,      # Y轴最大值 / Maximum Y coordinate
    'edge_size': 0.01,    # 矩形边长偏移量 / Rectangle edge offset
}

# ============================================================================
# 数据读取函数 / Data Reading Functions
# ============================================================================

def reading_data(file_path, save_path=None):
    """
    读取数据文件并保存为 npy 格式
    
    Parameters
    ----------
    file_path : str
        输入数据文件路径（TXT格式，每两行为一个点的x,y坐标）
        Input data file path (TXT format, every two lines are x,y coordinates of a point)
    save_path : str, optional
        输出 npy 文件路径，默认为 None（自动生成）
        Output npy file path, default is None (auto-generated)
    
    Returns
    -------
    rectangles : ndarray
        矩形数组，形状为 (N, 4)，每行为 [ll_x, ll_y, tr_x, tr_y]
        Rectangle array with shape (N, 4), each row is [ll_x, ll_y, tr_x, tr_y]
    
    Notes
    -----
    该函数将点坐标转换为矩形：以点为中心，向左下角扩展 edge_size 形成矩形
    This function converts point coordinates to rectangles: 
    expands edge_size to the lower-left corner with the point as center
    """
    print(f"Reading data from: {file_path}")
    
    # 统计文件总行数 / Count total lines in file
    with open(file_path, 'r') as f:
        total_lines = sum(1 for _ in f)

    # 使用 tqdm 显示读取进度，将所有行转换为浮点数
    # Use tqdm to show reading progress, convert all lines to float
    data = np.fromiter(
        tqdm((float(line.strip()) for line in open(file_path)), 
             total=total_lines, desc="Reading Data"), 
        dtype=np.float64, count=total_lines
    )
    
    # 从点坐标计算矩形坐标
    # Calculate rectangle coordinates from point coordinates
    # data[:-1:2]: 所有x坐标 / all x coordinates
    # data[1::2]: 所有y坐标 / all y coordinates
    # 矩形左下角 = (x - edge_size, y - edge_size)
    # Rectangle lower-left = (x - edge_size, y - edge_size)
    # 矩形右上角 = (x, y)
    # Rectangle upper-right = (x, y)
    edge_size = DEFAULT_CONFIG['edge_size']
    rectangles = np.column_stack((
        data[:-1:2] - edge_size, 
        data[1::2] - edge_size, 
        data[:-1:2], 
        data[1::2]
    ))
    
    # 如果未指定保存路径，则自动生成（与输入文件同名，扩展名为.npy）
    # If save path not specified, auto-generate (same name as input, .npy extension)
    if save_path is None:
        save_path = os.path.splitext(file_path)[0] + ".npy"
    
    # 保存为 NumPy 二进制格式 / Save as NumPy binary format
    np.save(save_path, rectangles)
    print(f"\n✓ Data saved to {save_path}")
    print(f"✓ Total Rectangles: {len(rectangles)}")
    
    return rectangles

# ============================================================================
# 数据生成函数 / Data Generation Functions
# ============================================================================

def generate_coordinates(x_min, x_max, y_min, y_max, data_size, output_dir, 
                        testing_flag=False, distribution_type="UNIFORM"):
    """
    生成指定分布和规模的坐标数据
    
    Parameters
    ----------
    x_min, x_max, y_min, y_max : float
        坐标范围 / Coordinate range
    data_size : str
        数据规模标识 (如 "W1", "HW1" 等)
        Data size identifier (e.g., "W1", "HW1")
    output_dir : str
        输出目录 / Output directory
    testing_flag : bool
        是否为测试数据（如果是，文件名会加 testing_ 前缀，并生成 .npy 格式）
        Whether it's test data (if True, filename gets 'testing_' prefix and generates .npy)
    distribution_type : str
        分布类型 ("NORMAL", "UNIFORM", "SKEW-NOR0" 等)
        Distribution type ("NORMAL", "UNIFORM", "SKEW-NOR0", etc.)
    
    Returns
    -------
    output_file : str
        生成的文件路径（TXT格式）
        Generated file path (TXT format)
    
    Notes
    -----
    主生成函数，根据分布类型调用相应的子函数生成坐标数据
    Main generation function, calls corresponding sub-functions based on distribution type
    """
    # 从配置表获取数据点数量 / Get number of data points from config table
    num_points = DATASIZE_TYPE[data_size]
    
    # 创建输出目录（如果不存在）/ Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建输出文件名：testing_前缀 + 规模 + 分布类型.txt
    # Build output filename: testing_ prefix + size + distribution type.txt
    prefix = "testing_" if testing_flag else ""
    base_filename = f"{prefix}{data_size}_{distribution_type}.txt"
    output_file = os.path.join(output_dir, base_filename)

    print(f"Generating {num_points:,} points with {distribution_type} distribution...")
    print(f"Output file: {output_file}")
    
    # 根据不同分布类型生成数据 / Generate data based on distribution type
    if distribution_type == "NORMAL":
        x_coords, y_coords = _generate_normal_distribution(
            num_points, x_min, x_max, y_min, y_max
        )
        
    elif distribution_type == "UNIFORM":
        # 均匀分布：在范围内完全随机生成
        # Uniform distribution: completely random within range
        x_coords = np.random.uniform(x_min, x_max, num_points)
        y_coords = np.random.uniform(y_min, y_max, num_points)
        
    elif distribution_type.startswith("SKEW-NOR"):
        # 偏态分布：根据alpha参数控制偏斜程度
        # Skewed distribution: control skewness by alpha parameter
        alpha_map = {"SKEW-NOR0": 0, "SKEW-NOR2": 2, "SKEW-NOR4": 4, "SKEW-NOR8": 8}
        alpha = alpha_map[distribution_type]
        x_coords, y_coords = _generate_skew_normal_distribution(
            num_points, x_min, x_max, y_min, y_max, alpha
        )
    
    elif distribution_type == "BIMODAL":
        # 双峰分布：两个高斯聚集中心
        # Bimodal distribution: two Gaussian cluster centers
        x_coords, y_coords = _generate_bimodal_distribution(
            num_points, x_min, x_max, y_min, y_max
        )
    
    elif distribution_type == "MULTI-MODAL":
        # 多峰分布：五个不同权重的聚类中心
        # Multi-modal distribution: five cluster centers with different weights
        x_coords, y_coords = _generate_multimodal_distribution(
            num_points, x_min, x_max, y_min, y_max
        )
    
    elif distribution_type == "CLUSTERED":
        # 聚类分布：8-12个随机聚类中心
        # Clustered distribution: 8-12 random cluster centers
        x_coords, y_coords = _generate_clustered_distribution(
            num_points, x_min, x_max, y_min, y_max
        )
    
    elif distribution_type == "CORRELATED":
        # 正相关分布：X和Y呈线性正相关
        # Positive correlated distribution: X and Y linearly positively correlated
        x_coords, y_coords = _generate_correlated_distribution(
            num_points, x_min, x_max, y_min, y_max
        )
    
    elif distribution_type == "ANTI-CORR":
        # 负相关分布：X和Y呈线性负相关
        # Negative correlated distribution: X and Y linearly negatively correlated
        x_coords, y_coords = _generate_anticorrelated_distribution(
            num_points, x_min, x_max, y_min, y_max
        )
    
    elif distribution_type == "GAUSSIAN-MIX":
        # 高斯混合模型：3个不同权重的高斯成分混合
        # Gaussian mixture model: 3 Gaussian components with different weights
        x_coords, y_coords = _generate_gaussian_mixture_distribution(
            num_points, x_min, x_max, y_min, y_max
        )
    
    elif distribution_type == "SPATIAL-TEMPORAL":
        # 时空数据：模拟移动轨迹（布朗运动风格）
        # Spatial-temporal data: simulate movement trajectories (Brownian motion style)
        x_coords, y_coords = _generate_spatial_temporal_distribution(
            num_points, x_min, x_max, y_min, y_max
        )
    
    elif distribution_type == "CITY-LIKE":
        # 城市网格状：沿道路网络分布
        # City-like grid: distributed along road networks
        x_coords, y_coords = _generate_city_like_distribution(
            num_points, x_min, x_max, y_min, y_max
        )
    
    elif distribution_type == "HOTSPOTS":
        # 热点区域：少数高密度核心 + 背景噪声
        # Hotspots: few high-density cores + background noise
        x_coords, y_coords = _generate_hotspots_distribution(
            num_points, x_min, x_max, y_min, y_max
        )
    
    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")
    
    # 检查文件是否存在，如果存在则询问是否覆盖
    # Check if file exists, ask for overwrite if it does
    if os.path.exists(output_file):
        response = input(f"⚠️  File '{output_file}' already exists. Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print("⚠️  Skipped saving. Existing file kept.")
            return output_file

    # 保存到 TXT 文件（每两行为一个点的x,y坐标）
    # Save to TXT file (every two lines are x,y coordinates of a point)
    with open(output_file, "w") as f:
        for x, y in zip(x_coords, y_coords):
            f.write(f"{x}\n{y}\n")
    
    print(f"✓ Successfully saved {num_points:,} points to {output_file}")
    
    # 生成 NPY 格式（将点坐标转换为矩形）
    # Generate NPY format (convert point coordinates to rectangles)
    output_npy = os.path.splitext(output_file)[0] + ".npy"
    reading_data(output_file, output_npy)
    
    return output_file


def _generate_normal_distribution(num_points, x_min, x_max, y_min, y_max):
    """
    生成截断正态分布
    
    Parameters
    ----------
    num_points : int
        数据点数量 / Number of data points
    x_min, x_max, y_min, y_max : float
        坐标范围 / Coordinate range
    
    Returns
    -------
    x_coords, y_coords : ndarray
        X和Y坐标数组 / X and Y coordinate arrays
    
    Notes
    -----
    使用截断正态分布，均值在中心(0.5)，标准差为0.15
    避免生成超出边界的点
    Uses truncated normal distribution, mean at center (0.5), std=0.15
    Avoids generating points outside boundaries
    """
    def truncated_normal(mean, std, lower, upper, size):
        """
        生成截断正态分布的辅助函数
        Helper function to generate truncated normal distribution
        """
        a, b = (lower - mean) / std, (upper - mean) / std
        return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)

    # 设置均值和标准差（相对于范围的比例）
    # Set mean and std (proportional to range)
    x_mean = y_mean = 0.5
    x_std = y_std = 0.15

    x_coords = truncated_normal(
        x_mean * (x_max - x_min), 
        x_std * (x_max - x_min), 
        x_min, x_max, num_points
    )
    y_coords = truncated_normal(
        y_mean * (y_max - y_min), 
        y_std * (y_max - y_min), 
        y_min, y_max, num_points
    )
    
    return x_coords, y_coords


def _generate_skew_normal_distribution(num_points, x_min, x_max, y_min, y_max, alpha):
    """
    生成偏态正态分布
    
    Parameters
    ----------
    num_points : int
        数据点数量 / Number of data points
    x_min, x_max, y_min, y_max : float
        坐标范围 / Coordinate range
    alpha : float
        偏态参数，控制偏斜程度（0=对称，正值右偏，负值左偏）
        Skewness parameter, controls skewness (0=symmetric, positive=right-skewed, negative=left-skewed)
    
    Returns
    -------
    x_coords, y_coords : ndarray
        X和Y坐标数组 / X and Y coordinate arrays
    
    Notes
    -----
    使用 scipy 的 skewnorm 生成偏态分布，然后归一化到指定范围
    Uses scipy's skewnorm to generate skewed distribution, then normalizes to specified range
    """
    # 生成原始偏态分布数据 / Generate raw skewed distribution data
    x_raw = skewnorm.rvs(alpha, size=num_points)
    y_raw = skewnorm.rvs(alpha, size=num_points)

    # 归一化到 [x_min, x_max] 和 [y_min, y_max] 范围
    # Normalize to [x_min, x_max] and [y_min, y_max] range
    x_coords = x_min + (x_raw - np.min(x_raw)) / (np.max(x_raw) - np.min(x_raw)) * (x_max - x_min)
    y_coords = y_min + (y_raw - np.min(y_raw)) / (np.max(y_raw) - np.min(y_raw)) * (y_max - y_min)
    
    return x_coords, y_coords


def _generate_bimodal_distribution(num_points, x_min, x_max, y_min, y_max):
    """
    生成双峰分布（Bimodal Distribution）
    两个高斯中心，模拟数据在两个区域的聚集
    
    Parameters
    ----------
    num_points : int
        数据点数量 / Number of data points
    x_min, x_max, y_min, y_max : float
        坐标范围 / Coordinate range
    
    Returns
    -------
    x_coords, y_coords : ndarray
        X和Y坐标数组 / X and Y coordinate arrays
    
    Notes
    -----
    适用于双城结构、两个商业中心等场景
    Suitable for dual-city structures, two commercial centers, etc.
    """
    # 两个中心的坐标（分别在30%和70%位置）
    # Two center coordinates (at 30% and 70% positions)
    center1_x = x_min + (x_max - x_min) * 0.3
    center1_y = y_min + (y_max - y_min) * 0.3
    center2_x = x_min + (x_max - x_min) * 0.7
    center2_y = y_min + (y_max - y_min) * 0.7
    
    # 标准差（范围的15%）/ Standard deviation (15% of range)
    std = (x_max - x_min) * 0.15
    
    # 每个峰的数据量（平均分配）/ Data points per peak (evenly distributed)
    num1 = num_points // 2
    num2 = num_points - num1
    
    # 生成两个高斯分布的数据 / Generate data from two Gaussian distributions
    x1 = np.random.normal(center1_x, std, num1)
    y1 = np.random.normal(center1_y, std, num1)
    x2 = np.random.normal(center2_x, std, num2)
    y2 = np.random.normal(center2_y, std, num2)
    
    # 合并并裁剪到范围内 / Merge and clip to range
    x_coords = np.clip(np.concatenate([x1, x2]), x_min, x_max)
    y_coords = np.clip(np.concatenate([y1, y2]), y_min, y_max)
    
    return x_coords, y_coords


def _sample_truncated_gaussian_2d(cx, cy, sx, sy, n, x_min, x_max, y_min, y_max):
    """
    重复采样，避免 np.clip 导致边界堆积
    
    Parameters
    ----------
    cx, cy : float
        高斯分布的中心坐标 / Center coordinates of Gaussian distribution
    sx, sy : float
        X和Y方向的标准差 / Standard deviations in X and Y directions
    n : int
        需要生成的点数 / Number of points to generate
    x_min, x_max, y_min, y_max : float
        坐标范围 / Coordinate range
    
    Returns
    -------
    xs, ys : ndarray
        生成的X和Y坐标数组 / Generated X and Y coordinate arrays
    
    Notes
    -----
    使用拒绝采样方法：批量生成点，只保留在范围内的点
    这样可以避免边界处点的堆积，保证分布的真实性
    Uses rejection sampling: batch generate points, keep only those within range
    This avoids point accumulation at boundaries, ensuring distribution authenticity
    """
    xs, ys = [], []
    batch = max(n * 2, 1000)  # 批量大小，至少1000个点 / Batch size, at least 1000 points
    
    while len(xs) < n:
        # 批量生成候选点 / Batch generate candidate points
        x = np.random.normal(cx, sx, batch)
        y = np.random.normal(cy, sy, batch)
        
        # 只保留在范围内的点 / Keep only points within range
        mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
        
        xs.extend(x[mask])
        ys.extend(y[mask])
    
    # 返回前n个点 / Return first n points
    return np.array(xs[:n]), np.array(ys[:n])


def _generate_multimodal_distribution(num_points, x_min, x_max, y_min, y_max):
    """
    五峰高斯混合分布（Multi-modal Distribution）
    模拟多个空间聚集中心，例如多个城市中心或兴趣点聚集区。
    
    Parameters
    ----------
    num_points : int
        数据点数量 / Number of data points
    x_min, x_max, y_min, y_max : float
        坐标范围 / Coordinate range
    
    Returns
    -------
    x_coords, y_coords : ndarray
        X和Y坐标数组 / X and Y coordinate arrays
    
    Notes
    -----
    特点：
    - 5个不同权重的聚类中心
    - 每个聚类的宽度随机化，更真实
    - 使用拒绝采样避免边界堆积
    - 符合期刊要求的可复现性
    
    Features:
    - 5 cluster centers with different weights
    - Randomized width for each cluster, more realistic
    - Uses rejection sampling to avoid boundary accumulation
    - Meets journal requirements for reproducibility
    """
    width = x_max - x_min
    height = y_max - y_min
    
    # 定义5个聚类中心的位置（固定位置以保证可复现性）
    # Define positions of 5 cluster centers (fixed for reproducibility)
    centers = [
        (x_min + width * 0.20, y_min + height * 0.25),
        (x_min + width * 0.75, y_min + height * 0.20),
        (x_min + width * 0.50, y_min + height * 0.50),
        (x_min + width * 0.25, y_min + height * 0.80),
        (x_min + width * 0.80, y_min + height * 0.75),
    ]
    
    # 定义权重（中间区域权重最高）/ Define weights (center region has highest weight)
    weights = np.array([0.20, 0.25, 0.30, 0.15, 0.10])
    weights = weights / weights.sum()  # 确保归一化 / Ensure normalization
    
    # 使用多项式分布分配点数到各个聚类
    # Use multinomial distribution to allocate points to each cluster
    counts = np.random.multinomial(num_points, weights)
    
    x_coords, y_coords = [], []
    
    for (cx, cy), n in zip(centers, counts):
        if n == 0:
            continue
        
        # 每个聚类的标准差随机化，增加真实性
        # Randomize standard deviation for each cluster, increase realism
        sx = width * np.random.uniform(0.045, 0.080)
        sy = height * np.random.uniform(0.045, 0.080)
        
        # 使用拒绝采样生成点 / Use rejection sampling to generate points
        xs, ys = _sample_truncated_gaussian_2d(
            cx, cy, sx, sy, n, x_min, x_max, y_min, y_max
        )
        
        x_coords.append(xs)
        y_coords.append(ys)
    
    return np.concatenate(x_coords), np.concatenate(y_coords)


def _generate_clustered_distribution(num_points, x_min, x_max, y_min, y_max):
    """
    生成聚类分布（Clustered Distribution）
    随机生成聚类中心，数据点围绕这些中心聚集
    
    Parameters
    ----------
    num_points : int
        数据点数量 / Number of data points
    x_min, x_max, y_min, y_max : float
        坐标范围 / Coordinate range
    
    Returns
    -------
    x_coords, y_coords : ndarray
        X和Y坐标数组 / X and Y coordinate arrays
    
    Notes
    -----
    改进：使用拒绝采样替代clip，避免边界堆积
    Improvement: uses rejection sampling instead of clip to avoid boundary accumulation
    """
    # 随机生成 8-12 个聚类中心 / Randomly generate 8-12 cluster centers
    num_clusters = random.randint(8, 12)
    width = x_max - x_min
    height = y_max - y_min
    
    # 聚类中心位置（留出5%边界）/ Cluster center positions (leave 5% border)
    centers_x = np.random.uniform(x_min + width * 0.05, x_max - width * 0.05, num_clusters)
    centers_y = np.random.uniform(y_min + height * 0.05, y_max - height * 0.05, num_clusters)
    
    # 每个聚类的标准差不同（2%-8%范围）/ Different std for each cluster (2%-8% of range)
    cluster_std_x = np.random.uniform(width * 0.02, width * 0.08, num_clusters)
    cluster_std_y = np.random.uniform(height * 0.02, height * 0.08, num_clusters)
    
    # 分配数据点到各个聚类 / Allocate data points to each cluster
    x_coords = []
    y_coords = []
    
    for i in range(num_points):
        # 随机选择一个聚类 / Randomly select a cluster
        cluster_idx = random.randint(0, num_clusters - 1)
        
        # 使用拒绝采样：只保留在范围内的点
        # Use rejection sampling: keep only points within range
        while True:
            x = np.random.normal(centers_x[cluster_idx], cluster_std_x[cluster_idx])
            y = np.random.normal(centers_y[cluster_idx], cluster_std_y[cluster_idx])
            
            if x_min <= x <= x_max and y_min <= y <= y_max:
                x_coords.append(x)
                y_coords.append(y)
                break
    
    return np.array(x_coords), np.array(y_coords)


def _generate_correlated_distribution(num_points, x_min, x_max, y_min, y_max):
    """
    生成正相关分布（Correlated Distribution）
    X 和 Y 呈正相关关系，模拟线性相关的数据
    
    Parameters
    ----------
    num_points : int
        数据点数量 / Number of data points
    x_min, x_max, y_min, y_max : float
        坐标范围 / Coordinate range
    
    Returns
    -------
    x_coords, y_coords : ndarray
        X和Y坐标数组 / X and Y coordinate arrays
    
    Notes
    -----
    相关系数约为0.8，加上10%的噪声
    适用于房价-面积、教育-收入等正相关场景
    Correlation coefficient ~0.8, plus 10% noise
    Suitable for positively correlated scenarios like price-area, education-income
    """
    # 生成 X 坐标（均匀分布）/ Generate X coordinates (uniform distribution)
    x_coords = np.random.uniform(x_min, x_max, num_points)
    
    # Y 与 X 正相关，加上一些噪声
    # Y positively correlated with X, plus some noise
    correlation = 0.8
    noise = np.random.normal(0, (y_max - y_min) * 0.1, num_points)
    y_coords = (y_min + (x_coords - x_min) / (x_max - x_min) * (y_max - y_min) * correlation + 
                (1 - correlation) * np.random.uniform(y_min, y_max, num_points) + noise)
    
    # 裁剪到范围内 / Clip to range
    y_coords = np.clip(y_coords, y_min, y_max)
    
    return x_coords, y_coords


def _generate_anticorrelated_distribution(num_points, x_min, x_max, y_min, y_max):
    """
    生成负相关分布（Anti-correlated Distribution）
    X 和 Y 呈负相关关系
    
    Parameters
    ----------
    num_points : int
        数据点数量 / Number of data points
    x_min, x_max, y_min, y_max : float
        坐标范围 / Coordinate range
    
    Returns
    -------
    x_coords, y_coords : ndarray
        X和Y坐标数组 / X and Y coordinate arrays
    
    Notes
    -----
    相关系数约为-0.8，加上10%的噪声
    适用于价格-需求、距离-密度等负相关场景
    Correlation coefficient ~-0.8, plus 10% noise
    Suitable for negatively correlated scenarios like price-demand, distance-density
    """
    # 生成 X 坐标（均匀分布）/ Generate X coordinates (uniform distribution)
    x_coords = np.random.uniform(x_min, x_max, num_points)
    
    # Y 与 X 负相关：X越大，Y越小
    # Y negatively correlated with X: larger X means smaller Y
    correlation = 0.8
    noise = np.random.normal(0, (y_max - y_min) * 0.1, num_points)
    y_coords = (y_max - (x_coords - x_min) / (x_max - x_min) * (y_max - y_min) * correlation + 
                (1 - correlation) * np.random.uniform(y_min, y_max, num_points) + noise)
    
    # 裁剪到范围内 / Clip to range
    y_coords = np.clip(y_coords, y_min, y_max)
    
    return x_coords, y_coords


def _generate_gaussian_mixture_distribution(num_points, x_min, x_max, y_min, y_max):
    """
    生成高斯混合模型分布（Gaussian Mixture Model）
    使用 3 个不同权重的高斯分布混合
    
    Parameters
    ----------
    num_points : int
        数据点数量 / Number of data points
    x_min, x_max, y_min, y_max : float
        坐标范围 / Coordinate range
    
    Returns
    -------
    x_coords, y_coords : ndarray
        X和Y坐标数组 / X and Y coordinate arrays
    
    Notes
    -----
    改进：使用拒绝采样替代clip
    三个成分的权重分别为0.5、0.3、0.2
    Improvement: uses rejection sampling instead of clip
    Three components with weights 0.5, 0.3, 0.2
    """
    from scipy.stats import multivariate_normal
    
    width = x_max - x_min
    height = y_max - y_min
    
    # 定义三个高斯成分的参数（均值和协方差矩阵）
    # Define parameters for three Gaussian components (means and covariance matrices)
    means = [
        [x_min + width * 0.3, y_min + height * 0.3],
        [x_min + width * 0.7, y_min + height * 0.5],
        [x_min + width * 0.5, y_min + height * 0.7],
    ]
    
    covs = [
        [[(width*0.15)**2, 0], [0, (height*0.15)**2]],
        [[(width*0.1)**2, 0], [0, (height*0.1)**2]],
        [[(width*0.2)**2, 0], [0, (height*0.2)**2]],
    ]
    
    weights = [0.5, 0.3, 0.2]  # 权重 / Weights
    
    # 从多项式分布中选择成分 / Select components from multinomial distribution
    component_choices = np.random.choice(3, size=num_points, p=weights)
    
    x_coords = []
    y_coords = []
    
    for i in range(3):
        n_i = np.sum(component_choices == i)
        if n_i > 0:
            # 从第i个高斯成分采样 / Sample from i-th Gaussian component
            samples = multivariate_normal.rvs(mean=means[i], cov=covs[i], size=n_i)
            # 过滤超出范围的点 / Filter out points outside range
            mask = ((samples[:, 0] >= x_min) & (samples[:, 0] <= x_max) & 
                    (samples[:, 1] >= y_min) & (samples[:, 1] <= y_max))
            x_coords.extend(samples[mask, 0])
            y_coords.extend(samples[mask, 1])
    
    # 如果点数不足，补充采样 / If insufficient points, sample extra
    remaining = num_points - len(x_coords)
    if remaining > 0:
        extra_x, extra_y = _sample_truncated_gaussian_2d(
            x_min + width * 0.5, y_min + height * 0.5,
            width * 0.15, height * 0.15,
            remaining, x_min, x_max, y_min, y_max
        )
        x_coords.extend(extra_x)
        y_coords.extend(extra_y)
    
    return np.array(x_coords[:num_points]), np.array(y_coords[:num_points])


def _generate_spatial_temporal_distribution(num_points, x_min, x_max, y_min, y_max):
    """
    生成时空数据分布（Spatial-Temporal Distribution）
    模拟带有时间维度的空间数据，如交通流量、移动轨迹等
    
    Parameters
    ----------
    num_points : int
        数据点数量 / Number of data points
    x_min, x_max, y_min, y_max : float
        坐标范围 / Coordinate range
    
    Returns
    -------
    x_coords, y_coords : ndarray
        X和Y坐标数组 / X and Y coordinate arrays
    
    Notes
    -----
    特点：在时间和空间上都有一定的连续性和聚集性
    模拟多条移动轨迹，每条轨迹约100个点
    Features: has continuity and clustering in both time and space
    Simulates multiple movement trajectories, ~100 points per trajectory
    """
    # 模拟多个移动轨迹或事件序列 / Simulate multiple movement trajectories or event sequences
    num_trajectories = max(10, num_points // 100)  # 每条轨迹约 100 个点 / ~100 points per trajectory
    points_per_traj = num_points // num_trajectories
    
    x_coords = []
    y_coords = []
    
    for _ in range(num_trajectories):
        # 随机起点（留出边界）/ Random start point (leave border)
        start_x = np.random.uniform(x_min + 10000, x_max - 10000)
        start_y = np.random.uniform(y_min + 10000, y_max - 10000)
        
        # 生成轨迹点（布朗运动风格）/ Generate trajectory points (Brownian motion style)
        current_x = start_x
        current_y = start_y
        
        for _ in range(points_per_traj):
            x_coords.append(current_x)
            y_coords.append(current_y)
            
            # 下一步的位置（带有一定惯性，步长标准差500）
            # Next position (with some inertia, step std=500)
            step_x = np.random.normal(0, 500)
            step_y = np.random.normal(0, 500)
            current_x = np.clip(current_x + step_x, x_min, x_max)
            current_y = np.clip(current_y + step_y, y_min, y_max)
    
    # 补充剩余的点（均匀分布）/ Fill remaining points (uniform distribution)
    remaining = num_points - len(x_coords)
    if remaining > 0:
        x_coords.extend(np.random.uniform(x_min, x_max, remaining))
        y_coords.extend(np.random.uniform(y_min, y_max, remaining))
    
    return np.array(x_coords[:num_points]), np.array(y_coords[:num_points])


def _generate_city_like_distribution(num_points, x_min, x_max, y_min, y_max):
    """
    城市网格状分布（City-like Grid Distribution）
    模拟道路网络、街区结构和城市中心密度更高的空间数据。
    
    Parameters
    ----------
    num_points : int
        数据点数量 / Number of data points
    x_min, x_max, y_min, y_max : float
        坐标范围 / Coordinate range
    
    Returns
    -------
    x_coords, y_coords : ndarray
        X和Y坐标数组 / X and Y coordinate arrays
    
    Notes
    -----
    特点：
    - 80%的点沿道路分布，20%为背景噪声
    - 城市中心区域密度更高
    - 道路宽度窄（更接近真实街道）
    - 使用拒绝采样确保点在范围内
    
    Features:
    - 80% points along roads, 20% background noise
    - Higher density in city center
    - Narrow road width (closer to real streets)
    - Uses rejection sampling to ensure points within range
    """
    width = x_max - x_min
    height = y_max - y_min
    
    num_vertical_streets = 10   # 垂直道路数量 / Number of vertical streets
    num_horizontal_streets = 10 # 水平道路数量 / Number of horizontal streets
    
    # 创建网格状的"街道"，留出边界
    # Create grid-like "streets", leave borders
    vertical_x = np.linspace(x_min + width * 0.08, x_max - width * 0.08, num_vertical_streets)
    horizontal_y = np.linspace(y_min + height * 0.08, y_max - height * 0.08, num_horizontal_streets)
    
    # 80% 点沿道路分布，20% 为背景点
    # 80% points along roads, 20% background points
    road_ratio = 0.80
    road_points = int(num_points * road_ratio)
    background_points = num_points - road_points
    
    x_coords, y_coords = [], []
    
    # 生成道路上的点 / Generate points on roads
    for _ in range(road_points):
        if np.random.rand() < 0.5:
            # 垂直道路 / Vertical street
            street_x = np.random.choice(vertical_x)
            
            # 城市中心区域出现概率更高（使用正态分布）
            # Higher probability in city center (use normal distribution)
            y = np.random.normal(y_min + height * 0.5, height * 0.25)
            x = np.random.normal(street_x, width * 0.005)  # 道路宽度很窄 / Very narrow road width
        else:
            # 水平道路 / Horizontal street
            street_y = np.random.choice(horizontal_y)
            
            x = np.random.normal(x_min + width * 0.5, width * 0.25)
            y = np.random.normal(street_y, height * 0.005)  # 道路宽度很窄 / Very narrow road width
        
        # 检查是否在范围内 / Check if within range
        if x_min <= x <= x_max and y_min <= y <= y_max:
            x_coords.append(x)
            y_coords.append(y)
    
    # 如果因为拒绝采样导致数量不足，继续补充道路点
    # If rejection sampling causes insufficient quantity, continue adding road points
    while len(x_coords) < road_points:
        street_x = np.random.choice(vertical_x)
        street_y = np.random.choice(horizontal_y)
        
        if np.random.rand() < 0.5:
            x = np.random.normal(street_x, width * 0.005)
            y = np.random.uniform(y_min, y_max)
        else:
            x = np.random.uniform(x_min, x_max)
            y = np.random.normal(street_y, height * 0.005)
        
        if x_min <= x <= x_max and y_min <= y <= y_max:
            x_coords.append(x)
            y_coords.append(y)
    
    # 背景点（均匀分布）/ Background points (uniform distribution)
    x_bg = np.random.uniform(x_min, x_max, background_points)
    y_bg = np.random.uniform(y_min, y_max, background_points)
    
    x_coords.extend(x_bg)
    y_coords.extend(y_bg)
    
    return np.array(x_coords[:num_points]), np.array(y_coords[:num_points])


def _generate_hotspots_distribution(num_points, x_min, x_max, y_min, y_max):
    """
    热点区域分布（Hotspots Distribution）
    模拟真实空间数据中少数高密度区域与大量低密度背景点共存的情况。
    
    Parameters
    ----------
    num_points : int
        数据点数量 / Number of data points
    x_min, x_max, y_min, y_max : float
        坐标范围 / Coordinate range
    
    Returns
    -------
    x_coords, y_coords : ndarray
        X和Y坐标数组 / X and Y coordinate arrays
    
    Notes
    -----
    特点：
    - 75%的数据在热点区域，25%为背景噪声
    - 5个热点，权重不均衡（使用Dirichlet分布）
    - 热点聚集性强（标准差小）
    - 使用拒绝采样避免边界堆积
    
    Features:
    - 75% data in hotspot areas, 25% background noise
    - 5 hotspots with unbalanced weights (using Dirichlet distribution)
    - Strong clustering in hotspots (small std)
    - Uses rejection sampling to avoid boundary accumulation
    """
    width = x_max - x_min
    height = y_max - y_min
    
    num_hotspots = 5
    
    # 75% 热点数据，25% 背景噪声
    # 75% hotspot data, 25% background noise
    hotspot_ratio = 0.75
    hotspot_points = int(num_points * hotspot_ratio)
    background_points = num_points - hotspot_points
    
    # 生成热点中心（留出10%边界）/ Generate hotspot centers (leave 10% border)
    centers_x = np.random.uniform(x_min + width * 0.10, x_max - width * 0.10, num_hotspots)
    centers_y = np.random.uniform(y_min + height * 0.10, y_max - height * 0.10, num_hotspots)
    
    # 热点权重不均衡，更接近真实城市热点（使用Dirichlet分布）
    # Unbalanced hotspot weights, closer to real urban hotspots (using Dirichlet distribution)
    weights = np.random.dirichlet(alpha=np.ones(num_hotspots) * 0.7)
    counts = np.random.multinomial(hotspot_points, weights)
    
    x_coords, y_coords = [], []
    
    for cx, cy, n in zip(centers_x, centers_y, counts):
        if n == 0:
            continue
        
        # 热点区域的标准差较小（更集中，2%-5%范围）
        # Smaller std in hotspot areas (more concentrated, 2%-5% of range)
        sx = width * np.random.uniform(0.020, 0.050)
        sy = height * np.random.uniform(0.020, 0.050)
        
        # 使用拒绝采样生成点 / Use rejection sampling to generate points
        xs, ys = _sample_truncated_gaussian_2d(
            cx, cy, sx, sy, n, x_min, x_max, y_min, y_max
        )
        
        x_coords.append(xs)
        y_coords.append(ys)
    
    # 背景噪声（均匀分布）/ Background noise (uniform distribution)
    x_bg = np.random.uniform(x_min, x_max, background_points)
    y_bg = np.random.uniform(y_min, y_max, background_points)
    
    x_coords.append(x_bg)
    y_coords.append(y_bg)
    
    return np.concatenate(x_coords), np.concatenate(y_coords)

# ============================================================================
# 可视化函数 / Visualization Functions
# ============================================================================

def data_visualization(data_size, output_dir, image_dir, testing_flag=False, 
                      distribution_type="NORMAL", max_display=10000):
    """
    数据可视化并保存图片
    
    Parameters
    ----------
    data_size : str
        数据规模标识 / Data size identifier
    output_dir : str
        数据文件所在目录 / Directory containing data files
    image_dir : str
        图片保存目录 / Directory to save images
    testing_flag : bool
        是否为测试数据 / Whether it's test data
    distribution_type : str
        分布类型 / Distribution type
    max_display : int
        最大显示矩形数量（避免过多导致图片无法查看）
        Maximum number of points to display (avoid too many making image unreadable)
    
    Returns
    -------
    image_path : str or None
        保存的图片路径，如果失败则返回None
        Saved image path, returns None if failed
    
    Notes
    -----
    使用散点图展示数据分布，自动限制显示数量以避免图片过于密集
    Uses scatter plot to show data distribution, automatically limits display count
    """
    # 构建数据文件名 / Build data filename
    prefix = "testing_" if testing_flag else ""
    data_filename = f"{prefix}{data_size}_{distribution_type}.txt"
    data_filepath = os.path.join(output_dir, data_filename)
    
    print(f"Loading data from: {data_filepath}")
    
    # 读取数据 - 正确的解析方式（每两行为一个点的x,y坐标）
    # Read data - correct parsing method (every two lines are x,y coordinates of a point)
    x_coords = []
    y_coords = []
    
    with open(data_filepath) as input_file:
        lines = input_file.readlines()
        
    # 每两行是一对坐标 (x, y) / Every two lines are a pair of coordinates (x, y)
    for i in range(0, len(lines), 2):
        if len(x_coords) >= max_display:
            print(f"Limiting to {max_display} points for visualization")
            break
        if i + 1 < len(lines):  # 确保有配对的 y 坐标 / Ensure paired y coordinate exists
            try:
                x = float(lines[i].strip())
                y = float(lines[i+1].strip())
                x_coords.append(x)
                y_coords.append(y)
            except ValueError:
                continue
    
    print(f"Visualizing {len(x_coords)} points...")
    
    if len(x_coords) == 0:
        print("Warning: No data points to visualize!")
        return None
    
    # 创建可视化 - 使用散点图展示数据点
    # Create visualization - use scatter plot to show data points
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 使用散点图而不是矩形 / Use scatter plot instead of rectangles
    ax.scatter(x_coords, y_coords, c='blue', s=5, alpha=0.5, edgecolors='none')
    
    # 设置显示范围 / Set display range
    x_min, x_max = DEFAULT_CONFIG['x_min'], DEFAULT_CONFIG['x_max']
    y_min, y_max = DEFAULT_CONFIG['y_min'], DEFAULT_CONFIG['y_max']
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X-axis', fontsize=12)
    ax.set_ylabel('Y-axis', fontsize=12)
    ax.set_title(
        f'{data_size} Dataset ({distribution_type})\n'
        f'Total: {len(x_coords)} points displayed',
        fontsize=14
    )
    ax.grid(True, alpha=0.3)
    
    # 保存图片 / Save image
    os.makedirs(image_dir, exist_ok=True)
    image_filename = f"{prefix}{data_size}_{distribution_type}.png"
    image_path = os.path.join(image_dir, image_filename)
    
    plt.savefig(image_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Image saved to: {image_path}")
    return image_path

# ============================================================================
# 便捷函数 / Convenience Functions
# ============================================================================

def generate_batch(configs, output_dir="generated_data", image_dir="images"):
    """
    批量生成多组数据
    
    Parameters
    ----------
    configs : list of dict
        配置列表，每个配置包含：
        - data_size: 数据规模
        - distribution_type: 分布类型
        - testing_flag: 是否测试数据 (可选)
        
        Configuration list, each config contains:
        - data_size: data size
        - distribution_type: distribution type
        - testing_flag: whether test data (optional)
    output_dir : str
        数据输出目录 / Data output directory
    image_dir : str
        图片输出目录 / Image output directory
    
    Returns
    -------
    results : list of dict
        生成结果列表，每个结果包含：
        - data_file: 数据文件路径
        - image_file: 图片文件路径
        - status: 状态 ('success' 或 'failed')
        
        Generation results list, each result contains:
        - data_file: data file path
        - image_file: image file path
        - status: status ('success' or 'failed')
    
    Notes
    -----
    批量处理多个配置，自动处理错误并返回结果摘要
    Processes multiple configurations in batch, handles errors automatically
    """
    print("=" * 70)
    print("Batch Data Generation")
    print("=" * 70)
    
    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Generating {config['data_size']} - {config['distribution_type']}")
        print("-" * 70)
        
        try:
            # 生成坐标数据 / Generate coordinate data
            output_file = generate_coordinates(
                x_min=DEFAULT_CONFIG['x_min'],
                x_max=DEFAULT_CONFIG['x_max'],
                y_min=DEFAULT_CONFIG['y_min'],
                y_max=DEFAULT_CONFIG['y_max'],
                data_size=config['data_size'],
                output_dir=output_dir,
                testing_flag=config.get('testing_flag', False),
                distribution_type=config['distribution_type']
            )
            
            # 生成可视化图片 / Generate visualization image
            image_path = data_visualization(
                data_size=config['data_size'],
                output_dir=output_dir,
                image_dir=image_dir,
                testing_flag=config.get('testing_flag', False),
                distribution_type=config['distribution_type']
            )
            
            results.append({
                'data_file': output_file,
                'image_file': image_path,
                'status': 'success'
            })
            
            # 删除临时TXT文件（只保留NPY格式）
            # Delete temporary TXT file (keep only NPY format)
            os.remove(output_file)
            
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append({
                'config': config,
                'status': 'failed',
                'error': str(e)
            })
    
    print("\n" + "=" * 70)
    print("Generation Summary")
    print("=" * 70)
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"Success: {success_count}/{len(results)}")
    
    return results

# ============================================================================
# 主函数 / Main Function
# ============================================================================

if __name__ == "__main__":
    # 获取项目根目录 / Get project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 配置输出目录 / Configure output directories
    data_output_dir = os.path.join(project_root, "generated_data")
    image_output_dir = os.path.join(project_root, "images")
    
    print("=" * 70)
    print("GSARTree Data Generator")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print(f"Data output: {data_output_dir}")
    print(f"Image output: {image_output_dir}")
    print("=" * 70)
    
    # 示例配置 / Example configurations
    configs = [
        {'data_size': 'W1', 'distribution_type': 'UNIFORM', 'testing_flag': True},
        {'data_size': 'W1', 'distribution_type': 'NORMAL', 'testing_flag': True},
        {'data_size': 'W2', 'distribution_type': 'UNIFORM', 'testing_flag': False},
    ]
    
    # 批量生成 / Batch generation
    results = generate_batch(
        configs=configs,
        output_dir=data_output_dir,
        image_dir=image_output_dir
    )
    
    print("\n✓ All generation tasks completed!")
