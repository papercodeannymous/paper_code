#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSARTree 数据生成命令行工具
提供简单的命令行接口来生成数据
"""

import sys
import os
import argparse

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)

from data_generator import generate_batch, generate_coordinates, data_visualization, DATASIZE_TYPE, DATADISTRIBUTION_TYPE
from data_config import QUICK_TEST_CONFIG, STANDARD_CONFIG, BENCHMARK_CONFIG

def list_sizes():
    """列出所有可用的数据规模"""
    print("\n可用的数据规模 (DATASIZE_TYPE):")
    print("-" * 60)
    for name, size in sorted(DATASIZE_TYPE.items(), key=lambda x: x[1]):
        print(f"  {name:8s} : {size:>12,} 个矩形")
    print()

def list_distributions():
    """列出所有可用的分布类型"""
    print("\n可用的分布类型 (DATADISTRIBUTION_TYPE):")
    print("=" * 80)
    
    categories = {
        '基础分布': ['UNIFORM', 'NORMAL', 'SKEW-NOR0', 'SKEW-NOR2', 'SKEW-NOR4', 'SKEW-NOR8'],
        '多峰分布': ['BIMODAL', 'MULTI-MODAL', 'CLUSTERED'],
        '复杂分布': ['CORRELATED', 'ANTI-CORR', 'GAUSSIAN-MIX'],
        '时空数据': ['SPATIAL-TEMPORAL', 'CITY-LIKE', 'HOTSPOTS']
    }
    
    descriptions = {
        'UNIFORM': '均匀分布（完全随机）',
        'NORMAL': '正态分布（中心聚集）',
        'SKEW-NOR0': '偏态分布 (α=0，接近均匀)',
        'SKEW-NOR2': '偏态分布 (α=2，轻度偏斜)',
        'SKEW-NOR4': '偏态分布 (α=4，中度偏斜)',
        'SKEW-NOR8': '偏态分布 (α=8，重度偏斜)',
        'BIMODAL': '双峰分布（两个聚集中心）',
        'MULTI-MODAL': '多峰分布（多个聚集中心）',
        'CLUSTERED': '聚类分布（随机聚类）',
        'CORRELATED': '正相关分布（X-Y 线性相关）',
        'ANTI-CORR': '负相关分布（X-Y 反向相关）',
        'GAUSSIAN-MIX': '高斯混合模型（多成分混合）',
        'SPATIAL-TEMPORAL': '时空数据（轨迹序列）',
        'CITY-LIKE': '城市网格状分布',
        'HOTSPOTS': '热点区域分布'
    }
    
    for category, dists in categories.items():
        print(f"\n{category}:")
        print("-" * 80)
        for dist in dists:
            desc = descriptions.get(dist, '')
            print(f"  {dist:20s} : {desc}")
    print()

def main():
    parser = argparse.ArgumentParser(
        description='GSARTree 数据生成器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            示例用法:
            # 快速测试
            python cli.py --quick
            
            # 生成单个数据集
            python cli.py --size W1 --dist UNIFORM --test
            
            # 批量生成标准测试数据
            python cli.py --standard
            
            # 自定义输出目录
            python cli.py --size W1 --dist NORMAL --output my_data --images my_images
                    """
    )
    
    # 预设配置选项
    preset_group = parser.add_argument_group('预设配置')
    preset_group.add_argument('--quick', action='store_true',
                             help='使用快速测试配置 (H1 规模)')
    preset_group.add_argument('--standard', action='store_true',
                             help='使用标准测试配置 (W1-W2 规模)')
    preset_group.add_argument('--benchmark', action='store_true',
                             help='使用基准测试配置 (W1-W5 多规模)')
    
    # 自定义配置选项
    custom_group = parser.add_argument_group('自定义配置')
    custom_group.add_argument('--size', type=str,
                             help='数据规模 (如 W1, W2, HW1 等)')
    custom_group.add_argument('--dist', type=str, choices=DATADISTRIBUTION_TYPE.keys(),
                             help='分布类型 (如 UNIFORM, NORMAL 等)')
    custom_group.add_argument('--test', action='store_true',
                             help='标记为测试数据（生成 .npy 格式）')
    custom_group.add_argument('--no-image', action='store_true',
                             help='不生成可视化图片')
    
    # 输出目录配置
    output_group = parser.add_argument_group('输出目录配置')
    output_group.add_argument('--output', '-o', type=str, default='generated_data',
                             help='数据输出目录 (默认：generated_data)')
    output_group.add_argument('--images', '-i', type=str, default='images',
                             help='图片输出目录 (默认：images)')
    
    # 信息选项
    info_group = parser.add_argument_group('信息查询')
    info_group.add_argument('--list-sizes', action='store_true',
                           help='列出所有可用的数据规模')
    info_group.add_argument('--list-dists', action='store_true',
                           help='列出所有可用的分布类型')
    
    args = parser.parse_args()
      
    # 显示信息
    if args.list_sizes:
        list_sizes()
        return
    
    if args.list_dists:
        list_distributions()
        return
    
    # 确定要使用的配置
    configs = []
    if args.quick:
        configs = QUICK_TEST_CONFIG
        print("✓ 使用快速测试配置")
    elif args.standard:
        configs = STANDARD_CONFIG
        print("✓ 使用标准测试配置")
    elif args.benchmark:
        configs = BENCHMARK_CONFIG
        print("✓ 使用基准测试配置")
    elif args.size and args.dist:
        configs = [{
            'data_size': args.size,
            'distribution_type': args.dist,
            'testing_flag': args.test
        }]
        print(f"✓ 自定义配置：{args.size} - {args.dist}")
    else:
        print("❌ 错误：请指定预设配置 (--quick/--standard/--benchmark) 或自定义配置 (--size 和 --dist)")
        print("使用 --help 查看帮助")
        return
    
    # 设置输出目录（相对于项目根目录）
    data_output_dir = os.path.join(project_root, args.output)
    image_output_dir = os.path.join(project_root, args.images)
    
    print(f"\n📁 数据输出目录：{data_output_dir}")
    print(f"🖼️  图片输出目录：{image_output_dir}")
    print()
    
    # 批量生成数据
    results = generate_batch(
        configs=configs,
        output_dir=data_output_dir,
        image_dir=image_output_dir if not args.no_image else None
    )
    
    # 显示结果摘要
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"\n✅ 成功：{success_count}/{len(results)}")
    
    if success_count > 0:
        print(f"\n💡 提示:")
        print(f"   - 查看生成的数据：ls -lh {args.output}/")
        print(f"   - 查看生成的图片：ls -lh {args.images}/")
        print(f"   - 测试 RTree: python3 gsartree/rtree.py")

if __name__ == "__main__":
    main()
