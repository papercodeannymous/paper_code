#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试所有数据分布类型
生成小规模数据集来验证每种分布类型的效果
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generator import generate_batch, DATADISTRIBUTION_TYPE
from data_config import QUICK_TEST_CONFIG

def test_all_distributions():
    """测试所有分布类型"""
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 配置输出目录
    data_output_dir = os.path.join(project_root, "generated_data", "test_distributions")
    image_output_dir = os.path.join(project_root, "images", "test_distributions")
    
    print("=" * 80)
    print("测试所有数据分布类型")
    print("=" * 80)
    print(f"📁 项目根目录：{project_root}")
    print(f"📂 数据输出目录：{data_output_dir}")
    print(f"🖼️  图片输出目录：{image_output_dir}")
    print("=" * 80)
    print()
    
    # 为每种分布类型生成一个小数据集（H1=100 个点）
    configs = []
    
    # 按类别组织分布类型
    categories = {
        '基础分布': ['UNIFORM', 'NORMAL', 'SKEW-NOR2'],
        '多峰分布': ['BIMODAL', 'MULTI-MODAL', 'CLUSTERED'],
        '复杂分布': ['CORRELATED', 'ANTI-CORR', 'GAUSSIAN-MIX'],
        '时空数据': ['SPATIAL-TEMPORAL', 'CITY-LIKE', 'HOTSPOTS']
    }
    
    for category, distributions in categories.items():
        print(f"\n{'='*80}")
        print(f"类别：{category}")
        print(f"{'='*80}")
        
        for dist_type in distributions:
            config = {
                'data_size': 'W1',  # 使用最小规模快速测试
                'distribution_type': dist_type,
                'testing_flag': True
            }
            configs.append(config)
            
            print(f"\n生成 {dist_type} 分布...")
            print("-" * 80)
            
            try:
                result = generate_batch(
                    configs=[config],
                    output_dir=data_output_dir,
                    image_dir=image_output_dir
                )
                
                if result[0]['status'] == 'success':
                    print(f"✓ {dist_type} 生成成功")
                    print(f"  数据文件：{result[0]['data_file']}")
                    print(f"  图片文件：{result[0]['image_file']}")
                else:
                    print(f"✗ {dist_type} 生成失败")
                
            except Exception as e:
                print(f"✗ {dist_type} 生成失败：{e}")
            
            print()
    
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"总共测试了 {len(configs)} 种分布类型")
    print(f"数据保存在：{data_output_dir}")
    print(f"图片保存在：{image_output_dir}")
    print()
    print("查看生成的图片:")
    print(f"  ls -lh {image_output_dir}/")
    print()
    print("下一步:")
    print("  1. 查看生成的分布图片，了解各种分布的特点")
    print("  2. 选择合适的分布类型进行大规模测试")
    print("  3. 使用 rtree.py 测试 RTree 在不同分布下的性能")
    print("=" * 80)


if __name__ == "__main__":
    test_all_distributions()
