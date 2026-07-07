#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试几种代表性的分布类型并验证可视化效果
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generator import generate_batch

def quick_test():
    """快速测试 5 种代表性分布"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    data_output_dir = os.path.join(project_root, "generated_data", "quick_test")
    image_output_dir = os.path.join(project_root, "images", "quick_test")
    
    # 测试 5 种代表性分布
    configs = [
        {'data_size': 'H1', 'distribution_type': 'UNIFORM', 'testing_flag': True},
        {'data_size': 'H1', 'distribution_type': 'BIMODAL', 'testing_flag': True},
        {'data_size': 'H1', 'distribution_type': 'CLUSTERED', 'testing_flag': True},
        {'data_size': 'H1', 'distribution_type': 'CORRELATED', 'testing_flag': True},
        {'data_size': 'H1', 'distribution_type': 'HOTSPOTS', 'testing_flag': True},
    ]
    
    print("=" * 80)
    print("快速测试：5 种代表性分布类型")
    print("=" * 80)
    
    results = generate_batch(
        configs=configs,
        output_dir=data_output_dir,
        image_dir=image_output_dir
    )
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"\n✅ 成功：{success_count}/{len(configs)}")
    
    if success_count > 0:
        print(f"\n💡 查看结果:")
        print(f"   数据：ls -lh {data_output_dir}/")
        print(f"   图片：ls -lh {image_output_dir}/")
        print(f"\n📊 提示：打开生成的 PNG 图片查看数据点分布效果")

if __name__ == "__main__":
    quick_test()