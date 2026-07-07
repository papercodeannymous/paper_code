#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速生成示例数据
演示如何使用数据生成器
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generator import generate_batch, generate_coordinates, data_visualization
from data_config import QUICK_TEST_CONFIG, STANDARD_CONFIG

def main():
    """主函数"""
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 配置输出目录（统一保存到项目目录）
    data_output_dir = os.path.join(project_root, "generated_data")
    image_output_dir = os.path.join(project_root, "images")
    
    print("=" * 70)
    print("GSARTree 快速数据生成示例")
    print("=" * 70)
    print()
    print(f"📁 项目根目录：{project_root}")
    print(f"📂 数据输出目录：{data_output_dir}")
    print(f"🖼️  图片输出目录：{image_output_dir}")
    print()
    
    # 选择要使用的配置
    print("可用的配置选项:")
    print("  1. QUICK_TEST_CONFIG   - 快速测试 (H1, 2 个数据集)")
    print("  2. STANDARD_CONFIG     - 标准测试 (W1-W2, 5 个数据集)")
    print("  3. CUSTOM              - 自定义配置")
    print()
    
    # 默认使用快速测试配置
    selected_config = QUICK_TEST_CONFIG
    print(f"✓ 使用配置：QUICK_TEST_CONFIG")
    print()
    
    # 批量生成数据
    results = generate_batch(
        configs=selected_config,
        output_dir=data_output_dir,
        image_dir=image_output_dir
    )
    
    print()
    print("=" * 70)
    print("生成完成！")
    print("=" * 70)
    print()
    print("生成的文件位置:")
    for result in results:
        if result['status'] == 'success':
            print(f"  ✓ 数据：{result['data_file']}")
            print(f"  ✓ 图片：{result['image_file']}")
            print()
    
    print("下一步:")
    print("  1. 查看生成的数据文件：ls -lh generated_data/")
    print("  2. 查看生成的图片：ls -lh images/")
    print("  3. 使用 rtree.py 测试 RTree 性能")
    print()

if __name__ == "__main__":
    main()
