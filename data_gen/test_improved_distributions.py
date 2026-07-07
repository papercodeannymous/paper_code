#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试改进后的三种分布：MULTI-MODAL、HOTSPOTS、CITY-LIKE
验证拒绝采样机制和分布质量
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_generator import (
    _generate_multimodal_distribution,
    _generate_hotspots_distribution,
    _generate_city_like_distribution,
    DEFAULT_CONFIG
)


def test_distribution(name, generator_func, num_points=10000):
    """测试单个分布并可视化"""
    print(f"\n{'='*70}")
    print(f"Testing {name} Distribution")
    print(f"{'='*70}")
    
    # 生成数据
    x_coords, y_coords = generator_func(
        num_points,
        DEFAULT_CONFIG['x_min'],
        DEFAULT_CONFIG['x_max'],
        DEFAULT_CONFIG['y_min'],
        DEFAULT_CONFIG['y_max']
    )
    
    # 基本统计检查
    print(f"Generated points: {len(x_coords)}")
    print(f"X range: [{x_coords.min():.2f}, {x_coords.max():.2f}]")
    print(f"Y range: [{y_coords.min():.2f}, {y_coords.max():.2f}]")
    print(f"X mean: {x_coords.mean():.2f}, std: {x_coords.std():.2f}")
    print(f"Y mean: {y_coords.mean():.2f}, std: {y_coords.std():.2f}")
    
    # 检查是否有越界点
    out_of_bounds = np.sum(
        (x_coords < DEFAULT_CONFIG['x_min']) | 
        (x_coords > DEFAULT_CONFIG['x_max']) |
        (y_coords < DEFAULT_CONFIG['y_min']) | 
        (y_coords > DEFAULT_CONFIG['y_max'])
    )
    print(f"Out of bounds points: {out_of_bounds} (should be 0)")
    
    if out_of_bounds > 0:
        print("⚠️  WARNING: Found out-of-bounds points!")
        return False
    
    # 可视化
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(x_coords, y_coords, c='blue', s=5, alpha=0.5, edgecolors='none')
    ax.set_xlim(DEFAULT_CONFIG['x_min'], DEFAULT_CONFIG['x_max'])
    ax.set_ylim(DEFAULT_CONFIG['y_min'], DEFAULT_CONFIG['y_max'])
    ax.set_xlabel('X-axis', fontsize=12)
    ax.set_ylabel('Y-axis', fontsize=12)
    ax.set_title(f'{name} Distribution ({num_points:,} points)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 保存图片
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images")
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, f"test_{name.lower().replace('-', '_')}.png")
    plt.savefig(image_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Visualization saved to: {image_path}")
    print(f"✓ {name} distribution test PASSED")
    
    return True


def main():
    """运行所有分布测试"""
    print("="*70)
    print("GSARTree Distribution Quality Test")
    print("="*70)
    
    tests = [
        ("MULTI-MODAL", _generate_multimodal_distribution),
        ("HOTSPOTS", _generate_hotspots_distribution),
        ("CITY-LIKE", _generate_city_like_distribution),
    ]
    
    results = []
    for name, func in tests:
        try:
            passed = test_distribution(name, func, num_points=10000)
            results.append((name, passed))
        except Exception as e:
            print(f"✗ {name} test FAILED with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 总结
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:20s} {status}")
    
    all_passed = all(passed for _, passed in results)
    print("="*70)
    if all_passed:
        print("✓ All tests PASSED!")
    else:
        print("✗ Some tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
