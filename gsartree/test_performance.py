#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTree 性能测试工具
测试 RTree 在各种数据分布下的查询性能
"""

import sys
import os

# 添加当前目录到路径（支持直接运行脚本）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 兼容两种导入方式：直接运行脚本 vs 作为模块导入
try:
    # 尝试相对导入（当作为模块时）
    from .rtree import tree_variants_query_test, comprehensive_distribution_test
except ImportError:
    # 直接运行脚本时使用绝对导入
    from rtree import tree_variants_query_test, comprehensive_distribution_test


def test_single_distribution():
    """测试单个分布类型"""
    print("=" * 80)
    print("RTree Single Distribution Test")
    print("=" * 80)
    
    # 配置测试参数
    data_size = 'W1'  # 使用小规模快速测试
    
    # 测试几种代表性分布
    distributions = [
        ('UNIFORM', '均匀分布'),
        ('BIMODAL', '双峰分布'),
        ('CLUSTERED', '聚类分布'),
        ('CORRELATED', '正相关分布'),
        ('HOTSPOTS', '热点分布')
    ]
    
    results = {}
    
    for dist_type, desc in distributions:
        print(f"\n{'='*80}")
        print(f"Testing: {dist_type} ({desc})")
        print(f"{'='*80}\n")
        
        try:
            result = tree_variants_query_test(
                data_size=data_size,
                distribution_type=dist_type,
                query_type='range',
                num_queries=1000
            )
            results[dist_type] = result
            print(f"✓ {dist_type} test completed\n")
        except Exception as e:
            print(f"✗ {dist_type} test failed: {e}\n")
            results[dist_type] = None
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    successful = sum(1 for r in results.values() if r is not None)
    print(f"Successful: {successful}/{len(results)}")
    print("=" * 80)


def test_all_distributions():
    """测试所有 15 种分布类型"""
    print("=" * 80)
    print("RTree Comprehensive Performance Test")
    print("Testing all 15 distribution types")
    print("=" * 80)
    
    results = comprehensive_distribution_test()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RTree Performance Testing Tool')
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'full', 'custom'],
                       help='Test mode: quick (5 representative), full (all 15), or custom')
    parser.add_argument('--size', type=str, default='W1',
                       help='Data size for custom mode')
    parser.add_argument('--dist', type=str, default='UNIFORM',
                       help='Distribution type for custom mode')
    parser.add_argument('--query', type=str, default='range',
                       choices=['range', 'knn'],
                       help='Query type')
    parser.add_argument('--num-queries', type=int, default=1000,
                       help='Number of queries')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        test_single_distribution()
    elif args.mode == 'full':
        test_all_distributions()
    else:  # custom
        tree_variants_query_test(
            data_size=args.size,
            distribution_type=args.dist,
            query_type=args.query,
            num_queries=args.num_queries
        )
