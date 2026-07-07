#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
奖励诊断脚本 - 验证环境是否能产生非零奖励

用法：
    python gsartree/debug_reward.py
"""
import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gsartree.config.default_config import create_training_config
from gsartree.environment.rtree_env import RTreeEnvironment
from gsartree.environment.reward_calculator import RewardCalculator


def test_random_insertion_reward():
    """测试随机插入策略下的奖励分布"""
    print("=" * 80)
    print("Test 1: Random Insertion Strategy - Reward Distribution")
    print("=" * 80)
    
    # 创建配置
    config = create_training_config(
        distribution='NORMAL',
        train_volume='W1',
        max_entry=20,
        feature_type=125
    )
    
    # 创建两个环境（Agent 和 Reference）
    agent_env = RTreeEnvironment(config, tree_type="acppo")
    ref_env = RTreeEnvironment(config, tree_type=config.tree.reference_tree_type)
    
    # 创建奖励计算器
    reward_calc = RewardCalculator(
        query_area_ratio=0.05,
        num_samples=10,
        reward_type="access_rate_diff"
    )
    
    # 生成随机数据
    n_samples = 10000
    dataset = []
    for _ in range(n_samples):
        x = np.random.uniform(0, 100000)
        y = np.random.uniform(0, 100000)
        w = np.random.uniform(50, 200)
        h = np.random.uniform(50, 200)
        dataset.append([x, y, x + w, y + h])
    
    # 模拟训练过程
    rewards = []
    rl_decisions = 0
    heuristic_decisions = 0
    
    for i, rect in enumerate(dataset):
        # 插入到参考树（使用 R*-Tree 策略）
        ref_env.insert_rectangle(rect)
        
        # 插入到 Agent 树（使用 AC-PPO 树的默认启发式策略）
        agent_env.insert_rectangle(rect)
        
        heuristic_decisions += 1
        
        # 每 10 步计算一次奖励
        if (i + 1) % 10 == 0:
            recent_rects = dataset[max(0, i - 9):i + 1]
            reward = reward_calc.calculate_reward(
                agent_env,
                ref_env,
                recent_rects,
                x_range=(config.data.x_min, config.data.x_max),
                y_range=(config.data.y_min, config.data.y_max)
            )
            rewards.append(reward)
            
            # 打印详细信息
            print(f"\nStep {i + 1}:")
            print(f"  Reward: {reward:.6f}")
            
            # 检查访问率
            query_rect = reward_calc.generate_random_query(
                x_range=(config.data.x_min, config.data.x_max),
                y_range=(config.data.y_min, config.data.y_max)
            )
            agent_rate = agent_env.access_rate(query_rect)
            ref_rate = ref_env.access_rate(query_rect)
            print(f"  Query Access Rate - Agent: {agent_rate:.4f}, Ref: {ref_rate:.4f}")
            
            # 检查节点访问数
            agent_access = agent_env.query(query_rect)
            ref_access = ref_env.query(query_rect)
            print(f"  Node Access - Agent: {agent_access}, Ref: {ref_access}")
    
    # 统计奖励分布
    print("\n" + "=" * 80)
    print("Reward Statistics:")
    print("=" * 80)
    print(f"Total samples: {len(rewards)}")
    print(f"Mean reward: {np.mean(rewards):.6f}")
    print(f"Std reward: {np.std(rewards):.6f}")
    print(f"Min reward: {np.min(rewards):.6f}")
    print(f"Max reward: {np.max(rewards):.6f}")
    print(f"Non-zero rewards: {np.sum(np.array(rewards) != 0)} / {len(rewards)}")
    
    if np.all(np.array(rewards) == 0):
        print("\n⚠️  WARNING: All rewards are ZERO!")
        print("Possible causes:")
        print("  1. Both trees have identical structure (heuristic vs heuristic)")
        print("  2. access_rate() or query() methods return same values")
        print("  3. Reward calculation logic has a bug")
    else:
        print("\n✓ Environment can produce non-zero rewards!")
    
    return rewards


def test_tree_structure_difference():
    """测试两棵树的结构差异"""
    print("\n" + "=" * 80)
    print("Test 2: Tree Structure Comparison")
    print("=" * 80)
    
    config = create_training_config(
        distribution='NORMAL',
        train_volume='W1',
        max_entry=50,
        feature_type=125
    )
    
    # 创建两种不同类型的树
    env1 = RTreeEnvironment(config, tree_type="acppo")
    env2 = RTreeEnvironment(config, tree_type="rstar")
    
    # 插入相同的数据
    n_samples = 50
    for _ in range(n_samples):
        x = np.random.uniform(0, 100000)
        y = np.random.uniform(0, 100000)
        w = np.random.uniform(50, 200)
        h = np.random.uniform(50, 200)
        rect = [x, y, x + w, y + h]
        
        env1.insert_rectangle(rect)
        env2.insert_rectangle(rect)
    
    # 比较树结构
    print(f"\nTree 1 (AC-PPO):")
    print(f"  Height: {env1.get_tree_height()}")
    print(f"  Num Nodes: {env1.get_num_nodes()}")
    print(f"  Fill Factor: {env1.get_fill_factor():.4f}")
    
    print(f"\nTree 2 (R*-Tree):")
    print(f"  Height: {env2.get_tree_height()}")
    print(f"  Num Nodes: {env2.get_num_nodes()}")
    print(f"  Fill Factor: {env2.get_fill_factor():.4f}")
    
    # 执行查询对比
    query_rect = [40000, 40000, 60000, 60000]
    access1 = env1.query(query_rect)
    access2 = env2.query(query_rect)
    
    print(f"\nQuery Performance ([40000, 40000, 60000, 60000]):")
    print(f"  Tree 1 node access: {access1}")
    print(f"  Tree 2 node access: {access2}")
    print(f"  Difference: {abs(access1 - access2)}")
    
    if access1 == access2:
        print("\n⚠️  WARNING: Both trees return same query result!")
    else:
        print("\n✓ Trees have different structures and query behaviors!")


if __name__ == "__main__":
    print("Starting Reward Diagnosis...\n")
    
    # Test 1: 奖励分布测试
    rewards = test_random_insertion_reward()
    
    # Test 2: 树结构差异测试
    test_tree_structure_difference()
    
    print("\n" + "=" * 80)
    print("Diagnosis Complete!")
    print("=" * 80)
