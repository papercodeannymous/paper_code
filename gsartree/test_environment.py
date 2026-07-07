#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试环境模块
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_environment():
    """测试 RTree 环境类"""
    print("*" * 80)
    print("Testing RTree Environment")
    print("*" * 80)
    
    try:
        from config.default_config import create_training_config
        from environment.rtree_env import RTreeEnvironment, MultiTreeEnvironment
        print("✓ Successfully imported environment classes")
        
        config = create_training_config(max_entry=5)
        
        test_rects = [
            #原始数据
            [100, 100, 200, 200],
            [150, 150, 250, 250],
            [300, 300, 400, 400],
            
            # 新增：高密度重叠区域 (测试节点分裂逻辑)
            [120, 120, 180, 180],
            [130, 130, 190, 190],
            [140, 140, 210, 210],
            [160, 160, 220, 220],
            
            # 新增：分散的大矩形 (测试覆盖范围)
            [500, 500, 800, 800],
            [10, 10, 50, 50],
            [900, 900, 950, 950],
            
            # 新增：细长矩形 (测试不同纵横比)
            [400, 100, 410, 300],
            [100, 400, 300, 410],
            
            # 新增：相邻但不重叠 (测试边界情况)
            [200, 200, 300, 300], 
        ]
        query_rect = [120, 120, 180, 180]
        
        """ 
        测试单个树环境 
        """
        print("\nTesting Single Tree Environment:")
        env = RTreeEnvironment(config, tree_type="rstar")
        print(f"✓ Created R*-Tree environment")
        print(f"  - Tree type: {env.tree_type}")
        print(f"  - Max entry: {config.tree.max_entry}")
        
        for rect in test_rects:
            env.insert_rectangle(rect)
        
        print(f"\n✓ Inserted {len(test_rects)} rectangles")
        print(f"  - Tree height: {env.get_tree_height()}")
        print(f"  - Num nodes: {env.get_num_nodes()}")
        print(f"  - Num objects: {env.get_num_objects()}")
        print(f"  - Fill factor: {env.get_fill_factor():.2%}")
        
        # 测试查询
        accessed = env.query(query_rect)
        print(f"\n✓ Query test:")
        print(f"  - Query rect: {query_rect}")
        print(f"  - Nodes accessed: {accessed}")
        
        # 测试统计信息
        stats = env.get_stats()
        print(f"\n✓ Tree statistics:")
        for key, value in stats.items():
            print(f"  - {key}: {value}")
        
        """ 
        测试多树环境
        """
        
        # 测试多树环境
        print("\n\nTesting Multi-Tree Environment:")
        multi_env = MultiTreeEnvironment(config)
        
        # 添加不同类型的树
        multi_env.add_tree("acppo", "acppo")
        multi_env.add_tree("rstar", "rstar")
        multi_env.add_tree("rtree", "rtree")
        print(f"✓ Created {len(multi_env.environments)} trees")
        
        # 向所有树插入相同数据
        # 注意：在测试阶段，我们只测试基本功能，不使用 RL（因为还没有训练好的 agent）
        # 实际使用时，可以通过 insert_to_all(rect, rl_trees=["acppo"], agent_dict={...}) 来启用 RL
        for rect in test_rects:
            multi_env.insert_to_all(rect)
        
        print(f"\n✓ Inserted {len(test_rects)} rectangles to all trees")
        
        # 比较性能
        comparison = multi_env.compare_performance(query_rect)
        print(f"\n✓ Performance comparison (without RL agent):")
        for name, metrics in comparison.items():
            print(f"  - {name}:")
            for key, value in metrics.items():
                print(f"      {key}: {value}")
        
        # 获取所有树的统计
        all_stats = multi_env.get_all_stats()
        print(f"\n✓ All trees statistics retrieved:")
        for name, stats in all_stats.items():
            print(f"  - {name}: height={stats['height']}, nodes={stats['num_nodes']}, objects={stats['num_objects']}")
        
        print("\nAll environment tests passed!")
        return True
        
    except Exception as e:
        print(f"\n[X] Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False



def test_reward_calculator_v0():
    """测试奖励计算器"""
    print("\n" + "=" * 80)
    print("Testing Reward Calculator")
    print("=" * 80)
    
    try:
        from config.default_config import create_training_config
        from environment.rtree_env import RTreeEnvironment
        from environment.reward_calculator import RewardCalculator, ZeroSumRewardCalculator
        
        print("✓ Successfully imported reward calculator classes")
        
        # 创建配置和环境
        config = create_training_config()
        
        agent_env = RTreeEnvironment(config, "acppo")
        ref_env = RTreeEnvironment(config, "rstar")
        
        # 插入一些数据
        test_rects = [
            [100, 100, 200, 200],
            [150, 150, 250, 250],
            [300, 300, 400, 400],
            [500, 500, 600, 600],
            [700, 700, 800, 800],
        ]
        
        for rect in test_rects:
            agent_env.insert_rectangle(rect)
            ref_env.insert_rectangle(rect)
        
        print(f"\n✓ Created two trees with {len(test_rects)} rectangles each")
        
        # 测试奖励计算器
        calculator = RewardCalculator(
            query_area_ratio=0.05,
            num_samples=3,
            reward_type="access_rate_diff"
        )
        
        reward = calculator.calculate_reward(
            agent_env,
            ref_env,
            test_rects
        )
        print(f"\n✓ Reward calculation:")
        print(f"  - Reward type: {calculator.reward_type}")
        print(f"  - Calculated reward: {reward:.4f}")
        print(f"  - Interpretation: {'Agent is better' if reward > 0 else 'Reference is better' if reward < 0 else 'Equal'}")
        
        # 测试零和奖励
        print("\n Testing Zero-Sum Reward:")
        zero_sum_calc = ZeroSumRewardCalculator()
        
        r1, r2 = zero_sum_calc.calculate_zero_sum_rewards(
            agent_env,
            ref_env,
            test_rects
        )
        
        print(f"  - Player 1 reward: {r1:.4f}")
        print(f"  - Player 2 reward: {r2:.4f}")
        print(f"  - Sum (should be ~0): {r1 + r2:.6f}")
        
        if abs(r1 + r2) < 1e-6:
            print("  ✓ Zero-sum property verified!")
        else:
            print("Warning: Zero-sum property not perfectly satisfied")
        
        print("\nAll reward calculator tests passed!")
        
        return True
        
    except Exception as e:
        print(f"\n[X] Reward calculator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reward_calculator():
    """测试奖励计算器"""
    print("\n" + "=" * 80)
    print("Testing Reward Calculator")
    print("=" * 80)
    
    try:
        from config.default_config import create_training_config
        from environment.rtree_env import RTreeEnvironment
        from environment.reward_calculator import RewardCalculator, ZeroSumRewardCalculator
        import random
        
        print("✓ Successfully imported reward calculator classes")
        
        # 创建配置和环境
        config = create_training_config(max_entry=5)
        # 稍微调整配置以增加树的复杂度，例如减小 max_entry 强制分裂
        # config.tree.max_entry = 4 
        
        agent_env = RTreeEnvironment(config, "acppo")
        ref_env = RTreeEnvironment(config, "rstar")
        
        # 生成更多、更复杂的测试数据
        random.seed(42) # 固定种子以便复现
        num_rects = 100
        test_rects = []
        for _ in range(num_rects):
            x = random.randint(0, 1000)
            y = random.randint(0, 1000)
            w = random.randint(10, 100)
            h = random.randint(10, 100)
            test_rects.append([x, y, x + w, y + h])
        
        print(f"\n📊 Inserting {num_rects} random rectangles...")
        for rect in test_rects:
            agent_env.insert_rectangle(rect)
            ref_env.insert_rectangle(rect)
        
        print(f"✓ Created two trees with {num_rects} rectangles each")
        print(f"  - Agent Env Stats: H={agent_env.get_tree_height()}, N={agent_env.get_num_nodes()}")
        print(f"  - Ref Env Stats:   H={ref_env.get_tree_height()}, N={ref_env.get_num_nodes()}")
        
        # 测试奖励计算器
        calculator = RewardCalculator(
            query_area_ratio=0.05,
            num_samples=10, # 增加采样次数以获得更稳定的平均值
            reward_type="access_rate_diff"
        )
        
        reward = calculator.calculate_reward(
            agent_env,
            ref_env,
            test_rects[:10] # 使用前10个作为查询源，或者使用随机查询
        )
        
        print(f"\n✓ Reward calculation:")
        print(f"  - Reward type: {calculator.reward_type}")
        print(f"  - Calculated reward: {reward:.4f}")
        
        # 注意：由于 acppo 和 rstar 在未加载模型时策略相同，奖励接近 0 是正常的
        # 这里主要测试代码是否跑通，以及数值是否合理（不应为 NaN 或 Inf）
        if abs(reward) < 1e-6:
            print(f"  - Interpretation: Equal (Expected for identical strategies without RL)")
        elif reward > 0:
            print(f"  - Interpretation: Agent is better")
        else:
            print(f"  - Interpretation: Reference is better")
        
        # 测试零和奖励
        print("\nTesting Zero-Sum Reward:")
        zero_sum_calc = ZeroSumRewardCalculator()
        
        r1, r2 = zero_sum_calc.calculate_zero_sum_rewards(
            agent_env,
            ref_env,
            test_rects[:10]
        )
        
        print(f"  - Player 1 reward: {r1:.4f}")
        print(f"  - Player 2 reward: {r2:.4f}")
        sum_reward = r1 + r2
        print(f"  - Sum (should be ~0): {sum_reward:.6f}")
        
        if abs(sum_reward) < 1e-6:
            print("  ✓ Zero-sum property verified!")
        else:
            print(f"[X] Error: Zero-sum property failed! Diff: {abs(sum_reward)}")
            return False
        
        print("\nAll reward calculator tests passed!")
        return True
        
    except Exception as e:
        print(f"\n[X] Reward calculator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rl_integration_example():
    """
    测试 RL 集成示例（展示如何使用训练好的 agent）
    
    注意：这个测试需要预先训练好的模型，如果模型不存在则跳过
    """
    print("\n" + "=" * 80)
    print("Testing RL Integration (Example)")
    print("=" * 80)
    
    try:
        from config.default_config import create_training_config
        from environment.rtree_env import MultiTreeEnvironment
        
        print("✓ Successfully imported required modules")
        
        # 创建配置和多树环境
        config = create_training_config()
        multi_env = MultiTreeEnvironment(config)
        
        # 添加树
        multi_env.add_tree("rl_agent", "acppo")
        multi_env.add_tree("baseline", "rstar")
        
        print(f"\n✓ Created {len(multi_env.environments)} trees")
        
        # 注意：这里演示的是接口用法，实际使用时需要加载训练好的 agent
        # from models.acppo_agent import ACPPOAgent
        # agent = ACPPOAgent(config)
        # agent.load_model("path/to/trained/model.pth")
        
        print("\n[!]Note: This is an example of how to use RL agents.")
        print("   To actually use RL, you need to:")
        print("   1. Train a model using scripts/train.py or scripts/self_play.py")
        print("   2. Load the trained agent")
        print("   3. Pass it to insert_to_all with rl_trees parameter")
        print("\n   Example code:")
        print("   ```python")
        print("   agent = ACPPOAgent(config)")
        print("   agent.load_model('checkpoints/best_model.pth')")
        print("   multi_env.insert_to_all(")
        print("       rect,")
        print("       rl_trees=['rl_agent'],")
        print("       agent_dict={'rl_agent': agent}")
        print("   )")
        print("   ```")
        
        print("\nRL integration example completed!")
        return True
        
    except Exception as e:
        print(f"\n[X] RL integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有环境测试"""
    print("\n" + "=" * 40)
    print("GSAR-Tree Environment Testing Suite")
    print("=" * 40 + "\n")
    
    # # 测试环境
    # env_passed = test_environment()
    
    # # 测试奖励计算器
    reward_passed = test_reward_calculator()
    
    # 测试 RL 集成示例
    # rl_passed = test_rl_integration_example()
    
    # # 总结
    # print("\n" + "=" * 80)
    # print("Test Summary")
    # print("=" * 80)
    # print(f"RTree Environment:     {'✅ PASSED' if env_passed else '❌ FAILED'}")
    # print(f"Reward Calculator:     {'✅ PASSED' if reward_passed else '❌ FAILED'}")
    # print(f"RL Integration Example:{'✅ PASSED' if rl_passed else '❌ FAILED'}")
    # print("=" * 80)
    
    # if env_passed and reward_passed and rl_passed:
    #     print("\n🎉 All environment tests passed! Ready for Phase 3.")
    #     return 0
    # else:
    #     print("\n⚠️  Some tests failed. Please check the errors above.")
    #     return 1


if __name__ == "__main__":
    main()





