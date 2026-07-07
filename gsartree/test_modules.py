#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试配置系统和模型模块
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config():
    """测试配置系统"""
    print("=" * 80)
    print("Testing Configuration System")
    print("=" * 80)
    
    try:
        from config.default_config import (
            GSARConfig, 
            create_training_config,
            create_self_play_config,
            DataConfig,
            TreeConfig,
            ModelConfig
        )
        print("✓ Successfully imported configuration classes")
        
        # 测试默认配置
        config = GSARConfig()
        print(f"\n✓ Default config created:")
        print(f"  - Distribution: {config.data.distribution}")
        print(f"  - Train volume: {config.data.train_volume}")
        print(f"  - Feature type: {config.model.feature_type}")
        print(f"  - Num features: {config.model.num_features}")
        print(f"  - State space: {config.model.state_space_size}")
        print(f"  - Action space: {config.model.action_space_size}")
        
        # 测试训练配置
        train_config = create_training_config(
            distribution="NORMAL",
            train_volume="W1",
            max_entry=50,
            feature_type=125
        )
        print(f"\n✓ Training config created:")
        print(f"  - Max entry: {train_config.tree.max_entry}")
        print(f"  - Reference tree: {train_config.tree.reference_tree_type}")
        print(f"  - Insert strategy: {train_config.tree.get_insert_strategy()}")
        print(f"  - Split strategy: {train_config.tree.get_split_strategy()}")
        
        # 测试 Self-Play 配置
        sp_config = create_self_play_config(train_config)
        print(f"\n✓ Self-play config created:")
        print(f"  - Reference tree: {sp_config.tree.reference_tree_type}")
        print(f"  - Use cross-validation: {sp_config.self_play.use_cross_validation}")
        
        # 测试模型文件路径生成
        model_path = train_config.get_model_filepath("train")
        print(f"\n✓ Model filepath generated:")
        print(f"  - {model_path}")
        
        # 测试配置拷贝
        copied_config = train_config.copy()
        copied_config.model.feature_type = 145
        print(f"\n✓ Config copy works:")
        print(f"  - Original feature type: {train_config.model.feature_type}")
        print(f"  - Copied feature type: {copied_config.model.feature_type}")
        
        print("\n[✓] All configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n[X] Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_models():
    """测试模型模块（需要 PyTorch）"""
    print("\n" + "=" * 80)
    print("Testing Model Modules")
    print("=" * 80)
    
    try:
        import torch
        print("✓ PyTorch available")
        
        from models.policy_net import PolicyNet
        from models.value_net import ValueNet
        from models.acppo_agent import ACPPOAgent
        from config.default_config import create_training_config
        
        print("✓ Successfully imported model classes")
        
        # 创建配置
        config = create_training_config()
        device = torch.device('cpu')
        
        # 测试 PolicyNet
        policy_net = PolicyNet(
            config.model.state_space_size,
            config.model.n_hidden,
            config.model.action_space_size
        )
        print(f"\n✓ PolicyNet created:")
        print(f"  - Input size: {config.model.state_space_size}")
        print(f"  - Hidden size: {config.model.n_hidden}")
        print(f"  - Output size: {config.model.action_space_size}")
        
        # 测试前向传播
        dummy_input = torch.randn(1, config.model.state_space_size)
        output = policy_net(dummy_input)
        print(f"  - Output shape: {output.shape}")
        print(f"  - Output sum (should be ~1.0): {output.sum().item():.4f}")
        
        # 测试 ValueNet
        value_net = ValueNet(
            config.model.state_space_size,
            config.model.n_hidden
        )
        print(f"\n✓ ValueNet created:")
        value_output = value_net(dummy_input)
        print(f"  - Output shape: {value_output.shape}")
        
        # 测试 ACPPOAgent
        agent = ACPPOAgent(config, device)
        print(f"\n✓ ACPPOAgent created:")
        print(f"  - Actor network: {type(agent.actor).__name__}")
        print(f"  - Critic network: {type(agent.critic).__name__}")
        print(f"  - Buffer size: {len(agent.buffer)}")
        
        # 测试动作选择
        import numpy as np
        dummy_state = np.random.randn(config.model.state_space_size)
        action, log_prob = agent.choose_action(dummy_state, explore=False)
        print(f"\n✓ Action selection works:")
        print(f"  - Selected action: {action}")
        print(f"  - Log probability: {log_prob.item():.4f}")
        
        print("\n[✓] All model tests passed!")
        return True
        
    except ImportError as e:
        print(f"\n[!] PyTorch not available, skipping model tests: {e}")
        print("   This is expected if running in a non-training environment")
        return None
    except Exception as e:
        print(f"\n[X] Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 40)
    print("GSAR-Tree Module Testing Suite")
    print("=" * 40 + "\n")
    
    # 测试配置系统
    config_passed = test_config()
    
    # 测试模型模块
    model_result = test_models()
    
    # # 总结
    # print("\n" + "=" * 80)
    # print("Test Summary")
    # print("=" * 80)
    # print(f"Configuration System: {'✅ PASSED' if config_passed else '❌ FAILED'}")
    
    # if model_result is True:
    #     print(f"Model Modules:        {'✅ PASSED'}")
    # elif model_result is None:
    #     print(f"Model Modules:        '⚠️  SKIPPED (PyTorch not available)'")
    # else:
    #     print(f"Model Modules:        '❌ FAILED'")
    
    # print("=" * 80)
    
    # if config_passed and (model_result is True or model_result is None):
    #     print("\n🎉 All critical tests passed! Ready for next phase.")
    #     return 0
    # else:
    #     print("\n⚠️  Some tests failed. Please check the errors above.")
    #     return 1


if __name__ == "__main__":
    exit(main())
