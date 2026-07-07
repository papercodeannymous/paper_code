#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSAR-Tree Self-Play 训练入口脚本

用法示例：
    # 基本 Self-Play 训练
    python scripts/self_play.py --distribution NORMAL --train-volume TW1
    
    # 自定义配置
    python scripts/self_play.py --num-episodes 30 --sync-freq 10 --eval-freq 5
    
    # 使用交叉验证
    python scripts/self_play.py --use-cross-validation True
"""
import sys
import os
import argparse
import json
import numpy as np
import random

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gsartree.config.default_config import create_self_play_config
from gsartree.models.acppo_agent import ACPPOAgent
from gsartree.training.self_play import SelfPlayTrainer
from gsartree.utils.logger import get_logger
from gsartree.utils.performance_monitor import PerformanceMonitor
from gsartree.utils.checkpoint_manager import CheckpointManager


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='GSAR-Tree Self-Play Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 数据配置
    parser.add_argument('--distribution', type=str, default='NORMAL',
                       choices=['UNIFORM', 'NORMAL', 'CITY-LIKE', 'HOTSPOTS', 'MULTI-MODAL'],
                       help='Data distribution type')
    parser.add_argument('--train-volume', type=str, default='TW1',
                       help='Training data volume')
    parser.add_argument('--data-dir', type=str, default='generated_data',
                       help='Data directory path')
    
    # 树配置
    parser.add_argument('--max-entry', type=int, default=50,
                       help='Maximum entries per node')
    
    # 模型配置
    parser.add_argument('--feature-type', type=int, default=125,
                       choices=[12, 14, 15, 124, 125, 145, 1245],
                       help='Feature type')
    parser.add_argument('--action-space-size', type=int, default=2,
                       help='Action space size')
    parser.add_argument('--enable-feature-ablation', action='store_true',
                       help='Enable feature ablation mode (include feature type in model filenames)')
    
    # Self-Play 配置
    parser.add_argument('--num-episodes', type=int, default=30,
                       help='Number of SP episodes')
    parser.add_argument('--eval-freq', type=int, default=5,
                       help='Evaluation frequency')
    parser.add_argument('--sync-freq', type=int, default=10,
                       help='Model synchronization frequency (course learning)')
    parser.add_argument('--use-cross-validation', type=bool, default=True,
                       help='Use cross-validation style self-play')
    
    # 设备配置
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cpu', 'cuda'],
                       help='Device to use')
    
    # 输出配置
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory')
    
    return parser.parse_args()


def load_dataset(data_dir: str, distribution: str, volume: str):
    """
    加载训练数据集（仅支持 .npy 格式）
    
    Args:
        data_dir: 数据目录
        distribution: 分布类型
        volume: 数据量标识
    
    Returns:
        矩形列表 [[ll_x, ll_y, tr_x, tr_y], ...]
    """
    logger = get_logger()
    
    # 构建训练数据文件名（不带 testing_ 前缀）：TW1_NORMAL.npy
    filename = f"{volume.upper()}_{distribution.upper()}.npy"
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        error_msg = f"Training dataset not found: {filepath}\n"
        
        # 列出当前目录中存在的相似文件
        if os.path.exists(data_dir):
            similar_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
            if similar_files:
                error_msg += f"Available .npy files in {data_dir}: {similar_files[:10]}"
        else:
            error_msg += f"Data directory does not exist: {data_dir}"
        
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        data = np.load(filepath)
        logger.success(f"Loaded training dataset: {filepath} ({len(data)} rectangles)")
        return data.tolist()
    except Exception as e:
        logger.error(f"Failed to load training dataset: {e}")
        raise


def main():
    """主函数"""
    args = parse_args()
    
    # 初始化性能监控
    perf_monitor = PerformanceMonitor()
    perf_monitor.start()
    
    # 初始化日志
    log_dir = os.path.join(args.output_dir, 'logs')
    logger = get_logger(log_dir=log_dir, console_output=True, file_output=True)
    
    logger.section("GSAR-Tree Self-Play Training")
    
    # 打印配置
    logger.subsection("Configuration")
    logger.info(f"Distribution:          {args.distribution}")
    logger.info(f"Train Volume:          {args.train_volume}")
    logger.info(f"Max Entry:             {args.max_entry}")
    logger.info(f"Feature Type:          {args.feature_type}")
    logger.info(f"Action Space:          {args.action_space_size}")
    logger.info(f"Feature Ablation:      {args.enable_feature_ablation}")
    logger.info(f"Num Episodes:          {args.num_episodes}")
    logger.info(f"Sync Frequency:        {args.sync_freq}")
    logger.info(f"Cross Validation:      {args.use_cross_validation}")
    logger.info(f"Eval Frequency:        {args.eval_freq}")
    logger.info(f"Device:                {args.device}")
    
    # 创建配置
    config = create_self_play_config(
        distribution=args.distribution,
        train_volume=args.train_volume,
        max_entry=args.max_entry,
        min_entry_factor=0.4,
        feature_type=args.feature_type,
        action_space_size=args.action_space_size,
        device=args.device,
        enable_feature_ablation=args.enable_feature_ablation
    )
    
    # 更新 Self-Play 配置
    config.self_play.num_episodes = args.num_episodes
    config.self_play.use_cross_validation = args.use_cross_validation
    config.training.eval_freq = args.eval_freq
    
    # 加载数据
    logger.subsection("Loading Dataset")
    dataset = load_dataset(args.data_dir, args.distribution, args.train_volume)
    logger.info(f"Dataset size: {len(dataset)} rectangles")
    
    # 创建设备
    import torch
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # ✅ 创建两个 Agent（设置不同的随机种子以确保多样性）
    logger.subsection("Creating Agents")
    
    # 为 Agent 设置种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    agent = ACPPOAgent(config, device)
    
    # 为 Competitor 设置不同的种子
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    competitor = ACPPOAgent(config, device)
    
    logger.success("Both agents created successfully (with different seeds)")
    
    # ⭐ 创建静态参考树（用于定期评估）
    logger.subsection("Building Static Reference Tree")
    from gsartree.environment.rtree_env import RTreeEnvironment
    static_ref_tree = RTreeEnvironment(config, tree_type=config.tree.reference_tree_type)
    logger.info(f"Building {config.tree.reference_tree_type.upper()} reference tree...")
    for rect in dataset:
        static_ref_tree.insert_rectangle(rect)
    logger.success(f"Static reference tree built with {len(dataset)} rectangles")
    
    # 初始化工具（移除可视化）
    # ✅ Self-Play 模型保存到 model 目录，使用规范化命名
    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=config.path.model_dir,  # 使用配置中的 model_dir
        max_checkpoints=1,  # ✅ 只保留最新的 1 个检查点，与单一训练保持一致
        metric_mode="min"
    )
    
    # 创建 Self-Play 训练器（传入 logger、静态参考树和性能监控器）
    logger.subsection("Starting Self-Play Training")
    sp_trainer = SelfPlayTrainer(
        config, agent, competitor, dataset, 
        static_ref_tree=static_ref_tree,  # ⭐ 传入静态参考树
        performance_monitor=perf_monitor,  # ⭐ 传入性能监控器
        logger=logger
    )
    
    # 开始训练（传入静态参考树）
    results = sp_trainer.train(
        num_episodes=args.num_episodes,
        eval_freq=args.eval_freq,
        sync_freq=args.sync_freq,
        static_ref_tree=static_ref_tree  # ⭐ 传入静态参考树用于定期评估
    ) 
    
    # 停止性能监控并生成报告
    perf_monitor.stop()
    
    # ⭐ 打印 Episode 时间统计报告
    perf_monitor.print_episode_time_report()
    
    # ⭐ 获取性能报告摘要
    perf_report = perf_monitor.get_performance_summary()
    
    # 打印性能报告
    logger.section("Performance Report")
    logger.info(f"Total Time:              {perf_report['total_time_seconds']:.2f}s ({perf_report['total_time_formatted']})")
    logger.info(f"Avg CPU Utilization:     {perf_report['average_cpu_percent']:.1f}%")
    logger.info(f"Peak Memory:             {perf_report['peak_memory_mb']:.2f} MB")
    
    if 'peak_gpu_memory_mb' in perf_report:
        logger.info(f"Peak GPU Memory:         {perf_report['peak_gpu_memory_mb']:.2f} MB")
        logger.info(f"Avg GPU Utilization:     {perf_report['average_gpu_utilization']:.1f}%")
    
    # 保存检查点
    if results.get('best_metric') is not None:
        # ⭐ 关键修复：使用 config.get_model_filepath() 生成正确的子目录路径
        best_episode = results.get('best_episode', args.num_episodes - 1)
        
        # 关键修复：best_episode 是从 0 开始的索引，需要 +1 转换为人类可读的 Episode 编号
        best_episode_number = best_episode + 1
        
        # ⭐ 使用配置类的方法生成路径（自动包含子目录）
        model_path = config.get_model_filepath("selfplay", episode=best_episode_number)
        
        # ⭐ 确保子目录存在
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)
        
        # ✅ 删除旧的同配置 Self-Play 模型文件
        import glob
        old_pattern = config.get_model_filepath("selfplay", episode=None).replace(
            "{episode}", "*"
        )
        old_models = glob.glob(old_pattern)
        for old_model in old_models:
            if old_model != model_path and os.path.exists(old_model):
                try:
                    os.remove(old_model)
                    logger.info(f"Removed old Self-Play model: {os.path.basename(old_model)}")
                except Exception as e:
                    logger.warning(f"Failed to remove old model {old_model}: {e}")
        
        # ✅ 保存新模型
        agent.save_checkpoint(model_path)
        logger.success(f"✓ Self-Play best model saved: {os.path.basename(model_path)}")
        logger.info(f"  Saved to: {model_path}")
        logger.info(f"  Episode: {best_episode_number}")
        logger.info(f"  Metric: {results['best_metric']:.4f}")

    # ✅ 打印 Self-Play 统计结果（直接使用 results，因为 train() 返回的就是 sp_statistics）
    logger.section("Self-Play Statistics")
    logger.info(f"Agent Wins:              {results.get('agent_wins', 0)}")
    logger.info(f"Competitor Wins:         {results.get('competitor_wins', 0)}")
    logger.info(f"Draws:                   {results.get('draws', 0)}")
    logger.info(f"Total Episodes:          {results.get('total_episodes', args.num_episodes)}")
    
    if results.get('performance_gaps'):
        avg_gap = np.mean(results['performance_gaps'])
        logger.info(f"Avg Performance Gap:     {avg_gap:.4f}")
    
    if results.get('best_episode') is not None:
        logger.info(f"Best Episode:          {results['best_episode'] + 1}")
        logger.info(f"Best Metric:           {results['best_metric']:.4f}")

    # 保存性能报告到 JSON
    perf_output_file = os.path.join(args.output_dir, 'sp_performance.json')
    with open(perf_output_file, 'w', encoding='utf-8') as f:
        json.dump(perf_report, f, indent=2, ensure_ascii=False)
    
    logger.success(f"Performance report saved to {perf_output_file}")
    logger.success("Self-Play training completed successfully!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
