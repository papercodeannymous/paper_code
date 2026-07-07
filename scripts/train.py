#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSAR-Tree 单智能体训练入口脚本

用法示例：
    # 基本训练
    python scripts/train.py --distribution NORMAL --train-volume W1 --max-entry 50
    
    # 自定义配置
    python scripts/train.py --feature-type 125 --num-episodes 30 --eval-freq 5
    
    # 使用 GPU
    python scripts/train.py --device cuda
"""
import sys
import os
import argparse
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gsartree.config.default_config import create_training_config
from gsartree.models.acppo_agent import ACPPOAgent
from gsartree.training.trainer import Trainer
from gsartree.utils.logger import get_logger
from gsartree.utils.performance_monitor import PerformanceMonitor
from gsartree.utils.checkpoint_manager import CheckpointManager
from gsartree.utils.visualizer import Visualizer


class PrintToLogger:
    """
    将 print 输出重定向到 logger
    
    Usage / 用法:
        sys.stdout = PrintToLogger(logger, level='info')
        print("This will be logged")  # 这将被记录到日志
    """
    def __init__(self, logger, level='info'):
        self.logger = logger
        self.level = level
        self.buffer = ''
    
    def write(self, text):
        if text.strip():  # 忽略空行 / Ignore empty lines
            # 移除 tqdm 的刷新字符 / Remove tqdm refresh characters
            text = text.replace('\r', '').strip()
            if text:
                getattr(self.logger, self.level)(text)
    
    def flush(self):
        pass


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='GSAR-Tree Single-Agent Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 数据配置
    parser.add_argument('--distribution', type=str, default='NORMAL',
                       choices=['SKEW-NOR4', 'UNIFORM', 'NORMAL', 'CITY-LIKE', 'HOTSPOTS', 'MULTI-MODAL'],
                       help='Data distribution type')
    parser.add_argument('--train-volume', type=str, default='TW1',
                       help='Training data volume (W1, W2, etc.)')
    parser.add_argument('--data-dir', type=str, default='generated_data/',
                       help='Data directory path')
    
    # 树配置
    parser.add_argument('--max-entry', type=int, default=50,
                       help='Maximum entries per node')
    parser.add_argument('--min-entry-factor', type=float, default=0.4,
                       help='Minimum entry factor')
    
    # 模型配置
    parser.add_argument('--feature-type', type=int, default=125,
                       choices=[0, 12, 15, 25, 125],
                       help='Feature type for state representation')
    parser.add_argument('--action-space-size', type=int, default=4,
                       help='Action space size for RL decisions')
    parser.add_argument('--enable-feature-ablation', action='store_true',
                       help='Enable feature ablation mode (include feature type in model filenames)')
    
    # 训练配置
    parser.add_argument('--num-episodes', type=int, default=20,
                       help='Number of training episodes')
    parser.add_argument('--eval-freq', type=int, default=15,
                       help='Evaluation frequency (every N episodes)')
    parser.add_argument('--load-best-freq', type=int, default=5,
                       help='Load best model frequency')
    parser.add_argument('--query-reward-freq', type=int, default=10,
                       help='Query reward calculation frequency')
    
    # 设备配置
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cpu', 'cuda'],
                       help='Device to use for training')
    
    # 输出配置
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for logs, checkpoints, images')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    parser.add_argument('--baseline', type=str, default='rstar',
                       choices=['rtree', 'rstar'],
                       help='Baseline tree type used in training')
    
    return parser.parse_args()


def load_dataset(data_dir: str, distribution: str, volume: str):
    """
    加载训练数据集
    
    Args:
        data_dir: 数据目录
        distribution: 分布类型
        volume: 数据量标识
    
    Returns:
        矩形列表 [[ll_x, ll_y, tr_x, tr_y], ...]
    """
    import numpy as np
    
    logger = get_logger()
    
    # 构建预期的完整文件名（严格基于测试参数）
    filename_npy = f"{volume.upper()}_{distribution.upper()}.npy"
    filepath_npy = os.path.join(data_dir, filename_npy)
    print(f"Looking for dataset file: {filepath_npy}")
    
    # filename_txt = f"testing_{volume.lower()}_{distribution.lower()}.txt"
    # filepath_txt = os.path.join(data_dir, filename_txt)
    
    # 1. 优先读取 .npy 格式文件（完全匹配）
    if os.path.exists(filepath_npy):
        try:
            data = np.load(filepath_npy)
            logger.success(f"Loaded dataset (npy): {filepath_npy} ({len(data)} rectangles)")
            return data.tolist()
        except Exception as e:
            logger.error(f"Failed to load npy dataset: {e}")
            raise
    
    # 2. 尝试读取 .txt 格式文件（完全匹配）
    # if os.path.exists(filepath_txt):
    #     try:
    #         logger.info(f"Loading txt dataset: {filepath_txt}")
    #         rectangles = []
    #         with open(filepath_txt, 'r') as f:
    #             n = 0
    #             ll_x = ll_y = tr_x = tr_y = 0
    #             for line in f:
    #                 line = line.strip()
    #                 if not line:
    #                     continue
    #                 if n % 2 == 0:
    #                     ll_x = float(line) - 1.0  # data_edge_size = 1.0
    #                     tr_x = float(line)
    #                 else:
    #                     ll_y = float(line) - 1.0
    #                     tr_y = float(line)
    #                     rectangles.append([ll_x, ll_y, tr_x, tr_y])
    #                 n += 1
            
    #         logger.success(f"Loaded dataset (txt): {filepath_txt} ({len(rectangles)} rectangles)")
    #         return rectangles
    #     except Exception as e:
    #         logger.error(f"Failed to load txt dataset: {e}")
    #         raise
    
    # 3. 自动降级机制：生成随机数据
    # logger.warning(f"Dataset not found:")
    # logger.warning(f"  Expected npy: {filepath_npy}")
    # # logger.warning(f"  Expected txt: {filepath_txt}")
    
    # # 列出当前目录中存在的相似文件（如有）
    # if os.path.exists(data_dir):
    #     similar_files = [f for f in os.listdir(data_dir) if 'testing' in f.lower()]
    #     if similar_files:
    #         logger.warning(f"  Available files in {data_dir}: {similar_files[:5]}")
    # else:
    #     logger.warning(f"  Data directory does not exist: {data_dir}")
    
    # logger.info("Generating random dataset as fallback...")
    
    # # 生成随机数据作为fallback
    # n_samples = 1000
    # rectangles = []
    # for _ in range(n_samples):
    #     x = np.random.uniform(0, 100000)
    #     y = np.random.uniform(0, 100000)
    #     w = np.random.uniform(50, 200)
    #     h = np.random.uniform(50, 200)
    #     rectangles.append([x, y, x + w, y + h])
    
    # logger.warning("Using randomly generated data. Results may not be meaningful.")
    # return rectangles


def main():
    """主函数"""
    args = parse_args()
    
    # 初始化日志
    log_dir = os.path.join(args.output_dir, 'logs')
    logger = get_logger(log_dir=log_dir, console_output=True, file_output=True)
    
    logger.section("GSAR-Tree Single-Agent Training")
    
    # 打印配置
    logger.subsection("Configuration")
    logger.info(f"Distribution:       {args.distribution}")
    logger.info(f"Train Volume:       {args.train_volume}")
    logger.info(f"Max Entry:          {args.max_entry}")
    logger.info(f"Feature Type:       {args.feature_type}")
    logger.info(f"Action Space Size:  {args.action_space_size}")
    logger.info(f"Feature Ablation:   {args.enable_feature_ablation}")
    logger.info(f"Num Episodes:       {args.num_episodes}")
    logger.info(f"Device:             {args.device}")
    
    # 创建配置
    config = create_training_config(
        distribution=args.distribution,
        train_volume=args.train_volume,
        max_entry=args.max_entry,
        min_entry_factor=0.4,
        feature_type=args.feature_type,
        action_space_size=args.action_space_size,
        device=args.device,
        enable_feature_ablation=args.enable_feature_ablation
    )
    
    # 更新训练配置
    config.training.num_episodes = args.num_episodes
    config.training.eval_freq = args.eval_freq
    config.training.load_best_model_freq = args.load_best_freq
    config.training.query_reward_freq = args.query_reward_freq
    config.tree.reference_tree_type = args.baseline
    
    # 加载数据
    logger.subsection("Loading Dataset")
    dataset = load_dataset(args.data_dir, args.distribution, args.train_volume)
    logger.info(f"Dataset size: {len(dataset)} rectangles")
    
    # 创建设备
    import torch
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 创建 Agent
    logger.subsection("Creating Agent")
    agent = ACPPOAgent(config, device)
    logger.success("Agent created successfully")
    
    # 初始化工具
    monitor = PerformanceMonitor()
    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=os.path.join(args.output_dir, 'checkpoints'),
        max_checkpoints=10,
        metric_mode="min"
    )
    viz = Visualizer(output_dir=os.path.join(args.output_dir, 'images'))
    
    # 创建训练器（传入性能监控器）
    logger.subsection("Starting Training")
    trainer = Trainer(config, agent, dataset, performance_monitor=monitor)
    
    # 开始训练
    monitor.start()
    
    history = trainer.train(
        num_episodes=args.num_episodes,
        eval_freq=args.eval_freq
    )
    
    monitor.stop()
    
    # 打印 Episode 时间统计报告
    monitor.print_episode_time_report()
    
    # 打印摘要
    trainer.print_summary()
    monitor.print_report()
    monitor.save_report(os.path.join(args.output_dir, 'performance.json'))
    
    # 保存检查点
    checkpoint_mgr.save_checkpoint(
        agent=agent,
        episode=args.num_episodes - 1,
        metric=trainer.best_efficiency_improvement,
        extra_info={'final': True},
        is_best=True
    )
    checkpoint_mgr.print_summary()

    
    if history.get('query_performance'):
        viz.plot_query_comparison(history['query_performance'][-1])
    
    logger.success("Training completed successfully!")
    logger.info(f"Results saved to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
