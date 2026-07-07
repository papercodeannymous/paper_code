#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSAR-Tree 测试评估入口脚本

用法示例：
    # 基本评估（自动匹配模型）
    python scripts/test.py --train-volume TW1 --distribution NORMAL --baseline rstar --max-entry 50 --action-space 4 --test-volumes TW2
    
    # 多数据集评估
    python scripts/test.py --train-volume TW1 --distribution NORMAL --test-volumes TW2,TW3,TW4
"""
import sys
import os
import argparse
import json
import re
import glob
import numpy as np
import torch
from typing import List
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gsartree.config.default_config import create_training_config
from gsartree.models.acppo_agent import ACPPOAgent
from gsartree.environment.rtree_env import RTreeEnvironment
from gsartree.training.evaluator import Evaluator
from gsartree.utils.logger import get_logger
from gsartree.utils.statistical_test import StatisticalTester
from gsartree.utils.performance_monitor import PerformanceMonitor


def evaluate_single_run(config, agent, dataset, baseline_type: str, 
                       query_ratios: List[float], num_queries: int,
                       seed: int = 42) -> dict:
    """
    单次评估运行
    
    Args:
        config: 配置对象
        agent: ACPPOAgent
        dataset: 测试数据集
        baseline_type: 基线类型
        query_ratios: 查询比例列表
        num_queries: 每个比例的查询数量
        seed: 随机种子（保证结果可复现）
    
    Returns:
        评估结果字典
    """
    # ✅ 设置随机种子，确保每次运行使用不同的独立查询集
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 构建树
    rl_tree = create_rl_tree(config, agent, dataset)
    baseline_tree = create_baseline_tree(config, baseline_type, dataset)
    
    # 执行评估
    evaluator = Evaluator(config)
    results = evaluator.evaluate_range_queries(
        test_tree=rl_tree,
        reference_tree=baseline_tree,
        query_ratios=query_ratios,
        num_queries_per_ratio=num_queries
    )
    
    return results


def multi_run_evaluation(config, agent, dataset, baseline_type: str,
                        query_ratios: List[float], num_queries: int,
                        num_runs: int) -> dict:
    """
    多次运行评估（用于统计显著性检验）
    
    Args:
        config: 配置对象
        agent: ACPPOAgent
        dataset: 测试数据集
        baseline_type: 基线类型
        query_ratios: 查询比例列表
        num_queries: 每个比例的查询数量
        num_runs: 运行次数
    
    Returns:
        聚合评估结果（包含统计检验信息）
    """
    logger = get_logger()
    
    # 收集所有运行的结果
    all_runs_results = []
    
    for run_idx in range(num_runs):
        logger.info(f"Running evaluation {run_idx + 1}/{num_runs}...")
        
        # 每次运行使用不同的随机种子（确保查询不同，用于统计检验）
        seed = 42 + run_idx * 1000
        
        # 执行单次评估（传入种子）
        run_result = evaluate_single_run(
            config, agent, dataset, baseline_type,
            query_ratios, num_queries,
            seed=seed  # 传入种子参数
        )
        
        all_runs_results.append(run_result)
        
        # ✅ 打印当前运行的详细结果
        logger.subsection(f"Run {run_idx + 1} Results")
        evaluator = Evaluator(config)
        evaluator.print_evaluation_report(run_result, f"Run {run_idx + 1} - Seed {seed}")
    
    # 聚合结果并执行统计检验
    aggregated_results = {}
    
    for query_name in query_ratios:
        query_key = f"{query_name}%"
        
        # 修复：使用正确的键名（test_mean 和 ref_mean）
        # 收集所有运行中该查询比例的指标
        agent_accesses = [run[query_key]['test_mean'] for run in all_runs_results]
        ref_accesses = [run[query_key]['ref_mean'] for run in all_runs_results]
        
        # 计算改进百分比（基于每次运行的均值）
        # Node Access Gain: (Ref - Test) / Test * 100%
        improvements = []
        for run in all_runs_results:
            ref_mean = run[query_key]['ref_mean']
            test_mean = run[query_key]['test_mean']
            if test_mean > 0:
                imp = (ref_mean - test_mean) / test_mean * 100
            else:
                imp = 0.0
            improvements.append(imp)
        
        # 计算均值和标准差
        mean_agent = np.mean(agent_accesses)
        std_agent = np.std(agent_accesses)
        mean_ref = np.mean(ref_accesses)
        std_ref = np.std(ref_accesses)
        mean_improvement = np.mean(improvements)
        std_improvement = np.std(improvements)
        
        # 执行配对 t-test
        tester = StatisticalTester()
        test_result = tester.paired_t_test(agent_accesses, ref_accesses)
        
        # 从结果字典中提取需要的值
        t_stat = test_result['t_statistic']
        p_value = test_result['p_value']
        cohens_d = test_result['cohens_d']
        
        # 判断显著性
        is_significant = test_result['significant']
        
        # 解释效应量
        if abs(cohens_d) < 0.2:
            effect_interpretation = "Negligible effect"
        elif abs(cohens_d) < 0.5:
            effect_interpretation = "Small effect"
        elif abs(cohens_d) < 0.8:
            effect_interpretation = "Medium effect"
        else:
            effect_interpretation = "Large effect"
        
        # 构建聚合结果
        aggregated_results[query_key] = {
            'agent_access': mean_agent,
            'agent_access_std': std_agent,
            'ref_access': mean_ref,
            'ref_access_std': std_ref,
            'improvement': mean_improvement,
            'improvement_std': std_improvement,
            'statistical_test': {
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': is_significant,
                'interpretation': effect_interpretation
            }
        }
    
    return aggregated_results


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='GSAR-Tree Model Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ✅ 模型配置 - 基于训练参数自动匹配
    parser.add_argument('--train-volume', type=str, default='TW1',
                       help='Training volume (e.g., TW1, TW2)')
    parser.add_argument('--distribution', type=str, default='NORMAL',
                       choices=['UNIFORM', 'NORMAL', 'CITY-LIKE', 'HOTSPOTS', 'MULTI-MODAL', 'SKEW-NOR4'],
                       help='Data distribution type used in training')
    parser.add_argument('--baseline', type=str, default='rstar',
                       choices=['rtree', 'rstar'],
                       help='Baseline tree type used in training')
    parser.add_argument('--max-entry', type=int, default=50,
                       help='Maximum entries per node')
    parser.add_argument('--action-space', type=int, default=4,
                       help='Action space size')
    parser.add_argument('--feature-type', type=int, default=125,
                       choices=[0, 12, 15, 25, 125],
                       help='Feature type used in training (must match the trained model)')
    parser.add_argument('--enable-feature-ablation', action='store_true',
                       help='Enable feature ablation mode (model filenames include feature type)')
    parser.add_argument('--model-dir', type=str, default='model',
                       help='Directory containing trained models')
    
    # ✅ 训练模式选择
    parser.add_argument('--training-mode', type=str, default='single',
                       choices=['single', 'selfplay'],
                       help='Training mode: single (traditional RL) or selfplay')

    # 数据配置
    parser.add_argument('--test-volumes', type=str, default='HW1',
                       help='Test dataset volumes (comma-separated, e.g., TW2,TW3)')
    parser.add_argument('--data-dir', type=str, default='generated_data',
                       help='Data directory path')
    
    # 评估配置
    parser.add_argument('--num-runs', type=int, default=1,
                       help='Number of evaluation runs for statistical significance (>=5 recommended)')
    parser.add_argument('--num-queries', type=int, default=1000,
                       help='Number of queries per evaluation')
    parser.add_argument('--query-ratios', type=str, default='2.0,1.0,0.5,0.05,0.01,0.005',
                       help='Query ratios to test (comma-separated)')
    
    # 设备配置
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cpu', 'cuda'],
                       help='Device to use')
    
    # 输出配置
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory')
    
    return parser.parse_args()


def find_best_model(model_dir: str, train_volume: str, distribution: str, 
                   baseline: str, max_entry: int, action_space: int,
                   training_mode: str = 'single', feature_type: int = None,
                   enable_feature_ablation: bool = False) -> str:
    """
    根据训练参数自动查找最佳效率模型
    
    Args:
        model_dir: 模型目录
        train_volume: 训练数据量（如 TW1）
        distribution: 数据分布（如 NORMAL）
        baseline: 基线类型（如 rstar）
        max_entry: 最大条目数
        action_space: 动作空间大小
        training_mode: 训练模式 ('single' 或 'selfplay')
        feature_type: 特征类型（仅在 enable_feature_ablation=True 时使用）
        enable_feature_ablation: 是否启用特征消融模式
    
    Returns:
        最佳模型路径
    """
    # ✅ 根据训练模式构建不同的文件名模式
    if training_mode == 'selfplay':
        prefix = 'selfplay'
    else:
        prefix = 'train'
    
    # ⭐ 根据 enable_feature_ablation flag 决定是否包含特征类型
    if enable_feature_ablation and feature_type is not None:
        feature_suffix = f"_F{feature_type}"
    else:
        feature_suffix = ""
    
    # 关键修复：构建子目录路径
    subdir_name = f"{train_volume}_{distribution}_{baseline}_{max_entry}-{action_space}"
    subdir_path = os.path.join(model_dir, subdir_name)
    
    # 在子目录中搜索模型
    pattern = f"{prefix}_{train_volume}_{distribution}_{baseline}_{max_entry}-{action_space}{feature_suffix}_BestEfficiency_Ep*.pth"
    search_path = os.path.join(subdir_path, pattern)
    
    matching_files = glob.glob(search_path)
    
    if not matching_files:
        raise FileNotFoundError(
            f"No best efficiency model found with pattern: {pattern}\n"
            f"Searched in: {subdir_path}\n"
            f"Training mode: {training_mode}\n"
            f"Feature ablation: {enable_feature_ablation}\n"
            f"Tip: Ensure the model was trained with these exact parameters."
        )
    
    # 选择最新的（Episode 最大的）模型
    def extract_episode(filepath):
        basename = os.path.basename(filepath)
        match = re.search(r'Ep(\d+)\.pth$', basename)
        return int(match.group(1)) if match else -1
    
    best_model = max(matching_files, key=extract_episode)
    
    return best_model


def load_test_dataset(data_dir: str, volume: str, distribution: str):
    """
    加载测试数据集（仅支持 .npy 格式）
    
    Args:
        data_dir: 数据目录
        volume: 数据量标识（如 TW2）
        distribution: 分布类型（如 NORMAL）
    
    Returns:
        矩形列表 [[ll_x, ll_y, tr_x, tr_y], ...]
    """
    logger = get_logger()
    
    # 构建预期的完整文件名：testing_TW2_NORMAL.npy
    filename = f"testing_{volume}_{distribution}.npy"
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        error_msg = f"Test dataset not found: {filepath}\n"
        
        # 列出当前目录中存在的相似文件
        if os.path.exists(data_dir):
            similar_files = [f for f in os.listdir(data_dir) if 'testing' in f.lower()]
            if similar_files:
                error_msg += f"Available test files in {data_dir}: {similar_files[:10]}"
        else:
            error_msg += f"Data directory does not exist: {data_dir}"
        
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        data = np.load(filepath)
        logger.success(f"Loaded test dataset: {filepath} ({len(data)} rectangles)")
        return data.tolist()
    except Exception as e:
        logger.error(f"Failed to load test dataset: {e}")
        raise


def create_rl_tree(config, agent, dataset, perf_monitor=None):
    """
    创建 RL 优化的树（与训练中相同的方式）
    
    Args:
        config: 配置对象
        agent: ACPPOAgent
        dataset: 矩形列表
        perf_monitor: 性能监控器（可选，用于记录构建时间）
    
    Returns:
        RTreeEnvironment
    """
    import time
    
    env = RTreeEnvironment(config, tree_type="acppo")
    
    start_time = time.time()
    rect_count = 0
    
    # 使用 tqdm 显示插入进度
    for rect in tqdm(dataset, desc="Building RL Tree", unit="rect"):
        # 使用 RL 决策插入（与训练中一致）
        env.insert_rectangle(
            rect,
            use_rl=True,
            agent=agent,
            feature_type=config.model.feature_type,
            action_space_size=config.model.action_space_size
        )
        rect_count += 1
        
        # 每 10000 个矩形记录一次中间状态
        if perf_monitor and rect_count % 10000 == 0:
            elapsed = time.time() - start_time
            rate = rect_count / elapsed if elapsed > 0 else 0
            
    build_time = time.time() - start_time
    
    # 记录构建树的性能指标
    if perf_monitor:
        extra_info = {
            'tree_type': 'RL_Tree',
            'build_time_seconds': build_time,
            'total_rectangles': len(dataset),
            'insertion_rate_per_second': len(dataset) / build_time if build_time > 0 else 0,
            'avg_time_per_rectangle_ms': (build_time / len(dataset) * 1000) if len(dataset) > 0 else 0
        }
        perf_monitor.record_episode(
            episode_idx=-1,  # 使用特殊索引表示构建树操作
            episode_time=build_time,
            reward=None,
            dataset_size=len(dataset),
            extra_info=extra_info
        )
    
    print(f"\n✓ RL Tree built in {build_time:.2f}s ({len(dataset)/build_time:.2f} rects/sec)")
    
    return env


def create_baseline_tree(config, baseline_type: str, dataset, perf_monitor=None):
    """
    创建基线树（传统策略）
    
    Args:
        config: 配置对象
        baseline_type: 基线类型（rstar, rtree, rrstar）
        dataset: 矩形列表
        perf_monitor: 性能监控器（可选，用于记录构建时间）
    
    Returns:
        RTreeEnvironment
    """
    import time
    
    env = RTreeEnvironment(config, tree_type=baseline_type)
    
    start_time = time.time()
    
    # ✅ 使用 tqdm 显示插入进度
    for rect in tqdm(dataset, desc=f"Building {baseline_type.upper()} Tree", unit="rect"):
        # ✅ 使用传统策略插入（与训练中参考树一致）
        env.insert_rectangle(rect)
    
    build_time = time.time() - start_time
    
    # 记录构建树的性能指标
    if perf_monitor:
        extra_info = {
            'tree_type': f'{baseline_type.upper()}_Tree',
            'build_time_seconds': build_time,
            'total_rectangles': len(dataset),
            'insertion_rate_per_second': len(dataset) / build_time if build_time > 0 else 0,
            'avg_time_per_rectangle_ms': (build_time / len(dataset) * 1000) if len(dataset) > 0 else 0
        }
        perf_monitor.record_episode(
            episode_idx=-2,  # 使用特殊索引表示构建基线树操作
            episode_time=build_time,
            reward=None,
            dataset_size=len(dataset),
            extra_info=extra_info
        )
    
    print(f"\n✓ {baseline_type.upper()} Tree built in {build_time:.2f}s ({len(dataset)/build_time:.2f} rects/sec)")
    
    return env


def main():
    """主函数"""
    args = parse_args()
    
    # ✅ 初始化性能监控
    perf_monitor = PerformanceMonitor()
    perf_monitor.start()
    
    # 初始化日志
    log_dir = os.path.join(args.output_dir, 'logs')
    logger = get_logger(log_dir=log_dir, console_output=True, file_output=True)
    
    logger.section("GSAR-Tree Model Evaluation")
    
    # ✅ 打印配置
    logger.subsection("Configuration")
    logger.info(f"Training Mode:       {args.training_mode}")
    logger.info(f"Training Volume:     {args.train_volume}")
    logger.info(f"Distribution:        {args.distribution}")
    logger.info(f"Baseline Type:       {args.baseline}")
    logger.info(f"Max Entry:           {args.max_entry}")
    logger.info(f"Action Space:        {args.action_space}")
    logger.info(f"Feature Type:        {args.feature_type}")
    logger.info(f"Feature Ablation:    {args.enable_feature_ablation}")
    logger.info(f"Test Volumes:        {args.test_volumes}")
    logger.info(f"Num Queries:         {args.num_queries}")
    logger.info(f"Num Runs:            {args.num_runs}")
    logger.info(f"Device:              {args.device}")
    
    # 解析查询比例
    query_ratios = [float(r) for r in args.query_ratios.split(',')]
    
    # 创建设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 创建配置
    config = create_training_config(
        distribution=args.distribution,
        max_entry=args.max_entry,
        feature_type=args.feature_type,  # ✅ 使用命令行传入的特征类型
        action_space_size=args.action_space,
        device=args.device
    )
    
    # 自动查找最佳模型
    logger.subsection("Loading Model")
    try:
        model_path = find_best_model(
            args.model_dir,
            args.train_volume,
            args.distribution,
            args.baseline,
            args.max_entry,
            args.action_space,
            training_mode=args.training_mode,
            feature_type=args.feature_type,
            enable_feature_ablation=args.enable_feature_ablation
        )
        logger.info(f"Found best model ({args.training_mode}): {os.path.basename(model_path)}")
        if args.enable_feature_ablation:
            logger.info(f"Feature ablation mode enabled (feature type: {args.feature_type})")
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    
    # 加载模型
    agent = ACPPOAgent(config, device)
    agent.load_checkpoint(model_path)
    logger.success("Model loaded successfully")
    
    # ✅ 解析测试数据集
    test_volumes = [v.strip() for v in args.test_volumes.split(',')]
    
    # 对每个数据集进行评估
    all_results = {}
    
    for volume in test_volumes:
        logger.section(f"Evaluating on Test Dataset: {volume}")
        
        # ✅ 加载测试数据
        try:
            dataset = load_test_dataset(args.data_dir, volume, args.distribution)
        except FileNotFoundError:
            logger.warning(f"Skipping test dataset {volume} (not found)")
            continue
        
        if args.num_runs == 1:
            # 单次评估（使用固定种子保证可复现性）
            logger.info("Building RL-optimized tree...")
            
            # 设置固定种子，保证每次运行结果一致
            # import random
            # fixed_seed = 42
            # random.seed(fixed_seed)
            # np.random.seed(fixed_seed)
            # torch.manual_seed(fixed_seed)
            # logger.info(f"Using fixed random seed: {fixed_seed} for reproducibility")
            
            # 传入 perf_monitor 以记录构建时间
            rl_tree = create_rl_tree(config, agent, dataset, perf_monitor=perf_monitor)
            
            logger.info(f"Building baseline tree ({args.baseline})...")
            # 传入 perf_monitor 以记录构建时间
            baseline_tree = create_baseline_tree(config, args.baseline, dataset, perf_monitor=perf_monitor)
            
            # 执行评估
            evaluator = Evaluator(config)
            results = evaluator.evaluate_range_queries(
                test_tree=rl_tree,
                reference_tree=baseline_tree,
                query_ratios=query_ratios,
                num_queries_per_ratio=args.num_queries
            )
            
            # 打印报告
            evaluator.print_evaluation_report(results, f"Results - {volume}")
            
            all_results[volume] = results
            
        else:
            # ✅ 多次运行（统计显著性检验）
            logger.info(f"Running {args.num_runs} evaluation runs for statistical significance...")
            
            aggregated_results = multi_run_evaluation(
                config, agent, dataset, args.baseline,
                query_ratios, args.num_queries, args.num_runs
            )
            
            # 打印聚合报告
            evaluator = Evaluator(config)
            evaluator.print_evaluation_report(
                aggregated_results,
                f"Aggregated Results ({args.num_runs} runs) - {volume}"
            )
            
            # ✅ 统计显著性分析
            logger.subsection("Statistical Significance Analysis")
            for query_name, metrics in aggregated_results.items():
                stat = metrics['statistical_test']
                sig_marker = "✓" if stat['significant'] else "✗"
                
                logger.info(f"{query_name}:")
                logger.info(f"  Agent Access: {metrics['agent_access']:.2f} ± {metrics['agent_access_std']:.2f}")
                logger.info(f"  Ref Access:   {metrics['ref_access']:.2f} ± {metrics['ref_access_std']:.2f}")
                logger.info(f"  Improvement:  {metrics['improvement']:+.2f}% ± {metrics['improvement_std']:.2f}%")
                
                # ✅ 优化 p-value 显示：如果极小则显示科学计数法
                p_val = stat['p_value']
                if p_val < 0.0001:
                    p_str = f"{p_val:.2e}"
                else:
                    p_str = f"{p_val:.4f}"
                
                logger.info(f"  p-value:      {p_str} {sig_marker}")
                logger.info(f"  Cohen's d:    {stat['cohens_d']:.4f} ({stat['interpretation']})")
                if stat['significant']:
                    if metrics['improvement'] > 0:
                        logger.success(f"  → RL model significantly OUTPERFORMS baseline!")
                    else:
                        logger.warning(f"  → RL model significantly UNDERPERFORMS baseline!")
                else:
                    logger.info(f"  → No significant difference detected.")
            
            all_results[volume] = aggregated_results
    
    # ✅ 停止性能监控并生成报告
    perf_monitor.stop()
    
    # ⭐ 打印树构建性能详细报告
    perf_monitor.print_tree_build_report()
    
    perf_report = perf_monitor.get_performance_summary()
    
    # 打印性能报告
    logger.section("Performance Report")
    logger.info(f"Total Time:              {perf_report['total_time_seconds']:.2f}s ({perf_report['total_time_formatted']})")
    logger.info(f"Avg CPU Utilization:     {perf_report['average_cpu_percent']:.1f}%")
    logger.info(f"Peak Memory:             {perf_report['peak_memory_mb']:.2f} MB")
    
    if 'peak_gpu_memory_mb' in perf_report:
        logger.info(f"Peak GPU Memory:         {perf_report['peak_gpu_memory_mb']:.2f} MB")
        logger.info(f"Avg GPU Utilization:     {perf_report['average_gpu_utilization']:.1f}%")
    
    # 保存结果
    output_file = os.path.join(args.output_dir, 'evaluation_results.json')
    
    # ✅ 辅助函数：递归转换所有 numpy 类型为 Python 原生类型
    def convert_to_serializable(obj):
        """递归地将 numpy 类型转换为 Python 原生类型"""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)  # 对于其他类型，转换为字符串
    
    # 转换为可序列化的格式
    serializable_results = convert_to_serializable(all_results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    # ✅ 保存性能报告到 JSON
    perf_output_file = os.path.join(args.output_dir, 'performance.json')
    with open(perf_output_file, 'w', encoding='utf-8') as f:
        json.dump(perf_report, f, indent=2, ensure_ascii=False)
    
    logger.success(f"Results saved to {output_file}")
    logger.success(f"Performance report saved to {perf_output_file}")
    logger.success("Evaluation completed successfully!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())