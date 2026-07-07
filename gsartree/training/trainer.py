"""
训练器 - GSAR-Tree 主训练流程
整合环境、Agent、奖励计算器，实现完整的训练循环
"""
import os
import time
import logging
import numpy as np
from typing import List, Dict, Optional, Callable
from collections import namedtuple

from gsartree.config.default_config import GSARConfig
from gsartree.models.acppo_agent import ACPPOAgent, Transition
from gsartree.environment.rtree_env import RTreeEnvironment
from gsartree.environment.reward_calculator import RewardCalculator
from gsartree.utils.logger import get_logger

# 获取 logger 实例
logger = logging.getLogger(__name__)


class Trainer:
    """
    GSAR-Tree 训练器
    
    负责管理完整的训练流程，包括：
    - Episode 循环
    - 数据插入（支持 RL 决策）
    - 奖励计算
    - Agent 学习更新
    - 模型保存和评估
    
    Attributes:
        config: 完整配置
        agent: ACPPO 智能体
        dataset: 训练数据集
        agent_env: Agent 操作的树环境
        ref_env: 参考树环境
        reward_calc: 奖励计算器
    """
    
    def __init__(
        self,
        config: GSARConfig,
        agent: ACPPOAgent,
        dataset: List[List[float]],
        performance_monitor=None  # ⭐ 新增：性能监控器（可选）
    ):
        """
        初始化训练器
        
        Args:
            config: 完整的 GSAR 配置
            agent: 已初始化的 ACPPOAgent
            dataset: 训练数据，格式为 [[ll_x, ll_y, tr_x, tr_y], ...]
            performance_monitor: 性能监控器实例（可选）
        """
        self.config = config
        self.agent = agent
        self.dataset = dataset
        self.performance_monitor = performance_monitor  # ⭐ 保存性能监控器
        
        # 创建环境
        self.agent_env = RTreeEnvironment(config, tree_type="acppo")
        self.ref_env = RTreeEnvironment(config, tree_type=config.tree.reference_tree_type)
        
        # ✅ 创建固定的参考树（Static Reference Tree）
        self.static_ref_env = RTreeEnvironment(config, tree_type=config.tree.reference_tree_type)
        print("Building static reference tree...")
        for rect in dataset:
            self.static_ref_env.insert_rectangle(rect)
        print(f"Static reference tree built with {len(dataset)} rectangles.")
        
        # 创建奖励计算器（使用访问率，与评估保持一致）
        train_cfg = config.training
        self.reward_calc = RewardCalculator(
            query_area_ratio=train_cfg.training_query_area_ratio,
            num_samples=train_cfg.query_reward_freq,
            reward_type="access_rate_diff"  # ✅ 使用访问率（除以树高），与评估器一致
        )
        
        # 训练统计
        self.training_history: Dict[str, List] = {
            'episode_rewards': [],
            'actor_losses': [],
            'critic_losses': [],
            'query_performance': [],
            'training_time': []
        }
        
        # 最佳模型跟踪（基于与固定参考树的查询对比）
        self.best_efficiency_improvement = float('-inf')  # 效率提升越大越好
        self.best_efficiency_episode = -1
        
        # 跟踪最佳效率检查点路径（用于清理旧文件）
        self.best_efficiency_checkpoint_path = None
        
        # 初始化日志器（仅写入文件，不输出到控制台）
        # 使用独立的 logger 实例，避免与全局 logger 冲突
        import logging
        self.file_logger = logging.getLogger(f"GSAR-Trainer-{id(self)}")
        self.file_logger.setLevel(logging.INFO)
        
        # 只添加文件 Handler
        os.makedirs("logs", exist_ok=True)
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        log_file = os.path.join("logs", f"trainer_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        self.file_logger.addHandler(file_handler)
        
        # 禁止传播到根 logger（避免重复输出）
        self.file_logger.propagate = False
        
        self.log_file_path = log_file
    
    def train_episode(self, episode_idx: int) -> Dict:
        """
        训练一个 episode
        
        Args:
            episode_idx: Episode 索引
        
        Returns:
            包含本 episode 统计信息的字典
        """
        episode_start = time.time()
        
        # 重置环境
        self.agent_env.reset()
        self.ref_env.reset()
        
        # 清空 Agent 记忆
        self.agent.clear_memory()
        self.agent.reset_training_history()
        
        episode_reward = 0.0
        rl_decisions = 0
        heuristic_decisions = 0
        
        # 遍历数据集
        for i, rect in enumerate(self.dataset):
            # 打印rect数据
            # print(f"[DEBUG] Inserting rect: {rect}")
            
            # 1. 插入到参考树（使用传统策略）
            self.ref_env.insert_rectangle(rect)
            
            # 2. 插入到 Agent 树（使用 RL 决策）
            success, decision_type = self.agent_env.insert_rectangle(
                rect,
                use_rl=True,
                agent=self.agent,
                feature_type=self.config.model.feature_type,
                action_space_size=self.config.model.action_space_size,
                record_memory=True  # 记录状态和动作
            )
            
            if decision_type == "rl":
                rl_decisions += 1
            else:
                heuristic_decisions += 1
            
            # 3. 定期计算奖励并学习
            if self._should_calculate_reward(i):
                # 修复：在计算奖励前记录动作数量（此时 action_memory 还未被清空）
                num_actions_before = len(self.agent.action_memory)
                
                reward = self._calculate_and_learn(i)
                episode_reward += reward
                
                # Logger 仅写入文件（不打印到控制台）
                if i % 10000 == 0:
                    self.file_logger.info(f"[Episode {episode_idx}] Step {i}: Reward={reward:.6f}, Actions={num_actions_before}")
                
                # Print 到控制台，使用 \r 覆盖同一行，flush=True 确保立即显示
                print(f"[Episode {episode_idx}] Step {i}: Reward={reward:.6f}, Actions={num_actions_before}", end="\r", flush=True)
                
                # 恢复：原始 gsar_tree.py 的设计就是在计算奖励后同步参考树
                # 这是为了加速训练，让下一个批次的比较在相似起点上进行
                # 关键是在计算奖励的那一刻，两棵树是不同的（因为插入策略不同）
                self.ref_env.copy_from(self.agent_env)
        
        episode_time = time.time() - episode_start
        
        # 收集统计信息
        stats = {
            'episode': episode_idx,
            'total_reward': episode_reward,
            'rl_decisions': rl_decisions,
            'heuristic_decisions': heuristic_decisions,
            'rl_ratio': rl_decisions / (rl_decisions + heuristic_decisions) if (rl_decisions + heuristic_decisions) > 0 else 0,
            'episode_time': episode_time
        }
        
        # 记录历史
        self.training_history['episode_rewards'].append(episode_reward)
        self.training_history['training_time'].append(episode_time)
        
        # 记录损失（如果有）
        if self.agent.training_history['actor_loss']:
            self.training_history['actor_losses'].extend(
                self.agent.training_history['actor_loss']
            )
            self.training_history['critic_losses'].extend(
                self.agent.training_history['critic_loss']
            )
        
        return stats
    
    def _should_calculate_reward(self, current_idx: int) -> bool:
        """
        判断是否应该计算奖励
        
        Args:
            current_idx: 当前数据索引
        
        Returns:
            是否计算奖励
        """
        freq = self.config.training.query_reward_freq
        mem_size = len(self.agent.state_memory)
        
        return (mem_size >= freq) and (mem_size % freq == 0)
    
    def _calculate_and_learn(self, current_idx: int) -> float:
        """
        计算奖励并执行学习更新
        
        Args:
            current_idx: 当前数据索引
        
        Returns:
            总奖励值
        """
        # 获取最近的矩形用于生成查询
        recent_rects = self.dataset[max(0, current_idx - 9):current_idx + 1]
        
        # 计算奖励
        reward = self.reward_calc.calculate_reward(
            self.agent_env,
            self.ref_env,
            recent_rects,
            x_range=(self.config.data.x_min, self.config.data.x_max),
            y_range=(self.config.data.y_min, self.config.data.y_max)
        )
        
        # 记录当前的动作数量（在清空之前）
        num_actions = len(self.agent.action_memory)
        
        # 存储 transitions
        self._store_transitions(reward)
        
        # 执行 PPO 学习
        loss_info = self.agent.ppo_learn(track_loss=True)
        
        # 清空临时记忆
        self.agent.clear_memory()
        
        # 修复：使用之前记录的 num_actions
        return reward * num_actions
    
    def _store_transitions(self, reward: float):
        """
        存储经验转移
        
        Args:
            reward: 奖励值
        """
        records_num = len(self.agent.action_memory)
        
        for idx in range(records_num):
            state = self.agent.state_memory[idx]
            action = self.agent.action_memory[idx]
            
            # 下一个状态（如果是最后一条，则使用当前状态）
            if idx < records_num - 1:
                next_state = self.agent.state_memory[idx + 1]
                done = False
            else:
                next_state = state
                done = True
            
            transition = Transition(state, action, reward, next_state, done)
            self.agent.store_transition(transition)
    
    def evaluate_against_static_ref(self, test_queries: List[float] = None) -> Dict:
        """
        将当前 Agent 树与固定的参考树进行查询效率对比
        
        Args:
            test_queries: 测试查询比例列表
        
        Returns:
            包含各查询比例下性能对比的字典
        """
        if test_queries is None:
            test_queries = self.config.query.range_ratios
        
        results = {}
        
        for ratio in test_queries:
            # 生成查询矩形
            query_rect = self.reward_calc.generate_random_query(
                x_range=(self.config.data.x_min, self.config.data.x_max),
                y_range=(self.config.data.y_min, self.config.data.y_max)
            )
            
            # 执行查询
            agent_access = self.agent_env.query(query_rect)
            static_ref_access = self.static_ref_env.query(query_rect)
            
            # 计算改进百分比：(Ref - Agent) / Ref
            # 如果 Agent 访问节点更少，improvement 为正数
            improvement = 0.0
            if static_ref_access > 0:
                improvement = ((static_ref_access - agent_access) / static_ref_access) * 100
            
            results[f"{ratio}%"] = {
                'agent_access': agent_access,
                'static_ref_access': static_ref_access,
                'improvement': improvement
            }
        
        return results

    def evaluate(self, test_queries: List[float] = None) -> Dict:
        """
        评估当前策略的性能
        
        Args:
            test_queries: 测试查询比例列表，如果为 None 则使用配置中的默认值
        
        Returns:
            评估结果字典
        """
        if test_queries is None:
            test_queries = self.config.query.range_ratios
        
        results = {}
        
        for ratio in test_queries:
            # 生成查询矩形
            query_rect = self.reward_calc.generate_random_query(
                x_range=(self.config.data.x_min, self.config.data.x_max),
                y_range=(self.config.data.y_min, self.config.data.y_max)
            )
            
            # 执行查询
            agent_access = self.agent_env.query(query_rect)
            ref_access = self.ref_env.query(query_rect)
            
            results[f"{ratio}%"] = {
                'agent_access': agent_access,
                'ref_access': ref_access,
                'improvement': ((ref_access - agent_access) / ref_access * 100) if ref_access > 0 else 0
            }
        
        return results
    
    def save_best_model(self, episode_idx: int, improvement: float):
        """
        保存最佳模型（基于与固定参考树的查询效率对比）
        
        Args:
            episode_idx: 当前 episode 索引（从 0 开始）
            improvement: 相比固定参考树的效率提升百分比（越大越好）
        """
        if improvement > self.best_efficiency_improvement:
            old_best = self.best_efficiency_improvement
            self.best_efficiency_improvement = improvement
            self.best_efficiency_episode = episode_idx
            
            # ⭐ 关键修复：直接传入 episode_idx + 1（转换为人类可读的 Episode 编号）
            checkpoint_path = self.config.get_model_filepath("train", episode=episode_idx + 1)
            
            # ⭐ 确保子目录存在
            checkpoint_dir = os.path.dirname(checkpoint_path)
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # ⭐ 删除旧的最佳效率检查点（同一配置下的所有旧模型）
            import glob
            pattern = self.config.get_model_filepath("train", episode=None).replace(
                "{episode}", "*"
            )
            old_models = glob.glob(pattern)
            for old_model in old_models:
                if old_model != checkpoint_path and os.path.exists(old_model):
                    try:
                        os.remove(old_model)
                        print(f"  Removed old best efficiency checkpoint: {os.path.basename(old_model)}")
                    except Exception as e:
                        print(f"  Warning: Failed to remove old checkpoint: {e}")
            
            # 保存新的最佳模型
            self.agent.save_checkpoint(checkpoint_path)
            self.best_efficiency_checkpoint_path = checkpoint_path
            
            print(f"✓ New best query efficiency! Episode {episode_idx + 1}: {old_best:.2f}% → {improvement:.2f}%")
            print(f"  Saved to: {checkpoint_path}")
    
    def _calculate_composite_metric(self, eval_results: Dict[str, Dict]) -> float:
        """
        计算复合评估指标（多个查询比例的平均性能）
        
        Args:
            eval_results: 评估结果字典，格式为 {"2.0%": {...}, "1.0%": {...}, ...}
        
        Returns:
            复合指标值（平均节点访问数）
        
        Note:
            使用加权平均：大查询范围权重更高（更接近实际使用场景）
            权重分配：2.0%(0.3), 1.0%(0.25), 0.5%(0.2), 0.05%(0.15), 0.01%(0.1)
        """
        # 定义查询比例的权重（可根据实际需求调整）
        weights = {
            "2.0%": 0.30,
            "1.0%": 0.25,
            "0.5%": 0.20,
            "0.05%": 0.15,
            "0.01%": 0.10
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for query_ratio, metrics in eval_results.items():
            if query_ratio in weights:
                weight = weights[query_ratio]
                weighted_sum += metrics['agent_access'] * weight
                total_weight += weight
        
        # 归一化（防止权重和不等于 1）
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            # 如果没有匹配的查询比例，返回简单平均
            accesses = [m['agent_access'] for m in eval_results.values()]
            return np.mean(accesses) if accesses else float('inf')
    
    def load_best_model(self, filepath: str = None):
        """
        加载最佳模型
        
        Args:
            filepath: 模型路径，如果为 None 则使用配置中的路径
        """
        if filepath is None:
            filepath = self.config.get_model_filepath("train")
        
        self.agent.load_checkpoint(filepath)
        print(f"✓ Loaded best model from episode {self.best_episode}")
    
    def train(self, num_episodes: int = None, eval_freq: int = 5) -> Dict:
        """
        完整训练流程
        
        Args:
            num_episodes: 训练 episode 数量，如果为 None 则使用配置中的值
            eval_freq: 评估频率（每多少个 episode 评估一次）
        
        Returns:
            完整的训练历史
        """
        if num_episodes is None:
            num_episodes = self.config.training.num_episodes
        
        print("\n" + "=" * 80)
        print(f"Starting Training: {num_episodes} episodes")
        print("=" * 80)
        
        for episode in range(num_episodes):
            # 训练一个 episode
            stats = self.train_episode(episode)
            
            # ⭐ 记录 episode 性能到监控器
            if self.performance_monitor:
                extra_info = {
                    'rl_decisions': stats['rl_decisions'],
                    'heuristic_decisions': stats['heuristic_decisions'],
                    'rl_ratio': stats['rl_ratio']
                }
                self.performance_monitor.record_episode(
                    episode_idx=episode,
                    episode_time=stats['episode_time'],
                    reward=stats['total_reward'],
                    dataset_size=len(self.dataset),
                    extra_info=extra_info
                )
            
            # 打印进度
            print(f"\nEpisode {episode + 1}/{num_episodes}:")
            print(f"  Time: {stats['episode_time']:.2f}s")
            print(f"  Total Reward: {stats['total_reward']:.4f}")
            print(f"  RL Decisions: {stats['rl_decisions']} ({stats['rl_ratio']:.1%})")
            print(f"  Heuristic: {stats['heuristic_decisions']}")
            
            # 每个 Episode 结束后，与固定参考树进行查询效率对比
            print(f"\n  Evaluating against static reference tree...")
            eval_results = self.evaluate_against_static_ref()
            
            # 计算综合效率提升（取所有查询比例的平均值）
            improvements = [metrics['improvement'] for metrics in eval_results.values()]
            avg_improvement = np.mean(improvements) if improvements else 0.0
            
            # 记录评估结果
            self.training_history['query_performance'].append(eval_results)
            
            # 打印评估结果
            for query_name, metrics in eval_results.items():
                print(f"    {query_name}: Agent={metrics['agent_access']:.2f}, "
                      f"StaticRef={metrics['static_ref_access']:.2f}, "
                      f"Impv={metrics['improvement']:+.2f}%")
            print(f"    Average Improvement: {avg_improvement:+.2f}%")
            
            # 如果效率提升创历史新高，保存为最佳模型
            if avg_improvement > self.best_efficiency_improvement:
                self.save_best_model(episode, avg_improvement)
            
            # 定期加载最佳模型（帮助稳定训练，防止灾难性遗忘）
            load_freq = self.config.training.load_best_model_freq
            if (episode + 1) % load_freq == 0 and episode > 0:
                if self.best_efficiency_checkpoint_path and os.path.exists(self.best_efficiency_checkpoint_path):
                    print(f"\n  🔄 Loading best efficiency model from episode {self.best_efficiency_episode} to stabilize training...")
                    self.agent.load_checkpoint(self.best_efficiency_checkpoint_path)
                    print(f"  ✓ Model reloaded successfully.")
            
            # 学习率衰减（帮助稳定后期训练）
            self.agent.decay_learning_rate()
        
        print("\n" + "=" * 80)
        print("Training Completed!")
        print(f"Best Model Episode:    {self.best_efficiency_episode}")
        print(f"Best Avg Improvement:  {self.best_efficiency_improvement:.2f}%")
        print("=" * 80 + "\n")
        
        # ✅ 返回训练历史，供后续分析和可视化使用
        return self.training_history
    
    def print_summary(self):
        """
        打印训练摘要信息
        
        显示训练的关键统计信息，包括：
        - Episode 数量
        - 平均奖励
        - 最佳/最差奖励
        - 总训练时间
        - 最佳模型信息
        """
        rewards = self.training_history['episode_rewards']
        times = self.training_history['training_time']
        
        if not rewards:
            print("No training data available.")
            return
        
        total_episodes = len(rewards)
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        best_reward = max(rewards)
        worst_reward = min(rewards)
        total_time = sum(times)
        avg_time = np.mean(times)
        