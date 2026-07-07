"""
Self-Play 训练器 - 交叉验证式对抗训练
实现两个智能体通过相互对抗共同提升
"""
import time
import os
import numpy as np
from typing import List, Dict, Optional
from collections import namedtuple

from gsartree.config.default_config import GSARConfig
from gsartree.models.acppo_agent import ACPPOAgent, Transition
from gsartree.environment.rtree_env import RTreeEnvironment, MultiTreeEnvironment
from gsartree.environment.reward_calculator import ZeroSumRewardCalculator
from gsartree.utils.logger import get_logger
from gsartree.utils.checkpoint_manager import CheckpointManager


class SelfPlayTrainer:
    """
    Self-Play 训练器（方案B：交叉验证式）
    
    核心思想：
    1. 两个智能体在相同数据上独立构建树
    2. 通过交叉验证消除位置偏差
    3. 零和奖励确保公平竞争
    4. 双方都从对弈中学习
    
    Attributes:
        config: 配置对象
        agent: 主智能体
        competitor: 对手智能体
        multi_env: 多树环境（管理两棵树）
        reward_calc: 零和奖励计算器
        logger: 日志记录器
    """
    
    def __init__(
        self,
        config: GSARConfig,
        agent: ACPPOAgent,
        competitor: ACPPOAgent,
        dataset: List[List[float]],
        static_ref_tree=None,  # ⭐ 新增：静态参考树环境
        performance_monitor=None,  # ⭐ 新增：性能监控器
        logger=None
    ):
        """
        初始化 Self-Play 训练器
        
        Args:
            config: 完整的 GSAR 配置
            agent: 主智能体（Agent）
            competitor: 对手智能体（Competitor）
            dataset: 训练数据集
            static_ref_tree: 静态参考树环境（用于评估，如 R*-Tree）
            performance_monitor: 性能监控器实例（可选）
            logger: 日志记录器（可选）
        """
        self.config = config
        self.agent = agent
        self.competitor = competitor
        self.dataset = dataset
        self.static_ref_tree = static_ref_tree  # ⭐ 保存静态参考树
        self.performance_monitor = performance_monitor  # ⭐ 保存性能监控器
        self.logger = logger if logger is not None else get_logger()
        
        # 创建多树环境
        self.multi_env = MultiTreeEnvironment(config)
        self.multi_env.add_tree("agent", "acppo")
        self.multi_env.add_tree("competitor", "acppo")
        
        # 获取单独的环境引用（方便访问）
        self.agent_env = self.multi_env.get_tree("agent")
        self.competitor_env = self.multi_env.get_tree("competitor")
        
        # ✅ 创建零和奖励计算器（基于访问率，与评估一致）
        sp_cfg = config.self_play
        self.reward_calc = ZeroSumRewardCalculator(
            query_area_ratio=config.training.training_query_area_ratio,
            num_samples=config.training.query_reward_freq,
            reward_type="access_rate_diff"  # ✅ 使用访问率（除以树高），与评估器一致
        )
        
        # Self-Play 统计
        self.sp_statistics = {
            'agent_wins': 0,
            'competitor_wins': 0,
            'draws': 0,
            'performance_gaps': [],
            'episode_rewards': [],
            'agent_losses': [],
            'competitor_losses': [],
            'total_episodes': 0
        }
        
        # ⭐ 最佳模型跟踪（基于查询性能提升）
        self.best_metric = float('inf')  # 平均节点访问数（越小越好）
        self.best_episode = -1
        self.best_avg_improvement = -float('inf')  # 平均提升百分比
        
        # ⭐ 创建 CheckpointManager（仅保存 Agent 的最佳效率模型）
        model_dir = os.path.dirname(config.get_model_filepath("selfplay"))
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=model_dir,  # ✅ 修正：使用 checkpoint_dir 而非 save_dir
            max_checkpoints=1  # 只保留最佳模型
        )
        

    def train_episode(self, episode_idx: int) -> Dict:
        """
        训练一个 Self-Play episode
        """
        episode_start = time.time()
        
        self.multi_env.reset_all()
        self.agent.clear_memory()
        self.competitor.clear_memory()
        
        use_agent_first = (episode_idx % 2 == 0)
        player1 = self.agent if use_agent_first else self.competitor
        player2 = self.competitor if use_agent_first else self.agent
        player1_env = self.agent_env if use_agent_first else self.competitor_env
        player2_env = self.competitor_env if use_agent_first else self.agent_env
        
        episode_reward = 0.0
        
        p1_states_mem = []
        p1_actions_mem = []
        p2_states_mem = []
        p2_actions_mem = []
        
        for i, rect in enumerate(self.dataset):
            # Player 1 插入
            prev_len_p1 = len(p1_actions_mem)
            success, decision_type = player1_env.insert_rectangle(
                rect,
                use_rl=True,
                agent=player1,
                feature_type=self.config.model.feature_type,
                action_space_size=self.config.model.action_space_size,
                record_memory=True,
                state_memory=p1_states_mem,
                action_memory=p1_actions_mem
            )
            # 仅当 RL 决策成功但动作未记录时才警告
            if success and decision_type == "rl" and len(p1_actions_mem) == prev_len_p1:
                self.logger.warning(f"⚠️ Player1 RL decision but action not recorded at index {i}")
            
            # Player 2 插入
            prev_len_p2 = len(p2_actions_mem)
            success, decision_type = player2_env.insert_rectangle(
                rect,
                use_rl=True,
                agent=player2,
                feature_type=self.config.model.feature_type,
                action_space_size=self.config.model.action_space_size,
                record_memory=True,
                state_memory=p2_states_mem,
                action_memory=p2_actions_mem
            )
            if success and decision_type == "rl" and len(p2_actions_mem) == prev_len_p2:
                self.logger.warning(f"⚠️ Player2 RL decision but action not recorded at index {i}")
            
            # 定期学习
            if self._should_learn(len(p1_states_mem)):
                reward_info = self._calculate_zero_sum_reward_and_learn(
                    player1_env, player2_env, i,
                    p1_states_mem, p1_actions_mem,
                    p2_states_mem, p2_actions_mem,
                    player1, player2
                )
                episode_reward += reward_info['total_reward']
                self._update_statistics(reward_info)
                
                p1_states_mem.clear()
                p1_actions_mem.clear()
                p2_states_mem.clear()
                p2_actions_mem.clear()
        
        episode_time = time.time() - episode_start
        self.sp_statistics['episode_rewards'].append(episode_reward)
        
        stats = {
            'episode': episode_idx,
            'total_reward': episode_reward,
            'episode_time': episode_time,
            'player1_is_agent': use_agent_first
        }
        return stats

    def _should_learn(self, memory_size: int) -> bool:
        """判断是否应该执行学习"""
        freq = self.config.training.query_reward_freq
        return (memory_size >= freq) and (memory_size % freq == 0)
    
        
    def _calculate_zero_sum_reward_and_learn(
        self,
        player1_env: RTreeEnvironment,
        player2_env: RTreeEnvironment,
        current_idx: int,
        p1_states: List,
        p1_actions: List,
        p2_states: List,
        p2_actions: List,
        player1: ACPPOAgent,
        player2: ACPPOAgent
    ) -> Dict:
        """
        计算零和奖励并执行双方学习
        """
        # 关键：任一玩家没有动作记录则无法学习
        if len(p1_actions) == 0 or len(p2_actions) == 0:
            self.logger.debug(f"Skipping learning: p1_actions={len(p1_actions)}, p2_actions={len(p2_actions)}")
            return {
                'p1_reward': 0.0,
                'p2_reward': 0.0,
                'total_reward': 0.0,
                'p1_access': 0.0,
                'p2_access': 0.0
            }

        recent_rects = self.dataset[max(0, current_idx - 9):current_idx + 1]
        
        p1_reward, p2_reward = self.reward_calc.calculate_zero_sum_rewards(
            player1_env,
            player2_env,
            recent_rects,
            x_range=(self.config.data.x_min, self.config.data.x_max),
            y_range=(self.config.data.y_min, self.config.data.y_max)
        )
        
        if np.isnan(p1_reward) or np.isinf(p1_reward):
            self.logger.warning(f"⚠️ Player 1 reward is NaN/Inf: {p1_reward}, replacing with 0")
            p1_reward = 0.0
        if np.isnan(p2_reward) or np.isinf(p2_reward):
            self.logger.warning(f"⚠️ Player 2 reward is NaN/Inf: {p2_reward}, replacing with 0")
            p2_reward = 0.0
        
        # 存储 transitions
        self._store_transitions_for_player(player1, p1_states, p1_actions, p1_reward)
        self._store_transitions_for_player(player2, p2_states, p2_actions, p2_reward)
        
        # 仅当 buffer 非空时学习
        p1_loss = player1.ppo_learn(track_loss=True) if len(player1.buffer) > 0 else {}
        p2_loss = player2.ppo_learn(track_loss=True) if len(player2.buffer) > 0 else {}
        
        if p1_loss and 'actor_loss' in p1_loss:
            self.sp_statistics['agent_losses'].append(p1_loss['actor_loss'])
        if p2_loss and 'actor_loss' in p2_loss:
            self.sp_statistics['competitor_losses'].append(p2_loss['actor_loss'])
        
        p1_states.clear()
        p1_actions.clear()
        p2_states.clear()
        p2_actions.clear()
        
        total_reward = p1_reward * len(p1_actions) + p2_reward * len(p2_actions)
        
        return {
            'p1_reward': p1_reward,
            'p2_reward': p2_reward,
            'total_reward': total_reward,
            'p1_access': abs(p1_reward),
            'p2_access': abs(p2_reward)
        }
        
    
    def _store_transitions_for_player(
        self,
        player: ACPPOAgent,
        states: List,
        actions: List,
        reward: float
    ):
        """为单个玩家存储 transitions"""
        records_num = len(actions)
        
        for idx in range(records_num):
            state = states[idx]
            action = actions[idx]
            
            if idx < records_num - 1:
                next_state = states[idx + 1]
                done = False
            else:
                next_state = state
                done = True
            
            transition = Transition(state, action, reward, next_state, done)
            player.store_transition(transition)
    
    def _update_statistics(self, reward_info: Dict):
        """
        更新 Self-Play 统计信息（使用原始奖励，而非绝对值）
        """
        p1_reward = reward_info['p1_reward']
        p2_reward = reward_info['p2_reward']
        
        # 性能差距 = 双方奖励差值的绝对值（反映了优劣差距大小）
        gap = abs(p1_reward - p2_reward)
        self.sp_statistics['performance_gaps'].append(gap)
        
        # 胜负判断：奖励高的一方胜出（零和奖励下，如果 p1_reward > 0 则 p1 胜）
        if p1_reward > p2_reward:
            self.sp_statistics['agent_wins'] += 1
        elif p2_reward > p1_reward:
            self.sp_statistics['competitor_wins'] += 1
        else:
            self.sp_statistics['draws'] += 1
    
    def train(
        self,
        num_episodes: int = None,
        eval_freq: int = 5,
        sync_freq: int = 10,
        static_ref_tree: RTreeEnvironment = None,
        rollback_freq: int = 10
    ) -> Dict:
        """
        完整 Self-Play 训练流程
        
        Args:
            num_episodes: 训练 episode 数量
            eval_freq: 评估频率
            sync_freq: 模型同步频率（课程学习）
            static_ref_tree: ⭐ 静态参考树（用于评估查询性能）
            rollback_freq: ⭐ 回滚频率（每隔 N 个 episode 检查并可能回滚到最佳模型）
        
        Returns:
            训练历史（包含 best_metric 和 best_episode）
        """
        if num_episodes is None:
            num_episodes = self.config.self_play.num_episodes
        
        self.logger.section("Starting Self-Play Training")
        self.logger.info(f"Total Episodes: {num_episodes}")
        self.logger.info(f"Evaluation Frequency: {eval_freq}")
        self.logger.info(f"Sync Frequency: {sync_freq}")
        self.logger.info(f"Rollback Frequency: {rollback_freq}")
        self.logger.info(f"Cross Validation: {self.config.self_play.use_cross_validation}")
        if static_ref_tree is not None:
            self.logger.info(f"Static Reference Tree: Enabled (for query performance evaluation)")
        
        for episode in range(num_episodes):
            episode_start = time.time()
            stats = self.train_episode(episode)
            episode_time = time.time() - episode_start
            
            # ⭐ 记录 episode 性能到监控器
            if self.performance_monitor:
                extra_info = {
                    'player1_is_agent': stats['player1_is_agent'],
                    'agent_wins': self.sp_statistics['agent_wins'],
                    'competitor_wins': self.sp_statistics['competitor_wins'],
                    'draws': self.sp_statistics['draws']
                }
                self.performance_monitor.record_episode(
                    episode_idx=episode,
                    episode_time=episode_time,
                    reward=stats['total_reward'],
                    dataset_size=len(self.dataset),
                    extra_info=extra_info
                )
            
            # 使用 logger 输出训练进度
            if (episode + 1) % eval_freq == 0 or episode == 0:
                self.logger.subsection(f"Episode {episode + 1}/{num_episodes}")
                self.logger.info(f"Time:                  {episode_time:.2f}s")
                self.logger.info(f"Total Reward:          {stats['total_reward']:.4f}")
                self.logger.info(f"Player 1 is Agent:     {stats['player1_is_agent']}")
                self.logger.info(f"Wins - Agent:          {self.sp_statistics['agent_wins']}")
                self.logger.info(f"Wins - Competitor:     {self.sp_statistics['competitor_wins']}")
                self.logger.info(f"Draws:                 {self.sp_statistics['draws']}")
                
                if self.sp_statistics['performance_gaps']:
                    avg_gap = np.mean(self.sp_statistics['performance_gaps'][-eval_freq:])
                    self.logger.info(f"Avg Performance Gap:   {avg_gap:.4f} (last {eval_freq} episodes)")
                
                # 定期评估查询性能（如果有静态参考树）
                if static_ref_tree is not None and (episode + 1) % eval_freq == 0:
                    self.logger.info(f"\nEvaluating against static reference tree...")
                    eval_results = self.evaluate_against_static_ref(static_ref_tree)
                    
                    # 打印每个查询比例的结果
                    for ratio in self.config.query.range_ratios:
                        key = f"{ratio}%"
                        if key in eval_results:
                            res = eval_results[key]
                            self.logger.info(
                                f"  {key}: Agent={res['agent_access']:.2f}, "
                                f"Ref={res['ref_access']:.2f}, "
                                f"Impv={res['improvement']:+.2f}%"
                            )
                    
                    # 计算平均提升
                    avg_improvement = eval_results.get('average_improvement', 0.0)
                    self.logger.info(f"  Average Improvement: {avg_improvement:+.2f}%")
                    
                    # 如果平均提升优于历史最佳，保存模型
                    if avg_improvement > self.best_avg_improvement:
                        old_improvement = self.best_avg_improvement
                        self.best_avg_improvement = avg_improvement
                        self.best_episode = episode
                        self.best_metric = avg_improvement  # 使用提升百分比作为 metric
                        
                        # ⭐ 使用配置类的方法生成路径（自动包含子目录）
                        checkpoint_path = self.config.get_model_filepath(
                            "selfplay", 
                            episode=episode + 1
                        )
                        
                        # ⭐ 确保子目录存在
                        checkpoint_dir = os.path.dirname(checkpoint_path)
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        
                        # ⭐ 删除旧的同配置 Self-Play 模型文件
                        import glob
                        pattern = self.config.get_model_filepath("selfplay", episode=None).replace(
                            "{episode}", "*"
                        )
                        old_models = glob.glob(pattern)
                        for old_model in old_models:
                            if old_model != checkpoint_path and os.path.exists(old_model):
                                try:
                                    os.remove(old_model)
                                    self.logger.info(f"Removed old Self-Play model: {os.path.basename(old_model)}")
                                except Exception as e:
                                    self.logger.warning(f"Failed to remove old model {old_model}: {e}")
                        
                        # 保存新模型
                        self.agent.save_checkpoint(checkpoint_path)
                        
                        self.logger.success(
                            f"✓ New best query efficiency! "
                            f"Episode {self.best_episode + 1}: "
                            f"{old_improvement:+.2f}% → {avg_improvement:+.2f}%"
                        )
                        self.logger.success(f"✓ Checkpoint saved to {checkpoint_path}")
            
            # 定期同步模型（课程学习）
            if sync_freq > 0 and (episode + 1) % sync_freq == 0:
                self.logger.info(f"Syncing models at episode {episode + 1}...")
                self.sync_models()
            
            # ⭐ 定期回滚到最佳模型（防止灾难性遗忘）
            if rollback_freq > 0 and (episode + 1) % rollback_freq == 0 and self.best_episode >= 0:
                if episode != self.best_episode:  # 如果当前不是最佳 episode
                    self.logger.info(
                        f"\n🔄 Loading best efficiency model from episode {self.best_episode + 1} "
                        f"to stabilize training..."
                    )
                    
                    # 查找最佳模型文件
                    dist = self.config.data.distribution
                    max_entry = self.config.tree.max_entry
                    action_space = self.config.model.action_space_size
                    
                    best_model_filename = (
                        f"selfplay_TW1_{dist}_rstar_{max_entry}-{action_space}"
                        f"_BestEfficiency_Ep{self.best_episode + 1}.pth"
                    )
                    best_model_path = os.path.join(
                        self.checkpoint_manager.checkpoint_dir,
                        best_model_filename
                    )
                    
                    if os.path.exists(best_model_path):
                        self.agent.load_checkpoint(best_model_path)
                        self.logger.success(f"✓ Model reloaded successfully.")
                    else:
                        self.logger.warning(f"⚠️ Best model file not found: {best_model_path}")
        
        self.sp_statistics['total_episodes'] = num_episodes
        
        # ✅ 计算并记录最佳性能指标（使用平均性能差距的负值作为 metric，越小越好）
        if self.sp_statistics['performance_gaps']:
            # 使用最后一个 eval_freq 个 episode 的平均性能差距作为评估指标
            recent_gaps = self.sp_statistics['performance_gaps'][-min(eval_freq, len(self.sp_statistics['performance_gaps'])):]
            current_metric = np.mean(recent_gaps)
            
            # 如果当前指标优于历史最佳，更新最佳记录
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                self.best_episode = num_episodes - 1
                self.logger.info(f"✓ New best model at episode {self.best_episode} (metric: {self.best_metric:.4f})")
        
        # ✅ 将最佳模型信息添加到返回结果中
        self.sp_statistics['best_metric'] = self.best_metric if self.best_episode >= 0 else None
        self.sp_statistics['best_episode'] = self.best_episode
        
        self.logger.section("Self-Play Training Completed!")
        self.logger.info(f"Final Stats:")
        self.logger.info(f"  Agent Wins:          {self.sp_statistics['agent_wins']}")
        self.logger.info(f"  Competitor Wins:     {self.sp_statistics['competitor_wins']}")
        self.logger.info(f"  Draws:               {self.sp_statistics['draws']}")
        
        if self.sp_statistics['performance_gaps']:
            final_avg_gap = np.mean(self.sp_statistics['performance_gaps'])
            self.logger.info(f"  Avg Performance Gap: {final_avg_gap:.4f}")
        
        if self.best_episode >= 0:
            self.logger.info(f"  Best Episode:        {self.best_episode + 1}")
            self.logger.info(f"  Best Metric:         {self.best_metric:.4f}")
        
        return self.sp_statistics

    def evaluate_against_static_ref(
        self,
        static_ref_tree: RTreeEnvironment,
        query_ratios: List[float] = None
    ) -> Dict:
        """
        ⭐ 评估 Agent 相对于静态参考树的性能提升
        
        Args:
            static_ref_tree: 静态参考树（如 R*-Tree）
            query_ratios: 查询比例列表
        
        Returns:
            评估结果字典，包含每个查询比例的性能和提升百分比
        """
        if query_ratios is None:
            query_ratios = self.config.query.range_ratios
        
        results = {}
        total_improvement = 0.0
        num_queries = len(query_ratios)
        
        for ratio in query_ratios:
            # 生成随机查询
            query_rect = self.reward_calc.generate_random_query(
                x_range=(self.config.data.x_min, self.config.data.x_max),
                y_range=(self.config.data.y_min, self.config.data.y_max)
            )
            
            # 执行查询
            agent_access = self.agent_env.query(query_rect)
            ref_access = static_ref_tree.query(query_rect)
            
            # 计算提升百分比
            improvement = ((ref_access - agent_access) / ref_access * 100) if ref_access > 0 else 0.0
            
            results[f"{ratio}%"] = {
                'agent_access': agent_access,
                'ref_access': ref_access,
                'improvement': improvement
            }
            
            total_improvement += improvement
        
        # 计算平均提升
        avg_improvement = total_improvement / num_queries if num_queries > 0 else 0.0
        results['average_improvement'] = avg_improvement
        
        return results

    def evaluate(self, test_queries: List[float] = None) -> Dict:
        """
        评估双方性能
        
        Args:
            test_queries: 测试查询比例列表
        
        Returns:
            评估结果
        """
        if test_queries is None:
            test_queries = self.config.query.range_ratios
        
        results = {
            'agent': {},
            'competitor': {}
        }
        
        for ratio in test_queries:
            query_rect = self.reward_calc.generate_random_query(
                x_range=(self.config.data.x_min, self.config.data.x_max),
                y_range=(self.config.data.y_min, self.config.data.y_max)
            )
            
            agent_access = self.agent_env.query(query_rect)
            comp_access = self.competitor_env.query(query_rect)
            
            results['agent'][f"{ratio}%"] = agent_access
            results['competitor'][f"{ratio}%"] = comp_access
        
        return results
    
    def save_best_model(self, episode_idx: int, metric: float, filepath: str = None):
        """保存最佳模型（保存 Agent）"""
        if metric < self.best_metric:
            self.best_metric = metric
            self.best_episode = episode_idx
            
            if filepath is None:
                filepath = self.config.get_model_filepath("self-play")
            
            self.agent.save_checkpoint(filepath)
            print(f"✓ New best SP model saved at episode {episode_idx} (metric: {metric:.4f})")
    
    def load_best_model(self, filepath: str = None):
        """加载最佳模型"""
        if filepath is None:
            filepath = self.config.get_model_filepath("self-play")
        
        self.agent.load_checkpoint(filepath)
        print(f"✓ Loaded best SP model from episode {self.best_episode}")
    
    def sync_models(self):
        """同步双方模型（用于课程式学习）"""
        # 将 Agent 的参数复制到 Competitor
        import torch
        
        self.competitor.actor.load_state_dict(self.agent.actor.state_dict())
        self.competitor.critic.load_state_dict(self.agent.critic.state_dict())
        self.competitor.actor_optimizer.load_state_dict(self.agent.actor_optimizer.state_dict())
        self.competitor.critic_optimizer.load_state_dict(self.agent.critic_optimizer.state_dict())
        
        print("✓ Models synced")
