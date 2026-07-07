"""
ACPPO Agent - Actor-Critic Proximal Policy Optimization 智能体
ACPPO Agent - Actor-Critic Proximal Policy Optimization Agent

整合了策略网络、价值网络和 PPO 学习算法
Integrates policy network, value network, and PPO learning algorithm
"""
import torch
import numpy as np
import random
from collections import namedtuple
from typing import List, Tuple, Optional, Dict

from gsartree.models.policy_net import PolicyNet
from gsartree.models.value_net import ValueNet
from gsartree.config.default_config import GSARConfig

# 定义转移数据结构 / Define transition data structure
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class ACPPOAgent:
    """
    ACPPO 智能体 / ACPPO Agent
    
    实现了基于 PPO 的 Actor-Critic 强化学习算法，用于优化 RTree 插入策略。
    Implements PPO-based Actor-Critic reinforcement learning algorithm for optimizing RTree insertion strategy.
    
    Attributes:
        config: 配置对象 / Configuration object
        device: 计算设备 (cuda/cpu) / Computing device
        actor: 策略网络 / Policy network
        critic: 价值网络 / Value network
        buffer: 经验回放缓冲区 / Experience replay buffer
    """
    
    def __init__(self, config: GSARConfig, device: torch.device):
        """
        初始化 ACPPO Agent / Initialize ACPPO Agent
        
        Args:
            config: 完整的 GSAR 配置 / Complete GSAR configuration
            device: PyTorch 设备 / PyTorch device
        """
        self.config = config
        self.device = device
        
        # 从配置中提取参数 / Extract parameters from configuration
        model_cfg = config.model
        train_cfg = config.training
        
        self.n_features = model_cfg.state_space_size  # 状态空间大小 / State space size
        self.n_actions = model_cfg.action_space_size  # 动作空间大小 / Action space size
        self.n_hidden = model_cfg.n_hidden  # 隐藏层大小 / Hidden layer size
        
        # PPO 超参数 / PPO hyperparameters
        self.gamma = train_cfg.gamma  # 折扣因子 / Discount factor
        self.lmbda = train_cfg.lmbda  # GAE lambda参数 / GAE lambda parameter
        self.epochs = train_cfg.epochs  # PPO更新轮数 / PPO update epochs
        self.eps = train_cfg.eps  # PPO裁剪参数 / PPO clipping parameter
        self.buffer_size = train_cfg.buffer_size  # 经验缓冲区大小 / Experience buffer size
        self.epsilon = train_cfg.epsilon  # 探索率（epsilon-greedy）/ Exploration rate
        
        # 学习率 / Learning rates
        self.lr_actor = model_cfg.lr_actor  # Actor学习率 / Actor learning rate
        self.lr_critic = model_cfg.lr_critic  # Critic学习率 / Critic learning rate
        
        # 梯度裁剪参数（防止梯度爆炸）/ Gradient clipping parameter (prevent gradient explosion)
        # self.max_grad_norm = 0.5
        self.max_grad_norm = 0.3  # 从 0.5 降低 / Reduced from 0.5
        
        # 学习率衰减参数 / Learning rate decay parameters
        self.lr_decay_rate = 0.99  # 每个 episode 后学习率乘以该系数 / Learning rate multiplied by this coefficient after each episode
        self.min_lr = 1e-6  # 最小学习率 / Minimum learning rate
        
        # 初始化网络 / Initialize networks
        self._build_networks()
        
        # 经验缓冲区 / Experience buffer
        self.buffer: List[Transition] = []
        
        # 训练历史 / Training history
        self.training_history: Dict[str, List[float]] = {
            'actor_loss': [],  # Actor损失历史 / Actor loss history
            'critic_loss': [],  # Critic损失历史 / Critic loss history
            'rewards': []  # 奖励历史 / Reward history
        }
        
        # 临时记忆（用于一个 episode 内）/ Temporary memory (for one episode)
        self.state_memory: List[np.ndarray] = []  # 状态记忆 / State memory
        self.action_memory: List[int] = []  # 动作记忆 / Action memory
        
        # 步数计数 / Step counter
        self.step = 0
        
        # Episode 计数器（用于学习率衰减）/ Episode counter (for learning rate decay)
        self.episode_count = 0
    
    def _build_networks(self):
        """构建 Actor 和 Critic 网络 / Build Actor and Critic networks"""
        # Actor (Policy Network) - 策略网络，输出动作概率分布
        # Actor (Policy Network) - outputs action probability distribution
        self.actor = PolicyNet(
            self.n_features,
            self.n_hidden,
            self.n_actions
        ).to(self.device)
        
        # Critic (Value Network) - 价值网络，输出状态价值
        # Critic (Value Network) - outputs state value
        self.critic = ValueNet(
            self.n_features,
            self.n_hidden
        ).to(self.device)
        
        # 优化器 / Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.lr_critic
        )

    def choose_action(
        self,
        state: np.ndarray,
        explore: bool = True,
        sp_flag: bool = False
    ) -> Tuple[int, torch.Tensor]:
        """
        根据当前状态选择动作 / Choose action based on current state
        
        Args:
            state: 状态向量 / State vector
            explore: 是否启用探索（epsilon-greedy）/ Whether to enable exploration (epsilon-greedy)
            sp_flag: Self-Play 标志（影响随机动作范围）/ Self-Play flag (affects random action range)
        
        Returns:
            (action, log_prob): 选择的动作和其对数概率 / (action, log_prob): chosen action and its log probability
        """
        # ✅ 防御：检查输入状态的有效性
        # ✅ Defense: check input state validity
        if state is None or np.any(np.isnan(state)) or np.any(np.isinf(state)):
            print(f"⚠️  Warning: Invalid state (NaN/Inf or None) in choose_action, using random action")
            action = random.randint(0, self.n_actions - 1)
            log_prob = torch.tensor(0.0, dtype=torch.float32).to(self.device)
            return action, log_prob
        
        # 转换为 tensor / Convert to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        try:
            # 获取动作概率 / Get action probabilities
            with torch.no_grad():
                action_probs = self.actor(state_tensor)
            
            # 增强检查：检测 NaN、Inf 和负值
            # Enhanced check: detect NaN, Inf, and negative values
            if torch.any(torch.isnan(action_probs)) or torch.any(torch.isinf(action_probs)):
                raise ValueError(f"Actor output contains NaN/Inf: {action_probs}")
            
            if torch.any(action_probs < 0):
                raise ValueError(f"Actor output contains negative values: {action_probs}")
            
            # 确保概率和接近 1（允许小的浮点误差）
            # Ensure probability sum is close to 1 (allow small floating-point error)
            prob_sum = action_probs.sum().item()
            if abs(prob_sum - 1.0) > 1e-5:
                print(f"⚠️  Warning: Action probs sum to {prob_sum:.6f}, normalizing...")
                action_probs = action_probs / (prob_sum + 1e-8)
            
            # 创建分布 / Create distribution
            dist = torch.distributions.Categorical(action_probs)
            
            # Epsilon-greedy 策略 / Epsilon-greedy strategy
            if explore and random.random() < self.epsilon:
                # 探索：随机选择动作 / Explore: randomly select action
                action = random.randint(0, self.n_actions - 1)
            else:
                # 利用：选择概率最大的动作 / Exploit: select action with highest probability
                action = action_probs.argmax().item()
            
            # 计算对数概率 / Calculate log probability
            log_prob = dist.log_prob(torch.tensor(action, dtype=torch.int64).to(self.device))
            
            self.step += 1
            
            return action, log_prob
        
        except (ValueError, RuntimeError) as e:
            print(f"⚠️  Exception in choose_action: {e}")
            print(f"    State shape: {state.shape}, State range: [{state.min():.4f}, {state.max():.4f}]")
            # 降级到随机动作 / Fallback to random action
            action = random.randint(0, self.n_actions - 1)
            log_prob = torch.tensor(0.0, dtype=torch.float32).to(self.device)
            return action, log_prob

    
    def store_transition(self, transition: Transition):
        """
        存储转移到经验缓冲区 / Store transition to experience buffer
        
        Args:
            transition: (state, action, reward, next_state, done)
        """
        self.buffer.append(transition)

    def ppo_learn(self, track_loss: bool = False) -> Dict[str, float]:
        """
        执行 PPO 学习更新（On-Policy）/ Execute PPO learning update (On-Policy)
        
        从 buffer 中提取经验数据进行学习，
        Extract experience data from buffer for learning,
        
        学习完成后清空 buffer。
        clear buffer after learning is complete.
        
        Args:
            track_loss: 是否跟踪损失值 / Whether to track loss values
        
        Returns:
            包含平均损失的字典（如果 track_loss=True）/ Dictionary containing average losses (if track_loss=True)
        """
        # 检查 buffer 是否有数据 / Check if buffer has data
        if len(self.buffer) == 0:
            print(f"    [WARN] Buffer is empty, skipping learning")
            return {}

        # 检查并修复 buffer 中的无效奖励
        # Check and fix invalid rewards in buffer
        for i, trans in enumerate(self.buffer):
            if np.isnan(trans.reward) or np.isinf(trans.reward):
                print(f"⚠️  Warning: Buffer transition {i} has invalid reward: {trans.reward}, replacing with 0")
                self.buffer[i] = Transition(
                    state=trans.state,
                    action=trans.action,
                    reward=0.0,
                    next_state=trans.next_state,
                    done=trans.done
                )

        # 转换为 batch / Convert to batch
        batch = Transition(*zip(*self.buffer))

        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.LongTensor(batch.action).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(batch.reward).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(batch.done).to(self.device).unsqueeze(1)

        # ===== 增加输入有效性检查 ===== / ===== Add input validity check =====
        def has_invalid(tensor, name):
            """检查张量是否包含无效值 / Check if tensor contains invalid values"""
            if torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)):
                print(f"⚠️  Warning: {name} contains NaN/Inf, skipping learning")
                return True
            return False

        if (has_invalid(states, "States") or
            has_invalid(actions, "Actions") or
            has_invalid(rewards, "Rewards") or
            has_invalid(next_states, "Next_states") or
            has_invalid(dones, "Dones")):
            self.buffer = []
            return {}

        # 计算 TD 目标和优势函数 / Calculate TD target and advantage function
        with torch.no_grad():
            next_values = self.critic(next_states)  # 下一状态的价值 / Value of next state
            # 检查 next_values 是否有效 / Check if next_values is valid
            if has_invalid(next_values, "Next_values"):
                self.buffer = []
                return {}
            td_target = rewards + self.gamma * next_values * (1 - dones)  # TD目标 / TD target

        current_values = self.critic(states)  # 当前状态的价值 / Value of current state
        advantages = td_target - current_values  # 优势函数 / Advantage function

        # 检查 advantages 是否有效 / Check if advantages is valid
        if has_invalid(advantages, "Advantages"):
            # 如果无效，将 advantages 置零 / If invalid, set advantages to zero
            print(f"⚠️  Warning: Advantages contain NaN/Inf, setting to zeros")
            advantages = torch.zeros_like(advantages)

        # GAE (Generalized Advantage Estimation) - 广义优势估计
        # GAE (Generalized Advantage Estimation)
        advantages_np = advantages.cpu().detach().numpy()
        gae_advantages = self._compute_gae(advantages_np)
        advantages = torch.FloatTensor(gae_advantages).to(self.device)

        if has_invalid(advantages, "GAE_Advantages"):
            print(f"⚠️  Warning: GAE Advantages contain NaN/Inf, setting to zeros")
            advantages = torch.zeros_like(advantages)

        # 标准化优势（提高稳定性），但需要避免除零或规模过小
        # Normalize advantages (improve stability), but avoid division by zero or too small scale
        # 只有当 batch 数量 > 1 且标准差大于阈值时才标准化
        # Only normalize when batch size > 1 and std is greater than threshold
        if advantages.numel() > 1:
            adv_std = advantages.std()
            if torch.isnan(adv_std) or torch.isinf(adv_std) or adv_std < 1e-8:
                print(f"⚠️  Warning: Advantage std is invalid or too small ({adv_std:.2e}), skipping normalization")
            else:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        # 旧的日志概率 / Old log probabilities
        with torch.no_grad():
            old_log_probs = torch.log(self.actor(states).gather(1, actions) + 1e-8)

        # PPO 多轮更新 / PPO multi-epoch update
        total_actor_loss = 0
        total_critic_loss = 0

        for epoch in range(self.epochs):
            # 新的动作概率 / New action probabilities
            log_probs = torch.log(self.actor(states).gather(1, actions) + 1e-8)
            # 重要性采样比率 / Importance sampling ratio
            ratio = torch.exp(log_probs - old_log_probs)
            # Clipped surrogate objective - 裁剪代理目标
            # Clipped surrogate objective
            surr1 = ratio * advantages  # 未裁剪的目标 / Unclipped objective
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages  # 裁剪后的目标 / Clipped objective
            # Actor loss - 取负号因为我们要最大化 / Take negative because we want to maximize
            actor_loss = -torch.mean(torch.min(surr1, surr2))
            # Critic loss - 均方误差 / Mean squared error
            critic_loss = torch.nn.functional.mse_loss(
                self.critic(states),
                td_target.detach()
            )

            # 检查损失是否有效 / Check if losses are valid
            if torch.isnan(actor_loss) or torch.isinf(actor_loss):
                print(f"⚠️  Warning: Actor loss is NaN/Inf at epoch {epoch}, skipping this epoch")
                continue
            if torch.isnan(critic_loss) or torch.isinf(critic_loss):
                print(f"⚠️  Warning: Critic loss is NaN/Inf at epoch {epoch}, skipping this epoch")
                continue

            # 更新 Actor / Update Actor
            self.actor_optimizer.zero_grad()  # 清零梯度 / Zero gradients
            actor_loss.backward()  # 反向传播 / Backward propagation
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)  # 梯度裁剪 / Gradient clipping

            # 检查梯度有效性 / Check gradient validity
            actor_grad_valid = True
            for param in self.actor.parameters():
                if param.grad is not None:
                    if torch.any(torch.isnan(param.grad)) or torch.any(torch.isinf(param.grad)):
                        actor_grad_valid = False
                        break
            if actor_grad_valid:
                self.actor_optimizer.step()  # 更新参数 / Update parameters
            else:
                print(f"⚠️  Warning: Actor gradient contains NaN/Inf, skipping update")
                self.actor_optimizer.zero_grad()

            # 更新 Critic / Update Critic
            self.critic_optimizer.zero_grad()  # 清零梯度 / Zero gradients
            critic_loss.backward()  # 反向传播 / Backward propagation
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)  # 梯度裁剪 / Gradient clipping

            critic_grad_valid = True
            for param in self.critic.parameters():
                if param.grad is not None:
                    if torch.any(torch.isnan(param.grad)) or torch.any(torch.isinf(param.grad)):
                        critic_grad_valid = False
                        break
            if critic_grad_valid:
                self.critic_optimizer.step()  # 更新参数 / Update parameters
            else:
                print(f"⚠️  Warning: Critic gradient contains NaN/Inf, skipping update")
                self.critic_optimizer.zero_grad()

            if track_loss:
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()

        # 清空 buffer / Clear buffer
        self.buffer = []

        if track_loss:
            avg_actor_loss = total_actor_loss / self.epochs
            avg_critic_loss = total_critic_loss / self.epochs

            self.training_history['actor_loss'].append(avg_actor_loss)
            self.training_history['critic_loss'].append(avg_critic_loss)

            return {
                'actor_loss': avg_actor_loss,
                'critic_loss': avg_critic_loss
            }

        return {}
        
    def _compute_gae(self, advantages: np.ndarray) -> np.ndarray:
        """
        计算广义优势估计 (GAE) / Compute Generalized Advantage Estimation (GAE)
        
        Args:
            advantages: TD errors - 时序差分误差 / Temporal difference errors
        
        Returns:
            GAE 优势值 / GAE advantage values
        """
        gae = 0
        gae_list = []
        
        # 从后往前计算 GAE / Calculate GAE from back to front
        for delta in advantages[::-1]:
            gae = self.gamma * self.lmbda * gae + delta
            gae_list.append(gae)
        
        gae_list.reverse()  # 反转列表 / Reverse list
        return np.array(gae_list)
    
    def decay_learning_rate(self):
        """
        学习率衰减（在每个 episode 后调用）/ Learning rate decay (called after each episode)
        
        使用指数衰减策略，但设置下限以避免学习率过低
        Uses exponential decay strategy, but sets lower bound to avoid too low learning rate
        """
        self.episode_count += 1
        
        # 计算新的学习率 / Calculate new learning rates
        new_lr_actor = self.lr_actor * (self.lr_decay_rate ** self.episode_count)
        new_lr_critic = self.lr_critic * (self.lr_decay_rate ** self.episode_count)
        
        # 确保不低于最小学习率 / Ensure not below minimum learning rate
        new_lr_actor = max(new_lr_actor, self.min_lr)
        new_lr_critic = max(new_lr_critic, self.min_lr)
        
        # 更新优化器的学习率 / Update optimizer learning rates
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = new_lr_actor
        
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = new_lr_critic
    
    def save_checkpoint(self, filepath: str):
        """
        保存检查点 / Save checkpoint
        
        Args:
            filepath: 保存路径 / Save path
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)  # 创建目录 / Create directory
        
        torch.save({
            'actor_state_dict': self.actor.state_dict(),  # Actor网络状态 / Actor network state
            'critic_state_dict': self.critic.state_dict(),  # Critic网络状态 / Critic network state
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),  # Actor优化器状态 / Actor optimizer state
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),  # Critic优化器状态 / Critic optimizer state
            'config': self.config,  # 配置 / Configuration
        }, filepath)
        
        print(f"✓ Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        加载检查点 / Load checkpoint
        
        Args:
            filepath: 检查点路径 / Checkpoint path
        """
        if not filepath or not isinstance(filepath, str):
            raise ValueError(f"Invalid filepath: {filepath}")
        
        import os
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        # 加载检查点 / Load checkpoint
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # 恢复网络和优化器状态 / Restore network and optimizer states
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        print(f"✓ Loaded checkpoint from {filepath}")
    
    def clear_memory(self):
        """清空临时记忆 / Clear temporary memory"""
        self.state_memory.clear()
        self.action_memory.clear()
    
    def reset_training_history(self):
        """重置训练历史 / Reset training history"""
        self.training_history = {
            'actor_loss': [],
            'critic_loss': [],
            'rewards': []
        }
