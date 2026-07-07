"""
奖励计算器
提供多种奖励函数用于强化学习训练
"""
import numpy as np
from typing import List, Dict, Optional
from environment.rtree_env import RTreeEnvironment

class RewardCalculator:
    """
    奖励计算器
    
    提供多种奖励函数设计，用于评估 RTree 插入策略的质量。
    
    支持的奖励类型：
    1. 查询访问率差异（主要）
    2. 树结构质量指标
    3. 组合奖励
    """

    def __init__(
        self,
        query_area_ratio: float = 0.05,
        num_samples: int = 10,
        reward_type: str = "access_rate_diff"
    ):
        """
        初始化奖励计算器
        
        Args:
            query_area_ratio: 查询区域占总面积的比例
            num_samples: 每次计算奖励时的采样查询数
            reward_type: 奖励类型
                - "access_rate_diff": 访问率差异
                - "node_access": 节点访问数
                - "combined": 组合奖励
        """
        self.query_area_ratio = query_area_ratio
        self.num_samples = num_samples
        self.reward_type = reward_type

    def calculate_reward(
        self,
        agent_tree: RTreeEnvironment,
        reference_tree: RTreeEnvironment,
        recent_rectangles: List[List[float]],
        x_range: tuple = (0, 100000),
        y_range: tuple = (0, 100000)
    ) -> float:
        """
        计算奖励
        
        Args:
            agent_tree: Agent 构建的树
            reference_tree: 参考树
            recent_rectangles: 最近插入的矩形列表（用于生成查询）
            x_range: x 坐标范围
            y_range: y 坐标范围
        
        Returns:
            奖励值（正值表示 Agent 优于参考树）
        """
        if self.reward_type == "access_rate_diff":
            return self._reward_access_rate_diff(
                agent_tree, reference_tree, recent_rectangles, x_range, y_range
            )
        elif self.reward_type == "node_access":
            return self._reward_node_access(
                agent_tree, reference_tree, recent_rectangles, x_range, y_range
            )
        elif self.reward_type == "combined":
            return self._reward_combined(
                agent_tree, reference_tree, recent_rectangles, x_range, y_range
            )
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")


    def _reward_access_rate_diff(
        self,
        agent_tree: RTreeEnvironment,
        reference_tree: RTreeEnvironment,
        recent_rectangles: List[List[float]],
        x_range: tuple,
        y_range: tuple
    ) -> float:
        """
        基于访问率差异的奖励
        
        Reward = avg(ref_access_rate - agent_access_rate)
        
        正值表示 Agent 的树查询效率更高
        """
        total_reward = 0.0
        num_queries = min(self.num_samples, len(recent_rectangles))
        
        if num_queries == 0:
            return 0.0
        
        for i in range(num_queries):
            # 基于最近的矩形生成查询
            rect = recent_rectangles[-(i + 1)]
            query_rect = self._generate_query_from_rectangle(
                rect, x_range, y_range
            )
            
            # 计算访问率
            agent_rate = agent_tree.access_rate(query_rect)
            ref_rate = reference_tree.access_rate(query_rect)
            # 打印两个的rate
            # print(f"  Query Access Rate - Agent: {agent_rate:.4f}, Ref: {ref_rate:.4f}")
            
            # 奖励：参考树访问率 - Agent 访问率
            # 如果 Agent 更好（访问率更低），奖励为正
            total_reward += (ref_rate - agent_rate)
        
        return total_reward / num_queries

    # def _reward_node_access(
    #     self,
    #     agent_tree: RTreeEnvironment,
    #     reference_tree: RTreeEnvironment,
    #     recent_rectangles: List[List[float]],
    #     x_range: tuple,
    #     y_range: tuple
    # ) -> float:
    #     """
    #     基于节点访问数的奖励
        
    #     Reward = avg(ref_node_access - agent_node_access)
    #     """
    #     total_reward = 0.0
    #     num_queries = min(self.num_samples, len(recent_rectangles))
        
    #     if num_queries == 0:
    #         return 0.0
        
    #     for i in range(num_queries):
    #         rect = recent_rectangles[-(i + 1)]
    #         query_rect = self._generate_query_from_rectangle(
    #             rect, x_range, y_range
    #         )
            
    #         # 直接查询节点访问数
    #         agent_access = agent_tree.query(query_rect)
    #         ref_access = reference_tree.query(query_rect)
            
    #         total_reward += (ref_access - agent_access)
        
    #     return total_reward / num_queries


    def _reward_node_access(
        self,
        agent_tree: RTreeEnvironment,
        reference_tree: RTreeEnvironment,
        recent_rectangles: List[List[float]],
        x_range: tuple,
        y_range: tuple
    ) -> float:
        """
        基于节点访问数的奖励
        
        Reward = avg(ref_node_access - agent_node_access)
        """
        total_reward = 0.0
        num_queries = min(self.num_samples, len(recent_rectangles))
        
        if num_queries == 0:
            return 0.0
        
        for i in range(num_queries):
            rect = recent_rectangles[-(i + 1)]
            query_rect = self._generate_query_from_rectangle(
                rect, x_range, y_range
            )
            
            agent_access = agent_tree.query(query_rect)
            ref_access = reference_tree.query(query_rect)
            # ✅ 调试打印
            print(f"DEBUG: agent_access={agent_access}, ref_access={ref_access}, diff={ref_access - agent_access}", end="\r")
            
            total_reward += (ref_access - agent_access)
        
        avg_reward = total_reward / num_queries
        return avg_reward

    def _reward_combined(
        self,
        agent_tree: RTreeEnvironment,
        reference_tree: RTreeEnvironment,
        recent_rectangles: List[List[float]],
        x_range: tuple,
        y_range: tuple
    ) -> float:
        """
        组合奖励 - 综合考虑多个指标
        
        Reward = w1 * access_rate_diff + w2 * tree_quality_diff
        """
        # 访问率差异（权重 0.7）
        access_reward = self._reward_access_rate_diff(
            agent_tree, reference_tree, recent_rectangles, x_range, y_range
        )
        
        # 树结构质量差异（权重 0.3）
        quality_reward = self._reward_tree_quality_diff(agent_tree, reference_tree)
        
        return 0.7 * access_reward + 0.3 * quality_reward

    def _reward_tree_quality_diff(
        self,
        agent_tree: RTreeEnvironment,
        reference_tree: RTreeEnvironment
    ) -> float:
        """
        基于树结构质量的奖励
        
        考虑因素：
        - 填充因子（越高越好）
        - 树高（越低越好）
        """
        agent_fill = agent_tree.get_fill_factor()
        ref_fill = reference_tree.get_fill_factor()
        
        agent_height = agent_tree.get_tree_height()
        ref_height = reference_tree.get_tree_height()
        
        # 填充因子差异（归一化到 [0, 1]）
        fill_diff = agent_fill - ref_fill
        
        # 树高差异（归一化）
        max_height = max(agent_height, ref_height, 1)
        height_diff = (ref_height - agent_height) / max_height
        
        # 组合
        return 0.6 * fill_diff + 0.4 * height_diff


    def _generate_query_from_rectangle(
        self,
        rect: List[float],
        x_range: tuple,
        y_range: tuple
    ) -> List[float]:
        """
        基于矩形生成查询区域
        
        Args:
            rect: 矩形 [ll_x, ll_y, tr_x, tr_y]
            x_range: x 范围
            y_range: y 范围
        
        Returns:
            查询矩形
        """
        # 计算矩形中心
        center_x = (rect[0] + rect[2]) / 2
        center_y = (rect[1] + rect[3]) / 2
        
        # 计算查询区域大小
        total_area = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0])
        query_area = self.query_area_ratio / 100 * total_area
        
        # 随机长宽比
        y_x_ratio = np.random.uniform(0.1, 1.0)
        y_length = np.sqrt(query_area * y_x_ratio)
        x_length = query_area / y_length
        
        # 生成查询矩形
        ll_x = center_x - x_length / 2
        ll_y = center_y - y_length / 2
        tr_x = center_x + x_length / 2
        tr_y = center_y + y_length / 2
        
        # 确保在范围内
        ll_x = max(ll_x, x_range[0])
        ll_y = max(ll_y, y_range[0])
        tr_x = min(tr_x, x_range[1])
        tr_y = min(tr_y, y_range[1])
        
        return [ll_x, ll_y, tr_x, tr_y]


    def generate_random_query(
        self,
        x_range: tuple = (0, 100000),
        y_range: tuple = (0, 100000)
    ) -> List[float]:
        """
        生成完全随机的查询
        
        Args:
            x_range: x 范围
            y_range: y 范围
        
        Returns:
            查询矩形
        """
        total_area = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0])
        query_area = self.query_area_ratio * total_area
        
        side = np.sqrt(query_area) / 2
        
        center_x = np.random.uniform(x_range[0] + side, x_range[1] - side)
        center_y = np.random.uniform(y_range[0] + side, y_range[1] - side)
        
        return [
            center_x - side,
            center_y - side,
            center_x + side,
            center_y + side
        ]

class ZeroSumRewardCalculator(RewardCalculator):
    """
    零和奖励计算器 - 专用于 Self-Play
    
    确保两个对手的奖励之和为零
    """
    
    def calculate_zero_sum_rewards(
        self,
        player1_tree: RTreeEnvironment,
        player2_tree: RTreeEnvironment,
        recent_rectangles: List[List[float]],
        x_range: tuple = (0, 100000),
        y_range: tuple = (0, 100000)
    ) -> tuple:
        """
        计算零和奖励
        
        Args:
            player1_tree: 玩家1的树
            player2_tree: 玩家2的树
            recent_rectangles: 最近的矩形
            x_range: x 范围
            y_range: y 范围
        
        Returns:
            (player1_reward, player2_reward)，满足 r1 + r2 = 0
        """
        # 计算 player1 相对于 player2 的优势
        relative_advantage = self.calculate_reward(
            player1_tree,
            player2_tree,
            recent_rectangles,
            x_range,
            y_range
        )
        
        # 零和：player1 的奖励 = 优势，player2 的奖励 = -优势
        player1_reward = relative_advantage
        player2_reward = -relative_advantage
        
        return player1_reward, player2_reward

