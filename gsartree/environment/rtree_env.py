"""
RTree 环境封装
RTree Environment Wrapper

提供统一的 RTree 操作接口，支持 RL 决策插入
Provides unified RTree operation interface, supports RL decision-based insertion
"""
import sys
import os
from typing import List, Optional, Tuple

# 添加父目录到路径以导入 rtree 模块
# Add parent directory to path to import rtree module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtree import RTree
from config.default_config import GSARConfig, TreeConfig


class RTreeEnvironment:
    """
    RTree 环境类 / RTree Environment Class
    
    封装所有 RTree 操作，提供清晰的接口用于强化学习训练。
    Encapsulates all RTree operations, provides clear interface for reinforcement learning training.
    
    支持多种树类型（rtree, rstar, rrstar）和插入策略。
    Supports multiple tree types (rtree, rstar, rrstar) and insertion strategies.
    
    Attributes:
        config: 树配置对象 / Tree configuration object
        tree: RTree 实例 / RTree instance
        tree_type: 当前树的类型标识 / Current tree type identifier
    """
    
    def __init__(self, config: GSARConfig, tree_type: str = "acppo"):
        """
        初始化 RTree 环境 / Initialize RTree environment
        
        Args:
            config: 完整的 GSAR 配置 / Complete GSAR configuration
            tree_type: 树类型 ("acppo", "rtree", "rstar", "rrstar") / Tree type
        """
        self.config = config
        self.tree_type = tree_type
        self.tree = None
        
        # 初始化树 / Initialize tree
        self.reset()
    
    def reset(self):
        """
        重置环境 - 创建新的空树 / Reset environment - create new empty tree
        
        Returns:
            新创建的 RTree 实例 / Newly created RTree instance
        """
        tree_cfg = self.config.tree
        
        # 创建新的 RTree / Create new RTree
        self.tree = RTree(
            tree_cfg.max_entry,
            tree_cfg.min_entry_factor
        )
        
        # 配置插入和分裂策略 / Configure insertion and split strategies
        self._configure_tree()
        
        self.tree.Clear()  # 清空树 / Clear tree
        
        return self.tree
    
    def _configure_tree(self):
        """根据树类型配置插入和分裂策略 / Configure insertion and split strategies based on tree type"""
        if self.tree_type == "rtree":
            # Guttman R-Tree
            self.tree.SetDefaultInsertStrategy("INS_AREA")  # 面积最小化 / Minimum area
            self.tree.SetDefaultSplitStrategy("SPL_QUADRATIC")  # 二次分裂 / Quadratic split
            
        elif self.tree_type == "rstar":
            # R*-Tree
            self.tree.SetDefaultInsertStrategy("INS_RSTAR")  # R*插入策略 / R* insertion strategy
            self.tree.SetDefaultSplitStrategy("SPL_MIN_OVERLAP")  # 重叠最小化 / Minimum overlap
            
        elif self.tree_type == "rrstar":
            # RR*-Tree
            self.tree.SetDefaultInsertStrategy("INS_RSTAR")  # R*插入策略 / R* insertion strategy
            self.tree.SetDefaultSplitStrategy("SPL_QUADRATIC")  # 二次分裂 / Quadratic split
            
        else:  # acppo (默认) / acppo (default)
            # 用于 RL 训练的树 / Tree for RL training
            self.tree.SetDefaultInsertStrategy("INS_AREA")  # 面积最小化 / Minimum area
            self.tree.SetDefaultSplitStrategy("SPL_MIN_MARGIN")  # 边距最小化 / Minimum margin
    
    def insert_rectangle(
        self,
        rect: List[float],
        use_rl: bool = False,
        agent = None,
        feature_type: int = 125,
        action_space_size: int = 20,
        record_memory: bool = True,
        state_memory: List = None,
        action_memory: List = None
    ) -> Tuple[bool, str]:
        """
        插入矩形到树中 / Insert rectangle into tree
        
        对于非 RL 场景（如 Reference Tree），直接使用 C++ 底层 API 一键插入，性能最优。
        For non-RL scenarios (e.g., Reference Tree), directly use C++ low-level API for one-step insertion, optimal performance.
        
        对于 RL 场景，使用循环进行逐步决策，并可选地记录状态和动作。
        For RL scenarios, use loop for step-by-step decision making, optionally record states and actions.
        
        Args:
            rect: 矩形坐标 [ll_x, ll_y, tr_x, tr_y] / Rectangle coordinates
            use_rl: 是否使用 RL 决策插入位置 / Whether to use RL decision for insertion position
            agent: ACPPOAgent 实例（当 use_rl=True 时需要）/ ACPPOAgent instance (required when use_rl=True)
            feature_type: 特征类型 / Feature type
            action_space_size: 动作空间大小 / Action space size
            record_memory: 是否记录状态和动作（仅 RL 模式有效）
                          Whether to record states and actions (only valid in RL mode)
            state_memory: 自定义状态记忆列表（如果为 None 且 record_memory=True，则使用 agent.state_memory）
                         Custom state memory list (if None and record_memory=True, uses agent.state_memory)
            action_memory: 自定义动作记忆列表（如果为 None 且 record_memory=True，则使用 agent.action_memory）
                          Custom action memory list (if None and record_memory=True, uses agent.action_memory)
        
        Returns:
            (success, decision_type) 元组 / (success, decision_type) tuple
            - success: 插入成功标志 (True/False) / Insertion success flag
            - decision_type: "rl" 或 "heuristic"，表示主要使用的决策类型
                           "rl" or "heuristic", indicates main decision type used
        """
        ll_x, ll_y, tr_x, tr_y = rect
        
        if not use_rl:
            # 非 RL 场景：直接使用 C++ 底层 API，一键插入（与原始参考代码一致）
            # Non-RL scenario: directly use C++ low-level API, one-step insertion (consistent with original reference code)
            # 这种方式性能最优，避免 Python 层循环开销
            # This approach has optimal performance, avoids Python layer loop overhead
            if self.tree_type == "rtree":
                self.tree.DefaultInsert(ll_x, ll_y, tr_x, tr_y)
            elif self.tree_type == "rstar":
                self.tree.DirectInsert(ll_x, ll_y, tr_x, tr_y)
                self.tree.DirectSplitWithReinsert()
            elif self.tree_type == "rrstar":
                self.tree.DirectRRInsert(ll_x, ll_y, tr_x, tr_y)
                self.tree.DirectRRSplit()
            else:  # acppo
                # 默认使用 R*-tree 策略 / Default to R*-tree strategy
                self.tree.DirectInsert(ll_x, ll_y, tr_x, tr_y)
                self.tree.DirectSplitWithReinsert()
            
            return True, "heuristic"
        
        else:
            # RL 场景：使用循环进行逐步决策
            # RL scenario: use loop for step-by-step decision making
            # 准备矩形 / Prepare rectangle
            self.tree.PrepareRectangle(ll_x, ll_y, tr_x, tr_y)
            
            decision_type = "heuristic"  # 默认为启发式 / Default to heuristic
            
            # 从根节点下降到叶子节点，每一步都是一个独立的动作
            # Descend from root node to leaf node, each step is an independent action
            while not self.tree.IsLeaf(self.tree.node_ptr):
                min_area_child = self.tree.GetMinAreaContainingChild()
                
                if min_area_child is None:
                    # 需要 RL 决策 / Need RL decision
                    num_features = self.config.model.FEATURES_MAP.get(feature_type, 4)
                    states = self.tree.RetrieveEvaluatedInsertStatesByType(
                        action_space_size,
                        num_features,
                        feature_type
                    )
                    
                    if states is not None and agent is not None:
                        # 智能体选择动作 / Agent chooses action
                        action, _ = agent.choose_action(states, explore=False)
                        
                        # 关键：记录状态和动作到 memory
                        # Key: record state and action to memory
                        if record_memory:
                            # 如果提供了自定义 memory 列表，使用它们；否则使用 agent 的默认 memory
                            # If custom memory lists are provided, use them; otherwise use agent's default memory
                            target_state_mem = state_memory if state_memory is not None else agent.state_memory
                            target_action_mem = action_memory if action_memory is not None else agent.action_memory
                            
                            target_state_mem.append(states)
                            target_action_mem.append(action)
                        
                        self.tree.InsertWithEvaluatedLoc(action)
                        decision_type = "rl"
                    else:
                        # 降级到启发式 / Fallback to heuristic
                        print("states is None")
                        self.tree.InsertWithLoc(0)
                else:
                    # 启发式决策（最小面积包含子节点）
                    # Heuristic decision (minimum area containing child)
                    self.tree.InsertWithLoc(min_area_child)
            
            # 插入到叶子节点 / Insert into leaf node
            self.tree.InsertWithLoc(0)
            
            # 执行分裂 / Execute split
            self._split_node()
            
            return True, decision_type

    def _split_node(self):
        """根据树类型执行相应的分裂操作 / Execute corresponding split operation based on tree type"""
        if self.tree_type == "rtree":
            self.tree.DefaultSplit()
        elif self.tree_type == "rstar":
            self.tree.DirectSplitWithReinsert()
        elif self.tree_type == "rrstar":
            self.tree.DirectRRSplit()
        else:  # acppo
            self.tree.DirectSplitWithReinsert()
    
    def query(self, query_rect: List[float]) -> int:
        """
        执行范围查询 / Execute range query
        
        Args:
            query_rect: 查询矩形 [ll_x, ll_y, tr_x, tr_y] / Query rectangle
        
        Returns:
            访问的节点数 / Number of accessed nodes
        """
        return self.tree.Query(tuple(query_rect))
    
    def knn_query(self, x: float, y: float, k: int) -> int:
        """
        执行 KNN 查询 / Execute KNN query
        
        Args:
            x: 查询点 x 坐标 / Query point x coordinate
            y: 查询点 y 坐标 / Query point y coordinate
            k: 最近邻数量 / Number of nearest neighbors
        
        Returns:
            访问的节点数 / Number of accessed nodes
        """
        return self.tree.KNNQuery(x, y, k)
    
    def access_rate(self, query_rect: List[float]) -> float:
        """
        计算查询访问率 / Calculate query access rate
        
        Args:
            query_rect: 查询矩形 / Query rectangle
        
        Returns:
            访问率（访问节点数 / 树高）/ Access rate (accessed nodes / tree height)
        """
        return self.tree.AccessRate(query_rect)
    
    def copy_from(self, other_env: 'RTreeEnvironment'):
        """
        从另一个环境复制树结构 / Copy tree structure from another environment
        
        Args:
            other_env: 源环境 / Source environment
        """
        # 使用 GetTreePtr() 方法获取树的指针
        # Use GetTreePtr() method to get tree pointer
        self.tree.CopyTree(other_env.tree.GetTreePtr())
    
    def clear(self):
        """清空树 / Clear tree"""
        self.tree.Clear()
    
    def get_tree_height(self) -> int:
        """获取树高 / Get tree height"""
        return self.tree.GetTreeHeight()
    
    def get_num_nodes(self) -> int:
        """获取节点总数 / Get total number of nodes"""
        return self.tree.GetNumNodes()
    
    def get_num_objects(self) -> int:
        """获取对象总数 / Get total number of objects"""
        return self.tree.GetNumObjects()
    
    def get_fill_factor(self) -> float:
        """
        计算填充因子 / Calculate fill factor
        
        Returns:
            平均填充率 / Average fill rate
        """
        num_nodes = self.get_num_nodes()
        if num_nodes == 0:
            return 0.0
        
        total_entries = self.tree.GetTotalEntries()
        max_capacity = num_nodes * self.config.tree.max_entry
        
        return total_entries / max_capacity if max_capacity > 0 else 0.0
    
    def get_stats(self) -> dict:
        """
        获取树的统计信息 / Get tree statistics
        
        Returns:
            包含各种统计指标的字典 / Dictionary containing various statistical metrics
        """
        return {
            'height': self.get_tree_height(),  # 树高 / Tree height
            'num_nodes': self.get_num_nodes(),  # 节点数 / Number of nodes
            'num_objects': self.get_num_objects(),  # 对象数 / Number of objects
            'fill_factor': self.get_fill_factor(),  # 填充因子 / Fill factor
            'max_entry': self.config.tree.max_entry,  # 最大条目数 / Maximum entries
            'tree_type': self.tree_type  # 树类型 / Tree type
        }
    
    def print_stats(self):
        """打印树的统计信息 / Print tree statistics"""
        stats = self.get_stats()
        print(f"\n{'='*60}")
        print(f"RTree Statistics ({stats['tree_type']})")
        print(f"{'='*60}")
        print(f"  Height:          {stats['height']}")
        print(f"  Total Nodes:     {stats['num_nodes']}")
        print(f"  Total Objects:   {stats['num_objects']}")
        print(f"  Fill Factor:     {stats['fill_factor']:.2%}")
        print(f"  Max Entry:       {stats['max_entry']}")
        print(f"{'='*60}")


class MultiTreeEnvironment:
    """
    多树环境 - 用于 Self-Play 和对比实验
    Multi-Tree Environment - for Self-Play and comparative experiments
    
    管理多个 RTree 实例，支持并行构建和比较
    Manages multiple RTree instances, supports parallel construction and comparison
    """
    
    def __init__(self, config: GSARConfig):
        """
        初始化多树环境 / Initialize multi-tree environment
        
        Args:
            config: 配置对象 / Configuration object
        """
        self.config = config
        self.environments = {}  # 存储多个树环境的字典 / Dictionary storing multiple tree environments
    
    def add_tree(self, name: str, tree_type: str = "acppo") -> RTreeEnvironment:
        """
        添加一个新的树环境 / Add a new tree environment
        
        Args:
            name: 树的名称标识 / Tree name identifier
            tree_type: 树类型 / Tree type
        
        Returns:
            创建的 RTreeEnvironment 实例 / Created RTreeEnvironment instance
        """
        env = RTreeEnvironment(self.config, tree_type)
        self.environments[name] = env
        return env
    
    def get_tree(self, name: str) -> RTreeEnvironment:
        """获取指定名称的树环境 / Get tree environment with specified name"""
        if name not in self.environments:
            raise KeyError(f"Tree '{name}' not found")
        return self.environments[name]
    
    def reset_all(self):
        """重置所有树 / Reset all trees"""
        for env in self.environments.values():
            env.reset()
    
    def clear_all(self):
        """清空所有树 / Clear all trees"""
        for env in self.environments.values():
            env.clear()
    
    def insert_to_all(
        self,
        rect: List[float],
        rl_trees: List[str] = None,
        agent_dict: dict = None,
        feature_type: int = 125,
        action_space_size: int = 20,
        record_memory: bool = True,
        state_memory_dict: dict = None,
        action_memory_dict: dict = None
    ):
        """
        向所有树插入同一个矩形 / Insert same rectangle into all trees
        
        Args:
            rect: 矩形坐标 / Rectangle coordinates
            rl_trees: 使用 RL 决策的树名称列表 / List of tree names using RL decision
            agent_dict: 树名到 agent 的映射 {tree_name: agent} / Mapping from tree name to agent
            feature_type: 特征类型 / Feature type
            action_space_size: 动作空间大小 / Action space size
            record_memory: 是否记录状态和动作 / Whether to record states and actions
            state_memory_dict: 树名到状态记忆列表的映射 {tree_name: state_memory_list}
                              Mapping from tree name to state memory list
            action_memory_dict: 树名到动作记忆列表的映射 {tree_name: action_memory_list}
                               Mapping from tree name to action memory list
        """
        if rl_trees is None:
            rl_trees = []
        
        if agent_dict is None:
            agent_dict = {}
        
        if state_memory_dict is None:
            state_memory_dict = {}
        
        if action_memory_dict is None:
            action_memory_dict = {}
        
        for name, env in self.environments.items():
            use_rl = name in rl_trees  # 判断该树是否使用RL / Check if this tree uses RL
            agent = agent_dict.get(name, None)  # 获取对应的agent / Get corresponding agent
            
            # 传递完整的参数以支持新的 insert_rectangle 接口
            # Pass complete parameters to support new insert_rectangle interface
            env.insert_rectangle(
                rect,
                use_rl=use_rl,
                agent=agent,
                feature_type=feature_type,
                action_space_size=action_space_size,
                record_memory=record_memory,
                state_memory=state_memory_dict.get(name, None),
                action_memory=action_memory_dict.get(name, None)
            )
    
    def compare_performance(
        self,
        query_rect: List[float]
    ) -> dict:
        """
        比较所有树在给定查询上的性能 / Compare performance of all trees on given query
        
        Args:
            query_rect: 查询矩形 / Query rectangle
        
        Returns:
            各树的性能指标 / Performance metrics of each tree
        """
        results = {}
        
        for name, env in self.environments.items():
            accessed = env.query(query_rect)  # 执行查询 / Execute query
            results[name] = {
                'accessed_nodes': accessed,  # 访问节点数 / Accessed nodes
                'tree_height': env.get_tree_height(),  # 树高 / Tree height
                'num_nodes': env.get_num_nodes()  # 节点数 / Number of nodes
            }
        
        return results
    
    def get_all_stats(self) -> dict:
        """获取所有树的统计信息 / Get statistics of all trees"""
        return {
            name: env.get_stats()
            for name, env in self.environments.items()
        }
