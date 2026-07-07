"""
默认配置模块 - 集中管理所有超参数和配置项
Default configuration module - Centralized management of all hyperparameters and configuration items
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class DataConfig:
    """数据配置 / Data configuration"""
    distribution: str = "NORMAL"  # 数据分布类型 / Data distribution type
    train_volume: str = "W1"  # 训练数据规模 / Training data volume
    test_volume: str = "HW1"  # 测试数据规模 / Testing data volume
    file_format: str = ".npy"  # 文件格式 / File format
    data_edge_size: float = 1.0  # 数据边长大小 / Data edge size
    
    # 数据范围 / Data range
    x_min: float = 0.0  # X轴最小值 / Minimum X coordinate
    x_max: float = 100000.0  # X轴最大值 / Maximum X coordinate
    y_min: float = 0.0  # Y轴最小值 / Minimum Y coordinate
    y_max: float = 100000.0  # Y轴最大值 / Maximum Y coordinate
    
    # 数据文件路径 / Data file path
    data_dir: str = "generated_data"  # 数据目录 / Data directory
    
    @property
    def train_filename(self) -> str:
        """获取训练文件名 / Get training filename"""
        return f"{self.data_dir}/{self.train_volume}_{self.distribution}.txt"
    
    @property
    def test_filename(self) -> str:
        """获取测试文件名 / Get testing filename"""
        return f"{self.data_dir}/testing_{self.test_volume}_{self.distribution}{self.file_format}"


@dataclass
class TreeConfig:
    """RTree 配置 / RTree configuration
    
    C++ 端的策略枚举：
    enum INSERT_STRATEGY{
        INS_AREA, INS_MARGIN, INS_OVERLAP, INS_RSTAR
    };

    enum SPLIT_STRATEGY{
        SPL_MIN_AREA, SPL_MIN_MARGIN, SPL_MIN_OVERLAP, SPL_QUADRATIC
    };
    """
    max_entry: int = 50  # RTree节点最大条目数 / Maximum entries per RTree node
    min_entry_factor: float = 0.4  # 最小条目因子（min_entry = max_entry * factor）/ Minimum entry factor
    
    # 参考树类型: rtree, rstar, rrstar / Reference tree type
    reference_tree_type: str = "rstar"
    # reference_tree_type: str = "rtree"
    
    # 插入策略映射 / Insert strategy mapping
    INSERT_STRATEGIES: Dict[str, str] = field(default_factory=lambda: {
        "rtree": "INS_AREA",  # Guttman R-Tree 使用面积最小化 / Guttman R-Tree uses minimum area
        "rstar": "INS_RSTAR",  # R*-Tree 使用R*策略 / R*-Tree uses R* strategy
        "rrstar": "INS_RSTAR"  # RR*-Tree 使用R*策略 / RR*-Tree uses R* strategy
    })
    
    # 分裂策略映射 / Split strategy mapping
    SPLIT_STRATEGIES: Dict[str, str] = field(default_factory=lambda: {
        "rtree": "SPL_MIN_AREA",  # Guttman R-Tree 使用面积最小化 / Guttman R-Tree uses minimum area
        "rstar": "SPL_MIN_OVERLAP",  # R*-Tree 使用重叠最小化 / R*-Tree uses minimum overlap
        "rrstar": "SPL_QUADRATIC"  # RR*-Tree 使用二次分裂 / RR*-Tree uses quadratic split
    })
    
    def get_insert_strategy(self, tree_type: str = None) -> str:
        """获取插入策略 / Get insert strategy"""
        ttype = tree_type or self.reference_tree_type
        return self.INSERT_STRATEGIES.get(ttype, "INS_AREA")
    
    def get_split_strategy(self, tree_type: str = None) -> str:
        """获取分裂策略 / Get split strategy"""
        ttype = tree_type or self.reference_tree_type
        return self.SPLIT_STRATEGIES.get(ttype, "SPL_QUADRATIC")

@dataclass
class ModelConfig:
    """神经网络模型配置 / Neural network model configuration
    特征类型说明：12、14、15、124、125、145、1245 代表不同的状态设计组合
    Feature types: 12, 14, 15, 124, 125, 145, 1245 represent different state design combinations
    """
    # 特征类型: 0(随机基线), 12, 14, 15, 124, 125, 145, 1245
    # Feature type: 0(random baseline), 12, 14, 15, 124, 125, 145, 1245
    feature_type: int = 125
    
    # 是否启用特征消融实验模式（影响模型文件命名等）
    # Whether to enable feature ablation experiment mode (affects model file naming, etc.)
    enable_feature_ablation: bool = False
    
    # 根据 feature_type 自动计算特征数量
    # Automatically calculate number of features based on feature_type
    FEATURES_MAP: Dict[int, int] = field(default_factory=lambda: {
        0: 4,   # 随机状态基线：4个特征（实际只用1个随机值，但保持维度一致）/ Random state baseline: 4 features (actually uses 1 random value, but keeps dimension consistent)
        12: 2,  # 特征组合12 / Feature combination 12
        # 14: 3, 
        25: 3,  # 特征组合25 / Feature combination 25
        15: 3,  # 特征组合15 / Feature combination 15
        # 124: 4, 
        125: 4,  # 特征组合125（默认）/ Feature combination 125 (default)
        # 145: 5,
        # 1245: 6
    })
    
    action_space_size: int = 10  # 动作空间大小（每个节点的子节点候选数）/ Action space size (number of child node candidates per node)
    n_hidden: int = 64  # 隐藏层神经元数量 / Number of hidden layer neurons
    
    # 学习率 / Learning rates
    lr_actor: float = 1e-4  # Actor网络学习率 / Actor network learning rate
    lr_critic: float = 1e-4  # Critic网络学习率 / Critic network learning rate
    # 在配置中修改 / Modify in configuration
    # lr_actor = 5e-5   # 从 1e-4 降低 / Reduced from 1e-4
    # lr_critic = 5e-5  # 从 1e-4 降低 / Reduced from 1e-4
    
    @property
    def num_features(self) -> int:
        """获取特征数量 / Get number of features"""
        return self.FEATURES_MAP.get(self.feature_type, 4)
    
    @property
    def state_space_size(self) -> int:
        """状态空间大小 = 动作空间 × 特征数量 / State space size = action_space × num_features"""
        return self.action_space_size * self.num_features

@dataclass
class TrainingConfig:
    """训练配置 / Training configuration"""
    num_episodes: int = 20  # 训练回合数 / Number of training episodes
    
    # PPO 超参数 / PPO hyperparameters
    gamma: float = 0.98  # 折扣因子 / Discount factor
    lmbda: float = 0.98  # GAE lambda参数 / GAE lambda parameter
    epochs: int = 10  # PPO更新轮数 / PPO update epochs
    # epochs: int = 5
    eps: float = 0.2  # PPO裁剪参数 / PPO clipping parameter
    buffer_size: int = 20  # 经验缓冲区大小 / Experience buffer size
    epsilon: float = 0.1  # 探索率（epsilon-greedy）/ Exploration rate (epsilon-greedy)
    
    # 奖励计算 / Reward calculation
    query_reward_freq: int = 10  # 查询奖励频率 / Query reward frequency
    training_query_area_ratio: float = 0.05  # 训练查询面积比例 / Training query area ratio
    
    # 模型保存 / Model saving
    load_best_model_freq: int = 5  # 加载最佳模型频率 / Load best model frequency
    model_save_dir: str = "model/"  # 模型保存目录 / Model save directory


@dataclass
class SelfPlayConfig:
    """Self-Play 配置 / Self-Play configuration"""
    num_episodes: int = 20  # Self-Play回合数 / Number of Self-Play episodes
    
    # 是否使用交叉验证（方案B）/ Whether to use cross-validation (Option B)
    use_cross_validation: bool = True
    
    # 对手池配置（可选，用于增强版）/ Opponent pool configuration (optional, for enhanced version)
    use_policy_pool: bool = False  # 是否使用策略池 / Whether to use policy pool
    pool_size: int = 10  # 策略池大小 / Policy pool size
    pool_update_freq: int = 5  # 策略池更新频率 / Policy pool update frequency
    
    # 收敛阈值 / Convergence threshold
    convergence_threshold: float = 0.01  # 性能提升小于此值时认为收敛 / Consider converged when performance improvement is below this value


@dataclass
class QueryConfig:
    """查询配置 / Query configuration"""
    # 范围查询比例列表 / Range query ratios list
    range_ratios: List[float] = field(default_factory=lambda: [
        2.0, 1.0, 0.5, 0.05, 0.01, 0.005
    ])
    
    # KNN 查询的 K 值列表 / K values for KNN queries
    knn_values: List[int] = field(default_factory=lambda: [
        1, 10, 50, 100, 250, 750
    ])
    
    # 查询次数 / Number of queries
    num_queries: int = 1000

@dataclass
class PathConfig:
    """路径配置 / Path configuration"""
    data_dir: str = "generated_data"  # 数据目录 / Data directory
    model_dir: str = "model/"  # 模型目录 / Model directory
    log_dir: str = "logs"  # 日志目录 / Log directory
    checkpoint_dir: str = "checkpoints"  # 检查点目录 / Checkpoint directory
    image_dir: str = "images"  # 图片目录 / Image directory


@dataclass
class GSARConfig:
    """
    完整的 GSAR-Tree 配置
    Complete GSAR-Tree configuration
    
    聚合所有子配置，提供统一的配置接口
    Aggregates all sub-configurations, provides unified configuration interface
    """
    data: DataConfig = field(default_factory=DataConfig)  # 数据配置 / Data configuration
    tree: TreeConfig = field(default_factory=TreeConfig)  # 树配置 / Tree configuration
    model: ModelConfig = field(default_factory=ModelConfig)  # 模型配置 / Model configuration
    training: TrainingConfig = field(default_factory=TrainingConfig)  # 训练配置 / Training configuration
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)  # Self-Play配置 / Self-Play configuration
    query: QueryConfig = field(default_factory=QueryConfig)  # 查询配置 / Query configuration
    path: PathConfig = field(default_factory=PathConfig)  # 路径配置 / Path configuration
    
    # 设备配置 / Device configuration
    device: str = "cuda"  # cuda 或 cpu / cuda or cpu
    seed: int = 1  # 随机种子 / Random seed
    
    def copy(self) -> 'GSARConfig':
        """创建配置的深拷贝 / Create a deep copy of the configuration"""
        import copy
        return copy.deepcopy(self)
    
    def get_model_filepath(self, operation: str = "train", episode: int = None) -> str:
        """
        生成模型文件路径 / Generate model file path
        
        Args:
            operation: 操作类型 (train, selfplay, test) / Operation type
            episode: Episode 编号（可选）。如果提供，将替换 {episode} 占位符；
                    如果为 None，保留 {episode} 占位符供调用者替换。
                    Episode number (optional). If provided, replaces {episode} placeholder;
                    if None, keeps {episode} placeholder for caller to replace.
        
        Returns:
            完整的模型文件路径 / Complete model file path
        """
        # 根据操作类型确定前缀 / Determine prefix based on operation type
        if operation == "selfplay":
            prefix = "selfplay"
        else:
            prefix = "train"
        
        # ⭐ 关键修复：如果提供了 episode 参数，直接替换占位符
        # ⭐ Key fix: If episode parameter is provided, directly replace placeholder
        if episode is not None:
            episode_str = str(episode)
        else:
            # 保留占位符，供调用者后续替换 / Keep placeholder for caller to replace later
            episode_str = "{episode}"
        
        # ⭐ 根据 enable_feature_ablation flag 决定是否包含特征类型
        # ⭐ Decide whether to include feature type based on enable_feature_ablation flag
        if self.model.enable_feature_ablation:
            feature_suffix = f"_F{self.model.feature_type}"
        else:
            feature_suffix = ""
        
        # 构建文件名 / Build filename
        filename = (
            f"{prefix}_"
            f"{self.data.train_volume}_"
            f"{self.data.distribution}_"
            f"{self.tree.reference_tree_type}_"
            f"{self.tree.max_entry}-"
            f"{self.model.action_space_size}"
            f"{feature_suffix}_"
            f"BestEfficiency_Ep{episode_str}.pth"
        )
        
        # ⭐ 构建子目录路径：model/{train_volume}_{distribution}_{baseline}_{max_entry}-{action_space}/
        # ⭐ Build subdirectory path
        subdir_name = (
            f"{self.data.train_volume}_"
            f"{self.data.distribution}_"
            f"{self.tree.reference_tree_type}_"
            f"{self.tree.max_entry}-"
            f"{self.model.action_space_size}"
        )
        
        # 确保使用正斜杠并创建完整路径 / Ensure using forward slashes and create complete path
        model_path = f"{self.path.model_dir.rstrip('/')}/{subdir_name}/{filename}"
        
        return model_path
    
    def get_selfplay_model_pattern(self) -> str:
        """
        生成 Self-Play 模型文件的搜索模式 / Generate Self-Play model file search pattern
        
        Returns:
            文件名模式（不含路径）/ Filename pattern (without path)
        """
        # ⭐ 根据 enable_feature_ablation flag 决定是否包含特征类型
        # ⭐ Decide whether to include feature type based on enable_feature_ablation flag
        if self.model.enable_feature_ablation:
            feature_suffix = f"_F{self.model.feature_type}"
        else:
            feature_suffix = ""
        
        return (
            f"selfplay_"
            f"{self.data.train_volume}_"
            f"{self.data.distribution}_"
            f"{self.tree.reference_tree_type}_"
            f"{self.tree.max_entry}-"
            f"{self.model.action_space_size}"
            f"{feature_suffix}_"
            f"BestEfficiency_Ep*.pth"
        )
    
    def get_train_model_pattern(self) -> str:
        """
        生成单一训练模型文件的搜索模式 / Generate single training model file search pattern
        
        Returns:
            文件名模式（不含路径）/ Filename pattern (without path)
        """
        # ⭐ 根据 enable_feature_ablation flag 决定是否包含特征类型
        # ⭐ Decide whether to include feature type based on enable_feature_ablation flag
        if self.model.enable_feature_ablation:
            feature_suffix = f"_F{self.model.feature_type}"
        else:
            feature_suffix = ""
        
        return (
            f"train_"
            f"{self.data.train_volume}_"
            f"{self.data.distribution}_"
            f"{self.tree.reference_tree_type}_"
            f"{self.tree.max_entry}-"
            f"{self.model.action_space_size}"
            f"{feature_suffix}_"
            f"BestEfficiency_Ep*.pth"
        )

# 便捷函数：创建常用配置 / Convenience functions: Create common configurations
def create_default_config() -> GSARConfig:
    """创建默认配置 / Create default configuration"""
    return GSARConfig()


def create_training_config(
    distribution: str = "NORMAL",
    train_volume: str = "W1",
    max_entry: int = 50,
    min_entry_factor: float = 0.4,
    feature_type: int = 125,
    action_space_size: int = 20,
    device: str = "cuda",
    enable_feature_ablation: bool = False
) -> GSARConfig:
    """
    创建训练配置 / Create training configuration
    
    Args:
        distribution: 数据分布类型 / Data distribution type
        train_volume: 训练数据规模 / Training data volume
        max_entry: RTree 最大条目数 / RTree maximum entries
        min_entry_factor: 最小条目因子 / Minimum entry factor
        feature_type: 特征类型 / Feature type
        action_space_size: 动作空间大小 / Action space size
        device: 计算设备 / Computing device
        enable_feature_ablation: 是否启用特征消融实验模式（影响模型文件命名）
                                Whether to enable feature ablation experiment mode (affects model file naming)
    
    Returns:
        配置对象 / Configuration object
    """
    config = GSARConfig()
    config.data.distribution = distribution
    config.data.train_volume = train_volume
    config.tree.max_entry = max_entry
    config.tree.min_entry_factor = min_entry_factor
    config.model.feature_type = feature_type
    config.model.action_space_size = action_space_size
    config.model.enable_feature_ablation = enable_feature_ablation
    config.device = device
    return config


def create_self_play_config(
    distribution: str = "NORMAL",
    train_volume: str = "W1",
    max_entry: int = 50,
    min_entry_factor: float = 0.4,
    feature_type: int = 125,
    action_space_size: int = 20,
    device: str = "cuda",
    base_config: GSARConfig = None,
    enable_feature_ablation: bool = False
) -> GSARConfig:
    """
    创建 Self-Play 配置 / Create Self-Play configuration
    
    Args:
        distribution: 数据分布类型 / Data distribution type
        train_volume: 训练数据规模 / Training data volume
        max_entry: RTree 最大条目数 / RTree maximum entries
        min_entry_factor: 最小条目因子 / Minimum entry factor
        feature_type: 特征类型 / Feature type
        action_space_size: 动作空间大小 / Action space size
        device: 计算设备 / Computing device
        base_config: 基础配置（如果提供，将基于此配置修改）
                    Base configuration (if provided, will modify based on this config)
        enable_feature_ablation: 是否启用特征消融实验模式（影响模型文件命名）
                                Whether to enable feature ablation experiment mode (affects model file naming)
    
    Returns:
        Self-Play 配置对象 / Self-Play configuration object
    """
    if base_config is not None:
        config = base_config.copy()
    else:
        config = GSARConfig()
        config.data.distribution = distribution
        config.data.train_volume = train_volume
        config.tree.max_entry = max_entry
        config.tree.min_entry_factor = min_entry_factor
        config.model.feature_type = feature_type
        config.model.action_space_size = action_space_size
        config.model.enable_feature_ablation = enable_feature_ablation
        # config.model.lr_actor = 5e-5
        # config.model.lr_critic = 5e-5
        # config.model.epochs = 5
        config.device = device
        
        # lr_actor: float = 1e-4
        # lr_critic: float = 1e-4
        # # 在配置中修改 / Modify in configuration
        # # lr_actor = 5e-5   # 从 1e-4 降低 / Reduced from 1e-4
        # # lr_critic = 5e-5  # 从 1e-4 降低 / Reduced from 1e-4
        
    
    
    # Self-Play 特定设置 / Self-Play specific settings
    # 修改：Reference 树使用 R*-Tree 策略，与 Agent 树的启发式降级策略区分开
    # Modified: Reference tree uses R*-Tree strategy, distinguished from Agent tree's heuristic fallback strategy
    # 原始 gsar_tree.py 中使用的是 REFTREE_TYPE[2] = "rstar-tree"
    # Original gsar_tree.py uses REFTREE_TYPE[2] = "rstar-tree"
    config.tree.reference_tree_type = "rstar"
    return config
