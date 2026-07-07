# 数据生成器配置文件
# 用于快速配置要生成的数据集

# ============================================================================
# 数据规模选项 (DATASIZE_TYPE)
# ============================================================================
# H1: 100         - 微型测试
# K5: 5,000       - 小型
# W1: 10,000      - 中小型
# W2: 20,000      - 中小型
# W3: 30,000      - 中小型
# W4: 40,000      - 中型
# W5: 50,000      - 中型
# TW1: 100,000    - 中型
# TW2: 200,000    - 大型
# TW5: 500,000    - 大型
# HW1: 1,000,000  - 超大型
# HW5: 5,000,000  - 超大型
# KW: 10,000,000  - 百万级
# WW1: 100,000,000 - 亿级

# ============================================================================
# 数据分布选项 (DATADISTRIBUTION_TYPE)
# ============================================================================
# 基础分布:
#   UNIFORM     - 均匀分布（完全随机）
#   NORMAL      - 正态分布（集中在中心区域）
#   SKEW-NOR0/2/4/8 - 偏态分布（α控制偏斜程度）
# 
# 多峰分布:
#   BIMODAL     - 双峰分布（两个聚集中心）
#   MULTI-MODAL - 多峰分布（多个聚集中心）
#   CLUSTERED   - 聚类分布（随机聚类中心）
# 
# 复杂分布:
#   CORRELATED  - 正相关分布（X-Y 线性正相关）
#   ANTI-CORR   - 负相关分布（X-Y 线性负相关）
#   GAUSSIAN-MIX - 高斯混合模型（多成分混合）
# 
# 时空数据:
#   SPATIAL-TEMPORAL - 时空数据（轨迹、序列）
#   CITY-LIKE        - 城市网格状分布
#   HOTSPOTS         - 热点区域分布

# ============================================================================
# 推荐配置组合
# ============================================================================

# 快速测试配置（适合开发和调试）
QUICK_TEST_CONFIG = [
    {'data_size': 'H1', 'distribution_type': 'UNIFORM', 'testing_flag': True},
    {'data_size': 'H1', 'distribution_type': 'NORMAL', 'testing_flag': True},
]

# 基础分布测试配置
BASIC_DISTRIBUTION_CONFIG = [
    {'data_size': 'W1', 'distribution_type': 'UNIFORM', 'testing_flag': True},
    {'data_size': 'W1', 'distribution_type': 'NORMAL', 'testing_flag': True},
    {'data_size': 'W1', 'distribution_type': 'SKEW-NOR2', 'testing_flag': True},
    {'data_size': 'W1', 'distribution_type': 'SKEW-NOR4', 'testing_flag': True},
]

# 多峰分布测试配置（新增）
MULTI_MODAL_CONFIG = [
    {'data_size': 'W1', 'distribution_type': 'BIMODAL', 'testing_flag': True},
    {'data_size': 'W1', 'distribution_type': 'MULTI-MODAL', 'testing_flag': True},
    {'data_size': 'W1', 'distribution_type': 'CLUSTERED', 'testing_flag': True},
]

# 复杂分布测试配置（新增）
COMPLEX_DISTRIBUTION_CONFIG = [
    {'data_size': 'W1', 'distribution_type': 'CORRELATED', 'testing_flag': True},
    {'data_size': 'W1', 'distribution_type': 'ANTI-CORR', 'testing_flag': True},
    {'data_size': 'W1', 'distribution_type': 'GAUSSIAN-MIX', 'testing_flag': True},
]

# 时空数据测试配置（新增）
SPATIAL_TEMPORAL_CONFIG = [
    {'data_size': 'W1', 'distribution_type': 'SPATIAL-TEMPORAL', 'testing_flag': True},
    {'data_size': 'W1', 'distribution_type': 'CITY-LIKE', 'testing_flag': True},
    {'data_size': 'W1', 'distribution_type': 'HOTSPOTS', 'testing_flag': True},
]

# 标准测试配置（适合性能对比）- 包含所有类型
STANDARD_CONFIG = [
    # 基础分布
    {'data_size': 'W1', 'distribution_type': 'UNIFORM', 'testing_flag': True},
    {'data_size': 'W1', 'distribution_type': 'NORMAL', 'testing_flag': True},
    {'data_size': 'W1', 'distribution_type': 'SKEW-NOR2', 'testing_flag': True},
    
    # 多峰分布
    {'data_size': 'W1', 'distribution_type': 'BIMODAL', 'testing_flag': True},
    {'data_size': 'W1', 'distribution_type': 'MULTI-MODAL', 'testing_flag': True},
    {'data_size': 'W1', 'distribution_type': 'CLUSTERED', 'testing_flag': True},
    
    # 复杂分布
    {'data_size': 'W1', 'distribution_type': 'CORRELATED', 'testing_flag': True},
    {'data_size': 'W1', 'distribution_type': 'ANTI-CORR', 'testing_flag': True},
    {'data_size': 'W1', 'distribution_type': 'GAUSSIAN-MIX', 'testing_flag': True},
    
    # 时空数据
    {'data_size': 'W1', 'distribution_type': 'SPATIAL-TEMPORAL', 'testing_flag': True},
    {'data_size': 'W1', 'distribution_type': 'CITY-LIKE', 'testing_flag': True},
    {'data_size': 'W1', 'distribution_type': 'HOTSPOTS', 'testing_flag': True},
]

# 完整基准测试配置
BENCHMARK_CONFIG = [
    # 不同规模 - 均匀分布
    {'data_size': 'W1', 'distribution_type': 'UNIFORM', 'testing_flag': True},
    {'data_size': 'W2', 'distribution_type': 'UNIFORM', 'testing_flag': True},
    {'data_size': 'W3', 'distribution_type': 'UNIFORM', 'testing_flag': True},
    {'data_size': 'W4', 'distribution_type': 'UNIFORM', 'testing_flag': True},
    {'data_size': 'W5', 'distribution_type': 'UNIFORM', 'testing_flag': True},
    
    # 不同规模 - 正态分布
    {'data_size': 'W1', 'distribution_type': 'NORMAL', 'testing_flag': True},
    {'data_size': 'W2', 'distribution_type': 'NORMAL', 'testing_flag': True},
    {'data_size': 'W3', 'distribution_type': 'NORMAL', 'testing_flag': True},
    
    # 代表性复杂分布
    {'data_size': 'W1', 'distribution_type': 'BIMODAL', 'testing_flag': True},
    {'data_size': 'W1', 'distribution_type': 'CLUSTERED', 'testing_flag': True},
    {'data_size': 'W1', 'distribution_type': 'HOTSPOTS', 'testing_flag': True},
]

# 大规模测试配置（需要较长时间和大量内存）
LARGE_SCALE_CONFIG = [
    {'data_size': 'TW1', 'distribution_type': 'UNIFORM', 'testing_flag': False},
    {'data_size': 'TW2', 'distribution_type': 'UNIFORM', 'testing_flag': False},
    {'data_size': 'HW1', 'distribution_type': 'UNIFORM', 'testing_flag': False},
]

# 全部数据类型测试配置（完整验证）
COMPREHENSIVE_CONFIG = [
    # 基础分布
    {'data_size': 'W1', 'distribution_type': 'UNIFORM', 'testing_flag': False},
    {'data_size': 'W1', 'distribution_type': 'NORMAL', 'testing_flag': False},
    {'data_size': 'W1', 'distribution_type': 'SKEW-NOR0', 'testing_flag': False},
    {'data_size': 'W1', 'distribution_type': 'SKEW-NOR2', 'testing_flag': False},
    {'data_size': 'W1', 'distribution_type': 'SKEW-NOR4', 'testing_flag': False},
    {'data_size': 'W1', 'distribution_type': 'SKEW-NOR8', 'testing_flag': False},
    
    # 多峰分布
    {'data_size': 'W1', 'distribution_type': 'BIMODAL', 'testing_flag': False},
    {'data_size': 'W1', 'distribution_type': 'MULTI-MODAL', 'testing_flag': False},
    {'data_size': 'W1', 'distribution_type': 'CLUSTERED', 'testing_flag': False},
    
    # 复杂相关分布
    {'data_size': 'W1', 'distribution_type': 'CORRELATED', 'testing_flag': False},
    {'data_size': 'W1', 'distribution_type': 'ANTI-CORR', 'testing_flag': False},
    {'data_size': 'W1', 'distribution_type': 'GAUSSIAN-MIX', 'testing_flag': False},
    
    # 时空数据
    {'data_size': 'W1', 'distribution_type': 'SPATIAL-TEMPORAL', 'testing_flag': False},
    {'data_size': 'W1', 'distribution_type': 'CITY-LIKE', 'testing_flag': False},
    {'data_size': 'W1', 'distribution_type': 'HOTSPOTS', 'testing_flag': False},
]

# ============================================================================
# 自定义配置示例
# ============================================================================

CUSTOM_CONFIG = [
    # 示例：生成多个不同规模和分布的组合
    {
        'data_size': 'W1', 
        'distribution_type': 'UNIFORM', 
        'testing_flag': True
    },
    # 添加更多自定义配置...
]
