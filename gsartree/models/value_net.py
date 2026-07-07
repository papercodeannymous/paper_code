"""
价值网络 (Critic Network)
负责评估状态的价值
"""
import torch
import torch.nn as nn


# class ValueNet(nn.Module):
#     """
#     价值网络 - Critic
    
#     Architecture:
#         Input (n_features) -> Linear -> ReLU -> Linear -> Output (1)
    
#     Args:
#         n_features: 输入特征维度
#         n_hidden: 隐藏层维度
#     """
    
#     def __init__(self, n_features: int, n_hidden: int):
#         super().__init__()
        
#         self.network = nn.Sequential(
#             nn.Linear(n_features, n_hidden),
#             nn.ReLU(),
#             nn.Linear(n_hidden, 1),
#         )
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         前向传播
        
#         Args:
#             x: 输入状态张量 [batch_size, n_features]
        
#         Returns:
#             状态价值 [batch_size, 1]
#         """
#         return self.network(x)


class ValueNet(nn.Module):
    def __init__(self, n_features: int, n_hidden: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )
        # ✅ 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
