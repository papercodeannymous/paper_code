"""
策略网络 (Actor Network)
负责根据状态输出动作概率分布
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# class PolicyNet(nn.Module):
#     """
#     策略网络 - Actor
    
#     Architecture:
#         Input (n_features) -> Linear -> ReLU -> Linear -> Softmax -> Output (n_actions)
    
#     Args:
#         n_features: 输入特征维度
#         n_hidden: 隐藏层维度
#         n_actions: 动作空间大小
#     """
    
#     def __init__(self, n_features: int, n_hidden: int, n_actions: int):
#         super().__init__()
        
#         self.network = nn.Sequential(
#             nn.Linear(n_features, n_hidden),
#             nn.ReLU(),
#             nn.Linear(n_hidden, n_actions),
#         )
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         前向传播
        
#         Args:
#             x: 输入状态张量 [batch_size, n_features]
        
#         Returns:
#             动作概率分布 [batch_size, n_actions]
#         """
#         logits = self.network(x)
#         return F.softmax(logits, dim=-1)


class PolicyNet(nn.Module):
    def __init__(self, n_features: int, n_hidden: int, n_actions: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_actions),
        )
        # ✅ 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(x)
        return F.softmax(logits, dim=-1)