"""
检查点管理器 - 管理模型保存和加载
"""
import os
import shutil
from typing import Dict, Optional, List
from datetime import datetime


class CheckpointManager:
    """
    检查点管理器
    
    功能：
    - 自动保存检查点
    - 保留最近的 N 个检查点
    - 跟踪最佳模型
    - 清理旧检查点
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        max_checkpoints: int = 10,
        metric_mode: str = "min"
    ):
        """
        初始化检查点管理器
        
        Args:
            checkpoint_dir: 检查点保存目录
            max_checkpoints: 最大保留的检查点数量
            metric_mode: 指标优化方向 ('min' 越小越好, 'max' 越大越好)
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.metric_mode = metric_mode
        
        # 创建目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 跟踪最佳模型
        self.best_metric = float('inf') if metric_mode == 'min' else float('-inf')
        self.best_checkpoint_path = None
        
        # 检查点历史
        self.checkpoints: List[Dict] = []
    
    def save_checkpoint(
        self,
        agent,
        episode: int,
        metric: float,
        extra_info: Dict = None,
        is_best: bool = False
    ) -> str:
        """
        保存检查点
        
        Args:
            agent: ACPPOAgent 实例
            episode: 当前 episode 索引
            metric: 评估指标
            extra_info: 额外信息
            is_best: 是否为最佳模型
        
        Returns:
            检查点文件路径
        """
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_ep{episode}_{timestamp}.pth"
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # 保存检查点
        agent.save_checkpoint(filepath)
        
        # 记录检查点信息
        checkpoint_info = {
            'path': filepath,
            'episode': episode,
            'metric': metric,
            'timestamp': timestamp,
            'is_best': is_best or self._is_new_best(metric)
        }
        
        if extra_info:
            checkpoint_info.update(extra_info)
        
        self.checkpoints.append(checkpoint_info)
        
        # 更新最佳模型
        if checkpoint_info['is_best']:
            self.best_metric = metric
            self.best_checkpoint_path = filepath
            
            # 复制为 best_model.pth
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            shutil.copy2(filepath, best_path)
            
            print(f"✓ New best model saved (metric: {metric:.4f})")
        
        # 清理旧检查点
        self._cleanup_old_checkpoints()
        
        return filepath
    
    def _is_new_best(self, metric: float) -> bool:
        """判断是否为新最佳模型"""
        if self.metric_mode == 'min':
            return metric < self.best_metric
        else:
            return metric > self.best_metric
    
    def _cleanup_old_checkpoints(self):
        """清理旧的检查点，只保留最近的 max_checkpoints 个"""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # 按 episode 排序
        sorted_checkpoints = sorted(
            self.checkpoints,
            key=lambda x: x['episode']
        )
        
        # 删除最旧的
        checkpoints_to_remove = sorted_checkpoints[:-self.max_checkpoints]
        
        for cp in checkpoints_to_remove:
            try:
                if os.path.exists(cp['path']):
                    os.remove(cp['path'])
            except Exception as e:
                print(f"Warning: Failed to remove {cp['path']}: {e}")
        
        # 更新列表
        self.checkpoints = sorted_checkpoints[-self.max_checkpoints:]
    
    def load_best_checkpoint(self, agent) -> bool:
        """
        加载最佳检查点
        
        Args:
            agent: ACPPOAgent 实例
        
        Returns:
            是否成功加载
        """
        if self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path):
            agent.load_checkpoint(self.best_checkpoint_path)
            print(f"✓ Loaded best checkpoint from episode")
            return True
        
        # 尝试加载 best_model.pth
        best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        if os.path.exists(best_path):
            agent.load_checkpoint(best_path)
            print(f"✓ Loaded best_model.pth")
            return True
        
        print("✗ No best checkpoint found")
        return False
    
    def list_checkpoints(self) -> List[Dict]:
        """列出所有检查点"""
        return self.checkpoints.copy()
    
    def get_latest_checkpoint(self) -> Optional[Dict]:
        """获取最新的检查点"""
        if not self.checkpoints:
            return None
        
        return max(self.checkpoints, key=lambda x: x['episode'])
    
    def print_summary(self):
        """打印检查点摘要"""
        print("\n" + "=" * 80)
        print("Checkpoint Summary".center(80))
        print("=" * 80)
        print(f"Total Checkpoints:       {len(self.checkpoints)}")
        print(f"Max Kept:                {self.max_checkpoints}")
        print(f"Best Metric:             {self.best_metric:.4f}")
        print(f"Best Checkpoint:         {self.best_checkpoint_path}")
        
        if self.checkpoints:
            latest = self.get_latest_checkpoint()
            print(f"Latest Episode:          {latest['episode']}")
            print(f"Latest Metric:           {latest['metric']:.4f}")
        
        print("=" * 80)
