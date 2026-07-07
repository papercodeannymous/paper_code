"""
性能监控器 - 记录训练和评估的性能指标
"""
import time
import json
import psutil
import os
from typing import Dict, List
from datetime import datetime


class PerformanceMonitor:
    """
    性能监控器
    
    监控和记录：
    - 训练时间
    - CPU/GPU 使用率
    - 内存占用
    - 各阶段耗时
    """
    
    def __init__(self):
        """初始化性能监控器"""
        self.start_time = None
        self.end_time = None
        self.checkpoints: List[Dict] = []
        self.process = psutil.Process(os.getpid())
    
    def start(self):
        """开始监控"""
        self.start_time = time.time()
        self._record_checkpoint("Training started")
    
    def stop(self):
        """停止监控"""
        self.end_time = time.time()
        self._record_checkpoint("Training completed")
    
    def _record_checkpoint(self, label: str):
        """记录检查点"""
        checkpoint = {
            'timestamp': time.time(),
            'label': label,
            'elapsed_time': time.time() - self.start_time if self.start_time else 0,
            'cpu_percent': self.process.cpu_percent(interval=0.1),
            'memory_mb': self.process.memory_info().rss / (1024 * 1024)
        }
        
        # 尝试获取 GPU 信息
        try:
            import torch
            if torch.cuda.is_available():
                checkpoint['gpu_memory_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
                checkpoint['gpu_utilization'] = torch.cuda.utilization()
        except:
            pass
        
        self.checkpoints.append(checkpoint)
    
    def record_episode(self, episode_idx: int, episode_time: float, reward: float = None, 
                      dataset_size: int = None, extra_info: Dict = None):
        """
        记录 episode 性能
        
        Args:
            episode_idx: Episode 索引
            episode_time: Episode 耗时（秒）
            reward: 奖励值（可选）
            dataset_size: 数据集大小（可选，用于计算每样本处理时间）
            extra_info: 额外信息字典（可选）
        """
        checkpoint = {
            'episode': episode_idx,
            'episode_time': episode_time,
            'reward': reward,
            'dataset_size': dataset_size,
            'cpu_percent': self.process.cpu_percent(interval=0.1),
            'memory_mb': self.process.memory_info().rss / (1024 * 1024),
            'timestamp': time.time()
        }
        
        # 添加额外信息
        if extra_info:
            checkpoint.update(extra_info)
        
        # 计算每样本处理时间（如果提供了数据集大小）
        if dataset_size and dataset_size > 0:
            checkpoint['time_per_sample'] = episode_time / dataset_size
        
        try:
            import torch
            if torch.cuda.is_available():
                checkpoint['gpu_memory_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
                checkpoint['gpu_utilization'] = torch.cuda.utilization()
        except:
            pass
        
        self.checkpoints.append(checkpoint)
    
    def get_total_time(self) -> float:
        """获取总耗时（秒）"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return 0
    
    def get_average_cpu_usage(self) -> float:
        """获取平均 CPU 使用率"""
        if not self.checkpoints:
            return 0
        
        cpu_values = [cp.get('cpu_percent', 0) for cp in self.checkpoints]
        return sum(cpu_values) / len(cpu_values)
    
    def get_peak_memory_mb(self) -> float:
        """获取峰值内存占用（MB）"""
        if not self.checkpoints:
            return 0
        
        memory_values = [cp.get('memory_mb', 0) for cp in self.checkpoints]
        return max(memory_values)
    
    def get_performance_summary(self) -> Dict:
        """
        获取性能摘要
        
        Returns:
            包含关键性能指标的字典
        """
        summary = {
            'total_time_seconds': self.get_total_time(),
            'total_time_formatted': self._format_time(self.get_total_time()),
            'average_cpu_percent': self.get_average_cpu_usage(),
            'peak_memory_mb': self.get_peak_memory_mb(),
            'num_checkpoints': len(self.checkpoints)
        }
        
        # 添加 GPU 信息（如果有）
        gpu_checkpoints = [cp for cp in self.checkpoints if 'gpu_memory_mb' in cp]
        if gpu_checkpoints:
            summary['peak_gpu_memory_mb'] = max(cp['gpu_memory_mb'] for cp in gpu_checkpoints)
            summary['average_gpu_utilization'] = sum(
                cp.get('gpu_utilization', 0) for cp in gpu_checkpoints
            ) / len(gpu_checkpoints)
        
        return summary
    
    def print_report(self):
        """打印性能报告"""
        summary = self.get_performance_summary()
        
        print("\n" + "=" * 80)
        print("Performance Report".center(80))
        print("=" * 80)
        print(f"Total Time:              {summary['total_time_formatted']}")
        print(f"Average CPU Usage:       {summary['average_cpu_percent']:.1f}%")
        print(f"Peak Memory:             {summary['peak_memory_mb']:.2f} MB")
        
        if 'peak_gpu_memory_mb' in summary:
            print(f"Peak GPU Memory:         {summary['peak_gpu_memory_mb']:.2f} MB")
            print(f"Avg GPU Utilization:     {summary['average_gpu_utilization']:.1f}%")
        
        print("=" * 80)
    
    def save_report(self, filepath: str):
        """
        保存性能报告到 JSON 文件
        
        Args:
            filepath: 保存路径
        """
        summary = self.get_performance_summary()
        summary['checkpoints'] = self.checkpoints
        summary['generated_at'] = datetime.now().isoformat()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Performance report saved to {filepath}")
    
    def get_episode_statistics(self) -> Dict:
        """
        获取 Episode 时间统计信息
        
        Returns:
            包含平均时间、总时间、最快/最慢 episode 等统计信息的字典
        """
        episode_checkpoints = [cp for cp in self.checkpoints if 'episode' in cp]
        
        if not episode_checkpoints:
            return {}
        
        episode_times = [cp['episode_time'] for cp in episode_checkpoints]
        
        stats = {
            'total_episodes': len(episode_checkpoints),
            'total_training_time': sum(episode_times),
            'average_episode_time': sum(episode_times) / len(episode_times),
            'min_episode_time': min(episode_times),
            'max_episode_time': max(episode_times),
            'std_episode_time': (sum((t - sum(episode_times)/len(episode_times))**2 for t in episode_times) / len(episode_times)) ** 0.5
        }
        
        # 如果有数据集大小信息，计算每样本统计
        samples_info = [cp for cp in episode_checkpoints if 'time_per_sample' in cp]
        if samples_info:
            times_per_sample = [cp['time_per_sample'] for cp in samples_info]
            stats['average_time_per_sample'] = sum(times_per_sample) / len(times_per_sample)
            stats['dataset_size'] = samples_info[0].get('dataset_size', None)
        
        return stats
    
    def print_episode_time_report(self):
        """打印 Episode 时间详细报告"""
        stats = self.get_episode_statistics()
        
        if not stats:
            print("No episode data recorded.")
            return
        
        print("\n" + "=" * 80)
        print("Episode Time Statistics".center(80))
        print("=" * 80)
        print(f"Total Episodes:          {stats['total_episodes']}")
        print(f"Total Training Time:     {self._format_time(stats['total_training_time'])}")
        print(f"Average Episode Time:    {stats['average_episode_time']:.2f}s")
        print(f"Min Episode Time:        {stats['min_episode_time']:.2f}s")
        print(f"Max Episode Time:        {stats['max_episode_time']:.2f}s")
        print(f"Std Episode Time:        {stats['std_episode_time']:.2f}s")
        
        if 'average_time_per_sample' in stats:
            print(f"\nPer-Sample Processing:")
            print(f"  Dataset Size:          {stats['dataset_size']}")
            print(f"  Avg Time per Sample:   {stats['average_time_per_sample']*1000:.4f}ms")
            print(f"  Samples per Second:    {1/stats['average_time_per_sample']:.2f}")
        
        print("=" * 80)
    
    def get_tree_build_statistics(self) -> Dict:
        """
        获取树构建时间统计信息
        
        Returns:
            包含 RL Tree 和 Baseline Tree 构建时间的详细统计
        """
        import numpy as np
        
        build_checkpoints = [cp for cp in self.checkpoints if 'tree_type' in cp]
        
        if not build_checkpoints:
            return {}
        
        # 按树类型分组
        rl_trees = [cp for cp in build_checkpoints if cp['tree_type'] == 'RL_Tree']
        baseline_trees = [cp for cp in build_checkpoints if cp['tree_type'] != 'RL_Tree']
        
        stats = {}
        
        # 统计 RL Tree
        if rl_trees:
            rl_times = [cp['build_time_seconds'] for cp in rl_trees]
            rl_memory = [cp.get('memory_mb', 0) for cp in rl_trees]
            rl_cpu = [cp.get('cpu_percent', 0) for cp in rl_trees]
            rl_gpu_mem = [cp.get('gpu_memory_mb', 0) for cp in rl_trees if 'gpu_memory_mb' in cp]
            
            stats['rl_tree'] = {
                'count': len(rl_trees),
                'total_build_time': sum(rl_times),
                'average_build_time': sum(rl_times) / len(rl_times),
                'min_build_time': min(rl_times),
                'max_build_time': max(rl_times),
                'std_build_time': (sum((t - sum(rl_times)/len(rl_times))**2 for t in rl_times) / len(rl_times)) ** 0.5,
                'dataset_size': rl_trees[0].get('total_rectangles', None),
                'avg_insertion_rate': np.mean([cp.get('insertion_rate_per_second', 0) for cp in rl_trees]),
                'avg_time_per_rect_ms': np.mean([cp.get('avg_time_per_rectangle_ms', 0) for cp in rl_trees]),
                # ⭐ 新增：资源开销统计
                'peak_memory_mb': max(rl_memory) if rl_memory else 0,
                'avg_memory_mb': np.mean(rl_memory) if rl_memory else 0,
                'avg_cpu_percent': np.mean(rl_cpu) if rl_cpu else 0,
                'peak_gpu_memory_mb': max(rl_gpu_mem) if rl_gpu_mem else 0,
                'avg_gpu_memory_mb': np.mean(rl_gpu_mem) if rl_gpu_mem else 0
            }
        
        # 统计 Baseline Trees（可能有多种类型）
        if baseline_trees:
            baseline_by_type = {}
            for cp in baseline_trees:
                tree_type = cp['tree_type']
                if tree_type not in baseline_by_type:
                    baseline_by_type[tree_type] = []
                baseline_by_type[tree_type].append(cp)
            
            stats['baseline_trees'] = {}
            for tree_type, checkpoints in baseline_by_type.items():
                times = [cp['build_time_seconds'] for cp in checkpoints]
                memory = [cp.get('memory_mb', 0) for cp in checkpoints]
                cpu = [cp.get('cpu_percent', 0) for cp in checkpoints]
                gpu_mem = [cp.get('gpu_memory_mb', 0) for cp in checkpoints if 'gpu_memory_mb' in cp]
                
                stats['baseline_trees'][tree_type] = {
                    'count': len(checkpoints),
                    'total_build_time': sum(times),
                    'average_build_time': sum(times) / len(times),
                    'min_build_time': min(times),
                    'max_build_time': max(times),
                    'std_build_time': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
                    'dataset_size': checkpoints[0].get('total_rectangles', None),
                    'avg_insertion_rate': np.mean([cp.get('insertion_rate_per_second', 0) for cp in checkpoints]),
                    'avg_time_per_rect_ms': np.mean([cp.get('avg_time_per_rectangle_ms', 0) for cp in checkpoints]),
                    # ⭐ 新增：资源开销统计
                    'peak_memory_mb': max(memory) if memory else 0,
                    'avg_memory_mb': np.mean(memory) if memory else 0,
                    'avg_cpu_percent': np.mean(cpu) if cpu else 0,
                    'peak_gpu_memory_mb': max(gpu_mem) if gpu_mem else 0,
                    'avg_gpu_memory_mb': np.mean(gpu_mem) if gpu_mem else 0
                }
        
        return stats
    
    def print_tree_build_report(self):
        """打印树构建时间详细报告"""
        stats = self.get_tree_build_statistics()
        
        if not stats:
            print("No tree build data recorded.")
            return
        
        print("\n" + "=" * 80)
        print("Tree Construction Time & Resource Statistics".center(80))
        print("=" * 80)
        
        # RL Tree 统计
        if 'rl_tree' in stats:
            rl = stats['rl_tree']
            print(f"\n📊 RL-Optimized Tree (Agent/Test Tree):")
            print(f"  Build Count:           {rl['count']}")
            print(f"  Total Build Time:      {self._format_time(rl['total_build_time'])}")
            print(f"  Average Build Time:    {rl['average_build_time']:.2f}s")
            print(f"  Min Build Time:        {rl['min_build_time']:.2f}s")
            print(f"  Max Build Time:        {rl['max_build_time']:.2f}s")
            print(f"  Std Deviation:         {rl['std_build_time']:.2f}s")
            print(f"  Dataset Size:          {rl['dataset_size']} rectangles")
            print(f"  Avg Insertion Rate:    {rl['avg_insertion_rate']:.2f} rects/sec")
            print(f"  Avg Time per Rect:     {rl['avg_time_per_rect_ms']:.4f}ms")
            
            # ⭐ 新增：资源开销
            print(f"\n  💻 Resource Consumption:")
            print(f"    Peak Memory:         {rl['peak_memory_mb']:.2f} MB")
            print(f"    Avg Memory:          {rl['avg_memory_mb']:.2f} MB")
            print(f"    Avg CPU Usage:       {rl['avg_cpu_percent']:.1f}%")
            if rl['peak_gpu_memory_mb'] > 0:
                print(f"    Peak GPU Memory:     {rl['peak_gpu_memory_mb']:.2f} MB")
                print(f"    Avg GPU Memory:      {rl['avg_gpu_memory_mb']:.2f} MB")
        
        # Baseline Trees 统计
        if 'baseline_trees' in stats:
            print(f"\n📊 Baseline Trees (Reference Trees):")
            for tree_type, tree_stats in stats['baseline_trees'].items():
                print(f"\n  {tree_type}:")
                print(f"    Build Count:           {tree_stats['count']}")
                print(f"    Total Build Time:      {self._format_time(tree_stats['total_build_time'])}")
                print(f"    Average Build Time:    {tree_stats['average_build_time']:.2f}s")
                print(f"    Min Build Time:        {tree_stats['min_build_time']:.2f}s")
                print(f"    Max Build Time:        {tree_stats['max_build_time']:.2f}s")
                print(f"    Std Deviation:         {tree_stats['std_build_time']:.2f}s")
                print(f"    Dataset Size:          {tree_stats['dataset_size']} rectangles")
                print(f"    Avg Insertion Rate:    {tree_stats['avg_insertion_rate']:.2f} rects/sec")
                print(f"    Avg Time per Rect:     {tree_stats['avg_time_per_rect_ms']:.4f}ms")
                
                # ⭐ 新增：资源开销
                print(f"\n    💻 Resource Consumption:")
                print(f"      Peak Memory:         {tree_stats['peak_memory_mb']:.2f} MB")
                print(f"      Avg Memory:          {tree_stats['avg_memory_mb']:.2f} MB")
                print(f"      Avg CPU Usage:       {tree_stats['avg_cpu_percent']:.1f}%")
                if tree_stats['peak_gpu_memory_mb'] > 0:
                    print(f"      Peak GPU Memory:     {tree_stats['peak_gpu_memory_mb']:.2f} MB")
                    print(f"      Avg GPU Memory:      {tree_stats['avg_gpu_memory_mb']:.2f} MB")
                
                # 对比 RL Tree 和 Baseline Tree
                if 'rl_tree' in stats:
                    rl_avg = stats['rl_tree']['average_build_time']
                    bl_avg = tree_stats['average_build_time']
                    speedup = rl_avg / bl_avg if bl_avg > 0 else 0
                    time_diff = rl_avg - bl_avg
                    
                    # ⭐ 资源对比
                    rl_mem = stats['rl_tree']['peak_memory_mb']
                    bl_mem = tree_stats['peak_memory_mb']
                    mem_diff = rl_mem - bl_mem
                    mem_ratio = rl_mem / bl_mem if bl_mem > 0 else 0
                    
                    print(f"\n    ⚖️  Comparison with RL Tree:")
                    print(f"      Time Difference:       {time_diff:+.2f}s ({'RL slower' if time_diff > 0 else 'RL faster'})")
                    print(f"      Speedup Ratio:         {speedup:.2f}x ({'Baseline faster' if speedup > 1 else 'RL faster'})")
                    print(f"      Memory Difference:     {mem_diff:+.2f} MB ({'RL higher' if mem_diff > 0 else 'RL lower'})")
                    print(f"      Memory Ratio:          {mem_ratio:.2f}x ({'RL uses more' if mem_ratio > 1 else 'RL uses less'})")
        
        print("\n" + "=" * 80)
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """格式化时间为可读字符串"""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f}m ({seconds:.0f}s)"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{hours:.2f}h ({minutes:.0f}m)"
