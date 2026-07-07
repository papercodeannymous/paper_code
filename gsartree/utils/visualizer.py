"""
可视化工具
"""
import os
import numpy as np
from typing import Dict, List, Optional

try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Visualization will be skipped.")


class Visualizer:
    def __init__(self, output_dir='images'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    def plot_query_comparison(self, query_perf: Dict):
        """
        绘制查询性能对比图
        
        Args:
            query_perf: 评估结果字典，格式为:
                       {
                         "2.0%": {'agent_access': 11, 'ref_access': 13, 'improvement': 15.38},
                         "1.0%": {'agent_access': 11, 'ref_access': 12, 'improvement': 8.33},
                         ...
                       }
        """
        if not HAS_MATPLOTLIB:
            print(f"⚠️  Skipping visualization (matplotlib not available)")
            return
        
        if not query_perf:
            print(f"⚠️  No query performance data to plot")
            return
        
        # 提取数据并排序
        ratios = []
        agent_access = []
        ref_access = []
        improvements = []
        
        for key in sorted(query_perf.keys(), key=lambda x: float(x.replace('%', ''))):
            metrics = query_perf[key]
            ratio_val = float(key.replace('%', ''))
            ratios.append(ratio_val)
            agent_access.append(metrics['agent_access'])
            
            # ✅ 兼容多种键名，优先使用新版 'static_ref_access'，回退到旧版 'ref_access'
            if 'static_ref_access' in metrics:
                ref_access.append(metrics['static_ref_access'])
            elif 'ref_access' in metrics:
                ref_access.append(metrics['ref_access'])
            else:
                # 如果都没有，记录警告并使用默认值 0
                print(f"⚠️  Warning: No reference access data found for {key}. Using 0 as default.")
                ref_access.append(0)
            
            improvements.append(metrics['improvement'])
        
        if not ratios:
            print(f"⚠️  No valid query performance data")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. 节点访问数对比
        x = np.arange(len(ratios))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, agent_access, width, label='Agent (RL)', 
                       color='#2196F3', alpha=0.8)
        bars2 = ax1.bar(x + width/2, ref_access, width, label='Reference (R*-Tree)', 
                       color='#FF5722', alpha=0.8)
        
        ax1.set_xlabel('Query Area Ratio (%)', fontsize=12)
        ax1.set_ylabel('Nodes Accessed', fontsize=12)
        ax1.set_title('Query Performance: Nodes Accessed', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{r:.2f}%' for r in ratios], rotation=45, ha='right')
        ax1.legend(fontsize=11)
        ax1.grid(True, axis='y', alpha=0.3)
        
        # 在柱状图上添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        # 2. 改进百分比
        colors = ['#4CAF50' if imp > 0 else '#F44336' for imp in improvements]
        bars3 = ax2.bar(x, improvements, width=0.6, color=colors, alpha=0.8, edgecolor='black')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_xlabel('Query Area Ratio (%)', fontsize=12)
        ax2.set_ylabel('Improvement (%)', fontsize=12)
        ax2.set_title('Performance Improvement (Positive is Better)', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{r:.2f}%' for r in ratios], rotation=45, ha='right')
        ax2.grid(True, axis='y', alpha=0.3)
        
        # 在柱状图上添加数值标签
        for bar, imp in zip(bars3, improvements):
            height = bar.get_height()
            va = 'bottom' if imp >= 0 else 'top'
            offset = 0.5 if imp >= 0 else -0.5
            ax2.text(bar.get_x() + bar.get_width()/2., height + offset,
                    f'{imp:.1f}%', ha='center', va=va, fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(self.output_dir, 'query_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Query comparison saved to {output_path}")
    
    def plot_self_play_statistics(self, stats: Dict):
        """
        绘制 Self-Play 统计信息
        
        Args:
            stats: 包含自我对弈统计数据的字典
        """
        if not HAS_MATPLOTLIB:
            print(f"⚠️  Skipping visualization (matplotlib not available)")
            return
        
        if not stats:
            print(f"⚠️  No self-play statistics to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. 胜率变化
        win_rates = stats.get('win_rates', [])
        if win_rates:
            episodes = range(1, len(win_rates) + 1)
            axes[0].plot(episodes, win_rates, 'b-o', linewidth=2, markersize=6)
            axes[0].axhline(y=0.5, color='r', linestyle='--', label='Random Baseline (50%)')
            axes[0].set_xlabel('Episode', fontsize=12)
            axes[0].set_ylabel('Win Rate', fontsize=12)
            axes[0].set_title('Self-Play Win Rate', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
        
        # 2. 平均奖励
        avg_rewards = stats.get('avg_rewards', [])
        if avg_rewards:
            episodes = range(1, len(avg_rewards) + 1)
            axes[1].plot(episodes, avg_rewards, 'g-s', linewidth=2, markersize=6)
            axes[1].axhline(y=0, color='r', linestyle='--', label='Zero Baseline')
            axes[1].set_xlabel('Episode', fontsize=12)
            axes[1].set_ylabel('Average Reward', fontsize=12)
            axes[1].set_title('Self-Play Average Reward', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
        
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(self.output_dir, 'self_play_stats.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Self-play statistics saved to {output_path}")