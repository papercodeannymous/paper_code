"""
评估器 - 全面的性能评估和统计分析
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import stats

from gsartree.config.default_config import GSARConfig
from gsartree.environment.rtree_env import RTreeEnvironment


class Evaluator:
    """
    性能评估器
    
    提供全面的评估功能：
    - 范围查询评估
    - KNN 查询评估
    - 统计显著性检验
    - 多轮实验聚合
    
    Attributes:
        config: 配置对象
    """
    
    def __init__(self, config: GSARConfig):
        """
        初始化评估器
        
        Args:
            config: 完整的 GSAR 配置
        """
        self.config = config
    
    def evaluate_range_queries(
        self,
        test_tree: RTreeEnvironment,
        reference_tree: RTreeEnvironment,
        query_ratios: List[float] = None,
        num_queries_per_ratio: int = 1000
    ) -> Dict:
        """
        评估范围查询性能
        
        Args:
            test_tree: 待评估的树
            reference_tree: 参考树
            query_ratios: 查询比例列表
            num_queries_per_ratio: 每个比例的查询次数
        
        Returns:
            评估结果字典
        """
        if query_ratios is None:
            query_ratios = self.config.query.range_ratios
        
        results = {}
        
        for ratio in query_ratios:
            result = self._test_single_ratio(
                test_tree, reference_tree, ratio, num_queries_per_ratio
            )
            results[f"{ratio}%"] = result
        
        return results
    
    def _test_single_ratio(
        self,
        test_tree: RTreeEnvironment,
        reference_tree: RTreeEnvironment,
        ratio: float,
        num_queries: int
    ) -> Dict:
        """
        测试单个查询比例
        
        Args:
            test_tree: 待评估树
            reference_tree: 参考树
            ratio: 查询比例
            num_queries: 查询次数
        
        Returns:
            单个比例的测试结果
        """
        # 计算查询区域大小
        total_area = (
            (self.config.data.x_max - self.config.data.x_min) *
            (self.config.data.y_max - self.config.data.y_min)
        )
        query_area = ratio / 100 * total_area
        side = np.sqrt(query_area) / 2
        
        test_accesses = []
        ref_accesses = []
        
        valid_queries = 0
        
        while valid_queries < num_queries:
            # 生成随机查询
            center_x = np.random.uniform(
                self.config.data.x_min + side,
                self.config.data.x_max - side
            )
            center_y = np.random.uniform(
                self.config.data.y_min + side,
                self.config.data.y_max - side
            )
            
            query_rect = [
                center_x - side,
                center_y - side,
                center_x + side,
                center_y + side
            ]
            
            # 执行查询并获取原始节点访问数
            test_access = test_tree.query(query_rect)  # 原始节点访问数
            ref_access = reference_tree.query(query_rect)  # 原始节点访问数
            
            test_accesses.append(test_access)
            ref_accesses.append(ref_access)
            
            valid_queries += 1
        
        # 计算统计量
        test_array = np.array(test_accesses)
        ref_array = np.array(ref_accesses)
        
        avg_test = np.mean(test_array)
        avg_ref = np.mean(ref_array)
        std_test = np.std(test_array)
        std_ref = np.std(ref_array)
        
        # 计算平均提升百分比 (Node Access Gain)
        # 定义：(Ref - Test) / Test * 100%
        improvement = ((avg_ref - avg_test) / avg_test * 100) if avg_test > 0 else 0
        
        # 计算每次查询的相对性能提升，然后求其标准差
        # 对于每次查询 i，计算: (ref_i - test_i) / test_i * 100
        # 这样可以反映性能提升的稳定性（波动程度）
        improvements_per_query = []
        for i in range(len(test_accesses)):
            if test_accesses[i] > 0:
                imp = (ref_accesses[i] - test_accesses[i]) / test_accesses[i]
                improvements_per_query.append(imp)
            else:
                improvements_per_query.append(0.0)
        
        improvements_array = np.array(improvements_per_query)
        improvement_std = np.std(improvements_array)  # 性能提升的标准差
        improvement_mean_from_queries = np.mean(improvements_array)  # 从单次查询计算的平均提升
        
        return {
            'ratio': ratio,
            'num_queries': num_queries,
            'test_mean': avg_test,
            'test_std': std_test,
            'ref_mean': avg_ref,
            'ref_std': std_ref,
            'improvement_percent': improvement,  # 基于均值的提升（原有逻辑）
            'improvement_mean': improvement_mean_from_queries,  # 基于单次查询的平均提升
            'improvement_std': improvement_std,  # 性能提升的标准差
            'test_accesses': test_accesses,
            'ref_accesses': ref_accesses,
            'improvements_per_query': improvements_per_query  # 每次查询的提升值列表
        }
    
    def evaluate_knn_queries(
        self,
        test_tree: RTreeEnvironment,
        reference_tree: RTreeEnvironment,
        k_values: List[int] = None,
        num_queries: int = 1000
    ) -> Dict:
        """
        评估 KNN 查询性能
        
        Args:
            test_tree: 待评估树
            reference_tree: 参考树
            k_values: K 值列表
            num_queries: 查询次数
        
        Returns:
            KNN 评估结果
        """
        if k_values is None:
            k_values = self.config.query.knn_values
        
        results = {}
        
        for k in k_values:
            result = self._test_single_knn(
                test_tree, reference_tree, k, num_queries
            )
            results[f"k={k}"] = result
        
        return results
    
    def _test_single_knn(
        self,
        test_tree: RTreeEnvironment,
        reference_tree: RTreeEnvironment,
        k: int,
        num_queries: int
    ) -> Dict:
        """测试单个 K 值的 KNN 查询"""
        test_accesses = []
        ref_accesses = []
        
        for _ in range(num_queries):
            x = np.random.uniform(
                self.config.data.x_min,
                self.config.data.x_max
            )
            y = np.random.uniform(
                self.config.data.y_min,
                self.config.data.y_max
            )
            
            test_access = test_tree.knn_query(x, y, k)
            ref_access = reference_tree.knn_query(x, y, k)
            
            test_accesses.append(test_access)
            ref_accesses.append(ref_access)
        
        test_array = np.array(test_accesses)
        ref_array = np.array(ref_accesses)
        
        avg_test = np.mean(test_array)
        avg_ref = np.mean(ref_array)
        improvement = ((avg_ref - avg_test) / avg_ref * 100) if avg_ref > 0 else 0
        
        return {
            'k': k,
            'num_queries': num_queries,
            'test_mean': avg_test,
            'ref_mean': avg_ref,
            'improvement_percent': improvement
        }
    
    def statistical_test(
        self,
        test_accesses: List[float],
        ref_accesses: List[float],
        alpha: float = 0.05
    ) -> Dict:
        """
        执行统计显著性检验（配对 t-test）
        
        Args:
            test_accesses: 待评估树的访问次数列表
            ref_accesses: 参考树的访问次数列表
            alpha: 显著性水平
        
        Returns:
            统计检验结果
        """
        if len(test_accesses) != len(ref_accesses):
            raise ValueError("Access lists must have the same length")
        
        if len(test_accesses) < 2:
            return {'error': 'Not enough samples for statistical test'}
        
        # 配对 t-test
        t_stat, p_value = stats.ttest_rel(test_accesses, ref_accesses)
        
        # 计算效应量 (Cohen's d)
        diff = np.array(test_accesses) - np.array(ref_accesses)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        
        # 置信区间
        n = len(test_accesses)
        se = std_diff / np.sqrt(n)
        ci_lower = mean_diff - stats.t.ppf(1 - alpha/2, n-1) * se
        ci_upper = mean_diff + stats.t.ppf(1 - alpha/2, n-1) * se
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'alpha': alpha,
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'cohens_d': cohens_d,
            'confidence_interval_95': (ci_lower, ci_upper),
            'interpretation': self._interpret_significance(p_value, alpha)
        }
    
    def _interpret_significance(self, p_value: float, alpha: float) -> str:
        """解释显著性结果"""
        if p_value < 0.001:
            return "Highly significant (p < 0.001)"
        elif p_value < 0.01:
            return "Very significant (p < 0.01)"
        elif p_value < alpha:
            return f"Significant (p < {alpha})"
        else:
            return f"Not significant (p = {p_value:.4f})"
    
    def multi_run_evaluation(
        self,
        create_test_tree_func,
        create_ref_tree_func,
        dataset: List[List[float]],
        num_runs: int = 5,
        query_ratios: List[float] = None
    ) -> Dict:
        """
        多轮实验评估（用于统计显著性分析）
        
        Args:
            create_test_tree_func: 创建待评估树的函数 () -> RTreeEnvironment
            create_ref_tree_func: 创建参考树的函数 () -> RTreeEnvironment
            dataset: 数据集
            num_runs: 运行次数
            query_ratios: 查询比例列表
        
        Returns:
            聚合的评估结果（包含均值、标准差等）
        """
        if query_ratios is None:
            query_ratios = self.config.query.range_ratios
        
        all_results = {ratio: {'test': [], 'ref': []} for ratio in query_ratios}
        
        print(f"\nRunning {num_runs} evaluation runs...")
        
        for run_idx in range(num_runs):
            print(f"  Run {run_idx + 1}/{num_runs}...")
            
            # 创建树
            test_tree = create_test_tree_func()
            ref_tree = create_ref_tree_func()
            
            # 插入数据
            for rect in dataset:
                test_tree.insert_rectangle(rect)
                ref_tree.insert_rectangle(rect)
            
            # 评估
            results = self.evaluate_range_queries(
                test_tree, ref_tree, query_ratios, num_queries=100
            )
            
            # 收集结果
            for ratio in query_ratios:
                key = f"{ratio}%"
                all_results[ratio]['test'].append(results[key]['test_mean'])
                all_results[ratio]['ref'].append(results[key]['ref_mean'])
        
        # 聚合统计
        aggregated = {}
        for ratio in query_ratios:
            key = f"{ratio}%"
            test_vals = np.array(all_results[ratio]['test'])
            ref_vals = np.array(all_results[ratio]['ref'])
            
            # 执行统计检验
            stat_test = self.statistical_test(test_vals.tolist(), ref_vals.tolist())
            
            # ⭐ 计算跨多次运行的性能提升标准差
            improvement_per_run = (ref_vals - test_vals) / ref_vals * 100
            
            aggregated[key] = {
                'test_mean': np.mean(test_vals),
                'test_std': np.std(test_vals),
                'ref_mean': np.mean(ref_vals),
                'ref_std': np.std(ref_vals),
                'improvement_mean': np.mean(improvement_per_run),
                'improvement_std': np.std(improvement_per_run),  # ⭐ 新增：跨运行的提升标准差
                'statistical_test': stat_test,
                'raw_test_values': test_vals.tolist(),
                'raw_ref_values': ref_vals.tolist()
            }
        
        return aggregated
    
    def print_evaluation_report(self, results: Dict, title: str = "Evaluation Report"):
        """
        打印评估报告
        
        Args:
            results: 评估结果字典
            title: 报告标题
        """
        print("\n" + "=" * 80)
        print(title.center(80))
        print("=" * 80)
        
        # 表头 - 增加 Improvement Std 列
        print(f"\n{'Query':<10} {'Test Mean':<12} {'Test Std':<12} {'Ref Mean':<12} {'Ref Std':<12} {'Impr %':<10} {'Impr Std':<10}")
        print("-" * 80)
        
        # 数据行
        for query_name, metrics in results.items():
            if isinstance(metrics, dict) and 'test_mean' in metrics:
                impr_pct = metrics.get('improvement_percent', metrics.get('improvement_mean', 0))
                impr_std = metrics.get('improvement_std', 0)
                print(f"{query_name:<10} {metrics['test_mean']:<12.4f} {metrics.get('test_std', 0):<12.4f} "
                      f"{metrics['ref_mean']:<12.4f} {metrics.get('ref_std', 0):<12.4f} "
                      f"{impr_pct:>+9.2f}% {impr_std:<10.2f}")
        
        print("=" * 80)
        
        # 统计显著性（如果有）
        has_stats = any('statistical_test' in m for m in results.values() if isinstance(m, dict))
        if has_stats:
            print("\nStatistical Significance:")
            for query_name, metrics in results.items():
                if isinstance(metrics, dict) and 'statistical_test' in metrics:
                    stat = metrics['statistical_test']
                    sig_marker = "✓" if stat.get('significant', False) else "✗"
                    print(f"  {query_name}: p = {stat['p_value']:.4f} {sig_marker} ({stat.get('interpretation', '')})")
        
        print("=" * 80)
