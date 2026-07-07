"""
统计检验工具 - 用于实验结果的显著性分析
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats


class StatisticalTester:
    """
    统计检验器
    
    提供常用的统计检验方法：
    - 配对 t 检验 (Paired t-test)
    - ANOVA 方差分析
    - 效应量计算 (Cohen's d)
    - 置信区间估计
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        初始化统计检验器
        
        Args:
            significance_level: 显著性水平（默认 0.05）
        """
        self.significance_level = significance_level
    
    def paired_t_test(
        self,
        group1: List[float],
        group2: List[float],
        alternative: str = 'two-sided'
    ) -> Dict:
        """
        执行配对 t 检验
        
        Args:
            group1: 第一组数据
            group2: 第二组数据
            alternative: 备择假设类型 ('two-sided', 'less', 'greater')
        
        Returns:
            包含检验结果的字典
        """
        if len(group1) != len(group2):
            raise ValueError("两组数据长度必须相同")
        
        if len(group1) < 2:
            raise ValueError("每组至少需要 2 个样本")
        
        t_statistic, p_value = stats.ttest_rel(
            group1, 
            group2, 
            alternative=alternative
        )
        
        # 计算效应量 (Cohen's d)
        cohens_d = self._calculate_cohens_d(group1, group2)
        
        # 计算均值差异的置信区间
        ci_lower, ci_upper = self._confidence_interval(
            np.array(group1) - np.array(group2)
        )
        
        result = {
            'test_type': 'paired_t_test',
            't_statistic': t_statistic,
            'p_value': p_value,
            'significant': p_value < self.significance_level,
            'cohens_d': cohens_d,
            'effect_size': self._interpret_effect_size(cohens_d),
            'ci_95': (ci_lower, ci_upper),
            'mean_diff': np.mean(group1) - np.mean(group2),
            'group1_mean': np.mean(group1),
            'group2_mean': np.mean(group2),
            'sample_size': len(group1)
        }
        
        return result
    
    def one_way_anova(self, *groups: List[float]) -> Dict:
        """
        执行单因素方差分析 (One-way ANOVA)
        
        Args:
            *groups: 多组数据
        
        Returns:
            包含检验结果的字典
        """
        if len(groups) < 2:
            raise ValueError("至少需要两组数据")
        
        f_statistic, p_value = stats.f_oneway(*groups)
        
        # 计算 eta-squared (效应量)
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        
        ss_between = sum(
            len(group) * (np.mean(group) - grand_mean) ** 2 
            for group in groups
        )
        ss_total = sum(
            (x - grand_mean) ** 2 
            for group in groups 
            for x in group
        )
        
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        result = {
            'test_type': 'one_way_anova',
            'f_statistic': f_statistic,
            'p_value': p_value,
            'significant': p_value < self.significance_level,
            'eta_squared': eta_squared,
            'effect_size': self._interpret_eta_squared(eta_squared),
            'num_groups': len(groups),
            'group_means': [np.mean(group) for group in groups],
            'group_sizes': [len(group) for group in groups]
        }
        
        return result
    
    def calculate_confidence_interval(
        self,
        data: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        计算均值的置信区间
        
        Args:
            data: 数据列表
            confidence: 置信水平（默认 0.95）
        
        Returns:
            (下限, 上限)
        """
        return self._confidence_interval(data, confidence)
    
    def _calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """计算 Cohen's d 效应量"""
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # 合并标准差
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0
        
        return (mean1 - mean2) / pooled_std
    
    def _confidence_interval(
        self,
        data: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """计算置信区间"""
        n = len(data)
        mean = np.mean(data)
        se = stats.sem(data)  # 标准误
        
        # t 分布的临界值
        h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
        
        return mean - h, mean + h
    
    @staticmethod
    def _interpret_effect_size(cohens_d: float) -> str:
        """解释 Cohen's d 效应量"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    @staticmethod
    def _interpret_eta_squared(eta_sq: float) -> str:
        """解释 eta-squared 效应量"""
        if eta_sq < 0.01:
            return "negligible"
        elif eta_sq < 0.06:
            return "small"
        elif eta_sq < 0.14:
            return "medium"
        else:
            return "large"
    
    def print_test_result(self, result: Dict):
        """
        格式化打印检验结果
        
        Args:
            result: 检验结果字典
        """
        print("\n" + "=" * 80)
        print(f"Statistical Test Result: {result['test_type'].upper()}".center(80))
        print("=" * 80)
        
        if result['test_type'] == 'paired_t_test':
            print(f"Sample Size:             {result['sample_size']}")
            print(f"Group 1 Mean:            {result['group1_mean']:.4f}")
            print(f"Group 2 Mean:            {result['group2_mean']:.4f}")
            print(f"Mean Difference:         {result['mean_diff']:.4f}")
            print(f"t-statistic:             {result['t_statistic']:.4f}")
            print(f"p-value:                 {result['p_value']:.6f}")
            print(f"Significant (α={self.significance_level}):   {'Yes ✓' if result['significant'] else 'No ✗'}")
            print(f"Cohen's d:               {result['cohens_d']:.4f} ({result['effect_size']})")
            print(f"95% CI of Difference:    [{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}]")
        
        elif result['test_type'] == 'one_way_anova':
            print(f"Number of Groups:        {result['num_groups']}")
            print(f"Group Means:             {[f'{m:.4f}' for m in result['group_means']]}")
            print(f"Group Sizes:             {result['group_sizes']}")
            print(f"F-statistic:             {result['f_statistic']:.4f}")
            print(f"p-value:                 {result['p_value']:.6f}")
            print(f"Significant (α={self.significance_level}):   {'Yes ✓' if result['significant'] else 'No ✗'}")
            print(f"Eta-squared:             {result['eta_squared']:.4f} ({result['effect_size']})")
        
        print("=" * 80)
