"""
Statistical Analysis Framework for Academic Research
Implements rigorous statistical methods for algorithm comparison
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon, friedmanchisquare, kruskal
from scipy.stats import shapiro, levene, bartlett
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ComparisonResult:
    """Statistical comparison result between two algorithms"""
    algorithm_1: str
    algorithm_2: str
    test_type: str
    test_statistic: float
    p_value: float
    effect_size: float
    effect_size_type: str
    significantly_different: bool
    confidence_interval: Tuple[float, float]
    practical_significance: bool
    interpretation: str


@dataclass
class MultipleComparisonResult:
    """Result of multiple algorithm comparison"""
    test_type: str
    test_statistic: float
    p_value: float
    post_hoc_results: List[ComparisonResult]
    ranking: List[Tuple[str, float]]
    statistical_summary: Dict[str, Any]


@dataclass
class DistributionAnalysis:
    """Distribution analysis result"""
    algorithm: str
    metric: str
    normality_test: Dict[str, float]
    descriptive_stats: Dict[str, float]
    distribution_type: str
    outliers: List[float]
    confidence_intervals: Dict[str, Tuple[float, float]]


class AcademicStatisticalAnalyzer:
    """
    Comprehensive statistical analysis framework for academic research
    Implements proper statistical tests and effect size calculations
    """
    
    def __init__(self, alpha: float = 0.05, power: float = 0.8):
        """
        Initialize statistical analyzer
        
        Args:
            alpha: Significance level (Type I error rate)
            power: Statistical power (1 - Type II error rate)
        """
        self.alpha = alpha
        self.power = power
        self.confidence_level = 1 - alpha
        
        # Effect size thresholds (Cohen's conventions)
        self.small_effect = 0.2
        self.medium_effect = 0.5
        self.large_effect = 0.8
        
        # Practical significance thresholds (domain-specific)
        self.practical_thresholds = {
            'throughput': 0.1,  # 10% improvement
            'delay': 0.05,      # 5% improvement
            'energy': 0.1,      # 10% improvement
            'coverage': 0.02,   # 2% improvement
            'fairness': 0.05    # 5% improvement
        }
    
    def compare_two_algorithms(self, algorithm_1_data: List[float], 
                              algorithm_2_data: List[float],
                              algorithm_1_name: str = "Algorithm 1",
                              algorithm_2_name: str = "Algorithm 2",
                              metric_name: str = "performance") -> ComparisonResult:
        """
        Compare two algorithms using appropriate statistical tests
        
        Args:
            algorithm_1_data: Performance data for algorithm 1
            algorithm_2_data: Performance data for algorithm 2
            algorithm_1_name: Name of algorithm 1
            algorithm_2_name: Name of algorithm 2
            metric_name: Name of the metric being compared
            
        Returns:
            ComparisonResult with detailed statistical analysis
        """
        data1 = np.array(algorithm_1_data)
        data2 = np.array(algorithm_2_data)
        
        # Check data validity
        if len(data1) == 0 or len(data2) == 0:
            return self._create_invalid_result(algorithm_1_name, algorithm_2_name, "Empty data")
        
        # Test for normality
        normal1 = self._test_normality(data1)
        normal2 = self._test_normality(data2)
        both_normal = normal1 and normal2
        
        # Test for equal variances (if both normal)
        equal_variances = self._test_equal_variances(data1, data2) if both_normal else False
        
        # Choose appropriate test
        if both_normal and equal_variances:
            # Independent t-test (two-tailed)
            test_stat, p_value = ttest_ind(data1, data2, equal_var=True)
            test_type = "Independent t-test (equal variances)"
            effect_size = self._cohen_d(data1, data2)
            effect_size_type = "Cohen's d"
        elif both_normal and not equal_variances:
            # Welch's t-test
            test_stat, p_value = ttest_ind(data1, data2, equal_var=False)
            test_type = "Welch's t-test (unequal variances)"
            effect_size = self._cohen_d(data1, data2)
            effect_size_type = "Cohen's d"
        else:
            # Mann-Whitney U test (non-parametric)
            test_stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
            test_type = "Mann-Whitney U test"
            effect_size = self._rank_biserial_correlation(data1, data2)
            effect_size_type = "Rank-biserial correlation"
        
        # Determine statistical significance
        significantly_different = p_value < self.alpha
        
        # Calculate confidence interval for difference in means
        confidence_interval = self._calculate_confidence_interval(data1, data2)
        
        # Assess practical significance
        practical_significance = self._assess_practical_significance(
            data1, data2, metric_name, effect_size
        )
        
        # Generate interpretation
        interpretation = self._generate_interpretation(
            algorithm_1_name, algorithm_2_name, p_value, effect_size, 
            significantly_different, practical_significance, test_type
        )
        
        return ComparisonResult(
            algorithm_1=algorithm_1_name,
            algorithm_2=algorithm_2_name,
            test_type=test_type,
            test_statistic=test_stat,
            p_value=p_value,
            effect_size=effect_size,
            effect_size_type=effect_size_type,
            significantly_different=significantly_different,
            confidence_interval=confidence_interval,
            practical_significance=practical_significance,
            interpretation=interpretation
        )
    
    def compare_multiple_algorithms(self, algorithm_data: Dict[str, List[float]],
                                   metric_name: str = "performance") -> MultipleComparisonResult:
        """
        Compare multiple algorithms using appropriate statistical tests
        
        Args:
            algorithm_data: Dictionary mapping algorithm names to performance data
            metric_name: Name of the metric being compared
            
        Returns:
            MultipleComparisonResult with comprehensive analysis
        """
        algorithm_names = list(algorithm_data.keys())
        data_arrays = [np.array(algorithm_data[name]) for name in algorithm_names]
        
        # Check if we have enough algorithms
        if len(algorithm_names) < 2:
            raise ValueError("Need at least 2 algorithms for comparison")
        
        # Test for normality across all groups
        all_normal = all(self._test_normality(data) for data in data_arrays)
        
        # Choose appropriate omnibus test
        if all_normal and len(algorithm_names) <= 10:
            # One-way ANOVA
            test_stat, p_value = stats.f_oneway(*data_arrays)
            test_type = "One-way ANOVA"
        else:
            # Kruskal-Wallis test (non-parametric)
            test_stat, p_value = kruskal(*data_arrays)
            test_type = "Kruskal-Wallis test"
        
        # Post-hoc pairwise comparisons
        post_hoc_results = []
        if p_value < self.alpha:  # Only if omnibus test is significant
            for i in range(len(algorithm_names)):
                for j in range(i + 1, len(algorithm_names)):
                    pairwise_result = self.compare_two_algorithms(
                        data_arrays[i], data_arrays[j],
                        algorithm_names[i], algorithm_names[j], metric_name
                    )
                    # Apply Bonferroni correction
                    num_comparisons = len(algorithm_names) * (len(algorithm_names) - 1) // 2
                    corrected_alpha = self.alpha / num_comparisons
                    pairwise_result.significantly_different = pairwise_result.p_value < corrected_alpha
                    post_hoc_results.append(pairwise_result)
        
        # Rank algorithms by mean performance
        mean_performances = [(name, np.mean(algorithm_data[name])) 
                           for name in algorithm_names]
        ranking = sorted(mean_performances, key=lambda x: x[1], reverse=True)
        
        # Statistical summary
        statistical_summary = self._generate_statistical_summary(algorithm_data)
        
        return MultipleComparisonResult(
            test_type=test_type,
            test_statistic=test_stat,
            p_value=p_value,
            post_hoc_results=post_hoc_results,
            ranking=ranking,
            statistical_summary=statistical_summary
        )
    
    def analyze_distribution(self, data: List[float], 
                           algorithm_name: str, metric_name: str) -> DistributionAnalysis:
        """
        Analyze the distribution of performance data
        
        Args:
            data: Performance data
            algorithm_name: Name of the algorithm
            metric_name: Name of the metric
            
        Returns:
            DistributionAnalysis with comprehensive distribution analysis
        """
        data_array = np.array(data)
        
        # Normality test
        normality_stat, normality_p = shapiro(data_array)
        normality_test = {
            'statistic': normality_stat,
            'p_value': normality_p,
            'is_normal': normality_p > self.alpha
        }
        
        # Descriptive statistics
        descriptive_stats = {
            'mean': np.mean(data_array),
            'median': np.median(data_array),
            'std': np.std(data_array, ddof=1),
            'variance': np.var(data_array, ddof=1),
            'min': np.min(data_array),
            'max': np.max(data_array),
            'range': np.max(data_array) - np.min(data_array),
            'skewness': stats.skew(data_array),
            'kurtosis': stats.kurtosis(data_array),
            'cv': np.std(data_array, ddof=1) / (np.mean(data_array) + 1e-12)
        }
        
        # Determine distribution type
        if normality_test['is_normal']:
            distribution_type = "Normal"
        elif descriptive_stats['skewness'] > 1:
            distribution_type = "Right-skewed"
        elif descriptive_stats['skewness'] < -1:
            distribution_type = "Left-skewed"
        else:
            distribution_type = "Non-normal"
        
        # Detect outliers using IQR method
        q1 = np.percentile(data_array, 25)
        q3 = np.percentile(data_array, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = data_array[(data_array < lower_bound) | (data_array > upper_bound)].tolist()
        
        # Confidence intervals
        confidence_intervals = {
            'mean': stats.t.interval(self.confidence_level, len(data_array) - 1,
                                   loc=descriptive_stats['mean'],
                                   scale=stats.sem(data_array)),
            'median': self._bootstrap_confidence_interval(data_array, np.median)
        }
        
        return DistributionAnalysis(
            algorithm=algorithm_name,
            metric=metric_name,
            normality_test=normality_test,
            descriptive_stats=descriptive_stats,
            distribution_type=distribution_type,
            outliers=outliers,
            confidence_intervals=confidence_intervals
        )
    
    def power_analysis(self, effect_size: float, sample_size: int, 
                      alpha: float = None) -> Dict[str, float]:
        """
        Perform power analysis for study design
        
        Args:
            effect_size: Expected effect size
            sample_size: Sample size per group
            alpha: Significance level (defaults to instance alpha)
            
        Returns:
            Dictionary with power analysis results
        """
        if alpha is None:
            alpha = self.alpha
        
        # Calculate power for two-sample t-test
        delta = effect_size * np.sqrt(sample_size / 2)
        critical_t = stats.t.ppf(1 - alpha/2, 2*sample_size - 2)
        
        # Non-centrality parameter
        ncp = delta * np.sqrt(2*sample_size - 2)
        
        # Power calculation
        power = 1 - stats.nct.cdf(critical_t, 2*sample_size - 2, ncp) + \
                stats.nct.cdf(-critical_t, 2*sample_size - 2, ncp)
        
        # Minimum detectable effect size for given power
        min_effect_size = self._calculate_minimum_effect_size(sample_size, alpha, self.power)
        
        # Required sample size for desired power
        required_n = self._calculate_required_sample_size(effect_size, alpha, self.power)
        
        return {
            'power': power,
            'effect_size': effect_size,
            'sample_size_per_group': sample_size,
            'alpha': alpha,
            'minimum_detectable_effect': min_effect_size,
            'required_sample_size': required_n
        }
    
    def generate_statistical_report(self, comparison_results: List[ComparisonResult],
                                   multiple_comparison: MultipleComparisonResult = None) -> str:
        """
        Generate comprehensive statistical report
        
        Args:
            comparison_results: List of pairwise comparison results
            multiple_comparison: Multiple comparison result
            
        Returns:
            Formatted statistical report
        """
        report = "# Statistical Analysis Report\n\n"
        
        # Executive summary
        report += "## Executive Summary\n\n"
        significant_comparisons = [r for r in comparison_results if r.significantly_different]
        report += f"- Total comparisons: {len(comparison_results)}\n"
        report += f"- Statistically significant: {len(significant_comparisons)}\n"
        report += f"- Significance level: α = {self.alpha}\n"
        report += f"- Confidence level: {self.confidence_level:.1%}\n\n"
        
        # Multiple comparison results
        if multiple_comparison:
            report += "## Overall Comparison\n\n"
            report += f"**Test:** {multiple_comparison.test_type}\n"
            report += f"**Test Statistic:** {multiple_comparison.test_statistic:.4f}\n"
            report += f"**p-value:** {multiple_comparison.p_value:.4e}\n"
            report += f"**Significant:** {'Yes' if multiple_comparison.p_value < self.alpha else 'No'}\n\n"
            
            report += "### Algorithm Ranking\n\n"
            for i, (name, score) in enumerate(multiple_comparison.ranking, 1):
                report += f"{i}. {name}: {score:.4f}\n"
            report += "\n"
        
        # Pairwise comparisons
        report += "## Pairwise Comparisons\n\n"
        for result in comparison_results:
            report += f"### {result.algorithm_1} vs {result.algorithm_2}\n\n"
            report += f"- **Test:** {result.test_type}\n"
            report += f"- **Test Statistic:** {result.test_statistic:.4f}\n"
            report += f"- **p-value:** {result.p_value:.4e}\n"
            report += f"- **Effect Size:** {result.effect_size:.4f} ({result.effect_size_type})\n"
            report += f"- **Statistically Significant:** {'Yes' if result.significantly_different else 'No'}\n"
            report += f"- **Practically Significant:** {'Yes' if result.practical_significance else 'No'}\n"
            report += f"- **95% CI:** [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]\n"
            report += f"- **Interpretation:** {result.interpretation}\n\n"
        
        # Effect size interpretation guide
        report += "## Effect Size Interpretation\n\n"
        report += "**Cohen's d:**\n"
        report += "- Small effect: d = 0.2\n"
        report += "- Medium effect: d = 0.5\n"
        report += "- Large effect: d = 0.8\n\n"
        
        report += "**Rank-biserial correlation:**\n"
        report += "- Small effect: r = 0.1\n"
        report += "- Medium effect: r = 0.3\n"
        report += "- Large effect: r = 0.5\n\n"
        
        return report
    
    def _test_normality(self, data: np.ndarray, min_samples: int = 3) -> bool:
        """Test if data follows normal distribution"""
        if len(data) < min_samples:
            return False  # Assume non-normal for very small samples
        
        try:
            _, p_value = shapiro(data)
            return p_value > self.alpha
        except:
            return False
    
    def _test_equal_variances(self, data1: np.ndarray, data2: np.ndarray) -> bool:
        """Test if two samples have equal variances"""
        try:
            _, p_value = levene(data1, data2)
            return p_value > self.alpha
        except:
            return False
    
    def _cohen_d(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""
        mean1, mean2 = np.mean(data1), np.mean(data2)
        std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
        n1, n2 = len(data1), len(data2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    def _rank_biserial_correlation(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate rank-biserial correlation effect size"""
        n1, n2 = len(data1), len(data2)
        
        # Mann-Whitney U statistic
        u_stat, _ = mannwhitneyu(data1, data2, alternative='two-sided')
        
        # Convert to rank-biserial correlation
        r = 1 - (2 * u_stat) / (n1 * n2)
        
        return abs(r)  # Return absolute value for effect size magnitude
    
    def _calculate_confidence_interval(self, data1: np.ndarray, data2: np.ndarray) -> Tuple[float, float]:
        """Calculate confidence interval for difference in means"""
        mean1, mean2 = np.mean(data1), np.mean(data2)
        sem1, sem2 = stats.sem(data1), stats.sem(data2)
        
        diff_mean = mean1 - mean2
        se_diff = np.sqrt(sem1**2 + sem2**2)
        
        # Degrees of freedom (Welch's approximation)
        df = (sem1**2 + sem2**2)**2 / (sem1**4/(len(data1)-1) + sem2**4/(len(data2)-1))
        
        # t-critical value
        t_crit = stats.t.ppf(1 - self.alpha/2, df)
        
        margin_error = t_crit * se_diff
        
        return (diff_mean - margin_error, diff_mean + margin_error)
    
    def _assess_practical_significance(self, data1: np.ndarray, data2: np.ndarray,
                                      metric_name: str, effect_size: float) -> bool:
        """Assess practical significance of difference"""
        # Use effect size threshold
        effect_threshold = self.large_effect  # Conservative approach
        
        # Use domain-specific threshold if available
        for key, threshold in self.practical_thresholds.items():
            if key in metric_name.lower():
                mean1, mean2 = np.mean(data1), np.mean(data2)
                relative_improvement = abs(mean1 - mean2) / (max(mean1, mean2) + 1e-12)
                return relative_improvement > threshold
        
        # Fallback to effect size
        return abs(effect_size) > effect_threshold
    
    def _generate_interpretation(self, alg1: str, alg2: str, p_value: float,
                               effect_size: float, stat_sig: bool, prac_sig: bool,
                               test_type: str) -> str:
        """Generate human-readable interpretation"""
        if not stat_sig:
            return f"No statistically significant difference between {alg1} and {alg2} (p = {p_value:.3f})."
        
        # Determine effect magnitude
        abs_effect = abs(effect_size)
        if abs_effect < self.small_effect:
            effect_desc = "negligible"
        elif abs_effect < self.medium_effect:
            effect_desc = "small"
        elif abs_effect < self.large_effect:
            effect_desc = "medium"
        else:
            effect_desc = "large"
        
        # Determine direction
        if effect_size > 0:
            direction = f"{alg1} outperforms {alg2}"
        else:
            direction = f"{alg2} outperforms {alg1}"
        
        interpretation = f"Statistically significant difference: {direction} with {effect_desc} effect size."
        
        if prac_sig:
            interpretation += " The difference is also practically significant."
        else:
            interpretation += " However, the difference may not be practically significant."
        
        return interpretation
    
    def _bootstrap_confidence_interval(self, data: np.ndarray, statistic_func,
                                      n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval"""
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic_func(bootstrap_sample))
        
        lower_percentile = (1 - self.confidence_level) / 2 * 100
        upper_percentile = (1 + self.confidence_level) / 2 * 100
        
        return (np.percentile(bootstrap_stats, lower_percentile),
                np.percentile(bootstrap_stats, upper_percentile))
    
    def _calculate_minimum_effect_size(self, sample_size: int, alpha: float, power: float) -> float:
        """Calculate minimum detectable effect size"""
        # Simplified calculation for two-sample t-test
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        effect_size = (z_alpha + z_beta) * np.sqrt(2 / sample_size)
        return effect_size
    
    def _calculate_required_sample_size(self, effect_size: float, alpha: float, power: float) -> int:
        """Calculate required sample size per group"""
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        return int(np.ceil(n))
    
    def _generate_statistical_summary(self, algorithm_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Generate statistical summary for multiple algorithms"""
        summary = {}
        
        for name, data in algorithm_data.items():
            data_array = np.array(data)
            summary[name] = {
                'n': len(data_array),
                'mean': np.mean(data_array),
                'std': np.std(data_array, ddof=1),
                'median': np.median(data_array),
                'min': np.min(data_array),
                'max': np.max(data_array),
                'q25': np.percentile(data_array, 25),
                'q75': np.percentile(data_array, 75)
            }
        
        return summary
    
    def _create_invalid_result(self, alg1: str, alg2: str, reason: str) -> ComparisonResult:
        """Create invalid comparison result"""
        return ComparisonResult(
            algorithm_1=alg1,
            algorithm_2=alg2,
            test_type="Invalid",
            test_statistic=0.0,
            p_value=1.0,
            effect_size=0.0,
            effect_size_type="None",
            significantly_different=False,
            confidence_interval=(0.0, 0.0),
            practical_significance=False,
            interpretation=f"Invalid comparison: {reason}"
        )