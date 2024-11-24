import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
class TrainingMetrics:
    def __init__(self, config):
        self.config = config
        self.current_episode = 0

        self.metrics = {
            'coverage_metrics': {
                'avg_coverage': [],
                'peak_coverage': [],
                'coverage_stability': [],
                'priority_coverage': []
            },
            'energy_metrics': {
                'avg_energy_consumption': [],
                'charging_frequency': [],
                'energy_efficiency': [],
                'low_energy_incidents': []
            },
            'cooperation_metrics': {
                'communication_density': [],
                'overlap_ratio': [],
                'task_sharing': [],
                'formation_stability': []
            },
            'task_metrics': {
                'mission_completion_rate': [],
                'avg_response_time': [],
                'priority_handling': [],
                'coverage_gaps': []
            },
            'system_metrics': {
                'collision_counts': [],
                'path_efficiency': [],
                'load_balance': [],
                'resource_utilization': []
            }
        }

        # 创建指标保存目录
        self.metrics_dir = os.path.join(config.base_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)

    def update(self, env, episode_info):
        """更新所有指标"""
        self.current_episode += 1

        # 从 episode_info 提取所需数据
        coverage = np.array(episode_info['coverage'])
        energy = np.array(episode_info['energy'])
        positions = episode_info['positions']
        collisions = episode_info['collisions']

        # 确保coverage数据的正确性
        if len(coverage.shape) == 1:
            # 如果是一维数组，直接使用
            covered_pois = coverage
        else:
            # 如果是二维数组，取最后一个时间步
            covered_pois = coverage[-1]

        # 更新各类指标
        self._update_coverage_metrics(env, covered_pois, positions)
        self._update_energy_metrics(env, energy)
        self._update_cooperation_metrics(env, positions)
        self._update_task_metrics(env, covered_pois, energy, positions)
        self._update_system_metrics(env, collisions, positions)

        # 定期保存指标
        if self.current_episode % self.config.log_frequency == 0:
            self.save_metrics()

    def _update_coverage_metrics(self, env, coverage, positions):
        """更新覆盖率相关指标，添加空值处理"""
        if isinstance(coverage, list):
            coverage_array = np.array(coverage)
        else:
            coverage_array = np.array([coverage])

        # 添加安全检查
        if coverage_array.size == 0:
            # 如果数组为空，使用默认值
            avg_coverage = 0.0
            peak_coverage = 0.0
            coverage_stability = 0.0
            priority_coverage = 0.0
        else:
            # 计算平均覆盖率
            avg_coverage = np.mean(coverage_array)
            peak_coverage = np.max(coverage_array)
            coverage_stability = np.std(coverage_array) if len(coverage_array) > 1 else 0.0

            # 获取最后一个时间步的覆盖情况
            if len(coverage_array.shape) > 1:
                latest_coverage = coverage_array[-1]
            else:
                latest_coverage = coverage_array

            # 计算高优先级POI的覆盖率
            high_priority_threshold = self.config.poi_priority_levels * 0.7
            high_priority_mask = env.poi_priorities >= high_priority_threshold

            # 确保维度匹配并处理空数组情况
            if len(latest_coverage) == len(high_priority_mask) and np.any(high_priority_mask):
                priority_coverage = np.mean(latest_coverage[high_priority_mask])
            else:
                # 如果没有高优先级POI或维度不匹配，使用当前覆盖率
                covered_pois = env._get_covered_pois()
                if np.any(high_priority_mask):
                    priority_coverage = np.mean(covered_pois[high_priority_mask])
                else:
                    priority_coverage = 0.0

        # 存储指标，确保使用float类型
        self.metrics['coverage_metrics']['avg_coverage'].append(float(avg_coverage))
        self.metrics['coverage_metrics']['peak_coverage'].append(float(peak_coverage))
        self.metrics['coverage_metrics']['coverage_stability'].append(float(coverage_stability))
        self.metrics['coverage_metrics']['priority_coverage'].append(float(priority_coverage))

    def _update_energy_metrics(self, env, energy):
        """更新能量相关指标，添加空值处理"""
        if isinstance(energy, list):
            energy_array = np.array(energy)
        else:
            energy_array = np.array([energy])

        # 添加安全检查
        if energy_array.size == 0:
            avg_consumption = 0.0
            low_energy_count = 0
            energy_efficiency = 0.0
        else:
            avg_consumption = np.mean(energy_array)
            low_energy_count = np.sum(energy_array < env.uav_energy_capacity * 0.2)

            # 安全计算能量效率
            coverage = np.mean(env._get_covered_pois())
            if avg_consumption > 0:
                energy_efficiency = coverage / avg_consumption
            else:
                energy_efficiency = 0.0

        self.metrics['energy_metrics']['avg_energy_consumption'].append(float(avg_consumption))
        self.metrics['energy_metrics']['charging_frequency'].append(float(env.episode_data['charging_events']))
        self.metrics['energy_metrics']['energy_efficiency'].append(float(energy_efficiency))
        self.metrics['energy_metrics']['low_energy_incidents'].append(float(low_energy_count))

    def _safe_mean(self, array):
        """安全计算平均值，处理空数组和除零情况"""
        if isinstance(array, list):
            array = np.array(array)

        if array.size == 0:
            return 0.0

        # 使用np.nanmean来处理可能的NaN值
        result = np.nanmean(array)

        # 检查是否是nan或inf
        if np.isnan(result) or np.isinf(result):
            return 0.0

        return float(result)

    def _update_cooperation_metrics(self, env, positions):
        """更新协作相关指标，添加空值处理"""
        # 计算通信密度
        comm_density = env._calculate_communication_density()

        # 计算覆盖重叠
        total_possible_links = env.num_agents * (env.num_agents - 1) / 2
        if total_possible_links > 0:
            overlap = len(env._get_communication_links()) / total_possible_links
        else:
            overlap = 0.0

        # 计算负载均衡
        load_balance = env._calculate_load_balance()

        # 存储指标
        self.metrics['cooperation_metrics']['communication_density'].append(float(comm_density))
        self.metrics['cooperation_metrics']['overlap_ratio'].append(float(overlap))
        self.metrics['cooperation_metrics']['task_sharing'].append(float(load_balance))
        self.metrics['cooperation_metrics']['formation_stability'].append(float(comm_density * load_balance))

    def _update_task_metrics(self, env, covered_pois, energy, positions):
        """更新任务相关指标，添加空值处理"""
        # 安全计算任务完成率
        mission_completion = self._safe_mean(env._get_covered_pois() > 0.8)

        # 计算平均响应时间（使用能量效率作为代理指标）
        energy_array = np.array(energy)
        if energy_array.size > 0:
            avg_response = np.mean(energy_array) / env.uav_energy_capacity
        else:
            avg_response = 0.0

        # 计算优先级处理效率
        if 'priority_coverage' in self.metrics['coverage_metrics'] and \
                len(self.metrics['coverage_metrics']['priority_coverage']) > 0:
            priority_handling = self.metrics['coverage_metrics']['priority_coverage'][-1]
        else:
            priority_handling = 0.0

        # 计算覆盖空白
        if 'avg_coverage' in self.metrics['coverage_metrics'] and \
                len(self.metrics['coverage_metrics']['avg_coverage']) > 0:
            coverage_gaps = 1.0 - self.metrics['coverage_metrics']['avg_coverage'][-1]
        else:
            coverage_gaps = 1.0

        # 存储指标
        self.metrics['task_metrics']['mission_completion_rate'].append(float(mission_completion))
        self.metrics['task_metrics']['avg_response_time'].append(float(avg_response))
        self.metrics['task_metrics']['priority_handling'].append(float(priority_handling))
        self.metrics['task_metrics']['coverage_gaps'].append(float(coverage_gaps))
    def _update_system_metrics(self, env, collisions, positions):
        """更新系统效率指标"""
        path_efficiency = env._calculate_resource_utilization()
        load_balance = env._calculate_load_balance()
        resource_util = (path_efficiency + load_balance) / 2

        self.metrics['system_metrics']['collision_counts'].append(float(collisions))
        self.metrics['system_metrics']['path_efficiency'].append(float(path_efficiency))
        self.metrics['system_metrics']['load_balance'].append(float(load_balance))
        self.metrics['system_metrics']['resource_utilization'].append(float(resource_util))

    def save_metrics(self):
        """保存指标数据"""
        save_path = os.path.join(self.metrics_dir, f"metrics_episode_{self.current_episode}.json")
        np.save(save_path, self.metrics)

    def get_summary(self):
        """生成当前训练状态摘要"""
        summary = {}
        for category in self.metrics:
            summary[category] = {}
            for metric, values in self.metrics[category].items():
                if values:
                    summary[category][metric] = {
                        'current': values[-1],
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
        return summary

    def plot_metrics(self, save_path=None):
        """绘制训练指标图表"""
        fig = make_subplots(
            rows=5, cols=1,
            subplot_titles=('Coverage Metrics', 'Energy Metrics',
                            'Cooperation Metrics', 'Task Metrics',
                            'System Metrics')
        )

        row = 1
        for category in self.metrics:
            for metric, values in self.metrics[category].items():
                if values:
                    fig.add_trace(
                        go.Scatter(y=values,
                                   name=metric,
                                   mode='lines+markers'),
                        row=row, col=1
                    )
            row += 1

        fig.update_layout(
            height=1500,
            showlegend=True,
            title_text="Training Metrics Overview"
        )

        if save_path:
            fig.write_html(save_path)

        return fig