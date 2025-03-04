from utils.visualizer import Visualizer
from utils.config import Config
import numpy as np
import json
import os
from datetime import datetime
import glob
import traceback
import sys


class TrainingVisualizer:
    def __init__(self):
        self.config = Config(mode='test')
        self.visualizer = Visualizer(self.config)
        self.runs_data = {}  # 存储所有训练运行的数据

        # 添加调试信息输出
        self.debug_mode = True
        self.debug_log_path = "viz_debug.log"
        self.init_debug_log()

    def debug_print(self, message):
        """输出调试信息并记录到文件"""
        if self.debug_mode:
            timestamp = datetime.now().strftime("%H:%M:%S")
            debug_msg = f"[{timestamp}] {message}"
            print(debug_msg)
            with open(self.debug_log_path, 'a', encoding='utf-8') as f:
                f.write(debug_msg + "\n")

    def init_debug_log(self):
        """初始化调试日志文件"""
        if self.debug_mode:
            with open(self.debug_log_path, 'w', encoding='utf-8') as f:
                f.write(f"=== 可视化调试日志开始 {datetime.now()} ===\n")
                f.write(f"Python版本: {sys.version}\n")
                f.write(f"NumPy版本: {np.__version__}\n\n")

    def scan_training_runs(self, base_dir="./runs"):
        """扫描所有训练结果"""
        self.debug_print(f"扫描训练目录: {base_dir}")
        runs = glob.glob(os.path.join(base_dir, "202*"))  # 匹配所有以202开头的目录
        self.debug_print(f"找到 {len(runs)} 个可能的训练运行目录")
        available_runs = []

        for run_dir in runs:
            run_id = os.path.basename(run_dir)
            try:
                # 检查是否存在必要的目录
                metrics_dir = os.path.join(run_dir, 'metrics')
                training_dir = os.path.join(run_dir, 'training_data')

                # 更宽松的文件检查 - 只要目录存在就认为可能有数据
                if os.path.exists(metrics_dir) or os.path.exists(training_dir):
                    # 获取训练时间
                    try:
                        timestamp = datetime.strptime(run_id.split('_')[0], '%Y%m%d')
                    except:
                        timestamp = datetime.now()

                    # 收集可能的指标和训练文件
                    metrics_files = []
                    if os.path.exists(metrics_dir):
                        metrics_files = glob.glob(os.path.join(metrics_dir, '*.npy'))

                    episode_files = []
                    if os.path.exists(training_dir):
                        episode_files = glob.glob(os.path.join(training_dir, '*.json'))

                    # 读取final_report获取训练信息
                    report_path = os.path.join(run_dir, 'final_report.json')
                    train_info = {}
                    if os.path.exists(report_path):
                        with open(report_path, 'r') as f:
                            try:
                                train_info = json.load(f)
                            except:
                                self.debug_print(f"警告: 无法解析 {report_path}")

                    # 加入可用运行列表
                    available_runs.append({
                        'run_id': run_id,
                        'timestamp': timestamp,
                        'path': run_dir,
                        'num_metrics_files': len(metrics_files),
                        'num_episode_files': len(episode_files),
                        'info': train_info
                    })

                    # 打印找到的文件
                    self.debug_print(f"找到运行: {run_id}")
                    self.debug_print(f"  指标文件: {len(metrics_files)}")
                    if metrics_files:
                        self.debug_print(f"    示例: {[os.path.basename(f) for f in metrics_files[:3]]}")
                    self.debug_print(f"  Episode文件: {len(episode_files)}")
                    if episode_files:
                        self.debug_print(f"    示例: {[os.path.basename(f) for f in episode_files[:3]]}")
            except Exception as e:
                self.debug_print(f"警告: 处理运行 {run_id} 时出错: {str(e)}")
                self.debug_print(traceback.format_exc())

        return sorted(available_runs, key=lambda x: x['timestamp'], reverse=True)

    def extract_episode_data_from_filename(self, filename):
        """从文件名中提取episode编号，更健壮的方法"""
        basename = os.path.basename(filename)

        # 尝试不同的文件名模式
        try:
            # 尝试'episode_X.json'格式
            if 'episode_' in basename:
                parts = basename.split('episode_')
                if len(parts) > 1:
                    number_part = parts[1].split('.')[0]
                    return int(number_part)

            # 尝试'metrics_episode_X.json.npy'格式
            if 'metrics_episode_' in basename:
                parts = basename.split('metrics_episode_')
                if len(parts) > 1:
                    number_part = parts[1].split('.')[0]
                    return int(number_part)

            # 尝试'X_其他内容.json'格式
            parts = basename.split('_')
            if parts and parts[0].isdigit():
                return int(parts[0])

            # 尝试'其他内容_X.其他'格式
            last_part = basename.split('_')[-1].split('.')[0]
            if last_part.isdigit():
                return int(last_part)

        except (ValueError, IndexError) as e:
            self.debug_print(f"警告: 无法从 {basename} 提取episode编号: {e}")

        return None  # 无法提取有效编号

    def load_episode_data(self, episode_file):
        """加载单个episode数据文件，更健壮的方法"""
        try:
            with open(episode_file, 'r') as f:
                data = json.load(f)

            # 直接从文件内容中获取episode编号
            if isinstance(data, dict) and 'episode' in data:
                episode_num = int(data['episode'])  # 确保是整数类型
                self.debug_print(f"从文件内容提取episode编号: {episode_num}")
            else:
                # 从文件名中提取
                episode_num = self.extract_episode_data_from_filename(episode_file)
                self.debug_print(f"从文件名提取episode编号: {episode_num}")

            return episode_num, data
        except Exception as e:
            self.debug_print(f"加载episode数据文件出错 {episode_file}: {e}")
            self.debug_print(traceback.format_exc())
            return None, None

    def load_metrics_data(self, metrics_file):
        """加载单个metrics数据文件，更健壮的方法"""
        try:
            data = np.load(metrics_file, allow_pickle=True).item()
            episode_num = self.extract_episode_data_from_filename(metrics_file)
            return episode_num, data
        except Exception as e:
            self.debug_print(f"加载metrics数据文件出错 {metrics_file}: {e}")
            self.debug_print(traceback.format_exc())
            return None, None

    def load_run_data(self, run_dir):
        """加载特定训练运行的数据，更健壮的方法"""
        try:
            self.debug_print(f"\n加载数据目录: {run_dir}")

            # 初始化数据结构
            metrics_data = {}
            episode_data = {}
            final_report_data = None

            # 加载metrics数据
            metrics_dir = os.path.join(run_dir, 'metrics')
            if os.path.exists(metrics_dir):
                metrics_files = glob.glob(os.path.join(metrics_dir, '*.npy'))
                self.debug_print(f"找到 {len(metrics_files)} 个metrics文件")

                for mf in metrics_files:
                    episode_num, data = self.load_metrics_data(mf)
                    if episode_num is not None and data is not None:
                        metrics_data[int(episode_num)] = data
                        self.debug_print(f"  加载了episode {episode_num}的metrics数据")
            else:
                self.debug_print("没有找到metrics目录")

            # 加载episode数据
            training_dir = os.path.join(run_dir, 'training_data')
            if os.path.exists(training_dir):
                episode_files = glob.glob(os.path.join(training_dir, '*.json'))
                self.debug_print(f"找到 {len(episode_files)} 个episode文件")

                for ef in episode_files:
                    episode_num, data = self.load_episode_data(ef)
                    if episode_num is not None and data is not None:
                        episode_data[int(episode_num)] = data
                        self.debug_print(f"  加载了episode {episode_num}的数据")
            else:
                self.debug_print("没有找到training_data目录")

            # 加载final_report.json数据
            final_report_path = os.path.join(run_dir, 'final_report.json')
            if os.path.exists(final_report_path):
                try:
                    with open(final_report_path, 'r') as f:
                        final_report_data = json.load(f)
                    self.debug_print("  加载了final_report数据")
                except Exception as e:
                    self.debug_print(f"  加载final_report出错: {e}")

            # 确保至少有一些数据可用
            if not metrics_data and not episode_data and not final_report_data:
                self.debug_print("警告: 在这个运行中没有找到可用数据!")
                return None

            # 创建环境可视化数据结构
            env_data = self._generate_env_data_from_episodes(episode_data, final_report_data)

            # 打印调试信息
            self.debug_print(f"env_data键列表: {list(env_data.keys())}")
            self.debug_print(f"env_data键类型: {[type(k) for k in env_data.keys()]}")
            if env_data:
                first_key = list(env_data.keys())[0]
                self.debug_print(f"第一个时间步键列表: {list(env_data[first_key].keys())}")

            # 汇总加载的数据
            self.debug_print(f"成功加载:")
            self.debug_print(f"  {len(metrics_data)} 个metrics条目")
            self.debug_print(f"  {len(episode_data)} 个episode条目")
            self.debug_print(f"  {len(env_data)} 个环境数据条目")

            return {
                'metrics': metrics_data,
                'episodes': episode_data,
                'env_data': env_data,
                'final_report': final_report_data
            }
        except Exception as e:
            self.debug_print(f"加载运行数据时出错: {str(e)}")
            self.debug_print(traceback.format_exc())
            return None

    def _generate_env_data_from_episodes(self, episode_data, final_report=None):
        """从episode数据生成环境可视化数据结构，为每个episode创建多个时间步"""
        try:
            self.debug_print("生成环境数据...")
            env_data = {}

            # 对每个episode创建模拟的环境数据
            for episode_num, ep_data in episode_data.items():
                # 确保键是整数
                episode_key = int(episode_num)
                env_data[episode_key] = {}

                # 为每个episode创建MAX_TIME_STEPS个时间步，使用插值生成连续变化的数据
                max_time_steps = self.config.max_time_steps

                # 创建基础数据 - 所有时间步共享的静态数据
                base_data = self._construct_base_env_data(ep_data)

                # 创建每个时间步的数据 - 用插值模拟动态变化
                for t in range(max_time_steps):
                    # 设置进度比例 (0.0 - 1.0)
                    progress = t / max_time_steps if max_time_steps > 1 else 0.5

                    # 创建当前时间步的环境数据，深拷贝基础数据并添加动态部分
                    timestep_data = self._create_timestep_data(base_data, progress, ep_data)

                    # 存储这个时间步的数据 - 确保时间步键是整数
                    timestep_key = int(t)
                    env_data[episode_key][timestep_key] = timestep_data

                self.debug_print(f"  为episode {episode_key}生成了{max_time_steps}个时间步的可视化数据")

            # 如果没有episodes数据，但有final_report，至少创建一个默认episode
            if not env_data and final_report is not None:
                default_data = {
                    'episode': 0,
                    'coverage': final_report.get('final_metrics', {}).get('coverage_metrics', {}).get('avg_coverage',
                                                                                                      {}).get('current',
                                                                                                              0.5),
                    'energy': final_report.get('final_metrics', {}).get('energy_metrics', {}).get(
                        'avg_energy_consumption', {}).get('current', 500),
                    'collisions': final_report.get('final_metrics', {}).get('system_metrics', {}).get(
                        'collision_counts', {}).get('current', 0),
                    'rewards': final_report.get('best_reward', 0),
                    'timestamp': datetime.now().isoformat()
                }

                # 使用整数0作为键
                env_data[0] = {}
                base_data = self._construct_base_env_data(default_data)

                for t in range(self.config.max_time_steps):
                    progress = t / self.config.max_time_steps if self.config.max_time_steps > 1 else 0.5
                    timestep_data = self._create_timestep_data(base_data, progress, default_data)
                    # 使用整数作为时间步键
                    env_data[0][int(t)] = timestep_data

                self.debug_print(f"  从final_report生成了默认可视化数据")

            # 确保所有键都是整数类型
            standardized_env_data = {}
            for ep_key, ep_value in env_data.items():
                standardized_timesteps = {}
                for ts_key, ts_value in ep_value.items():
                    standardized_timesteps[int(ts_key)] = ts_value
                standardized_env_data[int(ep_key)] = standardized_timesteps

            self.debug_print(f"环境数据生成完成，键类型: {[type(k) for k in standardized_env_data.keys()]}")
            return standardized_env_data

        except Exception as e:
            self.debug_print(f"生成环境数据时出错: {e}")
            self.debug_print(traceback.format_exc())
            return {}

    def _construct_base_env_data(self, episode_data):
        """构建基础环境数据"""
        # 创建随机但一致的位置数据
        np.random.seed(int(episode_data.get('episode', 0)))

        # 根据episode获取随机但一致的实体位置
        area_size = self.config.area_size
        satellites = np.random.uniform(0, area_size, (self.config.num_satellites, 2))
        uavs = np.random.uniform(0, area_size, (self.config.num_uavs, 2))
        ground_stations = np.random.uniform(0, area_size, (self.config.num_ground_stations, 2))
        pois = np.random.uniform(0, area_size, (self.config.num_pois, 2))
        poi_priorities = np.random.randint(1, self.config.poi_priority_levels + 1, self.config.num_pois)

        # 返回基础数据
        return {
            'satellites': satellites,
            'uavs': uavs,
            'ground_stations': ground_stations,
            'pois': pois,
            'poi_priorities': poi_priorities,
            'episode_data': episode_data
        }

    def _create_timestep_data(self, base_data, progress, episode_data):
        """为特定时间步创建环境数据"""
        # 提取基础数据
        satellites = base_data['satellites'].copy()
        uavs = base_data['uavs'].copy()
        ground_stations = base_data['ground_stations'].copy()
        pois = base_data['pois'].copy()
        poi_priorities = base_data['poi_priorities'].copy()

        # 根据进度创建动态变化
        # 1. 卫星沿轨道移动 (简单的圆形轨道)
        area_size = self.config.area_size
        center = area_size / 2
        for i in range(len(satellites)):
            angle = 2 * np.pi * progress + i * np.pi / len(satellites)
            radius = area_size * 0.3
            satellites[i, 0] = center + radius * np.cos(angle)
            satellites[i, 1] = center + radius * np.sin(angle)

        # 2. UAV进行简单的路径规划移动
        for i in range(len(uavs)):
            # 从起点到终点的简单线性插值
            start_pos = uavs[i].copy()
            # 随机但确定性的终点
            np.random.seed(int(episode_data.get('episode', 0)) * 100 + i)
            end_pos = np.random.uniform(0, area_size, 2)
            # 插值当前位置
            uavs[i] = start_pos + progress * (end_pos - start_pos)

        # 3. 地面站保持相对静止

        # 创建覆盖图
        coverage_map = self._calculate_coverage(satellites, uavs, ground_stations)

        # 计算通信链接
        communication_links = self._calculate_communication_links(satellites, uavs, ground_stations)

        # 构建完整的时间步数据
        coverage_percentage = float(episode_data.get('coverage', 0)) * 100
        collision_count = int(episode_data.get('collisions', 0))
        total_reward = float(episode_data.get('rewards', 0))
        avg_uav_energy = float(episode_data.get('energy', 500))

        return {
            'satellites': satellites,
            'uavs': uavs,
            'ground_stations': ground_stations,
            'pois': pois,
            'poi_priorities': poi_priorities,
            'coverage': coverage_map,
            'coverage_percentage': coverage_percentage,
            'communication_links': communication_links,
            'total_reward': total_reward,
            'avg_uav_energy': avg_uav_energy,
            'collision_count': collision_count
        }

    def _calculate_coverage(self, satellites, uavs, ground_stations):
        """计算覆盖图"""
        area_size = self.config.area_size
        coverage = np.zeros((area_size, area_size))

        try:
            # 简化的覆盖计算 - 使用高斯径向基函数
            x = np.arange(area_size)
            y = np.arange(area_size)
            xx, yy = np.meshgrid(x, y)

            # 为每个智能体添加覆盖
            for pos, range_val in [
                (satellites, self.config.satellite_range),
                (uavs, self.config.uav_range),
                (ground_stations, self.config.ground_station_range)
            ]:
                for agent_pos in pos:
                    dist = np.sqrt((xx - agent_pos[0]) ** 2 + (yy - agent_pos[1]) ** 2)
                    coverage += (dist <= range_val).astype(float)
        except Exception as e:
            self.debug_print(f"计算覆盖图时出错: {e}")
            self.debug_print(traceback.format_exc())

        return coverage

    def _calculate_communication_links(self, satellites, uavs, ground_stations):
        """计算通信链接"""
        links = []
        comm_range = self.config.communication_range

        try:
            # 合并所有智能体
            all_agents = []
            # 卫星
            for i, pos in enumerate(satellites):
                all_agents.append({
                    'type': 'satellites',
                    'id': i,
                    'position': pos
                })
            # UAV
            for i, pos in enumerate(uavs):
                all_agents.append({
                    'type': 'uavs',
                    'id': i,
                    'position': pos
                })
            # 地面站
            for i, pos in enumerate(ground_stations):
                all_agents.append({
                    'type': 'ground_stations',
                    'id': i,
                    'position': pos
                })

            # 计算链接
            for i in range(len(all_agents)):
                for j in range(i + 1, len(all_agents)):
                    pos1 = all_agents[i]['position']
                    pos2 = all_agents[j]['position']
                    dist = np.linalg.norm(pos1 - pos2)

                    if dist <= comm_range:
                        # 计算信号强度
                        signal_strength = 1.0 - (dist / comm_range)

                        links.append({
                            'points': (pos1, pos2),
                            'agents': (all_agents[i], all_agents[j]),
                            'signal_strength': signal_strength
                        })
        except Exception as e:
            self.debug_print(f"计算通信链接时出错: {e}")
            self.debug_print(traceback.format_exc())

        return links

    def _prepare_performance_data(self, run_data):
        """准备性能指标数据"""
        try:
            self.debug_print("准备性能指标数据...")
            # 初始化性能数据结构
            performance_data = {
                'episodes': [],
                'rewards': [],
                'coverages': [],
                'energies': [],
                'collision_counts': []
            }

            # 从episodes中获取基本数据
            episodes = run_data.get('episodes', {})
            if episodes:
                # 按照episode编号排序
                sorted_episodes = sorted(episodes.keys())
                self.debug_print(f"排序后的episode编号: {sorted_episodes}")

                for ep_num in sorted_episodes:
                    ep_data = episodes[ep_num]
                    performance_data['episodes'].append(float(ep_num))

                    # 提取各项指标
                    if 'rewards' in ep_data:
                        performance_data['rewards'].append(float(ep_data['rewards']))
                    else:
                        performance_data['rewards'].append(0.0)

                    if 'coverage' in ep_data:
                        performance_data['coverages'].append(float(ep_data['coverage']) * 100)  # 转换为百分比
                    else:
                        performance_data['coverages'].append(0.0)

                    if 'energy' in ep_data:
                        performance_data['energies'].append(float(ep_data['energy']))
                    else:
                        performance_data['energies'].append(0.0)

                    if 'collisions' in ep_data:
                        performance_data['collision_counts'].append(int(ep_data['collisions']))
                    else:
                        performance_data['collision_counts'].append(0)

            # 确保所有列表长度一致
            if performance_data['episodes']:
                max_len = len(performance_data['episodes'])
                for key in performance_data:
                    if key != 'episodes':
                        while len(performance_data[key]) < max_len:
                            # 使用最后一个值或0填充
                            last_val = performance_data[key][-1] if performance_data[key] else 0
                            performance_data[key].append(last_val)

            # 如果完全没有数据，创建一个虚拟的单点数据
            if not performance_data['episodes']:
                performance_data['episodes'] = [0.0]
                performance_data['rewards'] = [0.0]
                performance_data['coverages'] = [0.0]
                performance_data['energies'] = [0.0]
                performance_data['collision_counts'] = [0.0]

                # 如果有final_report，至少使用它的数据
                final_report = run_data.get('final_report')
                if final_report and 'best_reward' in final_report:
                    performance_data['rewards'][0] = float(final_report['best_reward'])

            # 转换为NumPy数组
            for key in performance_data:
                performance_data[key] = np.array(performance_data[key])
                self.debug_print(f"{key}: {performance_data[key]}")

            # 更新可视化器的性能数据
            self.visualizer.performance_data = performance_data

            self.debug_print(f"准备了{len(performance_data['episodes'])}个点的性能数据")

        except Exception as e:
            self.debug_print(f"准备性能数据时出错: {str(e)}")
            self.debug_print(traceback.format_exc())

            # 确保至少有一些数据可视化
            self.visualizer.performance_data = {
                'episodes': np.array([0.0]),
                'rewards': np.array([0.0]),
                'coverages': np.array([0.0]),
                'energies': np.array([0.0]),
                'collision_counts': np.array([0.0])
            }

    def setup_visualization(self):
        """设置可视化界面"""
        # 扫描可用的训练运行
        available_runs = self.scan_training_runs()

        if not available_runs:
            self.debug_print("没有找到训练运行!")
            return False

        self.debug_print("\n可用训练运行:")
        self.debug_print("-" * 60)
        for i, run in enumerate(available_runs):
            self.debug_print(f"{i + 1}. 运行ID: {run['run_id']}")
            self.debug_print(f"   日期: {run['timestamp']}")
            self.debug_print(f"   指标文件: {run['num_metrics_files']}")
            self.debug_print(f"   Episode文件: {run['num_episode_files']}")
            if run['info']:
                if 'best_reward' in run['info']:
                    self.debug_print(f"   最佳奖励: {run['info']['best_reward']:.2f}")
                if 'best_episode' in run['info']:
                    self.debug_print(f"   最佳Episode: {run['info']['best_episode']}")
            self.debug_print("-" * 60)

        try:
            choice = int(input("\n选择要可视化的训练运行 (输入编号): ")) - 1
            if 0 <= choice < len(available_runs):
                selected_run = available_runs[choice]
                self.debug_print(f"\n加载运行: {selected_run['run_id']}...")

                # 加载选中的训练运行数据
                run_data = self.load_run_data(selected_run['path'])
                if run_data:
                    self.debug_print("\n准备可视化数据...")

                    # 设置环境数据 - 确保所有键都是整数类型
                    if 'env_data' in run_data and run_data['env_data']:
                        # 检查visualizer对象的env_data结构
                        # 修复：确保所有键都是整数类型
                        standardized_env_data = {}
                        for ep_key, ep_value in run_data['env_data'].items():
                            ep_key_int = int(ep_key)
                            standardized_timesteps = {}
                            for ts_key, ts_value in ep_value.items():
                                standardized_timesteps[int(ts_key)] = ts_value
                            standardized_env_data[ep_key_int] = standardized_timesteps

                        # 输出详细的键信息以便调试
                        self.debug_print(f"标准化后的环境数据键: {list(standardized_env_data.keys())}")
                        self.debug_print(f"环境数据键类型: {[type(k) for k in standardized_env_data.keys()]}")
                        if standardized_env_data:
                            first_ep = list(standardized_env_data.keys())[0]
                            self.debug_print(f"时间步键: {list(standardized_env_data[first_ep].keys())}")
                            self.debug_print(
                                f"时间步键类型: {[type(k) for k in standardized_env_data[first_ep].keys()]}")

                        # 设置可视化器的环境数据
                        self.visualizer.env_data = standardized_env_data
                        self.debug_print(f"设置了环境数据，包含 {len(standardized_env_data)} 个episode")

                    # 准备性能指标数据
                    self._prepare_performance_data(run_data)

                    # 添加monkey patch以便在create_environment_figure中进行调试
                    original_create_environment_figure = self.visualizer.create_environment_figure

                    def create_environment_figure_debug(episode, timestep, show_trajectories, show_elements):
                        self.debug_print(
                            f"请求环境图形: episode={episode}, timestep={timestep}, 类型: episode={type(episode)}, timestep={type(timestep)}")
                        self.debug_print(
                            f"可用的环境数据键: {list(self.visualizer.env_data.keys())}, 类型: {[type(k) for k in self.visualizer.env_data.keys()]}")

                        # 确保键类型匹配 - 将episode和timestep转换为整数
                        episode_int = int(episode)
                        timestep_int = int(timestep)

                        # 检查键是否存在
                        if episode_int not in self.visualizer.env_data:
                            self.debug_print(f"警告: episode {episode_int} 不在环境数据中!")
                            return original_create_environment_figure(episode, timestep, show_trajectories,
                                                                      show_elements)

                        if timestep_int not in self.visualizer.env_data[episode_int]:
                            self.debug_print(f"警告: timestep {timestep_int} 不在episode {episode_int}的数据中!")
                            # 尝试找到最近的时间步
                            available_timesteps = list(self.visualizer.env_data[episode_int].keys())
                            if available_timesteps:
                                nearest_timestep = min(available_timesteps, key=lambda x: abs(x - timestep_int))
                                self.debug_print(f"使用最近的时间步: {nearest_timestep}")
                                timestep = nearest_timestep

                        # 调用原始函数
                        return original_create_environment_figure(episode_int, timestep_int, show_trajectories,
                                                                  show_elements)

                    # 替换方法
                    self.visualizer.create_environment_figure = create_environment_figure_debug

                    print("\nVisualization ready!")
                    print("\nControls:")
                    print("- Episode Slider: Select training episode")
                    print("- Timestep Slider: Navigate within episode (simulated data)")
                    print("- Show/Hide Elements:")
                    print("  • Trajectories (Satellites, UAVs, Ground Stations)")
                    print("  • Coverage Map")
                    print("  • Communication Links")
                    print("  • Points of Interest (POIs)")
                    print("- Playback Controls:")
                    print("  • Play/Pause: Animate episode")
                    print("  • Speed Control: Adjust animation speed")
                    print("\nStarting visualization server...")
                    return True
                else:
                    print("Failed to load run data.")
                    return False
            else:
                print("Invalid selection!")
                return False

        except ValueError:
            print("Invalid input!")
            return False
        except Exception as e:
            self.debug_print(f"Error setting up visualization: {str(e)}")
            self.debug_print(traceback.format_exc())
            return False

    def run(self, port=8050):
        """运行可视化器"""
        if self.setup_visualization():
            print(f"\nOpen your browser and navigate to: http://localhost:{port}")
            print("Press Ctrl+C to stop the server")

            try:
                self.visualizer.run(debug=False, port=port)
            except KeyboardInterrupt:
                print("\nVisualization server stopped.")
            except Exception as e:
                self.debug_print(f"Error running visualization server: {str(e)}")
                self.debug_print(traceback.format_exc())

def main():
    try:
        viz = TrainingVisualizer()
        viz.run()
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()