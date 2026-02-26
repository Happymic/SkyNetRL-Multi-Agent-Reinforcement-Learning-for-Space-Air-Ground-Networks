# SkyNetRL: Comprehensive Project Knowledge Base

## 1. Project Overview

**SkyNetRL** (Multi-Agent Reinforcement Learning for Space-Air-Ground Networks) is a comprehensive research framework that applies multi-agent reinforcement learning (MARL) to optimize coverage and resource allocation in Space-Air-Ground Integrated Networks (SAGIN). The framework features an attention-enhanced MADDPG algorithm as its primary contribution, alongside multiple comparison algorithms, realistic 3GPP network simulation, rigorous statistical evaluation, and professional visualization tools.

### Research Problem
In SAGIN environments, heterogeneous agents (satellites, UAVs, ground stations) must cooperatively provide network coverage to Points of Interest (POIs) while managing energy constraints, avoiding collisions, and optimizing resource utilization. This is a multi-agent coordination problem with continuous action spaces, partial observability, and mixed cooperative-competitive dynamics.

### Key Innovation
The **AE-MADDPG** (Attention-Enhanced Multi-Agent Deep Deterministic Policy Gradient) algorithm introduces a hierarchical multi-head attention mechanism operating at three levels:
1. **Spatial Attention**: Selectively focuses on relevant POIs, obstacles, and charging stations
2. **Agent Attention**: Identifies and attends to relevant agents for coordination
3. **Task Attention**: Prioritizes tasks based on importance and completion status

This enables agents to dynamically prioritize environmental elements and coordinate more effectively compared to standard MADDPG.

---

## 2. Complete File-by-File Reference

### 2.1 Entry Points

#### `main.py` - Primary CLI Entry Point
- Argparse-based command line interface
- Supports modes: standard training, comparison with statistical analysis, video generation
- CLI flags: `--algorithm`, `--comparison`, `--3gpp-channels`, `--mac-protocols`, `--statistical-runs`, `--episodes`, `--video`, `--realtime-3d`, `--seed`
- Loads config from `configs/default_config.json`, overrides with CLI args
- Routes to `CompleteExperiment`, `EnhancedCompleteExperiment`, or `ComparisonExperiment`

#### `train.py` - Training Script
- Alternative entry point focused on training
- Supports config file loading and override

#### `run_clean_training.py` - Clean Training Runner
- 100-episode training with clean visualization output
- Integrates professional report generation

#### `run_simple_training_viz.py` - Simple Training with Visualization
- Lightweight training with basic plots

#### `run_complete_training_with_movement.py` - Training with Agent Movement Visualization
- Full training pipeline with agent movement tracking and trajectory output

#### `generate_movement_demo.py` - Movement Demonstration
- Generates demo of agent movement patterns without full training

---

### 2.2 Configuration Files (`configs/`)

#### `configs/default_config.json`
- **Environment**: area_size=1200, 2 satellites, 5 UAVs, 3 ground stations, 15 POIs, 5 obstacles, 200 max steps
- **Training**: 1000 episodes, batch_size=256, buffer_size=100000
- **Algorithm defaults**: embed_dim=256, num_heads=8, actor_lr=3e-4, critic_lr=1e-3, gamma=0.99, tau=0.005
- **Ablation configs**: no_attention, no_spatial_attention, no_agent_attention, no_task_attention, reduced_heads
- **Scalability configs**: small (4 agents), medium (8 agents), large (15 agents)

#### `configs/experiment_config.json`
- Balanced comparison setup: area_size=1000, 1 satellite, 3 UAVs, 2 ground stations, 10 POIs
- 100 episodes, 5 statistical runs
- Algorithms: ae_maddpg, baseline_maddpg, qmix, independent_ppo, greedy_heuristic, random_policy
- Optional 3GPP channels and MAC protocols

#### `configs/research_config.json`
- Large-scale research: 5000 episodes, 12 agents, area_size=1200
- Extended ablation study matrix
- All visualization types enabled

---

### 2.3 Algorithms (`src/algorithms/`)

#### AE-MADDPG (`src/algorithms/ae_maddpg/`) - PRIMARY ALGORITHM

**`agent.py`** - Main agent class `AEMADDPGAgent`:
- Implements centralized training with decentralized execution (CTDE)
- Each agent has an AttentionEnhancedActor (policy) and AttentionEnhancedCritic (value)
- Target networks with soft updates (tau=0.005)
- Exploration via Gaussian noise with decay (noise_scale starts at 0.1, decays by 0.995/episode, minimum 0.01)
- Agent type differentiation: satellite, UAV, ground_station (encoded as one-hot or type ID)
- `act(obs)`: Returns continuous 2D actions, applies noise during training
- `update()`: Samples from replay buffer, computes TD target, updates critic then actor, soft-updates target networks
- Attention regularization loss (weight=0.01) encourages diverse attention patterns
- Entropy regularization (0.001) for exploration

**`attention_modules.py`** - Attention mechanisms:
- `EnhancedMultiHeadAttention`: Scaled dot-product attention with num_heads=8, embed_dim=256
  - Projects Q, K, V via linear layers
  - Attention weights = softmax(QK^T / sqrt(d_k))
  - Output = attention_weights * V, projected back
  - Layer normalization + residual connections
- `SpatialAttentionModule`: Processes environmental features
  - Separate encoders for POIs (5d: x, y, priority, covered, importance), obstacles (3d: x, y, radius), charging stations (4d: x, y, availability, importance), self-state (9d)
  - Cross-attention between self-state and each spatial element type
  - Outputs fused spatial representation

**`networks.py`** - Neural network architectures:
- `AttentionEnhancedActor`:
  - Input: observation (184d) -> spatial encoder -> agent encoder -> task encoder
  - Attention layers process each component
  - MLP head: concatenated features -> hidden layers -> 2D action (tanh activation)
  - Output: continuous action in [-1, 1]
- `AttentionEnhancedCritic`:
  - Input: all agents' observations + all agents' actions (centralized)
  - Same attention architecture as actor for state processing
  - Additional action encoder
  - MLP head -> scalar Q-value
- `AttentionAnalyzer`: Extracts and analyzes attention weight patterns for interpretability

#### Baseline MADDPG (`src/algorithms/baseline_maddpg/`)

**`agent.py`** - Standard MADDPG without attention:
- `BaselineMADDPGAgent`: Same CTDE framework as AE-MADDPG but with standard MLP networks
- Purpose: ablation baseline to quantify the benefit of attention mechanisms
- `BaselineActor`: MLP with hidden layers [256, 128], ReLU activations, tanh output
- `BaselineCritic`: MLP processing concatenated observations and actions

#### QMIX (`src/algorithms/qmix/`)

**`agent.py`** - QMIX with value decomposition:
- `QMIXMultiAgent`: Cooperative MARL with monotonic value decomposition
- Individual Q-networks per agent (RNN option for partial observability)
- Mixing network: takes individual Q-values, outputs Q_tot with monotonicity constraint
- Hypernetwork generates mixing weights conditioned on global state
- Discrete action conversion: continuous 2D space discretized into grid (default 9 actions: 3x3 grid of directions)
- Double QMIX variant available (use target Q for action selection, online Q for evaluation)
- Epsilon-greedy exploration: epsilon starts at 1.0, decays to 0.05 over training

#### Independent PPO (`src/algorithms/independent_ppo/`)

**`agent.py`** - Independent PPO agents:
- Each agent learns independently using PPO (no centralized training)
- `PPOAgent` with actor-critic architecture
- GAE (Generalized Advantage Estimation) with lambda=0.95
- PPO clipping: epsilon=0.2 for policy ratio, value clipping=0.2
- Entropy coefficient=0.01 for exploration
- Mini-batch updates: K_epochs=4 per data collection
- Shared network option: agents can share parameters (optional)

#### Heuristic Baselines (`src/algorithms/baselines/`)

**`heuristic_agents.py`**:
- `GreedyHeuristicAgent`: Moves toward nearest uncovered high-priority POI. No learning.
- `RandomPolicyAgent`: Uniform random actions from action space. Lower bound baseline.
- `AdaptiveGreedyAgent`: Enhanced greedy with energy management for UAVs (seeks charging station when energy < 20%)

#### Demo Algorithm (`src/algorithms/demo_algorithm/`)
- Minimal implementation for testing the framework interface
- Random actions, placeholder storage and update methods

---

### 2.4 Environments (`src/environments/`)

#### `enhanced_sagin_env.py` - Main SAGIN Environment

Gymnasium-based environment simulating a Space-Air-Ground Integrated Network.

**Agent Configuration**:
| Type | Coverage Radius | Max Speed | Energy Capacity | Altitude Range |
|------|----------------|-----------|-----------------|----------------|
| Satellite | 250m | 3 m/s | Unlimited | High |
| UAV | 120m | 6 m/s | 1200 Joules | Medium |
| Ground Station | 80m | 2 m/s | Unlimited | Ground |

**State Space** (184 dimensions per agent):
- Self-state (9d): [x, y, vx, vy, energy_ratio, agent_type_0, agent_type_1, agent_type_2, coverage_contribution]
- Spatial observations (80d): Encoded POIs (positions, priorities, coverage status), obstacles (positions, radii)
- Agent observations (90d): Other agents' positions, velocities, types, distances
- Task observations (5d): Global coverage ratio, remaining POIs, avg priority of uncovered, time ratio, energy availability

**POI Properties**: position (x,y), priority (1-5), importance weight, covered boolean, coverage radius threshold
**Obstacle Properties**: position (x,y), radius, agents cannot enter obstacle regions
**Charging Stations**: position (x,y), availability status, UAVs can recharge when within range

**Episode Dynamics**:
- Max steps: 50-200 (configurable)
- Agents move based on continuous 2D velocity actions scaled by max_speed
- UAVs consume energy proportional to movement + base consumption
- POIs become "covered" when any agent is within coverage radius
- Episode terminates when max_steps reached or all POIs covered

**Reward Computation**:
- Coverage reward (+2.0 per newly covered POI)
- Priority bonus (+1.5 for high-priority POIs, scaled by priority level)
- Efficiency reward (+1.0 for coverage/energy ratio)
- Cooperation reward (+0.8 for coordinated coverage, agents covering different POIs)
- Energy penalty (-0.3 for UAV energy expenditure)
- Collision penalty (-2.0 to -5.0 for agent-agent collisions, distance < 10m)
- Redundancy penalty (-0.5 for multiple agents covering same POI)
- Exploration bonus (+0.4 for visiting new areas)
- Diversity bonus (+0.3 for spatial spread)
- Time penalty (-0.1 per step to encourage efficiency)

#### `robust_environment.py` - Robust Environment with Randomization

Extends enhanced_sagin_env with domain randomization for training robustness:
- **Agent position noise**: Gaussian noise (std=100m) on initial positions
- **POI density variation**: +/-20% random variation in number of POIs
- **Dynamic obstacles**: Obstacles can move during episode
- **Weather effects**: Multiplier (0.5-1.5x) on communication quality and coverage
- **Wind effects**: Random wind vectors affecting UAV movement
- **Communication noise**: Gaussian noise on inter-agent communication channels

#### `channel_model.py` - 3GPP Channel Models

Implements realistic wireless channel modeling per 3GPP standards:

**Path Loss Models**:
- Satellite-to-Ground: Free-space path loss + atmospheric attenuation + shadow fading
- Satellite-to-UAV: Similar to S2G with reduced atmospheric effects
- UAV-to-Ground: 3GPP TR 36.777 air-to-ground model with LoS probability
- Ground-to-Ground: 3GPP TR 38.901 urban macro model

**Channel Parameters**:
- Carrier frequency: 2.0 GHz (configurable)
- Bandwidth: 20 MHz
- Noise figure: 9 dB
- Shadowing std dev: 4 dB (log-normal)
- Multipath components: 6 (Rayleigh fading)
- Doppler shift: up to 100 Hz (velocity-dependent)

**Computed Metrics**:
- Received signal power (dBm)
- Signal-to-Noise Ratio (SNR in dB)
- Channel capacity (Shannon: C = B * log2(1 + SNR))
- Bit Error Rate (BER) estimation
- Link budget analysis

---

### 2.5 Network Protocols (`src/protocols/`)

#### `mac_layer.py` - MAC Layer Protocol Simulation

Simulates OFDMA-based medium access control:
- **Resource blocks**: 50 configurable blocks (time-frequency grid)
- **Resource types**: Time slots, frequency blocks, spatial streams, code sequences
- **Scheduling**: Proportional fair scheduling (balances throughput and fairness)
- **User requests**: Data size, priority class (EMERGENCY > REAL_TIME > HIGH_PRIORITY > BEST_EFFORT), delay tolerance
- **Channel state information**: Quality indicators, SNR, path loss, interference level
- **Allocation output**: Assigned resource blocks, estimated throughput, delay

#### `qos_manager.py` - Quality of Service Management

5G NR-inspired QoS framework:
- **Service classes**: eMBB, URLLC, mMTC, Broadcast
- **QoS parameters**: Guaranteed bit rate, maximum bit rate, packet delay budget (100ms default), packet error rate (1e-3), priority levels (1-15)
- **Admission control**: Checks if requested QoS can be satisfied
- **Flow management**: Per-flow tracking of throughput, delay, jitter, packet loss
- **QoS satisfaction**: Reports percentage of flows meeting their requirements

---

### 2.6 Reward System (`src/rewards/`)

#### `multi_objective_rewards.py` - Multi-Objective Reward Mechanism

**Reward Components and Default Weights**:
| Component | Weight | Description |
|-----------|--------|-------------|
| coverage_reward | 2.0 | POI coverage achievement |
| priority_bonus | 1.5 | High-priority POI bonus |
| efficiency_reward | 1.0 | Energy-coverage ratio |
| cooperation_reward | 0.8 | Multi-agent coordination |
| exploration_bonus | 0.4 | New area exploration |
| diversity_bonus | 0.3 | Spatial distribution |
| energy_penalty | 0.3 | Energy waste penalty |
| redundancy_penalty | 0.5 | Duplicate coverage penalty |
| collision_penalty | 2.0 | Agent collision penalty |
| time_penalty | 0.1 | Step-wise time cost |

**Adaptive Scaling Mechanism**:
- Training phase detection: early (first 30%), middle, late (last 20%)
- Early phase: Exploration bonus 2x, coverage reward 0.8x (encourage exploration)
- Late phase: Exploration bonus 0.2x, coverage reward 1.5x (refine exploitation)
- Performance-based: If coverage > 80%, increase cooperation weight; if coverage < 30%, increase exploration
- Dynamic range: Each component scales between 0.2x and 2.0x its base weight

---

### 2.7 Neural Networks (`src/networks/`)

#### `hierarchical_attention.py` - Hierarchical Attention Architecture

**Three-Level Architecture**:

**Level 1 - Spatial Attention Module**:
- POI Encoder: Linear(5, embed_dim) -> ReLU -> Linear(embed_dim, embed_dim)
  - Input features: [x, y, priority, is_covered, importance_weight]
- Obstacle Encoder: Linear(3, embed_dim) -> ReLU -> Linear(embed_dim, embed_dim)
  - Input features: [x, y, radius]
- Charging Station Encoder: Linear(4, embed_dim) -> ReLU -> Linear(embed_dim, embed_dim)
  - Input features: [x, y, is_available, importance]
- Self-State Encoder: Linear(9, embed_dim) -> ReLU -> Linear(embed_dim, embed_dim)
- Cross-attention: self_state attends to each encoded set (Q=self, K=V=entities)
- Output: Concatenation of attended features -> Linear projection

**Level 2 - Agent Attention Module**:
- Agent Encoder: Linear(agent_obs_dim, embed_dim) per other agent
- Self-attention across all agent representations
- Captures inter-agent relationships and potential for cooperation
- Output: Aggregated agent context vector

**Level 3 - Task Attention Module**:
- Task Encoder: Linear(5, embed_dim) for task features
- Attention over task priorities weighted by completion status
- Guides agent toward highest-value incomplete tasks
- Output: Task-weighted context vector

**Final Fusion**: Concatenate [spatial_context, agent_context, task_context] -> MLP -> action/value

**Attention Parameters**:
- Embedding dimension: 256 (default, configurable 128-512)
- Number of heads: 8 (default, configurable 4-16)
- Dropout: 0.1
- Layer normalization after each attention + FFN block
- Residual connections throughout

---

### 2.8 Evaluation (`src/evaluation/`)

#### `metrics.py` - Comprehensive Evaluation Metrics

**Coverage Metrics**:
- `coverage_rate`: (# covered POIs) / (total POIs), range [0, 1]
- `priority_weighted_coverage`: Sum of (priority * covered) / sum of priorities
- `coverage_persistence`: Fraction of time POIs remain covered after first coverage
- `spatial_coverage_uniformity`: How evenly coverage is distributed across the area

**Energy Metrics**:
- `energy_efficiency`: Total data transmitted / total energy consumed (bits/Joule)
- `per_agent_energy`: Energy breakdown by agent type
- `energy_fairness`: Jain's fairness index of energy usage across agents

**Cooperation Metrics**:
- `cooperation_index`: Measures how much agents coordinate vs. operate independently
- `communication_utilization`: Fraction of time agents are in communication range
- `task_allocation_efficiency`: How well tasks are distributed among agents

**Fairness Metrics**:
- `jains_fairness`: J(x) = (sum(xi))^2 / (n * sum(xi^2)), range [1/n, 1]
- `proportional_fairness`: sum(log(xi))
- `max_min_fairness`: min(xi) / max(xi)

#### `academic_metrics.py` - Standardized Academic Metrics

ITU-R and 3GPP compliant metrics for publication:
- **Coverage probability**: P(SNR > threshold) for each link type
- **Aggregate throughput**: Sum of per-user Shannon capacity
- **Average delay**: Mean packet delay across all flows
- **Packet delivery ratio**: Successfully delivered / total sent
- **Spectral efficiency**: Total throughput / total bandwidth (bps/Hz)
- **Energy per bit**: Total energy / total bits transmitted (J/bit)

**Convergence Analysis**:
- Convergence episode: First episode where running average stays within 5% of final value for 50+ episodes
- Convergence rate: Slope of performance curve at convergence point
- Stability measure: Std dev of performance in last 20% of training
- Oscillation frequency: Number of sign changes in performance derivative

**Scalability Metrics**:
- Training time vs. number of agents
- Memory usage vs. observation space size
- Communication overhead vs. number of agents
- Performance degradation factor with scale

#### `statistical_analysis.py` - Statistical Comparison Framework

**Hypothesis Testing**:
- Normality check: Shapiro-Wilk test (alpha=0.05)
- Variance homogeneity: Levene's test
- Parametric tests: Independent t-test, one-way ANOVA (if normal + homogeneous variance)
- Non-parametric tests: Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis (otherwise)
- Friedman test for repeated measures

**Effect Sizes**:
- Cohen's d for parametric: d = (mean1 - mean2) / pooled_std
  - Small: 0.2, Medium: 0.5, Large: 0.8
- Rank-biserial r for non-parametric: r = 1 - 2U/(n1*n2)
- Interpretation with practical significance thresholds

**Multiple Comparisons**:
- Bonferroni correction: alpha_adjusted = alpha / num_comparisons
- Post-hoc pairwise comparisons for significant ANOVA/Kruskal-Wallis
- 95% confidence intervals for all reported metrics
- Algorithm ranking across metrics

**Practical Significance Thresholds**:
- Coverage: 2% improvement considered meaningful
- Throughput: 10% improvement
- Delay: 5% improvement
- Energy efficiency: 10% improvement
- Fairness: 5% improvement

---

### 2.9 Experiments (`src/experiments/`)

#### `comparison_experiment.py` - Multi-Algorithm Comparison

**Algorithm Registry**:
```python
available_algorithms = {
    'ae_maddpg': AEMADDPGAgent,
    'baseline_maddpg': BaselineMADDPGAgent,
    'qmix': QMIXMultiAgent,
    'independent_ppo': IndependentPPO,
    'greedy_heuristic': GreedyHeuristicAgent,
    'random_policy': RandomPolicyAgent,
    'demo_algorithm': DemoAlgorithm
}
```

**Workflow**:
1. Create environment with optional 3GPP channels and MAC protocols
2. For each algorithm x num_statistical_runs:
   - Initialize algorithm with config
   - Train for N episodes, evaluate every eval_interval episodes
   - Record per-episode metrics (reward, coverage, energy, cooperation)
3. Run statistical analysis across all algorithms
4. Generate comparison visualizations and reports
5. Output: JSON results, markdown report, statistical test results, plots

**Automatic Mock Generation**: If an algorithm import fails, creates a mock agent with realistic performance curves for framework testing.

#### `complete_experiment.py` - Full Training Pipeline
- Single algorithm training with comprehensive monitoring
- Periodic evaluation on separate episodes
- Checkpoint saving at best performance
- Visualization generation throughout training
- Final report compilation

#### `enhanced_complete_experiment.py` - Enhanced with Video
- Extends complete experiment with video generation
- Real-time 3D visualization option
- Agent movement trajectory recording
- High-quality animation output

---

### 2.10 Training System (`src/training/`)

#### `integrated_trainer.py` - Integrated Training System
- Wraps algorithm + environment + monitoring + visualization
- Standardized training loop with proper episode management
- Checkpoint saving and loading
- HTML dashboard generation for training progress
- Professional report output

#### `stability_improvements.py` - Training Stability
- **Gradient clipping**: Max gradient norm = 0.5 (configurable)
- **Learning rate scheduling**: ReduceLROnPlateau and cosine annealing options
- **Early stopping**: Stop if no improvement for N episodes (patience configurable)
- **Target network smoothing**: Soft updates with tau, optional hard updates every N steps
- **Batch normalization**: Optional for critic network inputs

---

### 2.11 Monitoring (`src/monitoring/`)

#### `comprehensive_monitor.py` - Training Monitor

Tracks during training:
- **Losses**: Actor loss, critic loss, attention regularization loss
- **Gradients**: Per-layer gradient norms, gradient statistics (mean, std, max)
- **Rewards**: Per-episode total reward, per-component reward breakdown
- **Attention**: Attention weight entropy, attention pattern diversity
- **Learning rates**: Current LR for actor and critic
- **Coverage**: Episode coverage rate, priority-weighted coverage
- **Energy**: Per-agent energy usage, total energy efficiency
- **Cooperation**: Cooperation index per episode

Uses circular buffer for memory-efficient storage. Provides rolling statistics and trend analysis.

---

### 2.12 Utils (`src/utils/`)

#### `replay_buffer.py` - Experience Replay Buffer
- Standard replay buffer with configurable capacity (default 100,000)
- Stores (obs, action, reward, next_obs, done) tuples
- Random sampling for mini-batch training
- Per-agent storage option for independent learners

#### `training_pipeline.py` - Standardized Output Pipeline
- Creates standardized output directory structure
- Manages file naming conventions
- Handles logging setup

#### `training_utils.py` - Training Utilities
- Seed setting for reproducibility (torch, numpy, random)
- Device detection (CPU/GPU)
- Learning rate scheduling helpers

#### `output_manager.py` - Output Management
- Creates directory structure: logs/, plots/, videos/, models/, analysis/
- Saves config alongside results for reproducibility
- HTML report generation with embedded figures

#### `visualization.py` & `advanced_visualization.py` - Basic/Advanced Visualization
- Training curve plotting
- Reward breakdown visualization
- Coverage map generation

---

### 2.13 Visualization System (`src/visualization/`)

12 specialized visualization modules:

#### `trajectory_plotter.py` - Agent Trajectory Visualization
- 2D top-down view of agent movements
- Color-coded by type: Satellite=#FF6B6B (red), UAV=#4ECDC4 (teal), Ground Station=#45B7D1 (blue)
- POI markers with priority-scaled size
- Obstacle regions as shaded circles
- Charging stations as special markers
- Trail fade effect showing movement history
- Coverage radius circles

#### `video_generator.py` - Video Creation
- Generates MP4 videos of agent movements
- Configurable FPS (default 10) and quality
- Frame-by-frame rendering with matplotlib
- Supports multi-panel layouts

#### `agent_3d_viewer.py` - 3D Agent Visualization
- 3D scatter plots with altitude differentiation
- Satellites at high altitude, UAVs at medium, ground stations at ground level
- Coverage volume visualization (spheres/cylinders)

#### `enhanced_3d_viewer.py` - Enhanced 3D Viewer
- Improved 3D graphics with better lighting and materials
- Multiple camera angles: top-down, side, perspective
- Agent type-specific 3D markers

#### `realtime_3d_viewer.py` - Real-time 3D Visualization
- OpenGL-based rendering for real-time display
- Interactive camera controls (rotate, zoom, pan)
- Live updates during training
- Requires OpenGL support

#### `professional_visualizer.py` - Publication-Quality Plots
- IEEE/ACM paper-ready figure generation
- Proper fonts, sizing, and formatting
- Multi-panel figures with shared legends
- LaTeX-compatible label formatting
- High-DPI output (300+ DPI PNG, vector PDF)

#### `clean_training_visualizer.py` - Clean Training Visualization
- Training performance curves (reward vs. episode)
- Multi-metric dashboard (coverage, energy, cooperation, fairness)
- Smoothed curves with confidence bands
- Algorithm comparison plots with error bars

#### `training_integration.py` - Visualization Integration Module
- Unified API for all visualization types
- Auto-selects appropriate visualization based on data
- Configuration-driven visualization pipeline

#### `video_integration.py` - Video System Integration
- Video file management and encoding
- Format conversion utilities
- Quality optimization for file size vs. visual quality

#### Additional Modules:
- `enhanced_training_visualizer.py` - Enhanced training curves with annotations
- `simple_training_visualizer.py` - Lightweight training visualization
- `visualizer.py` - Base visualization utilities

---

### 2.14 Pre-generated Figures (`paper_figures/`)
- Contains pre-generated research figures for publications
- Includes trajectory plots, performance comparisons, attention analysis
- Publication-ready formatting

---

## 3. Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Configuration (JSON)                       │
│  default_config.json / experiment_config.json                │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Experiment Framework                         │
│  ComparisonExperiment / CompleteExperiment                    │
│  - Manages training loop, evaluation, statistical runs       │
└──────────┬──────────────────────────────────┬───────────────┘
           │                                  │
           ▼                                  ▼
┌──────────────────────┐         ┌──────────────────────────┐
│   SAGIN Environment  │         │    RL Algorithm           │
│                      │         │                          │
│ • Agent positions    │◄───────►│ • AE-MADDPG (primary)   │
│ • POI states         │  obs/   │ • Baseline MADDPG       │
│ • Obstacle dynamics  │  action │ • QMIX                  │
│ • Energy tracking    │  reward │ • Independent PPO       │
│ • Coverage calc      │         │ • Heuristic baselines   │
│                      │         │                          │
│ Optional:            │         │ Neural Networks:         │
│ • 3GPP Channel Model │         │ • Attention Actor/Critic │
│ • MAC Layer          │         │ • Hierarchical Attention │
│ • QoS Manager        │         │ • Replay Buffer          │
└──────────────────────┘         └──────────────────────────┘
           │                                  │
           ▼                                  ▼
┌──────────────────────┐         ┌──────────────────────────┐
│  Multi-Objective     │         │   Training System        │
│  Reward System       │         │                          │
│                      │         │ • Gradient clipping      │
│ • Adaptive scaling   │         │ • LR scheduling          │
│ • Phase-aware weights│         │ • Target network updates │
│ • 10 reward components│        │ • Stability monitoring   │
└──────────────────────┘         └──────────────────────────┘
                                              │
                          ┌───────────────────┼───────────────────┐
                          ▼                   ▼                   ▼
               ┌─────────────────┐ ┌──────────────────┐ ┌──────────────────┐
               │   Evaluation    │ │   Monitoring     │ │  Visualization   │
               │                 │ │                  │ │                  │
               │ • Coverage rate │ │ • Loss tracking  │ │ • 2D trajectories│
               │ • Energy eff.  │ │ • Gradient norms │ │ • 3D viewer      │
               │ • Fairness     │ │ • Reward breakdown│ │ • Videos        │
               │ • Statistical  │ │ • Attention      │ │ • Training curves│
               │   analysis     │ │   analysis       │ │ • Pub-quality   │
               │ • Convergence  │ │ • Convergence    │ │   figures        │
               └────────┬────────┘ └────────┬─────────┘ └────────┬─────────┘
                        │                   │                    │
                        ▼                   ▼                    ▼
               ┌──────────────────────────────────────────────────────────┐
               │                    Output Manager                        │
               │  results/experiment_name/                                │
               │  ├── logs/          (training logs)                      │
               │  ├── plots/         (performance plots, trajectories)    │
               │  ├── videos/        (agent movement videos)             │
               │  ├── models/        (saved checkpoints)                 │
               │  ├── analysis/      (statistical reports)               │
               │  ├── config.json    (reproducibility)                   │
               │  └── report.html    (interactive dashboard)             │
               └──────────────────────────────────────────────────────────┘
```

---

## 4. Algorithm Comparison Summary

| Feature | AE-MADDPG | Baseline MADDPG | QMIX | Independent PPO |
|---------|-----------|-----------------|------|-----------------|
| **Training Paradigm** | CTDE | CTDE | CTDE | Independent |
| **Action Space** | Continuous | Continuous | Discrete (converted) | Continuous |
| **Exploration** | Gaussian noise + decay | Gaussian noise | Epsilon-greedy | Entropy bonus |
| **Value Function** | Centralized critic | Centralized critic | Decomposed + mixer | Per-agent critic |
| **Key Innovation** | Hierarchical attention | Standard MLP | Value monotonicity | PPO clipping |
| **Scalability** | Good (attention efficient) | Moderate | Good (decomposition) | Excellent (independent) |
| **Communication** | Implicit (centralized critic) | Implicit | Implicit (global state) | None |
| **Best For** | Heterogeneous cooperation | Baseline comparison | Fully cooperative | Scalable training |

---

## 5. Key Hyperparameters Reference

### AE-MADDPG (Primary)
```
embed_dim: 256          # Attention embedding dimension
num_heads: 8            # Multi-head attention heads
actor_lr: 3e-4          # Actor learning rate
critic_lr: 1e-3         # Critic learning rate
gamma: 0.99             # Discount factor
tau: 0.005              # Target network soft update rate
batch_size: 256         # Training batch size
buffer_size: 100000     # Replay buffer capacity
noise_scale: 0.1        # Initial exploration noise
noise_decay: 0.995      # Noise decay per episode
noise_min: 0.01         # Minimum noise
attention_reg: 0.01     # Attention regularization weight
entropy_reg: 0.001      # Entropy regularization weight
gradient_clip: 0.5      # Max gradient norm
```

### Environment
```
area_size: 1000-1200    # Environment area (meters)
num_satellites: 1-3     # Satellite count
num_uavs: 3-8          # UAV count
num_ground_stations: 2-4 # Ground station count
num_pois: 8-25          # Points of interest
num_obstacles: 3-5      # Obstacles
max_episode_steps: 50-200 # Max steps per episode
communication_range: 200-300 # Inter-agent comm range (meters)
```

### Training
```
num_episodes: 100-5000  # Total training episodes
eval_interval: 10       # Evaluate every N episodes
num_eval_episodes: 10   # Episodes per evaluation
num_statistical_runs: 3-10 # Independent runs for statistics
seed: 42                # Default random seed
```

---

## 6. Running the Framework

### Basic Commands
```bash
# Standard training with AE-MADDPG
python main.py --algorithm ae_maddpg --episodes 100

# Full algorithm comparison with statistics
python main.py --comparison --algorithm all --episodes 100 --statistical-runs 5

# With realistic network simulation
python main.py --comparison --algorithm all --3gpp-channels --mac-protocols

# With video generation
python main.py --algorithm ae_maddpg --video --video-mode overview

# Development/quick test mode
python main.py --development-mode --episodes 10

# Research-scale experiment
python main.py --comparison --algorithm all --episodes 1000 --statistical-runs 10 --3gpp-channels --mac-protocols
```

### Configuration Override
Edit `configs/experiment_config.json` or `configs/default_config.json` to change environment, training, algorithm, or visualization parameters without modifying code.

### Adding a New Algorithm
1. Create `src/algorithms/your_algo/agent.py` implementing: `__init__(config)`, `act(obs)`, `store_experience(...)`, `update()`, `set_eval_mode()`
2. Add import in `src/experiments/comparison_experiment.py`
3. Add to `available_algorithms` registry
4. Test: `python main.py --comparison --algorithm your_algo`

---

## 7. Dependencies

### Core (Required)
- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Gymnasium >= 0.28.0
- Pandas >= 1.3.0
- Matplotlib >= 3.4.0
- Seaborn >= 0.11.0
- Plotly >= 5.0.0
- TQDM >= 4.60.0
- TensorBoard >= 2.7.0

### Optional
- wandb (experiment tracking)
- dash (interactive dashboards)
- jupyter (notebook analysis)
- OpenGL (real-time 3D visualization)

---

## 8. Research Context

### Target Publication Venues
- IEEE Transactions on Wireless Communications
- IEEE JSAC (Selected Areas in Communications)
- IEEE ICC / GLOBECOM conferences
- ACM MobiCom / MobiSys

### Related Research Areas
- Multi-Agent Reinforcement Learning (MARL)
- Space-Air-Ground Integrated Networks (SAGIN)
- UAV-Assisted Communications
- Network Coverage Optimization
- Attention Mechanisms in RL
- 3GPP New Radio (NR) / 5G
- Resource Allocation in Heterogeneous Networks

### Key References
- MADDPG: Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"
- QMIX: Rashid et al., "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning"
- PPO: Schulman et al., "Proximal Policy Optimization Algorithms"
- 3GPP TR 36.777: Study on Enhanced LTE Support for Aerial Vehicles
- 3GPP TR 38.901: Study on Channel Model for Frequencies from 0.5 to 100 GHz
