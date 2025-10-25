# SAGIN Network Framework - Usage Guide

## Overview

This framework provides comprehensive tools for evaluating multi-agent reinforcement learning algorithms in Space-Air-Ground Integrated Networks (SAGIN) with:

- **Realistic Network Modeling** (3GPP channel models, MAC layer, QoS management)
- **Statistical Analysis** (rigorous algorithm comparison with significance testing)
- **Rich Visualizations** (agent trajectories, performance metrics, network topology)
- **Performance Evaluation** (standardized metrics for analysis)

## Quick Start

### 1. Basic Algorithm Testing

Test a single algorithm with visualization:

```bash
python main.py --comparison --algorithm ae_maddpg --episodes 50 --statistical-runs 5
```

### 2. Algorithm Comparison

Compare multiple algorithms with statistical analysis:

```bash
python main.py --comparison --algorithm all --3gpp-channels --mac-protocols --statistical-runs 5
```

### 3. Standard Training

Run standard training without comparison mode:

```bash
python main.py --algorithm ae_maddpg --episodes 100
```

### 4. Custom Configuration

Modify `configs/experiment_config.json` for your specific requirements.

## Command Line Options

### Core Options
- `--algorithm {ae_maddpg,baseline_maddpg,qmix,independent_ppo,greedy_heuristic,random_policy,all}`: Algorithm to test
- `--episodes EPISODES`: Number of training episodes (default: 50)
- `--statistical-runs RUNS`: Number of independent runs for statistical analysis (default: 5)

### Comparison Mode
- `--comparison`: Enable algorithm comparison with statistical analysis and visualization
- `--3gpp-channels`: Enable 3GPP-compliant channel models
- `--mac-protocols`: Enable MAC layer and QoS protocol simulation

### Visualization
- `--video`: Generate video visualization
- `--video-mode {overview,tracking,orbiting,split_screen}`: Video camera mode

## Configuration Files

### Test Configuration (`configs/test_config.json`)

Small-scale configuration for quick testing:
- **Area**: 800m × 800m
- **Agents**: 1 satellite, 2 UAVs, 1 ground station
- **POIs**: 8 points of interest
- **Episodes**: 20 training episodes, 5 evaluation episodes
- **Runs**: 3 statistical runs

### Custom Configuration

Create your own configuration by modifying:

```json
{
  "experiment_name": "my_experiment",
  "environment": {
    "area_size": 1000,
    "num_satellites": 1,
    "num_uavs": 3,
    "num_ground_stations": 2,
    "num_pois": 10
  },
  "visualization": {
    "save_trajectories": true,
    "save_performance_plots": true,
    "create_animations": true
  }
}
```

## Generated Outputs

### Result Files

All results are saved to `results/sagin_multi_algorithm_evaluation/`:

- `evaluation_report.md`: Comprehensive evaluation report
- `statistical_comparisons.json`: Statistical test results
- `{algorithm}_results.json`: Individual algorithm performance data

### Visualizations

Generated in `results/sagin_multi_algorithm_evaluation/visualizations/`:

- **Trajectory Plots**: Agent movement patterns over time
- **Performance Metrics**: Coverage, energy, fairness over episodes
- **Network Topology**: Communication links and connectivity
- **Animations**: Dynamic GIFs showing agent behavior
- **Summary Reports**: Markdown files linking all visualizations

## Performance Metrics

### Network Performance
- **Coverage Probability**: Fraction of POIs successfully covered
- **Spectral Efficiency**: Data rate per unit bandwidth
- **Energy Efficiency**: Bits transmitted per unit energy
- **Throughput**: Total data transmission rate

### Algorithm Performance
- **Convergence Rate**: Episodes to reach stable performance
- **Training Stability**: Variance in performance over time
- **Resource Utilization**: Efficiency of network resource usage
- **Fairness Indices**: Jain's fairness, proportional fairness

### Statistical Analysis
- **Significance Testing**: t-tests, Mann-Whitney U, ANOVA
- **Effect Sizes**: Cohen's d, rank-biserial correlation
- **Confidence Intervals**: 95% confidence bounds
- **Multiple Comparisons**: Bonferroni correction

## Algorithm Integration

### Using Mock Algorithms

The framework includes mock algorithms that simulate realistic performance patterns:

```python
# Mock algorithms are automatically used when real implementations aren't available
algorithms = ['ae_maddpg', 'baseline_maddpg', 'qmix', 'independent_ppo']
```

### Adding Real Algorithms

To integrate your own algorithm:

1. Create algorithm class with required methods:
   ```python
   class MyAlgorithm:
       def __init__(self, config): ...
       def act(self, obs): ...
       def store_experience(self, ...): ...
       def update(self): ...
       def set_eval_mode(self): ...
   ```

2. Add to algorithm registry in `comparison_experiment.py`

3. Test with the framework:
   ```bash
   python main.py --comparison --algorithm my_algorithm
   ```

## Visualization Customization

### Trajectory Visualization

Customize agent colors and markers in `trajectory_plotter.py`:

```python
self.agent_colors = {
    'satellite': '#FF6B6B',    # Red
    'uav': '#4ECDC4',          # Teal  
    'ground_station': '#45B7D1' # Blue
}
```

### Performance Plots

Modify plot styles and metrics in visualization configuration:

```json
{
  "visualization": {
    "plot_interval": 10,
    "save_format": "png",
    "animation_fps": 5,
    "plot_style": "seaborn"
  }
}
```

## Research Applications

### Academic Papers

The framework generates publication-ready results:

1. **Statistical rigor**: Proper significance testing and effect sizes
2. **Standardized metrics**: Industry-standard performance measures
3. **Reproducible results**: Seeded random number generation
4. **Professional visualizations**: High-quality plots and animations

### Benchmark Studies

Compare algorithms across multiple dimensions:

```bash
# Comprehensive benchmark
python main.py --research-mode --algorithm all --episodes 100 --statistical-runs 10
```

### Ablation Studies

Test specific components:

```bash
# Test with 3GPP channels
python main.py --comparison --3gpp-channels --algorithm ae_maddpg

# Test with MAC protocols
python main.py --comparison --mac-protocols --algorithm ae_maddpg
```

## Troubleshooting

### Common Issues

1. **No visualizations generated**: Ensure `--comparison` is enabled
2. **Import errors**: Check that `src/` is in Python path
3. **Performance issues**: Reduce episodes or statistical runs for testing

### Performance Optimization

- Use smaller configurations for development
- Increase statistical runs for publication
- Enable realistic models only when needed

## Examples

### Quick Performance Test

```bash
python test_single_algorithm.py
```

### Full Research Evaluation

```bash
python main.py --comparison --algorithm all --3gpp-channels --mac-protocols --episodes 100 --statistical-runs 10
```

### Visualization Demo

```bash
python demo_visualization.py
```

## Next Steps

1. **Modify configurations** in `configs/` for your specific use case
2. **Integrate real algorithms** by following the algorithm integration guide
3. **Customize visualizations** by modifying the plotting functions
4. **Extend metrics** by adding new evaluation criteria
5. **Scale experiments** by increasing episodes and statistical runs

For questions or contributions, refer to the project documentation and source code comments.