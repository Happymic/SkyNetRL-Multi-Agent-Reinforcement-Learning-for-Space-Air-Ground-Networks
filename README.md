# SkyNetRL: Multi-Agent RL for Space-Air-Ground Networks

A comprehensive framework for evaluating multi-agent reinforcement learning algorithms in Space-Air-Ground Integrated Networks (SAGIN).

## Features

- **Realistic SAGIN Environment**: 3GPP-compliant channel models, MAC layer, QoS management
- **Algorithm Comparison**: Statistical analysis with significance testing and effect sizes
- **Visualization**: Agent trajectories, performance metrics, network topology
- **Standardized Metrics**: Coverage, energy efficiency, throughput, fairness

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Algorithm Comparison

```bash
# Compare all available algorithms
python main.py --comparison --algorithm all --episodes 100 --statistical-runs 5

# Test single algorithm with statistical analysis
python main.py --comparison --algorithm ae_maddpg --episodes 50 --statistical-runs 3
```

### 3. Standard Training

```bash
# Standard training without comparison
python main.py --algorithm ae_maddpg --episodes 100
```

### 4. Advanced Features

```bash
# Enable 3GPP channel models and MAC protocol simulation
python main.py --comparison --algorithm all --3gpp-channels --mac-protocols
```

## Results

Results are saved to `results/sagin_multi_algorithm_evaluation/` with:

- **evaluation_report.md**: Comprehensive performance analysis
- **statistical_comparisons.json**: Statistical test results
- **visualizations/**: Agent trajectories, performance plots, animations
- **{algorithm}_results.json**: Individual algorithm data

## Algorithm Implementation

### Current Algorithms

The framework supports these algorithm types:
- `ae_maddpg`: Attention-Enhanced Multi-Agent DDPG
- `baseline_maddpg`: Standard Multi-Agent DDPG
- `qmix`: QMIX
- `independent_ppo`: Independent PPO
- `greedy_heuristic`: Greedy baseline
- `random_policy`: Random baseline

### Adding Your Algorithm

1. Create algorithm class in `src/algorithms/your_algorithm/`:

```python
class YourAlgorithm:
    def __init__(self, config):
        # Initialize your algorithm
        pass
    
    def act(self, obs):
        # Return actions for all agents
        return actions
    
    def store_experience(self, obs, actions, rewards, next_obs, done):
        # Store experience for training
        pass
    
    def update(self):
        # Update algorithm parameters
        pass
    
    def set_eval_mode(self):
        # Set to evaluation mode
        pass
```

2. Add import in `src/experiments/comparison_experiment.py`:

```python
try:
    from algorithms.your_algorithm.your_algorithm import YourAlgorithm
except ImportError:
    YourAlgorithm = None
```

3. Add to algorithm registry:

```python
available_algorithms = {
    # ... existing algorithms
    'your_algorithm': YourAlgorithm,
}
```

4. Test your algorithm:

```bash
python main.py --comparison --algorithm your_algorithm --episodes 50
```

## Configuration

Modify `configs/experiment_config.json` to customize:

- **Environment**: Area size, number of agents, POIs, obstacles
- **Training**: Episodes, evaluation episodes, statistical runs
- **Network**: Channel models, MAC layer, QoS parameters
- **Visualization**: Plot types, animation settings

## Environment Details

### SAGIN Network Structure

- **Satellites**: High altitude, large coverage, orbital movement
- **UAVs**: Medium altitude, flexible movement, energy constraints
- **Ground Stations**: Fixed/limited mobility, reliable communication

### Performance Metrics

- **Coverage Probability**: Fraction of POIs successfully covered
- **Energy Efficiency**: Data transmission per unit energy
- **Spectral Efficiency**: Data rate per unit bandwidth
- **Fairness Indices**: Jain's fairness, proportional fairness
- **Network Utilization**: Resource usage efficiency

### Network Modeling

- **3GPP-Compliant Channels**: Satellite, UAV, and terrestrial path loss models
- **MAC Layer Protocols**: Resource allocation, scheduling algorithms
- **QoS Management**: Traffic classes, admission control
- **Interference Models**: Co-channel interference, noise modeling

## File Structure

```
src/
├── algorithms/          # Algorithm implementations
├── environments/        # SAGIN environment and channel models
├── experiments/         # Experiment frameworks
├── evaluation/          # Metrics and statistical analysis
├── protocols/          # MAC layer and QoS management
└── visualization/      # Plotting and animation tools

configs/                # Configuration files
results/                # Experiment results
main.py                 # Main entry point
```

## Usage Examples

### Massive Experimentation

```bash
# Run comprehensive comparison
python main.py --comparison --algorithm all --episodes 200 --statistical-runs 10 --3gpp-channels --mac-protocols
```

### Quick Development Testing

```bash
# Fast testing with reduced episodes
python main.py --comparison --algorithm ae_maddpg --episodes 20 --statistical-runs 3
```

### Visualization Only

```bash
# Generate visualizations without training
python main.py --algorithm ae_maddpg --video --video-mode overview
```

## Troubleshooting

### No Algorithms Available

If you see "Warning: No algorithm implementations found":

1. Implement algorithms in `src/algorithms/` directory
2. Follow the algorithm implementation guide above
3. Ensure proper imports in comparison experiment

### Performance Issues

- Reduce episodes/statistical runs for development
- Use `--test-mode` for quick validation
- Disable visualization for faster training

### Import Errors

- Ensure `src/` is in Python path
- Check algorithm implementations are complete
- Verify all dependencies are installed

## Contributing

1. Implement your algorithm following the interface
2. Add comprehensive evaluation metrics
3. Include visualization components
4. Test with the comparison framework
5. Document performance characteristics

For detailed documentation, see `USAGE_GUIDE.md`.