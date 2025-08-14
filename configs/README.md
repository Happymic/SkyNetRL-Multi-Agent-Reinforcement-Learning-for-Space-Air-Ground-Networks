# Configuration Files

This directory contains configuration files for different training scenarios and experiments.

## Available Configurations

### `default_config.json`
The standard configuration for typical training runs with balanced parameters:
- **Episodes**: 1000 training episodes
- **Environment**: 8 agents (2 satellites, 4 UAVs, 2 ground stations)
- **Area**: 800x800 grid with 12 POIs
- **Features**: Full optimization suite with randomization enabled

### `quick_test_config.json`
Lightweight configuration for rapid testing and development:
- **Episodes**: 10 training episodes (quick validation)
- **Environment**: 4 agents in 400x400 area with 6 POIs
- **Features**: Simplified setup, minimal monitoring overhead
- **Use Case**: Code testing, debugging, initial development

### `research_config.json`
Comprehensive configuration for research experiments and publications:
- **Episodes**: 5000 training episodes with extensive evaluation
- **Environment**: 12 agents in 1200x1200 area with 20 POIs
- **Features**: Full optimization suite, comprehensive monitoring, ablation studies
- **Use Case**: Research experiments, paper results, scalability testing

## Configuration Structure

Each configuration file contains the following main sections:

```json
{
  "experiment_name": "...",
  "environment": {
    "area_size": "...",
    "num_agents": "...",
    "randomization": { "..." }
  },
  "algorithm": {
    "embed_dim": "...",
    "learning_rates": "..."
  },
  "hierarchical_attention": { "..." },
  "stability": { "..." },
  "monitoring": { "..." },
  "visualization": { "..." }
}
```

## Key Parameters

### Environment Parameters
- `area_size`: Simulation area dimensions
- `num_agents`: Total number of agents (satellites + UAVs + ground stations)
- `max_episode_steps`: Maximum steps per episode
- `randomization`: Environment randomization settings for robustness

### Algorithm Parameters
- `embed_dim`: Neural network embedding dimension
- `num_heads`: Number of attention heads
- `learning_rates`: Actor/critic learning rates
- `gamma`: Discount factor

### Optimization Features
- `hierarchical_attention`: Multi-level attention mechanism settings
- `stability`: Training stability improvements (gradient clipping, scheduling)
- `monitoring`: Comprehensive monitoring and analysis settings

## Usage

Use configurations with the main training script:

```bash
# Quick test
python train.py --config configs/quick_test_config.json --algorithm ae_maddpg

# Standard training
python train.py --config configs/default_config.json --algorithm ae_maddpg --episodes 1000

# Research experiment
python train.py --config configs/research_config.json --algorithm ae_maddpg --video
```

## Customization

1. **Copy an existing config**: Start with the closest match to your needs
2. **Modify parameters**: Adjust environment size, agent counts, training parameters
3. **Save with descriptive name**: Use clear naming like `custom_large_scale_config.json`
4. **Test with quick_test first**: Validate changes with short runs

## Notes

- All configurations include intelligent randomization for training robustness
- Research config enables comprehensive ablation studies and scalability testing  
- Monitor GPU memory usage with larger configurations (research_config)
- Configurations are designed to work with all optimization components