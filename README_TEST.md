# Traffic Signal RL - Minimal Testing

This file contains instructions for running a minimal test of the Traffic Signal RL environment on any hardware before moving to more powerful training on a dedicated GPU.

## Running the Minimal Test

1. Make sure all dependencies are installed:
   ```
   pip install tensorflow numpy matplotlib
   ```

2. Run the minimal test script:
   ```
   python src/test_minimal.py
   ```

This script:
- Creates a minimal configuration with conservative settings
- Tests for 2 episodes using the DQN agent
- Runs with minimal memory requirements
- Works on both CPU and GPU environments
- Disables all advanced features that might cause issues

The test should complete in just a few minutes. If successful, you'll see output showing:
- Hardware detection (CPU or GPU)
- Episode progress
- Final rewards
- Time taken

## Troubleshooting

If you encounter any issues:

1. **GPU-related errors**: The script automatically falls back to CPU if needed
2. **Memory errors**: Reduce batch size in the script if needed
3. **SUMO errors**: Make sure SUMO is properly installed and in your PATH

## Moving to Full Training

Once the minimal test is successful, you can proceed with the full training on more powerful hardware:

```
python src/test_all_agents.py
```

Or to train specific agents only:

```
python src/test_all_agents.py dqn,ppo 