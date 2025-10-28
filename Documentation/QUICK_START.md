# Quick Start Guide

## Getting Started

### System Requirements
- **Python**: 3.9.7 or compatible
- **OS**: Windows, macOS, or Linux
- **RAM**: 4GB minimum (8GB recommended)
- **Disk Space**: ~500MB for dependencies and models

### Installation

1. **Clone or Download the Repository**
   ```bash
   git clone <repository-url>
   cd Multi-Agent-Faction-Wars
   ```

2. **First Run Setup**
   
   When you run the project for the first time:
   ```bash
   python main.py
   ```
   
   The system will automatically:
   - Check for required dependencies
   - Install missing packages if needed
   - Prompt you for HEADLESS_MODE preference
   - Create a `settings.json` file to save your preferences

3. **Install Dependencies Manually (Optional)**
   
   If you prefer to install dependencies manually:
   ```bash
   pip install -r UTILITIES/requirements.txt
   ```

### First Launch

1. **Startup Options**
   - Choose "Run in Terminal" or "Train Headless"
   - Terminal: See visual gameplay
   - Headless: Faster training without graphics

2. **Main Menu**
   - Select number of factions (2-6)
   - Configure episode and step settings
   - Choose to start fresh or load existing models

3. **Gameplay Controls**
   - **ESC**: Pause/unpause
   - **Q**: Quit (while paused)
   - **M**: Restart episode (while paused)
   - **Mouse Wheel**: Zoom in/out
   - **Arrow Keys**: Pan camera

### Running Modes

#### Visual Mode (Recommended for First Time)
- See gameplay in real-time
- Better understanding of agent behavior
- Inspect game state with tooltips
- Useful for debugging and research

#### Headless Mode
- Faster training (no rendering overhead)
- Useful for long training sessions
- Monitor progress via TensorBoard
- Good for production training

### Quick Configuration

Edit `UTILITIES/utils_config.py` to customize:

```python
# Number of agents per faction
DEFAULT_AGENTS_PER_FACTION = 4

# Number of factions
DEFAULT_FACTION_COUNT = 3

# Episodes to run
DEFAULT_EPISODES = 30

# Steps per episode
DEFAULT_STEPS_PER_EPISODE = 5000
```

### Settings Menu

Access settings during gameplay or from main menu:

- **AI Training**: Learning rates, batch sizes, gradient clipping
- **Curriculum Learning**: Difficulty adjustment over time
- **Advanced Loss**: Huber loss configuration
- **System**: Headless mode, dependency checker

Settings are saved to `settings.json` automatically.

### Monitoring Progress

#### TensorBoard (Recommended)
```bash
tensorboard --logdir=RUNTIME_LOGS/Tensorboard_logs
```
Open browser to: `http://localhost:6006`

**Key Metrics to Monitor**:
- Episode Rewards (per faction)
- Agent Performance (success rates by role)
- Resource Collection
- Victory Rates
- Network Losses

#### Logs
- **Error Logs**: `RUNTIME_LOGS/Error_Logs/`
- **General Logs**: `RUNTIME_LOGS/General_Logs/`
- **Profiling**: `Profiling_Stats/`

### Tips for First Run

1. **Start Small**
   - Use 2-3 factions initially
   - Run 1-2 episodes to familiarize with the system
   - Monitor TensorBoard to understand behavior

2. **Observe Agent Behavior**
   - Watch how agents explore
   - Notice collaboration patterns
   - Identify resource collection strategies

3. **Experiment with Settings**
   - Adjust learning rates if training is too slow/fast
   - Modify agent counts to see emergent behaviors
   - Try different episode lengths

4. **Save Early and Often**
   - Training takes time
   - Models are saved automatically (best performance)
   - Check `NEURAL_NETWORK/saved_models/` for saved models

### Common Issues

#### Import Errors
**Problem**: Module not found errors
**Solution**: Run dependency installer from settings menu or manually `pip install -r UTILITIES/requirements.txt`

#### Performance Issues
**Problem**: Slow rendering or low FPS
**Solution**: Enable HEADLESS_MODE or reduce agent count in `utils_config.py`

#### Out of Memory
**Problem**: Memory errors during training
**Solution**: Reduce `MAX_MEMORY_SIZE` in `utils_config.py` or decrease batch size

#### TensorBoard Not Loading
**Problem**: Blank dashboard
**Solution**: Check logs are being created in `RUNTIME_LOGS/Tensorboard_logs/`

### Next Steps

1. **Understand the System**
   - Read `ARCHITECTURE.md` for system overview
   - Check `AGENT/agent_base.py` for agent mechanics
   - Review `NEURAL_NETWORK/PPO_Agent_Network.py` for learning

2. **Customize Behavior**
   - Modify rewards in `agent_behaviours.py`
   - Add new strategies in `agent_factions.py`
   - Create custom behaviors in `Agent_Behaviours/`

3. **Research Opportunities**
   - Experiment with different reward structures
   - Test various network architectures
   - Study emergent behaviors
   - Analyze faction dynamics

4. **Contribute**
   - Add new agent roles
   - Implement new strategies
   - Optimize performance
   - Create visualizations

### Getting Help

- Check logs in `RUNTIME_LOGS/` for errors
- Review `CHANGELOG.md` for recent changes
- Inspect `ARCHITECTURE.md` for system design
- Enable debug logging in `utils_config.py`

### Example Workflow

1. **Initial Training** (1 hour)
   ```
   - Start with 3 factions
   - Run 10 episodes
   - Monitor TensorBoard
   ```

2. **Analysis** (15 minutes)
   ```
   - Check victory rates
   - Analyze resource collection
   - Review agent behaviors
   ```

3. **Refinement** (30 minutes)
   ```
   - Adjust rewards if needed
   - Modify strategies
   - Fine-tune parameters
   ```

4. **Extended Training** (4-6 hours)
   ```
   - Continue from saved models
   - Run 50+ episodes
   - Headless mode for speed
   ```

5. **Evaluation**
   ```
   - Load best models
   - Run evaluation episodes
   - Generate reports
   - Export data for analysis
   ```

### Command Reference

```bash
# Run in visual mode
python main.py

# Run in headless mode (after first setup)
# - Enable in settings or set in utils_config.py

# Launch TensorBoard
tensorboard --logdir=RUNTIME_LOGS/Tensorboard_logs

# Profiling mode
python -m cProfile -o profile.stats main.py

# Analyze profiles
python -m pstats profile.stats
```

### Performance Tips

- Use **headless mode** for long training sessions
- Reduce **render frequency** if visual mode is slow
- Increase **log buffer size** for less I/O overhead
- Cache **sprites and textures** (already implemented)
- Limit **memory size** to prevent overflow

### Resources

- **Project Structure**: See `ARCHITECTURE.md`
- **Configuration**: See `UTILITIES/utils_config.py`
- **Recent Changes**: See `CHANGELOG.md`
- **Code Examples**: Check `AGENT/` and `NEURAL_NETWORK/` directories

