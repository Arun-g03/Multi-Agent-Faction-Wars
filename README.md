# MultiAgent Faction Wars

A Python-based multi-agent simulation game that explores emergent behaviors through competitive and collaborative agent interactions. This project demonstrates advanced AI concepts including reinforcement learning, neural networks, and complex multi-agent systems.

## Overview

MultiAgent Faction Wars is an interactive simulation environment where multiple AI agents compete and collaborate within faction-based groups. The system features:

- **Multi-Agent Non-Static Environment**: Dynamic, ever-changing simulation where agents must adapt to evolving conditions
- **Multi-Agent Reinforcement Learning**: Agents learn through experience using PPO algorithms (DQN Future works)
- **Faction-Based Competition**: Agents form alliances and compete for resources and territory
- **Emergent Behaviors**: Complex strategies emerge from simple agent interactions
- **Real-Time Visualisation**: Live game rendering with customizable settings
- **TensorBoard Integration**: Comprehensive metrics tracking and analysis
- **Neural Network Architecture**: Custom attention mechanisms and hierarchical networks

The project serves as both an educational tool for understanding multi-agent systems and a research platform for studying emergent behaviors in competitive environments.

**Developed in Python 3.9.7**

## Getting Started

Before running the project:

1. Ensure **Python 3.9.7** is installed. (https://www.python.org/ftp/python/3.9.7/)
2. Make sure you're in the correct project directory:`/MultiAgent Faction Wars`
3. Run the project using:

The system includes a basic dependency checker. If any dependencies are missing, it will prompt you to install them.

### Multi-Agent System

Collaboration and competitiveness are core to this project.
However, agent behaviour might not always be clear during runtime.

TensorBoard Integration
TensorBoard is used to track various metrics and help identify trends over time.
It can be launched while the main app is running or afterwards.
Metric data is logged either in real-time or at the end of each episode, depending on the specific metric.

### Customisation

You can customise project settings in the utils_config.py file.
Examples include:

- Number of agents
- Factions
- World and screen settings

Look for comments like "Customise as needed!" throughout the code to easily find configurable sections.

## Screenshots

### Main Game Interface
![Main Game Interface](Documentation/image.png)

### Setup and Configuration
![Settings Screenshot 1](Documentation/image-1.png)

![Settings Screenshot 2](Documentation/image-2.png)

![Settings Screenshot 3](Documentation/image-3.png)

### Simulation Setup
![Select Simulation Mode Screenshot](Documentation/image-6.png)

![Train fresh or Load model Screenshot](Documentation/image-7.png)

### Gameplay
![Gameplay Screenshot 1](Documentation/image-8.png)

![Gameplay Screenshot 2](Documentation/image-9.png)

### Monitoring and Credits
![Tensorboard Screenshot 5](Documentation/image-5.png)

![Credits Screenshot 4](Documentation/image-4.png)