**Developed in Python 3.9.7**

## Getting Started

Before running the project:

1. Ensure **Python 3.9.7** is installed. (64bit python 3.9.7 link https://www.python.org/ftp/python/3.9.7/python-3.9.7-amd64.exe)
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

To run TensorBoard:
Use a second terminal/command prompt/PowerShell window and run:

```
tensorboard --logdir="Tensorboard_logs" --port=6006
```

Once started, access TensorBoard at:

```
http://localhost:6006
```

### Customisation

You can customise project settings in the utils_config.py file.
Examples include:

- Number of agents
- Factions
- World and screen settings

Look for comments like "Customise as needed!" throughout the code to easily find configurable sections.
