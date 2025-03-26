ascii Art generator: https://patorjk.com/software/taag/#p=display&f=Standard&t=This%20is%20an%20example





DEVELOPED IN PYTHON 3.9.7

you may 
1. ensure 3.9.7 is installed 
2. ensure currently located in correct project directory (/MultiAgent Faction Wars)
3. run the command "py -3.9 game_main.py" to ensure that the correct python version executes the main file to start the project.

The system includes a basic dependency checker, if you are missing any dependencies, the system will prompt you to install them.

Multi agents, collaboration and competitiveness is a core part of this project, 
it may not always be easy to identify during runtime what agents are doing, 
so Tensorboard is used to track various metrics and supports identifying trends overtime.
Tensorboard can be launched and viewed whilst the main app is running or after. metric data is added either in realtime or after each episode, depending on the metric

Running both the project and tensorboard requires that a second terminal/command prompt/windows powershell is used to run TensorBoard in the background

Command to run tensorboard from a specific path

tensorboard --logdir="""C:*CHANGE TO ACTUAL PATH TO WHERE THE PROJECT IS RUNNING*\MultiAgent Faction Wars - Recent Backup\logs""" --port=6006

access once started from http://localhost:6006



🔧 **Customisation:**  
You can view the "utils_config.py" file to see the different customisable parameters that affect the project. e.g., the number of agents, factions, world settings, screen settings, etc.
Using your own diligence, look for the `"Customise as needed!"` comments throughout the code to find configurable settings.  

