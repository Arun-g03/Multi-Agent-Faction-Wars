
from enum import Enum
from collections import namedtuple


"""
This file contains configuration settings for the game.
So i dont have to change the internal code every time I want to change something

"""


#    ____  _____ ____  _   _  ____    ___  ____ _____ ___ ___  _   _ ____
#   |  _ \| ____| __ )| | | |/ ___|  / _ \|  _ \_   _|_ _/ _ \| \ | / ___|
#   | | | |  _| |  _ \| | | | |  _  | | | | |_) || |  | | | | |  \| \___ \
#   | |_| | |___| |_) | |_| | |_| | | |_| |  __/ | |  | | |_| | |\  |___) |
#   |____/|_____|____/ \___/ \____|  \___/|_|    |_| |___\___/|_| \_|____/
#

HEADLESS_MODE = False
"""Disable pygame game rendering for performance"""
# Customisable
ENABLE_PROFILE_BOOL = False
"""Enable profiling for performance analysis- function calls and execution time"""
"""Used to enable visual debugging"""
ENABLE_LOGGING = True
"""Enable logging for debugging"""

ENABLE_TENSORBOARD = False
"""Enable tensorboard for visualisation"""

ENABLE_PLOTS = False
"""Enable plot and CSV generation for visualization and analysis"""




#    _____ ____ ___ ____   ___  ____  _____   ____  _____ _____ _____ ___ _   _  ____ ____
#   | ____|  _ \_ _/ ___| / _ \|  _ \| ____| / ___|| ____|_   _|_   _|_ _| \ | |/ ___/ ___|
#   |  _| | |_) | |\___ \| | | | | | |  _|   \___ \|  _|   | |   | |  | ||  \| | |  _\___ \
#   | |___|  __/| | ___) | |_| | |_| | | |___   ___) | |___  | |   | |  | || |\  | |_| |___) |
#   |_____|_|  |___|____/ \___/|____/|_____| |____/|_____| |_|   |_| |___|_| \_|\____|____/
#


# Customise as needed!
# More training equals better results... probably.

EPISODES_LIMIT = 30 
"""How many episodes or games to train for"""

STEPS_PER_EPISODE = 20000
"""How many steps to take per episode/ How long should an episode last"""
# Estimated 15k steps in around 5 minutes, need to reconfirm (Depends on
# hardware)

# ===== ADVANCED TRAINING CONFIGURATION =====
# Enhanced training parameters for better AI performance

# Learning Rate Configuration
INITIAL_LEARNING_RATE_PPO = 3e-4  # Optimal PPO learning rate
INITIAL_LEARNING_RATE_HQ = 1e-4   # HQ network learning rate
MIN_LEARNING_RATE = 1e-6          # Minimum learning rate
LEARNING_RATE_DECAY = 0.95        # Learning rate decay factor
LEARNING_RATE_STEP_SIZE = 500     # Steps between LR updates

# Training Optimization
BATCH_SIZE = 64                   # Training batch size
MIN_MEMORY_SIZE = 128             # Minimum memory before training
GAE_LAMBDA = 0.95                 # GAE lambda parameter
VALUE_LOSS_COEFF = 0.5           # Value function loss coefficient
ENTROPY_COEFF = 0.01             # Entropy bonus coefficient
GRADIENT_CLIP_NORM = 0.5         # Gradient clipping norm
PPO_CLIP_RATIO = 0.2             # PPO clip ratio

# Curriculum Learning
ENABLE_CURRICULUM_LEARNING = True
CURRICULUM_DIFFICULTY_STEPS = [5, 10, 15, 20, 25]  # Episodes to increase difficulty
INITIAL_RESOURCE_SPAWN_RATE = 0.3  # Start with fewer resources
FINAL_RESOURCE_SPAWN_RATE = 1.0    # End with normal spawn rate

# Experience Replay
ENABLE_EXPERIENCE_REPLAY = True
REPLAY_BUFFER_SIZE = 10000        # Experience replay buffer size
PRIORITY_REPLAY_ALPHA = 0.6       # Priority replay alpha
PRIORITY_REPLAY_BETA = 0.4        # Priority replay beta

# Multi-Agent Training
ENABLE_MULTI_AGENT_TRAINING = True
INTER_AGENT_COMMUNICATION = True   # Allow agents to share experiences
FACTION_COORDINATION_BONUS = 0.1   # Bonus for coordinated actions

# Training Monitoring
ENABLE_TRAINING_MONITORING = True
SAVE_CHECKPOINT_EVERY = 5         # Save checkpoint every N episodes
EVALUATION_FREQUENCY = 3          # Evaluate performance every N episodes
EARLY_STOPPING_PATIENCE = 10      # Episodes without improvement before stopping

# Advanced Loss Functions
USE_HUBER_LOSS = True             # Use Huber loss for value function
USE_FOCAL_LOSS = False            # Use focal loss for policy (experimental)
LOSS_NORMALIZATION = True         # Normalize losses for stability

# ===== END ADVANCED TRAINING CONFIGURATION =====


#    __  __      _        _
#   |  \/  | ___| |_ _ __(_) ___ ___
#   | |\/| |/ _ \ __| '__| |/ __/ __|
#   | |  | |  __/ |_| |  | | (__\__ \
#   |_|  |_|\___|\__|_|  |_|\___|___/
#

ModelMetrics_Path = "logs/"
"""
Metric path for tensorboard

I suggest leaving this as default
"""

#    ____
#   / ___|  ___ _ __ ___  ___ _ __
#   \___ \ / __| '__/ _ \/ _ \ '_ \
#    ___) | (__| | |  __/  __/ | | |
#   |____/ \___|_|  \___|\___|_| |_|
#

# Pygame Settings
FPS = 60  # Frames per second # Customise as needed!

# Screen Dimensions

ASPECT_RATIO = 0.6  # Customise as needed!
"Allows to change the size of the game window without changing x,y resolution"

SCREEN_WIDTH = 1920 * ASPECT_RATIO
SCREEN_HEIGHT = 1080 * ASPECT_RATIO


# FPS cap and screen dimenions
# Change these to change the game window size


#   __        __         _     _
#   \ \      / /__  _ __| | __| |
#    \ \ /\ / / _ \| '__| |/ _` |
#     \ V  V / (_) | |  | | (_| |
#      \_/\_/ \___/|_|  |_|\__,_|
#

# The size of each cell in the grid
CELL_SIZE = 20

# Number of cells in the world
num_cells_width = 40  # Number of cells horizontally
num_cells_height = 40  # Number of cells vertically

# Precision setting
SUB_TILE_PRECISION = False

# First, calculate the world dimensions (in pixels)
WORLD_WIDTH = num_cells_width * CELL_SIZE
WORLD_HEIGHT = num_cells_height * CELL_SIZE

# Then, calculate scaling based on precision
GRID_WIDTH = num_cells_width
GRID_HEIGHT = num_cells_height

if SUB_TILE_PRECISION:
    WORLD_SCALE_X = WORLD_WIDTH
    WORLD_SCALE_Y = WORLD_HEIGHT
else:
    WORLD_SCALE_X = GRID_WIDTH
    WORLD_SCALE_Y = GRID_HEIGHT



# Perlin Noise Settings
# If RandomiseTerrainBool is set to false, the seed will use Terrain_Seed
RandomiseTerrainBool = True # Customise as needed!
Terrain_Seed = 65   # Customise as needed, 65 is a good default seed in
# combination with the current perlin noise settings


# colours
# General colours used in the game
GREEN = (34, 139, 34)  # Land
BLUE = (30, 144, 255)     # Water
RED = (255, 0, 0)  # Red
APPLE_TREE_colour = (0, 255, 0)  # A brighter green for apple trees
GOLD_colour = (255, 215, 0)  # Gold colour



Grass_Texture_Path = "RENDER\IMAGES\Grass Tiles/Grass 001.png"  # Grass texture path
Water_Texture_Path = "RENDER\IMAGES\Water+.png"  # Water texture path


WaterAnimationToggle = False  # Toggle water animation #Turn off for performance


#    ____
#   |  _ \ ___  ___  ___  _   _ _ __ ___ ___
#   | |_) / _ \/ __|/ _ \| | | | '__/ __/ _ \
#   |  _ <  __/\__ \ (_) | |_| | | | (_|  __/
#   |_| \_\___||___/\___/ \__,_|_|  \___\___|
#


# Perlin Noise Parameters
# Together, these parameters control the terrain's appearance.
# Good as default
NOISE_SCALE = 100 
"""Higher values create larger features, lower values create smaller features Customise as needed!"""
# Higher values add more detail, lower values create smoother terrain #
# Customise as needed!
NOISE_OCTAVES = 4
# Higher values make details more pronounced, lower values make them
# subtler # Customise as needed!
NOISE_PERSISTENCE = 0.7
# Higher values increase the frequency of octaves, lower values decrease
# it # Customise as needed!
NOISE_LACUNARITY = 1.5
# Percentage of the terrain that should be water # Customise as needed!
WATER_COVERAGE = 0.3

# Resource Parameters
TREE_DENSITY = 0.05# Density of apple trees on land # Customise as needed!  #DEFAULT 0.05
# Time in seconds for an apple tree to regrow an apple # Customise as needed!
APPLE_REGEN_TIME = 20 #DEFAULT 20
# Probability of spawning a gold zone # Customise as needed!
GOLD_ZONE_PROBABILITY = 0.05 #Default 0.05
GOLD_SPAWN_DENSITY = 0.03  #Default 0.03 # Density of gold in gold zones # Customise as needed!

Apple_Base_quantity = 5  #default 5 # Base quantity of apples on a tree # Customise as needed!
GoldLump_base_quantity = 5  #default 5 # Base quantity of gold in a gold zone # Customise as needed!

RESOURCE_VICTORY_TARGET_RATIO = 0.7 
"""
Percentage of resources needed to win an episode.
Calculated based on the total resources spawned at the start of the episode.
Default 0.7
 """

# File paths for resource images
# Okay as is
# Tree image path
TREE_IMAGE_PATH = "RENDER\IMAGES\\PixelFlush - Pixel Tree Mega Pack\\pngs\\Apple Tree.png"
GOLD_IMAGE_PATH = "RENDER\IMAGES\\Gold.png"  # Gold image path
# Scale of the gold lump image, needed to match the image size with the
# interactable area
GoldLump_Scale_Img = 2
# Scale of the tree image, needed to match the image size with the
# interactable area
Tree_Scale_Img = 3


#    _____         _      ____  _        _
#   |_   _|_ _ ___| | __ / ___|| |_ __ _| |_ ___
#     | |/ _` |/ __| |/ / \___ \| __/ _` | __/ _ \
#     | | (_| \__ \   <   ___) | || (_| | ||  __/
#     |_|\__,_|___/_|\_\ |____/ \__\__,_|\__\___|
#


class TaskState(Enum):
    """
    State of a task in the task manager.

    Allows for tracking the progress of a task or action.

    TaskState Enum is used to track the state of a task and uses several keywords to track the progress of a task.

    WARNING: DO NOT MODIFY! Changing these values may break functionality.
       """
    NONE = "none"            # No task assigned
    ONGOING = "ongoing"      # Task is actively being executed
    PENDING = "pending"      # Task is assigned but not started
    SUCCESS = "success"      # Task completed successfully
    FAILURE = "failure"      # Task failed
    INTERRUPTED = "interrupted"  # Task was interrupted
    BLOCKED = "blocked"      # Task cannot proceed
    ABANDONED = "abandoned"  # Task was abandoned
    # Task is not valid or cannot be executed (e.g. invalid parameters,
    # unsupported action type for task)
    INVALID = "invalid"
    UNASSIGNED = "unassigned"  # A Task is ready to be picked up


TASK_TYPE_MAPPING = {
    "none": 0,
    "gather": 1,
    "eliminate": 2,
    "explore": 3,
    "move_to": 4
}
"""
Define task type mappings to an integer value.

  """


NETWORK_TYPE_MAPPING: dict = {
    "none": 0,
    "PPOModel": 1,
    "DQNModel": 2,
    "HQ_Network": 3,

}
"""
Mapping of network types to their corresponding interger IDs.

 """

TASK_METHODS_MAPPING = {
    "eliminate": "handle_eliminate_task",
    "gather": "handle_gather_task",
    "explore": "handle_explore_task",
    "move_to": "handle_move_to_task",
    #
}
"""
Connectes a mapping of task type to its handler methods.

 """


HQ_STRATEGY_OPTIONS = [
    "DEFEND_HQ",
    "ATTACK_THREATS",
    "COLLECT_GOLD",
    "COLLECT_FOOD",
    "RECRUIT_GATHERER",
    "RECRUIT_PEACEKEEPER",
    "SWAP_TO_GATHERER",
    "SWAP_TO_PEACEKEEPER",
    "NO_PRIORITY",
]
"""
HQ Strategy options for the HQ.

Defines the broad categories of strategies that the HQ can employ.

"""


ROLE_ACTIONS_MAP = {
    "gatherer": [
        "move_up",
        "move_down",
        "move_left",
        "move_right",
        "mine_gold",
        "forage_apple",
        "heal_with_apple",
        "explore"
    ],
    "peacekeeper": [
        "move_up",
        "move_down",
        "move_left",
        "move_right",
        "patrol",
        "heal_with_apple",
        "eliminate_threat",
        "explore"
    ],

}
"""
Core actions that can be performed by each role.
Makes it easier to define actions for each role.

"""


STATE_FEATURES_MAP = {
    "global_state": [
        "HQ_health",
        "gold_balance",
        "food_balance",
        "resource_count",   # Total resources
        "threat_count",     # Total threats
    ],
    "local_perception": [
        "position_x",
        "position_y",
        "health",
        "nearby_resource_count",  # Count of nearby resources
        "nearby_threat_count"     # Count of nearby threats
    ],
    "task_features": [
        "task_type",        # Encoded task type
        "task_target_x",    # Target X position
        "task_target_y",    # Target Y position
        "current_action"    # Current action
    ]
}
"""
State features mapping structure for the agent.


"""


#       _                    _
#      / \   __ _  ___ _ __ | |_
#     / _ \ / _` |/ _ \ '_ \| __|
#    / ___ \ (_| |  __/ | | | |_
#   /_/   \_\__, |\___|_| |_|\__|
#           |___/

# AGENT
# Agent Render Scale Factor
AGENT_SCALE_FACTOR = 0.08  # Agent render scale factor # Recommend keep default

Agent_field_of_view = 5  # Agent field of view # Customise as needed!
Agent_Interact_Range = 2  # Agent interact range, Anything inside will be interactable


# File paths for agent images
Peacekeeper_PNG = 'RENDER\IMAGES\peacekeeper.png'  # Path to peacekeeper image
Gatherer_PNG = 'RENDER\IMAGES\gatherer.png'  # Path to gatherer image

Gold_Cost_for_Agent = 10  # Gold cost for an agent
Gold_Cost_for_Agent_Swap = 5  # Gold cost for swapping an existing agent to a different role


DEF_AGENT_STATE_SIZE = 18 + len(TASK_TYPE_MAPPING)  # Updated to support enhanced state: 8 core + 2 role + N task one-hot + 6 task info + 2 context

"""
State size breakdown:
- Core state (8): pos_x, pos_y, health, threat_proximity, threat_distance, resource_proximity, resource_distance, hq_proximity
- Role vector (2): gatherer_onehot, peacekeeper_onehot
- Task one-hot: len(TASK_TYPE_MAPPING)
- Task info (6): target_x, target_y, action_norm, norm_dist, task_urgency, task_progress
- Context (2): threat_count_norm, resource_count_norm
"""


#    _____          _   _
#   |  ___|_ _  ___| |_(_) ___  _ __
#   | |_ / _` |/ __| __| |/ _ \| '_ \
#   |  _| (_| | (__| |_| | (_) | | | |
#   |_|  \__,_|\___|\__|_|\___/|_| |_|
#

# HQ
HQ_SPAWN_RADIUS = 2  # Radius around HQ to spawn other HQs
HQ_Agent_Spawn_Radius = 5  # Radius around HQ to spawn agents
Faction_PNG = "RENDER\IMAGES\\castle-7440761_1280.png"

# Team Composition
FACTON_COUNT = 3 # Number of factons # Customise as needed!
# Initial number of gatherers for a single faction # Customise as needed!
INITAL_GATHERER_COUNT = 2
# Initial number of peacekeepers for a single faction # Customise as needed!
INITAL_PEACEKEEPER_COUNT = 2

MAX_AGENTS = 10  # Maximum number of agents per faction

#     ____
#    / ___|__ _ _ __ ___   ___ _ __ __ _
#   | |   / _` | '_ ` _ \ / _ \ '__/ _` |
#   | |__| (_| | | | | | |  __/ | | (_| |
#    \____\__,_|_| |_| |_|\___|_|  \__,_|
#
# Customise as needed!
# Camera
START_CAMERA_X = SCREEN_HEIGHT / 2
START_CAMERA_Y = SCREEN_WIDTH / 2
START_CELL_SIZE = 20
# Customise as needed!

# Zoom and Scaling Parameters
SCALING_FACTOR = 1  # Factor to scale images when zooming in/out
MIN_CELL_SIZE = 2  # Minimum zoom level (cell size)
MAX_ZOOM_OUT_LIMIT = 20  # Maximum zoom-out level (depends on screen size)


#     ___        _                 _                _     ___ ___
#    / __|  _ __| |_ ___ _ __     /_\  __ _ ___ _ _| |_  |_ _|   \
#   | (_| || (_-<  _/ _ \ '  \   / _ \/ _` / -_) ' \  _|  | || |) |
#    \___\_,_/__/\__\___/_|_|_| /_/ \_\__, \___|_||_\__| |___|___/
#                                     |___/


AgentIDStruc = namedtuple("AgentID", ["faction_id", "agent_id"])
"""Identification tag for agents, combining faction ID and agent ID."""


def create_task(self, task_type, target, task_id=None):
    """
    Create a standardised task object.

    Args:
        task_type (str): The type of the task (e.g., "gather", "eliminate", "explore").
        target (any): The target of the task, format depends on task type (e.g., location, resource, threat).
        task_id (str, optional): A unique identifier for the task. Defaults to None.

    Returns:
        dict: A standardised task object including tracking state.

    WARNING: DO NOT MODIFY STRUCTURE!
    """
    task = {
        "type": task_type,
        "target": target,
        "state": TaskState.PENDING  # Track the task's lifecycle from assignment
    }
    if task_id:
        task["id"] = task_id

    return task
