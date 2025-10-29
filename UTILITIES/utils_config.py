from enum import Enum
from collections import namedtuple
import warnings

# Suppress PyTorch TracerWarnings - these are informational and not errors
try:
    import torch

    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
except ImportError:
    pass  # PyTorch not available yet

# Import settings manager for persistent settings
try:
    from UTILITIES.settings_manager import settings_manager

    _load_persistent_settings = True
except ImportError:
    _load_persistent_settings = False
    settings_manager = None


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

# Load persistent settings or use defaults
HEADLESS_MODE = False
"""Disable pygame game rendering for performance"""

if _load_persistent_settings and settings_manager:
    HEADLESS_MODE = settings_manager.is_headless_mode()
# Customisable
ENABLE_PROFILE_BOOL = False
"""Enable profiling for performance analysis- function calls and execution time"""
"""Used to enable visual debugging"""
ENABLE_LOGGING = True
"""Enable logging for debugging"""

ENABLE_TENSORBOARD = False
"""Enable tensorboard for visualisation"""

VERBOSE_TENSOR_LOGGING = False
"""Enable verbose tensor-to-string conversions in logs (expensive operation)"""

ENABLE_PLOTS = False
"""Enable plot and CSV generation for visualization and analysis"""

FORCE_DEPENDENCY_CHECK = False
"""Force dependency checker to run on next startup"""


# ===== LOAD PERSISTENT CONFIG SETTINGS =====
# Load any saved settings from the settings menu, if available
# Note: This runs before all other constants are defined, so we'll load them later in a function
def load_persistent_config():
    """Load persisted config settings from storage"""
    if _load_persistent_settings and settings_manager:
        saved_config = settings_manager.get_config_settings()
        if saved_config:
            # Update config values from saved settings
            # This allows settings changed in the Settings menu to persist across runs
            for key, value in saved_config.items():
                # Update the module-level constants
                try:
                    globals()[key] = value
                    #  print(f"[CONFIG] Loaded persisted setting: {key} = {value}")
                except Exception as e:
                    print(f"[WARNING] Could not load setting {key}: {e}")


# Call this after all constants are defined
# We'll use a marker to call it at the end of the file


#    _____ ____ ___ ____   ___  ____  _____   ____  _____ _____ ___ _   _  ____ ____
#   | ____|  _ \_ _/ ___| / _ \|  _ \| ____| / ___|| ____|_   _|_   _|_ _| \ | |/ ___/ ___|
#   |  _| | |_) | |\___ \| | | | | | |  _|   \___ \|  _|   | |   | |  | ||  \| | |  _\___ \
#   | |___|  __/| | ___) | |_| | |_| | | |___   ___) | |___  | |   | |  | || |\  | |_| |___) |
#   |_____|_|  |___|____/ \___/|____/|_____| |____/|_____| |_|   |_| |___|_| \_|\____|____/
#


# Customise as needed!
# More training equals better results... probably.

EPISODES_LIMIT = 30
"""How many episodes or games to train for"""

STEPS_PER_EPISODE = 5000
"""How many steps to take per episode/ How long should an episode last"""
# Estimated 15k steps in around 5 minutes, need to reconfirm (Depends on
# hardware)

# ===== ADVANCED TRAINING CONFIGURATION =====
# Enhanced training parameters for better AI performance

# Learning Rate Configuration
INITIAL_LEARNING_RATE_PPO = 3e-4  # Optimal PPO learning rate
INITIAL_LEARNING_RATE_HQ = 1e-4  # HQ network learning rate
MIN_LEARNING_RATE = 1e-6  # Minimum learning rate
LEARNING_RATE_DECAY = 0.95  # Learning rate decay factor
LEARNING_RATE_STEP_SIZE = 500  # Steps between LR updates

# Training Optimization
BATCH_SIZE = 64  # Training batch size
MIN_MEMORY_SIZE = 128  # Minimum memory before training
GAE_LAMBDA = 0.95  # GAE lambda parameter
VALUE_LOSS_COEFF = 0.5  # Value function loss coefficient
ENTROPY_COEFF = 0.01  # Entropy bonus coefficient
GRADIENT_CLIP_NORM = 0.5  # Gradient clipping norm
PPO_CLIP_RATIO = 0.2  # PPO clip ratio

# Curriculum Learning
ENABLE_CURRICULUM_LEARNING = True
CURRICULUM_DIFFICULTY_STEPS = [5, 10, 15, 20, 25]  # Episodes to increase difficulty
INITIAL_RESOURCE_SPAWN_RATE = 0.3  # Start with fewer resources
FINAL_RESOURCE_SPAWN_RATE = 1.0  # End with normal spawn rate

# Experience Replay
ENABLE_EXPERIENCE_REPLAY = True
REPLAY_BUFFER_SIZE = 10000  # Experience replay buffer size
PRIORITY_REPLAY_ALPHA = 0.6  # Priority replay alpha
PRIORITY_REPLAY_BETA = 0.4  # Priority replay beta

# Multi-Agent Training
ENABLE_MULTI_AGENT_TRAINING = True
INTER_AGENT_COMMUNICATION = True  # Allow agents to share experiences
FACTION_COORDINATION_BONUS = 0.1  # Bonus for coordinated actions

# Training Monitoring
ENABLE_TRAINING_MONITORING = True
SAVE_CHECKPOINT_EVERY = 5  # Save checkpoint every N episodes
EVALUATION_FREQUENCY = 3  # Evaluate performance every N episodes
EARLY_STOPPING_PATIENCE = 10  # Episodes without improvement before stopping

# Advanced Loss Functions
USE_HUBER_LOSS = True  # Use Huber loss for value function
USE_FOCAL_LOSS = False  # Use focal loss for policy (experimental)
LOSS_NORMALIZATION = True  # Normalize losses for stability

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
RandomiseTerrainBool = True  # Customise as needed!
Terrain_Seed = 65  # Customise as needed, 65 is a good default seed in
# combination with the current perlin noise settings


# colours
# General colours used in the game
GREEN = (34, 139, 34)  # Land
BLUE = (30, 144, 255)  # Water
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
TREE_DENSITY = (
    0.05  # Density of apple trees on land # Customise as needed!  #DEFAULT 0.05
)
# Time in seconds for an apple tree to regrow an apple # Customise as needed!
APPLE_REGEN_TIME = 20  # DEFAULT 20
# Probability of spawning a gold zone # Customise as needed!
GOLD_ZONE_PROBABILITY = 0.05  # Default 0.05
GOLD_SPAWN_DENSITY = (
    0.03  # Default 0.03 # Density of gold in gold zones # Customise as needed!
)

Apple_Base_quantity = (
    5  # default 5 # Base quantity of apples on a tree # Customise as needed!
)
GoldLump_base_quantity = (
    5  # default 5 # Base quantity of gold in a gold zone # Customise as needed!
)

RESOURCE_VICTORY_TARGET_RATIO = 0.7
"""
Percentage of resources needed to win an episode.
Calculated based on the total resources spawned at the start of the episode.
Default 0.7
 """

# File paths for resource images
# Okay as is
# Tree image path
TREE_IMAGE_PATH = (
    "RENDER\IMAGES\\PixelFlush - Pixel Tree Mega Pack\\pngs\\Apple Tree.png"
)
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

    NONE = "none"  # No task assigned
    ONGOING = "ongoing"  # Task is actively being executed
    PENDING = "pending"  # Task is assigned but not started
    SUCCESS = "success"  # Task completed successfully
    FAILURE = "failure"  # Task failed
    INTERRUPTED = "interrupted"  # Task was interrupted
    BLOCKED = "blocked"  # Task cannot proceed
    ABANDONED = "abandoned"  # Task was abandoned
    # Task is not valid or cannot be executed (e.g. invalid parameters,
    # unsupported action type for task)
    INVALID = "invalid"
    UNASSIGNED = "unassigned"  # A Task is ready to be picked up


TASK_TYPE_MAPPING = {"none": 0, "gather": 1, "eliminate": 2, "explore": 3, "move_to": 4}
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
    "PLANT_TREES",
    "PLANT_GOLD_VEINS",
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


# ============================================================================
# PARAMETRIC STRATEGY SYSTEM
# ============================================================================

HQ_STRATEGY_CATEGORIES = {
    "RECRUIT": ["RECRUIT_GATHERER", "RECRUIT_PEACEKEEPER"],
    "SWAP": ["SWAP_TO_GATHERER", "SWAP_TO_PEACEKEEPER"],
    "RESOURCE": ["COLLECT_GOLD", "COLLECT_FOOD", "PLANT_TREES", "PLANT_GOLD_VEINS"],
    "COMBAT": ["ATTACK_THREATS", "DEFEND_HQ"],
    "PASSIVE": ["NO_PRIORITY"],
}
"""
Groups strategies into high-level categories for parametric learning.
"""

# ============================================================================
# MISSION-ORIENTED TASK SYSTEM
# ============================================================================

class MissionType(Enum):
    """
    High-level mission types that agents can interpret and execute autonomously.
    These replace specific task assignments with flexible objectives.
    """
    SECURE_AREA = "secure_area"          # Secure/patrol a specific area
    GATHER_RESOURCES = "gather_resources"  # Collect resources (any type)
    ELIMINATE_THREATS = "eliminate_threats"  # Find and eliminate threats
    EXPLORE_TERRITORY = "explore_territory"  # Explore unknown areas
    DEFEND_POSITION = "defend_position"   # Defend a specific location
    SUPPORT_ALLIES = "support_allies"     # Support other agents
    SCOUT_INFORMATION = "scout_information"  # Gather intelligence
    ESCORT_MISSION = "escort_mission"    # Escort/protect other agents
    INTERCEPT_TARGET = "intercept_target"  # Intercept moving targets
    CONSOLIDATE_CONTROL = "consolidate_control"  # Strengthen territorial control

class MissionPriority(Enum):
    """Mission priority levels for agent decision making."""
    CRITICAL = "critical"    # Must be done immediately
    HIGH = "high"           # Important, should be prioritized
    MEDIUM = "medium"       # Normal priority
    LOW = "low"             # Can be delayed
    BACKGROUND = "background"  # Low priority, background activity

class MissionState(Enum):
    """Mission execution states."""
    ASSIGNED = "assigned"        # Mission assigned but not started
    IN_PROGRESS = "in_progress"  # Mission actively being executed
    COMPLETED = "completed"      # Mission completed successfully
    FAILED = "failed"          # Mission failed
    ABANDONED = "abandoned"     # Mission abandoned
    SUSPENDED = "suspended"    # Mission temporarily suspended
    INTERRUPTED = "interrupted"  # Mission interrupted by higher priority

MISSION_PARAMETERS = {
    # Mission-specific parameters that agents can learn to interpret
    "target_area": {
        "type": "area",  # Circular area with center and radius
        "description": "Target area for the mission (center_x, center_y, radius)",
    },
    "resource_preference": {
        "type": "preference",  # Resource type preference
        "options": ["gold", "food", "any"],
        "description": "Preferred resource type for gathering missions",
    },
    "threat_tolerance": {
        "type": "continuous",
        "range": [0.0, 1.0],
        "description": "How much risk to accept (0=avoid all threats, 1=ignore threats)",
    },
    "coordination_level": {
        "type": "continuous", 
        "range": [0.0, 1.0],
        "description": "How much to coordinate with other agents (0=independent, 1=highly coordinated)",
    },
    "urgency_factor": {
        "type": "continuous",
        "range": [0.0, 1.0], 
        "description": "How urgently to execute the mission (affects speed vs efficiency trade-offs)",
    },
    "success_criteria": {
        "type": "criteria",
        "description": "What constitutes mission success (e.g., 'collect 5 gold', 'eliminate all threats in area')",
    },
    "time_limit": {
        "type": "discrete",
        "range": [1, 1000],  # Steps
        "description": "Maximum time to spend on this mission",
    },
    "fallback_mission": {
        "type": "mission_type",
        "description": "What to do if this mission fails",
    },
}

# Mission complexity levels for agent learning progression
MISSION_COMPLEXITY_LEVELS = {
    "BEGINNER": ["GATHER_RESOURCES", "EXPLORE_TERRITORY"],
    "INTERMEDIATE": ["SECURE_AREA", "DEFEND_POSITION", "ELIMINATE_THREATS"],
    "ADVANCED": ["SUPPORT_ALLIES", "SCOUT_INFORMATION", "INTERCEPT_TARGET"],
    "EXPERT": ["ESCORT_MISSION", "CONSOLIDATE_CONTROL"],
}

# ============================================================================
# ADAPTIVE AGENT ACTION SELECTION SYSTEM
# ============================================================================

class AdaptiveBehavior(Enum):
    """Types of adaptive behaviors agents can exhibit when facing obstacles."""
    PERSISTENT = "persistent"          # Keep trying the same approach
    EXPLORATORY = "exploratory"        # Try alternative approaches
    COLLABORATIVE = "collaborative"    # Seek help from other agents
    DEFENSIVE = "defensive"          # Switch to defensive/survival mode
    OPPORTUNISTIC = "opportunistic"   # Take advantage of new opportunities
    RETREAT = "retreat"              # Fall back to safer position
    ESCALATE = "escalate"            # Request higher-level intervention

class FailureType(Enum):
    """Types of failures that can occur during task execution."""
    RESOURCE_UNAVAILABLE = "resource_unavailable"  # Target resource not found/depleted
    THREAT_TOO_STRONG = "threat_too_strong"        # Enemy too powerful
    PATH_BLOCKED = "path_blocked"                  # Cannot reach target
    TIME_EXCEEDED = "time_exceeded"                # Task taking too long
    HEALTH_LOW = "health_low"                     # Agent health critical
    COMMUNICATION_LOST = "communication_lost"      # Lost contact with HQ/allies
    UNKNOWN_OBSTACLE = "unknown_obstacle"         # Unexpected situation

class AdaptiveStrategy(Enum):
    """Adaptive strategies for handling different failure types."""
    RETRY_WITH_MODIFICATION = "retry_with_modification"  # Try again with different approach
    SWITCH_TARGET = "switch_target"                     # Find alternative target
    REQUEST_SUPPORT = "request_support"                 # Ask for help from other agents
    ESCALATE_TO_HQ = "escalate_to_hq"                   # Request new mission from HQ
    EMERGENCY_PROTOCOL = "emergency_protocol"           # Switch to survival mode
    OPPORTUNISTIC_ACTION = "opportunistic_action"       # Take advantage of current situation
    RETREAT = "retreat"                                 # Fall back to safer position

ADAPTIVE_RESPONSES = {
    FailureType.RESOURCE_UNAVAILABLE: [
        AdaptiveStrategy.SWITCH_TARGET,
        AdaptiveStrategy.RETRY_WITH_MODIFICATION,
        AdaptiveStrategy.OPPORTUNISTIC_ACTION,
    ],
    FailureType.THREAT_TOO_STRONG: [
        AdaptiveStrategy.REQUEST_SUPPORT,
        AdaptiveStrategy.RETREAT,
        AdaptiveStrategy.ESCALATE_TO_HQ,
    ],
    FailureType.PATH_BLOCKED: [
        AdaptiveStrategy.RETRY_WITH_MODIFICATION,
        AdaptiveStrategy.OPPORTUNISTIC_ACTION,
        AdaptiveStrategy.SWITCH_TARGET,
    ],
    FailureType.TIME_EXCEEDED: [
        AdaptiveStrategy.ESCALATE_TO_HQ,
        AdaptiveStrategy.OPPORTUNISTIC_ACTION,
        AdaptiveStrategy.EMERGENCY_PROTOCOL,
    ],
    FailureType.HEALTH_LOW: [
        AdaptiveStrategy.EMERGENCY_PROTOCOL,
        AdaptiveStrategy.RETREAT,
        AdaptiveStrategy.REQUEST_SUPPORT,
    ],
    FailureType.COMMUNICATION_LOST: [
        AdaptiveStrategy.EMERGENCY_PROTOCOL,
        AdaptiveStrategy.OPPORTUNISTIC_ACTION,
        AdaptiveStrategy.RETREAT,
    ],
    FailureType.UNKNOWN_OBSTACLE: [
        AdaptiveStrategy.ESCALATE_TO_HQ,
        AdaptiveStrategy.EMERGENCY_PROTOCOL,
        AdaptiveStrategy.OPPORTUNISTIC_ACTION,
    ],
}

# Adaptive behavior parameters for agent learning
ADAPTIVE_PARAMETERS = {
    "failure_tolerance": {
        "type": "continuous",
        "range": [0.0, 1.0],
        "description": "How many failures before giving up (0=give up immediately, 1=never give up)",
    },
    "exploration_tendency": {
        "type": "continuous", 
        "range": [0.0, 1.0],
        "description": "How likely to try alternative approaches (0=stick to plan, 1=always explore)",
    },
    "collaboration_willingness": {
        "type": "continuous",
        "range": [0.0, 1.0], 
        "description": "How likely to request help from other agents (0=independent, 1=always collaborate)",
    },
    "risk_tolerance": {
        "type": "continuous",
        "range": [0.0, 1.0],
        "description": "How much risk to accept in adaptive actions (0=very cautious, 1=very bold)",
    },
    "escalation_threshold": {
        "type": "continuous",
        "range": [0.0, 1.0],
        "description": "When to escalate to HQ (0=escalate immediately, 1=try everything first)",
    },
}

# ============================================================================
# HIERARCHICAL REWARD SYSTEM
# ============================================================================

class RewardLevel(Enum):
    """Levels of reward in the hierarchical system."""
    AGENT_TACTICAL = "agent_tactical"      # Individual agent task execution
    AGENT_COORDINATION = "agent_coordination"  # Multi-agent coordination
    HQ_STRATEGIC = "hq_strategic"          # HQ strategy selection and execution
    HQ_META = "hq_meta"                   # HQ meta-learning and adaptation

class RewardComponent(Enum):
    """Components of hierarchical rewards."""
    # Agent-level components
    TASK_COMPLETION = "task_completion"           # Success/failure of individual tasks
    EFFICIENCY = "efficiency"                    # How efficiently tasks are completed
    COORDINATION = "coordination"                 # How well agents coordinate
    ADAPTATION = "adaptation"                     # How well agents adapt to failures
    SURVIVAL = "survival"                         # Agent survival and health maintenance
    
    # HQ-level components
    STRATEGY_SELECTION = "strategy_selection"     # Quality of strategy choice
    STRATEGY_EXECUTION = "strategy_execution"     # How well strategies are executed
    RESOURCE_MANAGEMENT = "resource_management"    # Resource allocation and usage
    AGENT_MANAGEMENT = "agent_management"         # Agent recruitment and role assignment
    THREAT_RESPONSE = "threat_response"          # Response to threats and opportunities
    MISSION_SUCCESS = "mission_success"           # Overall mission objective achievement

# Hierarchical reward weights and scaling factors
HIERARCHICAL_REWARD_CONFIG = {
    # Agent-level reward weights
    "agent_weights": {
        RewardComponent.TASK_COMPLETION: 1.0,     # Base task completion reward
        RewardComponent.EFFICIENCY: 0.3,          # Efficiency bonus/penalty
        RewardComponent.COORDINATION: 0.4,        # Multi-agent coordination bonus
        RewardComponent.ADAPTATION: 0.2,          # Adaptive behavior bonus
        RewardComponent.SURVIVAL: 0.5,            # Survival and health maintenance
    },
    
    # HQ-level reward weights
    "hq_weights": {
        RewardComponent.STRATEGY_SELECTION: 2.0,   # Strategy choice quality
        RewardComponent.STRATEGY_EXECUTION: 1.5,   # Strategy execution quality
        RewardComponent.RESOURCE_MANAGEMENT: 1.0,   # Resource management
        RewardComponent.AGENT_MANAGEMENT: 1.2,     # Agent management
        RewardComponent.THREAT_RESPONSE: 1.8,      # Threat response
        RewardComponent.MISSION_SUCCESS: 3.0,      # Mission success (highest weight)
    },
    
    # Hierarchical scaling factors
    "scaling": {
        "agent_to_hq_feedback": 0.1,              # How much agent success affects HQ reward
        "hq_to_agent_guidance": 0.2,              # How much HQ strategy affects agent rewards
        "coordination_bonus": 0.3,                 # Bonus for coordinated actions
        "adaptation_bonus": 0.15,                  # Bonus for adaptive behavior
        "mission_progress": 0.25,                  # Mission progress contribution
    },
    
    # Reward normalization factors
    "normalization": {
        "max_agent_reward": 5.0,                  # Maximum agent reward per step
        "max_hq_reward": 10.0,                    # Maximum HQ reward per step
        "episode_reward_scale": 0.1,              # Episode-level reward scaling
    },
}

# Experience reporting configuration
EXPERIENCE_REPORTING_CONFIG = {
    "agent_report_frequency": 1,                 # Steps between agent reports to HQ
    "hq_evaluation_frequency": 5,                # Steps between HQ strategy evaluations
    "coordination_window": 10,                    # Steps to look back for coordination rewards
    "adaptation_window": 5,                      # Steps to look back for adaptation rewards
    "mission_evaluation_steps": 20,               # Steps between mission progress evaluations
}

# Mission success criteria for hierarchical rewards
MISSION_SUCCESS_CRITERIA = {
    "resource_collection": {
        "gold_target": 0.3,                       # Fraction of available gold to collect
        "food_target": 0.3,                       # Fraction of available food to collect
        "efficiency_threshold": 0.7,              # Minimum efficiency for success
    },
    "threat_elimination": {
        "elimination_rate": 0.8,                  # Fraction of threats to eliminate
        "response_time": 50,                       # Maximum steps to respond to threats
        "casualty_threshold": 0.2,                # Maximum casualty rate
    },
    "territory_control": {
        "control_radius": 100,                    # Radius of territory control
        "maintenance_time": 100,                  # Steps to maintain control
        "expansion_bonus": 0.5,                   # Bonus for territory expansion
    },
    "agent_coordination": {
        "coordination_threshold": 0.6,            # Minimum coordination score
        "communication_efficiency": 0.8,          # Communication efficiency threshold
        "task_distribution": 0.7,                 # Balanced task distribution threshold
    },
}

# ============================================================================
# MULTI-AGENT COORDINATION LEARNING SYSTEM
# ============================================================================

class CommunicationType(Enum):
    """Types of learned communication between agents."""
    RESOURCE_SHARING = "resource_sharing"          # Share resource locations
    THREAT_WARNING = "threat_warning"              # Warn about threats
    TASK_COORDINATION = "task_coordination"        # Coordinate task execution
    HELP_REQUEST = "help_request"                  # Request assistance
    STATUS_UPDATE = "status_update"                # Share current status
    STRATEGY_SYNC = "strategy_sync"                # Synchronize strategies
    FORMATION_REQUEST = "formation_request"        # Request formation changes
    EMERGENCY_ALERT = "emergency_alert"           # Emergency situations

class CoordinationStrategy(Enum):
    """Learned coordination strategies."""
    INDEPENDENT = "independent"                    # Work independently
    PAIRED = "paired"                              # Work in pairs
    FORMATION = "formation"                        # Work in formations
    SWARM = "swarm"                               # Swarm behavior
    HIERARCHICAL = "hierarchical"                 # Hierarchical coordination
    EMERGENCY_RESPONSE = "emergency_response"     # Emergency coordination

# Communication network configuration
COMMUNICATION_CONFIG = {
    # Communication range and frequency
    "communication_range": 150,                    # Maximum communication distance
    "communication_frequency": 5,                 # Steps between communication attempts
    "message_queue_size": 10,                     # Maximum messages per agent
    "communication_cost": 0.1,                    # Cost of communication (energy/time)
    
    # Learning parameters
    "learning_rate": 0.001,                       # Learning rate for communication network
    "memory_size": 1000,                          # Size of communication memory buffer
    "coordination_window": 20,                     # Steps to look back for coordination rewards
    "communication_success_threshold": 0.7,       # Threshold for successful communication
    
    # Message types and their properties
    "message_types": {
        CommunicationType.RESOURCE_SHARING: {
            "priority": 1,                        # Message priority (1=highest)
            "expiry_steps": 50,                    # Steps before message expires
            "max_recipients": 3,                   # Maximum recipients
            "success_reward": 0.2,                # Reward for successful communication
        },
        CommunicationType.THREAT_WARNING: {
            "priority": 1,
            "expiry_steps": 30,
            "max_recipients": 5,
            "success_reward": 0.3,
        },
        CommunicationType.TASK_COORDINATION: {
            "priority": 2,
            "expiry_steps": 40,
            "max_recipients": 2,
            "success_reward": 0.25,
        },
        CommunicationType.HELP_REQUEST: {
            "priority": 1,
            "expiry_steps": 20,
            "max_recipients": 3,
            "success_reward": 0.4,
        },
        CommunicationType.STATUS_UPDATE: {
            "priority": 3,
            "expiry_steps": 60,
            "max_recipients": 4,
            "success_reward": 0.1,
        },
        CommunicationType.STRATEGY_SYNC: {
            "priority": 2,
            "expiry_steps": 80,
            "max_recipients": 6,
            "success_reward": 0.15,
        },
        CommunicationType.FORMATION_REQUEST: {
            "priority": 2,
            "expiry_steps": 30,
            "max_recipients": 4,
            "success_reward": 0.2,
        },
        CommunicationType.EMERGENCY_ALERT: {
            "priority": 1,
            "expiry_steps": 10,
            "max_recipients": 8,
            "success_reward": 0.5,
        },
    },
    
    # Coordination strategies and their properties
    "coordination_strategies": {
        CoordinationStrategy.INDEPENDENT: {
            "communication_frequency": 1,          # Low communication
            "coordination_bonus": 0.0,            # No coordination bonus
            "learning_rate_multiplier": 0.5,       # Slower learning
        },
        CoordinationStrategy.PAIRED: {
            "communication_frequency": 3,
            "coordination_bonus": 0.2,
            "learning_rate_multiplier": 1.0,
        },
        CoordinationStrategy.FORMATION: {
            "communication_frequency": 5,
            "coordination_bonus": 0.4,
            "learning_rate_multiplier": 1.2,
        },
        CoordinationStrategy.SWARM: {
            "communication_frequency": 8,
            "coordination_bonus": 0.6,
            "learning_rate_multiplier": 1.5,
        },
        CoordinationStrategy.HIERARCHICAL: {
            "communication_frequency": 4,
            "coordination_bonus": 0.5,
            "learning_rate_multiplier": 1.3,
        },
        CoordinationStrategy.EMERGENCY_RESPONSE: {
            "communication_frequency": 10,
            "coordination_bonus": 0.8,
            "learning_rate_multiplier": 2.0,
        },
    },
    
    # State encoding for communication
    "communication_state_size": 20,               # Size of communication state vector
    "message_encoding_size": 16,                   # Size of message encoding
    "coordination_context_size": 12,               # Size of coordination context
    
    # Reward shaping for coordination
    "coordination_rewards": {
        "successful_communication": 0.1,          # Reward for successful communication
        "coordination_success": 0.3,               # Reward for successful coordination
        "formation_maintenance": 0.2,              # Reward for maintaining formation
        "emergency_response": 0.5,                 # Reward for emergency coordination
        "communication_efficiency": 0.15,         # Reward for efficient communication
        "coordination_learning": 0.1,               # Reward for learning new coordination patterns
    },
}

# Communication state components
COMMUNICATION_STATE_COMPONENTS = {
    "agent_position": 2,                          # Agent's position (x, y)
    "agent_health": 1,                           # Agent's health
    "agent_role": 2,                              # Agent's role (one-hot)
    "current_task": 4,                            # Current task type (one-hot)
    "task_progress": 1,                           # Task progress (0-1)
    "nearby_agents": 1,                           # Number of nearby agents (normalized)
    "communication_history": 1,                   # Recent communication success rate
    "coordination_score": 1,                      # Current coordination score
    "emergency_level": 1,                         # Emergency level (0-1)
    "resource_availability": 1,                   # Resource availability in area
}

# Total communication state size
COMMUNICATION_STATE_SIZE = sum(COMMUNICATION_STATE_COMPONENTS.values())

# ============================================================================
# EXPERIENCE SHARING SYSTEM
# ============================================================================

class ExperienceType(Enum):
    """Types of experiences that can be shared between agents."""
    SUCCESSFUL_TASK = "successful_task"              # Successful task completion
    FAILED_TASK = "failed_task"                      # Failed task attempt
    ADAPTIVE_RECOVERY = "adaptive_recovery"          # Successful adaptive recovery
    COORDINATION_SUCCESS = "coordination_success"    # Successful coordination
    COMMUNICATION_SUCCESS = "communication_success"  # Successful communication
    STRATEGY_INSIGHT = "strategy_insight"            # Strategic insights
    THREAT_ENCOUNTER = "threat_encounter"            # Threat encounter experience
    RESOURCE_DISCOVERY = "resource_discovery"        # Resource discovery experience

class SharingStrategy(Enum):
    """Strategies for sharing experiences between agents."""
    IMMEDIATE = "immediate"                          # Share immediately
    BATCHED = "batched"                              # Share in batches
    SELECTIVE = "selective"                          # Share only high-value experiences
    HIERARCHICAL = "hierarchical"                    # Share through hierarchy
    PEER_TO_PEER = "peer_to_peer"                    # Direct peer-to-peer sharing
    COLLECTIVE = "collective"                        # Collective sharing pool

# Experience sharing configuration
EXPERIENCE_SHARING_CONFIG = {
    # Sharing parameters
    "sharing_frequency": 10,                         # Steps between sharing attempts
    "max_shared_experiences": 50,                     # Maximum experiences to share per agent
    "experience_lifetime": 100,                       # Steps before experience expires
    "sharing_range": 200,                            # Maximum distance for sharing
    "sharing_cost": 0.05,                            # Cost of sharing experiences
    
    # Learning parameters
    "shared_learning_rate": 0.0005,                 # Learning rate for shared experiences
    "experience_weight_decay": 0.95,                 # Weight decay for experience importance
    "similarity_threshold": 0.7,                     # Threshold for experience similarity
    "value_threshold": 0.3,                          # Minimum value threshold for sharing
    
    # Experience types and their properties
    "experience_types": {
        ExperienceType.SUCCESSFUL_TASK: {
            "priority": 1,                           # High priority
            "sharing_probability": 0.8,               # High sharing probability
            "learning_weight": 1.0,                  # Full learning weight
            "expiry_steps": 150,                     # Longer expiry
        },
        ExperienceType.FAILED_TASK: {
            "priority": 2,
            "sharing_probability": 0.6,
            "learning_weight": 0.7,
            "expiry_steps": 100,
        },
        ExperienceType.ADAPTIVE_RECOVERY: {
            "priority": 1,
            "sharing_probability": 0.9,
            "learning_weight": 1.2,
            "expiry_steps": 200,
        },
        ExperienceType.COORDINATION_SUCCESS: {
            "priority": 1,
            "sharing_probability": 0.8,
            "learning_weight": 1.1,
            "expiry_steps": 120,
        },
        ExperienceType.COMMUNICATION_SUCCESS: {
            "priority": 2,
            "sharing_probability": 0.7,
            "learning_weight": 0.9,
            "expiry_steps": 100,
        },
        ExperienceType.STRATEGY_INSIGHT: {
            "priority": 1,
            "sharing_probability": 0.9,
            "learning_weight": 1.3,
            "expiry_steps": 300,
        },
        ExperienceType.THREAT_ENCOUNTER: {
            "priority": 1,
            "sharing_probability": 0.8,
            "learning_weight": 1.0,
            "expiry_steps": 80,
        },
        ExperienceType.RESOURCE_DISCOVERY: {
            "priority": 2,
            "sharing_probability": 0.7,
            "learning_weight": 0.8,
            "expiry_steps": 200,
        },
    },
    
    # Sharing strategies and their properties
    "sharing_strategies": {
        SharingStrategy.IMMEDIATE: {
            "delay_steps": 0,                        # No delay
            "batch_size": 1,                         # Share one at a time
            "efficiency": 0.8,                       # High efficiency
        },
        SharingStrategy.BATCHED: {
            "delay_steps": 5,                        # Small delay
            "batch_size": 10,                        # Share in batches
            "efficiency": 0.9,                       # Very high efficiency
        },
        SharingStrategy.SELECTIVE: {
            "delay_steps": 2,                        # Small delay
            "batch_size": 5,                         # Small batches
            "efficiency": 0.7,                       # Medium efficiency
        },
        SharingStrategy.HIERARCHICAL: {
            "delay_steps": 3,                        # Medium delay
            "batch_size": 8,                         # Medium batches
            "efficiency": 0.8,                       # High efficiency
        },
        SharingStrategy.PEER_TO_PEER: {
            "delay_steps": 1,                        # Minimal delay
            "batch_size": 3,                         # Small batches
            "efficiency": 0.6,                       # Lower efficiency
        },
        SharingStrategy.COLLECTIVE: {
            "delay_steps": 4,                        # Medium delay
            "batch_size": 15,                        # Large batches
            "efficiency": 0.9,                       # Very high efficiency
        },
    },
    
    # Experience encoding parameters
    "experience_encoding_size": 32,                  # Size of experience encoding
    "state_encoding_size": 64,                       # Size of state encoding
    "action_encoding_size": 16,                      # Size of action encoding
    "context_encoding_size": 24,                     # Size of context encoding
    
    # Reward shaping for experience sharing
    "sharing_rewards": {
        "successful_sharing": 0.1,                   # Reward for successful sharing
        "valuable_experience": 0.2,                  # Reward for sharing valuable experiences
        "learning_from_others": 0.15,                # Reward for learning from others
        "teaching_others": 0.1,                      # Reward for teaching others
        "collective_improvement": 0.25,              # Reward for collective improvement
    },
}

# Experience similarity metrics
EXPERIENCE_SIMILARITY_METRICS = {
    "state_similarity_weight": 0.4,                 # Weight for state similarity
    "action_similarity_weight": 0.3,                 # Weight for action similarity
    "context_similarity_weight": 0.2,                # Weight for context similarity
    "outcome_similarity_weight": 0.1,                # Weight for outcome similarity
}

# Collective learning parameters
COLLECTIVE_LEARNING_CONFIG = {
    "collective_memory_size": 2000,                  # Size of collective memory pool
    "experience_pool_update_frequency": 20,          # Steps between pool updates
    "collective_learning_rate": 0.0003,              # Learning rate for collective learning
    "experience_diversity_bonus": 0.1,               # Bonus for diverse experiences
    "knowledge_transfer_efficiency": 0.8,            # Efficiency of knowledge transfer
}

# ============================================================================
# LEARNED STATE REPRESENTATION SYSTEM
# ============================================================================

class StateRepresentationType(Enum):
    """Types of learned state representations."""
    RAW_FEATURES = "raw_features"                    # Raw feature extraction
    ABSTRACT_CONCEPTS = "abstract_concepts"          # Abstract concept learning
    TEMPORAL_PATTERNS = "temporal_patterns"         # Temporal pattern recognition
    SPATIAL_RELATIONS = "spatial_relations"          # Spatial relationship learning
    CAUSAL_MODELS = "causal_models"                  # Causal relationship modeling
    HIERARCHICAL_FEATURES = "hierarchical_features"  # Hierarchical feature learning

class StateEncoderType(Enum):
    """Types of state encoders."""
    CONVOLUTIONAL = "convolutional"                  # Convolutional encoder
    RECURRENT = "recurrent"                          # Recurrent encoder (LSTM/GRU)
    TRANSFORMER = "transformer"                      # Transformer encoder
    ATTENTION = "attention"                          # Attention-based encoder
    AUTOENCODER = "autoencoder"                      # Autoencoder
    VARIATIONAL = "variational"                      # Variational autoencoder

# Learned state representation configuration
LEARNED_STATE_CONFIG = {
    # State representation parameters
    "representation_size": 128,                      # Size of learned state representation
    "input_state_size": 29,                          # Size of input state (current HQ state size)
    "hidden_size": 256,                              # Size of hidden layers
    "num_attention_heads": 8,                        # Number of attention heads
    "attention_dropout": 0.1,                        # Dropout for attention layers
    
    # Learning parameters
    "learning_rate": 0.001,                          # Learning rate for state encoder
    "representation_learning_rate": 0.0005,           # Learning rate for representation learning
    "contrastive_learning_rate": 0.0003,             # Learning rate for contrastive learning
    "temporal_window": 20,                           # Steps to look back for temporal patterns
    "spatial_window": 50,                            # Radius for spatial relationship learning
    
    # State representation types and their properties
    "representation_types": {
        StateRepresentationType.RAW_FEATURES: {
            "priority": 3,                           # Lower priority
            "learning_weight": 0.5,                  # Lower learning weight
            "update_frequency": 10,                  # Update every 10 steps
            "complexity": 1,                         # Low complexity
        },
        StateRepresentationType.ABSTRACT_CONCEPTS: {
            "priority": 1,                           # High priority
            "learning_weight": 1.0,                  # Full learning weight
            "update_frequency": 5,                   # Update every 5 steps
            "complexity": 3,                         # High complexity
        },
        StateRepresentationType.TEMPORAL_PATTERNS: {
            "priority": 1,                           # High priority
            "learning_weight": 1.2,                  # Higher learning weight
            "update_frequency": 3,                   # Update every 3 steps
            "complexity": 4,                         # Very high complexity
        },
        StateRepresentationType.SPATIAL_RELATIONS: {
            "priority": 2,                           # Medium priority
            "learning_weight": 0.8,                  # Medium learning weight
            "update_frequency": 7,                   # Update every 7 steps
            "complexity": 2,                         # Medium complexity
        },
        StateRepresentationType.CAUSAL_MODELS: {
            "priority": 1,                           # High priority
            "learning_weight": 1.1,                  # Higher learning weight
            "update_frequency": 5,                   # Update every 5 steps
            "complexity": 4,                         # Very high complexity
        },
        StateRepresentationType.HIERARCHICAL_FEATURES: {
            "priority": 1,                           # High priority
            "learning_weight": 1.0,                  # Full learning weight
            "update_frequency": 4,                   # Update every 4 steps
            "complexity": 3,                         # High complexity
        },
    },
    
    # State encoder types and their properties
    "encoder_types": {
        StateEncoderType.CONVOLUTIONAL: {
            "kernel_size": 3,                        # Convolutional kernel size
            "stride": 1,                             # Convolutional stride
            "padding": 1,                            # Convolutional padding
            "efficiency": 0.8,                       # Computational efficiency
        },
        StateEncoderType.RECURRENT: {
            "hidden_size": 128,                      # RNN hidden size
            "num_layers": 2,                         # Number of RNN layers
            "dropout": 0.1,                         # RNN dropout
            "efficiency": 0.7,                       # Computational efficiency
        },
        StateEncoderType.TRANSFORMER: {
            "num_layers": 4,                         # Number of transformer layers
            "d_model": 128,                          # Model dimension
            "d_ff": 512,                             # Feed-forward dimension
            "efficiency": 0.6,                       # Computational efficiency
        },
        StateEncoderType.ATTENTION: {
            "num_heads": 8,                          # Number of attention heads
            "key_dim": 64,                           # Key dimension
            "value_dim": 64,                         # Value dimension
            "efficiency": 0.7,                       # Computational efficiency
        },
        StateEncoderType.AUTOENCODER: {
            "encoder_layers": [256, 128, 64],        # Encoder layer sizes
            "decoder_layers": [64, 128, 256],        # Decoder layer sizes
            "latent_dim": 32,                        # Latent dimension
            "efficiency": 0.8,                       # Computational efficiency
        },
        StateEncoderType.VARIATIONAL: {
            "encoder_layers": [256, 128, 64],        # Encoder layer sizes
            "decoder_layers": [64, 128, 256],        # Decoder layer sizes
            "latent_dim": 32,                        # Latent dimension
            "kl_weight": 0.1,                        # KL divergence weight
            "efficiency": 0.7,                       # Computational efficiency
        },
    },
    
    # State representation learning objectives
    "learning_objectives": {
        "reconstruction_loss": 0.3,                  # Weight for reconstruction loss
        "contrastive_loss": 0.2,                     # Weight for contrastive loss
        "temporal_consistency": 0.2,                 # Weight for temporal consistency
        "spatial_consistency": 0.15,                 # Weight for spatial consistency
        "causal_consistency": 0.15,                  # Weight for causal consistency
    },
    
    # Reward shaping for state representation learning
    "representation_rewards": {
        "successful_encoding": 0.1,                  # Reward for successful encoding
        "representation_quality": 0.15,               # Reward for high-quality representations
        "pattern_discovery": 0.2,                    # Reward for discovering patterns
        "concept_formation": 0.25,                   # Reward for forming concepts
        "temporal_prediction": 0.2,                  # Reward for temporal predictions
        "spatial_understanding": 0.15,               # Reward for spatial understanding
    },
}

# State representation components
STATE_REPRESENTATION_COMPONENTS = {
    "raw_features": 32,                              # Raw feature extraction
    "abstract_concepts": 24,                         # Abstract concept learning
    "temporal_patterns": 28,                         # Temporal pattern recognition
    "spatial_relations": 20,                         # Spatial relationship learning
    "causal_models": 16,                             # Causal relationship modeling
    "hierarchical_features": 8,                      # Hierarchical feature learning
}

# Total state representation size
STATE_REPRESENTATION_SIZE = sum(STATE_REPRESENTATION_COMPONENTS.values())

# ============================================================================
# STRATEGY COMPOSITION SYSTEM
# ============================================================================

class StrategyCompositionType(Enum):
    """Types of strategy composition."""
    SEQUENTIAL = "sequential"                    # Execute strategies in sequence
    PARALLEL = "parallel"                        # Execute strategies simultaneously
    CONDITIONAL = "conditional"                  # Execute based on conditions
    HIERARCHICAL = "hierarchical"               # Nested strategy composition
    ADAPTIVE = "adaptive"                       # Adapt based on execution results
    EMERGENT = "emergent"                       # Emergent strategy discovery

class StrategySequenceType(Enum):
    """Types of strategy sequences."""
    LINEAR = "linear"                           # Simple linear sequence
    BRANCHING = "branching"                     # Conditional branching
    LOOPING = "looping"                         # Repetitive execution
    CONVERGENT = "convergent"                   # Multiple paths to same goal
    DIVERGENT = "divergent"                     # Single path to multiple goals
    RECURSIVE = "recursive"                     # Self-referential strategies

class StrategyGoalType(Enum):
    """Types of strategy goals."""
    RESOURCE_ACQUISITION = "resource_acquisition"  # Acquire specific resources
    THREAT_ELIMINATION = "threat_elimination"        # Eliminate threats
    TERRITORY_CONTROL = "territory_control"         # Control territory
    AGENT_MANAGEMENT = "agent_management"           # Manage agent roles/counts
    DEFENSIVE_POSITIONING = "defensive_positioning" # Defensive positioning
    OFFENSIVE_EXPANSION = "offensive_expansion"     # Offensive expansion
    ECONOMIC_GROWTH = "economic_growth"             # Economic development
    STRATEGIC_ADVANTAGE = "strategic_advantage"     # Gain strategic advantage

# Strategy composition configuration
STRATEGY_COMPOSITION_CONFIG = {
    # Composition parameters
    "max_sequence_length": 10,                    # Maximum strategies in a sequence
    "max_parallel_strategies": 5,                 # Maximum parallel strategies
    "composition_depth": 3,                       # Maximum nesting depth
    "strategy_timeout": 100,                      # Steps before strategy times out
    "composition_learning_rate": 0.001,          # Learning rate for composition
    "sequence_learning_rate": 0.0005,            # Learning rate for sequences
    
    # Strategy composition types and their properties
    "composition_types": {
        StrategyCompositionType.SEQUENTIAL: {
            "priority": 1,                        # High priority
            "learning_weight": 1.0,               # Full learning weight
            "execution_efficiency": 0.8,          # Execution efficiency
            "complexity": 2,                      # Medium complexity
        },
        StrategyCompositionType.PARALLEL: {
            "priority": 1,                        # High priority
            "learning_weight": 1.2,               # Higher learning weight
            "execution_efficiency": 0.9,          # High execution efficiency
            "complexity": 3,                      # High complexity
        },
        StrategyCompositionType.CONDITIONAL: {
            "priority": 1,                        # High priority
            "learning_weight": 1.1,               # Higher learning weight
            "execution_efficiency": 0.7,          # Medium execution efficiency
            "complexity": 4,                      # Very high complexity
        },
        StrategyCompositionType.HIERARCHICAL: {
            "priority": 2,                        # Medium priority
            "learning_weight": 0.9,               # Medium learning weight
            "execution_efficiency": 0.6,          # Lower execution efficiency
            "complexity": 5,                      # Very high complexity
        },
        StrategyCompositionType.ADAPTIVE: {
            "priority": 1,                        # High priority
            "learning_weight": 1.3,               # Highest learning weight
            "execution_efficiency": 0.8,          # High execution efficiency
            "complexity": 4,                      # Very high complexity
        },
        StrategyCompositionType.EMERGENT: {
            "priority": 3,                        # Lower priority
            "learning_weight": 0.7,               # Lower learning weight
            "execution_efficiency": 0.5,          # Lower execution efficiency
            "complexity": 5,                      # Very high complexity
        },
    },
    
    # Strategy sequence types and their properties
    "sequence_types": {
        StrategySequenceType.LINEAR: {
            "execution_time": 1.0,                # Base execution time
            "success_rate": 0.8,                  # Base success rate
            "learning_difficulty": 1.0,           # Base learning difficulty
        },
        StrategySequenceType.BRANCHING: {
            "execution_time": 1.5,                # Longer execution time
            "success_rate": 0.7,                  # Lower success rate
            "learning_difficulty": 1.5,           # Higher learning difficulty
        },
        StrategySequenceType.LOOPING: {
            "execution_time": 2.0,                # Much longer execution time
            "success_rate": 0.6,                  # Lower success rate
            "learning_difficulty": 2.0,           # Much higher learning difficulty
        },
        StrategySequenceType.CONVERGENT: {
            "execution_time": 1.8,                # Longer execution time
            "success_rate": 0.75,                 # Medium success rate
            "learning_difficulty": 1.8,           # Higher learning difficulty
        },
        StrategySequenceType.DIVERGENT: {
            "execution_time": 1.3,                # Slightly longer execution time
            "success_rate": 0.65,                 # Lower success rate
            "learning_difficulty": 1.7,           # Higher learning difficulty
        },
        StrategySequenceType.RECURSIVE: {
            "execution_time": 3.0,                # Much longer execution time
            "success_rate": 0.5,                  # Lower success rate
            "learning_difficulty": 3.0,           # Much higher learning difficulty
        },
    },
    
    # Strategy goal types and their properties
    "goal_types": {
        StrategyGoalType.RESOURCE_ACQUISITION: {
            "priority": 1,                        # High priority
            "success_criteria": {"resource_count": 20.0, "resource_efficiency": 0.8},
            "time_horizon": 50,                   # Steps to achieve goal
            "complexity": 2,                      # Medium complexity
        },
        StrategyGoalType.THREAT_ELIMINATION: {
            "priority": 1,                        # High priority
            "success_criteria": {"threat_count": 0.0, "casualty_rate": 0.1},
            "time_horizon": 30,                   # Steps to achieve goal
            "complexity": 3,                      # High complexity
        },
        StrategyGoalType.TERRITORY_CONTROL: {
            "priority": 2,                        # Medium priority
            "success_criteria": {"territory_radius": 100.0, "control_duration": 50.0},
            "time_horizon": 100,                  # Steps to achieve goal
            "complexity": 4,                      # Very high complexity
        },
        StrategyGoalType.AGENT_MANAGEMENT: {
            "priority": 1,                        # High priority
            "success_criteria": {"agent_count": 5.0, "role_balance": 0.5},
            "time_horizon": 20,                   # Steps to achieve goal
            "complexity": 1,                      # Low complexity
        },
        StrategyGoalType.DEFENSIVE_POSITIONING: {
            "priority": 1,                        # High priority
            "success_criteria": {"defense_strength": 0.8, "vulnerability": 0.2},
            "time_horizon": 40,                   # Steps to achieve goal
            "complexity": 2,                      # Medium complexity
        },
        StrategyGoalType.OFFENSIVE_EXPANSION: {
            "priority": 2,                        # Medium priority
            "success_criteria": {"expansion_rate": 0.5, "territory_gained": 50.0},
            "time_horizon": 80,                   # Steps to achieve goal
            "complexity": 3,                      # High complexity
        },
        StrategyGoalType.ECONOMIC_GROWTH: {
            "priority": 2,                        # Medium priority
            "success_criteria": {"resource_balance": 500.0, "efficiency": 0.8},
            "time_horizon": 60,                   # Steps to achieve goal
            "complexity": 2,                      # Medium complexity
        },
        StrategyGoalType.STRATEGIC_ADVANTAGE: {
            "priority": 1,                        # High priority
            "success_criteria": {"advantage_score": 0.8, "opponent_disadvantage": 0.6},
            "time_horizon": 120,                  # Steps to achieve goal
            "complexity": 4,                      # Very high complexity
        },
    },
    
    # Strategy composition learning objectives
    "learning_objectives": {
        "sequence_success": 0.4,                  # Weight for sequence success
        "goal_achievement": 0.3,                  # Weight for goal achievement
        "execution_efficiency": 0.2,              # Weight for execution efficiency
        "adaptation_quality": 0.1,                 # Weight for adaptation quality
    },
    
    # Reward shaping for strategy composition
    "composition_rewards": {
        "successful_composition": 0.2,             # Reward for successful composition
        "goal_achievement": 0.3,                  # Reward for achieving goals
        "efficient_execution": 0.15,              # Reward for efficient execution
        "adaptive_behavior": 0.1,                 # Reward for adaptive behavior
        "emergent_discovery": 0.25,               # Reward for emergent strategy discovery
    },
}

# Strategy composition state components
STRATEGY_COMPOSITION_STATE_COMPONENTS = {
    "current_strategy": 1,                        # Current strategy index
    "strategy_history": 10,                       # Recent strategy history
    "goal_progress": 8,                           # Progress toward goals
    "execution_state": 4,                         # Current execution state
    "composition_context": 6,                     # Composition context
    "adaptation_signals": 3,                      # Adaptation signals
}

# Total strategy composition state size
STRATEGY_COMPOSITION_STATE_SIZE = sum(STRATEGY_COMPOSITION_STATE_COMPONENTS.values())

# ============================================================================
# META-LEARNING FOR STRATEGY DISCOVERY SYSTEM
# ============================================================================

class MetaLearningType(Enum):
    """Types of meta-learning approaches."""
    GRADIENT_BASED = "gradient_based"            # Gradient-based meta-learning (MAML)
    EVOLUTIONARY = "evolutionary"                 # Evolutionary strategy discovery
    REINFORCEMENT = "reinforcement"              # Reinforcement learning meta-learning
    MEMORY_BASED = "memory_based"                 # Memory-based meta-learning
    TRANSFER_LEARNING = "transfer_learning"      # Transfer learning between contexts
    CURRICULUM_LEARNING = "curriculum_learning"   # Curriculum-based learning

class StrategyDiscoveryMethod(Enum):
    """Methods for discovering new strategies."""
    PATTERN_ANALYSIS = "pattern_analysis"         # Analyze successful patterns
    GENETIC_ALGORITHM = "genetic_algorithm"        # Genetic algorithm approach
    NEURAL_ARCHITECTURE_SEARCH = "neural_architecture_search"  # NAS approach
    REINFORCEMENT_SEARCH = "reinforcement_search" # RL-based search
    TRANSFER_FROM_SIMILAR = "transfer_from_similar" # Transfer from similar contexts
    EMERGENT_COMBINATION = "emergent_combination"  # Emergent combination of existing strategies

class StrategyEvaluationMetric(Enum):
    """Metrics for evaluating discovered strategies."""
    SUCCESS_RATE = "success_rate"                 # Overall success rate
    EFFICIENCY = "efficiency"                     # Resource efficiency
    ADAPTABILITY = "adaptability"                 # Adaptability to different situations
    NOVELTY = "novelty"                          # Novelty compared to existing strategies
    ROBUSTNESS = "robustness"                     # Robustness to perturbations
    SCALABILITY = "scalability"                   # Scalability to different scenarios

# Meta-learning configuration
META_LEARNING_CONFIG = {
    # Meta-learning parameters
    "meta_learning_rate": 0.001,                  # Meta-learning rate
    "inner_learning_rate": 0.01,                  # Inner loop learning rate
    "meta_batch_size": 16,                        # Meta-learning batch size
    "inner_steps": 5,                             # Number of inner loop steps
    "meta_steps": 10,                             # Number of meta-learning steps
    "strategy_population_size": 50,                # Population size for evolutionary approaches
    "mutation_rate": 0.1,                         # Mutation rate for genetic algorithms
    "crossover_rate": 0.8,                        # Crossover rate for genetic algorithms
    
    # Strategy discovery parameters
    "discovery_frequency": 100,                   # Steps between strategy discovery attempts
    "evaluation_episodes": 10,                     # Episodes to evaluate new strategies
    "novelty_threshold": 0.3,                     # Minimum novelty threshold for new strategies
    "success_threshold": 0.6,                     # Minimum success threshold for keeping strategies
    "max_strategies": 20,                         # Maximum number of strategies to maintain
    
    # Meta-learning types and their properties
    "meta_learning_types": {
        MetaLearningType.GRADIENT_BASED: {
            "priority": 1,                        # High priority
            "learning_weight": 1.0,              # Full learning weight
            "computational_cost": 0.8,            # High computational cost
            "convergence_speed": 0.7,             # Medium convergence speed
        },
        MetaLearningType.EVOLUTIONARY: {
            "priority": 1,                       # High priority
            "learning_weight": 1.2,              # Higher learning weight
            "computational_cost": 0.6,            # Medium computational cost
            "convergence_speed": 0.5,             # Slower convergence
        },
        MetaLearningType.REINFORCEMENT: {
            "priority": 1,                       # High priority
            "learning_weight": 1.1,              # Higher learning weight
            "computational_cost": 0.7,            # Medium-high computational cost
            "convergence_speed": 0.6,             # Medium convergence speed
        },
        MetaLearningType.MEMORY_BASED: {
            "priority": 2,                       # Medium priority
            "learning_weight": 0.9,              # Medium learning weight
            "computational_cost": 0.4,            # Low computational cost
            "convergence_speed": 0.8,             # Fast convergence
        },
        MetaLearningType.TRANSFER_LEARNING: {
            "priority": 2,                       # Medium priority
            "learning_weight": 0.8,              # Medium learning weight
            "computational_cost": 0.5,            # Medium computational cost
            "convergence_speed": 0.9,             # Very fast convergence
        },
        MetaLearningType.CURRICULUM_LEARNING: {
            "priority": 3,                       # Lower priority
            "learning_weight": 0.7,              # Lower learning weight
            "computational_cost": 0.3,            # Low computational cost
            "convergence_speed": 0.6,             # Medium convergence speed
        },
    },
    
    # Strategy discovery methods and their properties
    "discovery_methods": {
        StrategyDiscoveryMethod.PATTERN_ANALYSIS: {
            "priority": 1,                       # High priority
            "success_rate": 0.7,                  # High success rate
            "novelty_score": 0.6,                 # Medium novelty
            "computational_cost": 0.4,            # Low computational cost
        },
        StrategyDiscoveryMethod.GENETIC_ALGORITHM: {
            "priority": 1,                       # High priority
            "success_rate": 0.8,                 # High success rate
            "novelty_score": 0.9,                 # High novelty
            "computational_cost": 0.7,            # High computational cost
        },
        StrategyDiscoveryMethod.NEURAL_ARCHITECTURE_SEARCH: {
            "priority": 2,                       # Medium priority
            "success_rate": 0.6,                 # Medium success rate
            "novelty_score": 0.8,                 # High novelty
            "computational_cost": 0.9,            # Very high computational cost
        },
        StrategyDiscoveryMethod.REINFORCEMENT_SEARCH: {
            "priority": 1,                       # High priority
            "success_rate": 0.75,                # High success rate
            "novelty_score": 0.7,                 # Medium-high novelty
            "computational_cost": 0.8,            # High computational cost
        },
        StrategyDiscoveryMethod.TRANSFER_FROM_SIMILAR: {
            "priority": 2,                       # Medium priority
            "success_rate": 0.8,                 # High success rate
            "novelty_score": 0.4,                 # Low novelty
            "computational_cost": 0.3,            # Low computational cost
        },
        StrategyDiscoveryMethod.EMERGENT_COMBINATION: {
            "priority": 1,                       # High priority
            "success_rate": 0.65,                # Medium-high success rate
            "novelty_score": 0.8,                 # High novelty
            "computational_cost": 0.5,            # Medium computational cost
        },
    },
    
    # Strategy evaluation metrics and their weights
    "evaluation_metrics": {
        StrategyEvaluationMetric.SUCCESS_RATE: {
            "weight": 0.3,                       # High weight
            "importance": 1.0,                   # Very important
        },
        StrategyEvaluationMetric.EFFICIENCY: {
            "weight": 0.25,                      # High weight
            "importance": 0.9,                   # Very important
        },
        StrategyEvaluationMetric.ADAPTABILITY: {
            "weight": 0.2,                       # Medium weight
            "importance": 0.8,                   # Important
        },
        StrategyEvaluationMetric.NOVELTY: {
            "weight": 0.15,                      # Medium weight
            "importance": 0.7,                   # Important
        },
        StrategyEvaluationMetric.ROBUSTNESS: {
            "weight": 0.1,                       # Low weight
            "importance": 0.6,                   # Somewhat important
        },
        StrategyEvaluationMetric.SCALABILITY: {
            "weight": 0.1,                       # Low weight
            "importance": 0.5,                   # Somewhat important
        },
    },
    
    # Meta-learning objectives
    "learning_objectives": {
        "strategy_discovery": 0.4,               # Weight for strategy discovery
        "strategy_refinement": 0.3,               # Weight for strategy refinement
        "knowledge_transfer": 0.2,                # Weight for knowledge transfer
        "adaptation_speed": 0.1,                 # Weight for adaptation speed
    },
    
    # Reward shaping for meta-learning
    "meta_learning_rewards": {
        "successful_discovery": 0.3,             # Reward for successful strategy discovery
        "novel_strategy": 0.2,                    # Reward for novel strategies
        "efficient_strategy": 0.15,               # Reward for efficient strategies
        "adaptive_strategy": 0.1,                 # Reward for adaptive strategies
        "knowledge_transfer": 0.1,                # Reward for successful knowledge transfer
        "meta_learning_progress": 0.15,            # Reward for meta-learning progress
    },
}

# Meta-learning state components
META_LEARNING_STATE_COMPONENTS = {
    "strategy_performance": 10,                   # Performance of current strategies
    "discovery_history": 8,                       # History of strategy discoveries
    "evaluation_metrics": 6,                      # Current evaluation metrics
    "meta_learning_progress": 4,                  # Meta-learning progress
    "context_similarity": 4,                      # Similarity to previous contexts
}

# Total meta-learning state size
META_LEARNING_STATE_SIZE = sum(META_LEARNING_STATE_COMPONENTS.values())

# ============================================================================
# STRATEGY INTERPRETABILITY AND VISUALIZATION SYSTEM
# ============================================================================

class VisualizationType(Enum):
    """Types of visualizations available."""
    STRATEGY_PERFORMANCE = "strategy_performance"        # Strategy performance over time
    PARAMETER_ANALYSIS = "parameter_analysis"           # Parameter importance and relationships
    DECISION_TREES = "decision_trees"                    # Decision tree visualization
    ATTENTION_MAPS = "attention_maps"                     # Attention mechanism visualization
    STRATEGY_COMPOSITION = "strategy_composition"         # Strategy composition flow
    META_LEARNING_PROGRESS = "meta_learning_progress"    # Meta-learning progress tracking
    COMMUNICATION_NETWORKS = "communication_networks"     # Agent communication networks
    EXPERIENCE_SHARING = "experience_sharing"           # Experience sharing patterns
    STATE_REPRESENTATION = "state_representation"        # Learned state representations
    REWARD_COMPONENTS = "reward_components"              # Reward component breakdown

class InterpretabilityMethod(Enum):
    """Methods for strategy interpretability."""
    SHAP_VALUES = "shap_values"                          # SHAP (SHapley Additive exPlanations)
    LIME = "lime"                                        # Local Interpretable Model-agnostic Explanations
    GRADIENT_ATTRIBUTION = "gradient_attribution"        # Gradient-based attribution
    ATTENTION_ANALYSIS = "attention_analysis"             # Attention mechanism analysis
    FEATURE_IMPORTANCE = "feature_importance"            # Feature importance analysis
    DECISION_BOUNDARIES = "decision_boundaries"         # Decision boundary visualization
    STRATEGY_CLUSTERING = "strategy_clustering"          # Strategy clustering analysis
    TEMPORAL_PATTERNS = "temporal_patterns"              # Temporal pattern analysis

class PerformanceMetric(Enum):
    """Performance metrics for visualization."""
    WIN_RATE = "win_rate"                                # Win rate over time
    SURVIVAL_RATE = "survival_rate"                      # Survival rate over time
    EFFICIENCY_SCORE = "efficiency_score"               # Efficiency score over time
    COORDINATION_SCORE = "coordination_score"           # Coordination score over time
    RESOURCE_UTILIZATION = "resource_utilization"        # Resource utilization over time
    STRATEGY_SUCCESS_RATE = "strategy_success_rate"      # Strategy success rate over time
    META_LEARNING_PROGRESS = "meta_learning_progress"   # Meta-learning progress over time
    COMMUNICATION_EFFECTIVENESS = "communication_effectiveness"  # Communication effectiveness over time
    EXPERIENCE_SHARING_RATE = "experience_sharing_rate"  # Experience sharing rate over time
    STRATEGY_DIVERSITY = "strategy_diversity"            # Strategy diversity over time

# Visualization configuration
VISUALIZATION_CONFIG = {
    # Visualization parameters
    "update_frequency": 10,                             # Steps between visualization updates
    "history_length": 1000,                              # Length of performance history to keep
    "plot_resolution": (1920, 1080),                    # Plot resolution (width, height)
    "dpi": 100,                                          # Dots per inch for plots
    "color_palette": "viridis",                          # Color palette for plots
    "animation_fps": 30,                                 # Frames per second for animations
    
    # Visualization types and their properties
    "visualization_types": {
        VisualizationType.STRATEGY_PERFORMANCE: {
            "priority": 1,                               # High priority
            "update_frequency": 5,                       # Update every 5 steps
            "complexity": 2,                             # Medium complexity
            "memory_usage": 0.3,                         # Medium memory usage
        },
        VisualizationType.PARAMETER_ANALYSIS: {
            "priority": 1,                               # High priority
            "update_frequency": 10,                      # Update every 10 steps
            "complexity": 3,                             # High complexity
            "memory_usage": 0.4,                         # High memory usage
        },
        VisualizationType.DECISION_TREES: {
            "priority": 2,                               # Medium priority
            "update_frequency": 20,                      # Update every 20 steps
            "complexity": 4,                             # Very high complexity
            "memory_usage": 0.6,                         # Very high memory usage
        },
        VisualizationType.ATTENTION_MAPS: {
            "priority": 1,                               # High priority
            "update_frequency": 5,                       # Update every 5 steps
            "complexity": 3,                             # High complexity
            "memory_usage": 0.5,                         # High memory usage
        },
        VisualizationType.STRATEGY_COMPOSITION: {
            "priority": 1,                               # High priority
            "update_frequency": 10,                      # Update every 10 steps
            "complexity": 2,                             # Medium complexity
            "memory_usage": 0.3,                         # Medium memory usage
        },
        VisualizationType.META_LEARNING_PROGRESS: {
            "priority": 1,                               # High priority
            "update_frequency": 10,                      # Update every 10 steps
            "complexity": 2,                             # Medium complexity
            "memory_usage": 0.3,                         # Medium memory usage
        },
        VisualizationType.COMMUNICATION_NETWORKS: {
            "priority": 2,                               # Medium priority
            "update_frequency": 15,                      # Update every 15 steps
            "complexity": 3,                             # High complexity
            "memory_usage": 0.4,                         # High memory usage
        },
        VisualizationType.EXPERIENCE_SHARING: {
            "priority": 2,                               # Medium priority
            "update_frequency": 15,                      # Update every 15 steps
            "complexity": 2,                             # Medium complexity
            "memory_usage": 0.3,                         # Medium memory usage
        },
        VisualizationType.STATE_REPRESENTATION: {
            "priority": 2,                               # Medium priority
            "update_frequency": 20,                      # Update every 20 steps
            "complexity": 4,                             # Very high complexity
            "memory_usage": 0.6,                         # Very high memory usage
        },
        VisualizationType.REWARD_COMPONENTS: {
            "priority": 1,                               # High priority
            "update_frequency": 5,                       # Update every 5 steps
            "complexity": 1,                             # Low complexity
            "memory_usage": 0.2,                         # Low memory usage
        },
    },
    
    # Interpretability methods and their properties
    "interpretability_methods": {
        InterpretabilityMethod.SHAP_VALUES: {
            "priority": 1,                               # High priority
            "computational_cost": 0.8,                    # High computational cost
            "accuracy": 0.9,                             # High accuracy
            "interpretability": 0.9,                     # High interpretability
        },
        InterpretabilityMethod.LIME: {
            "priority": 1,                               # High priority
            "computational_cost": 0.6,                    # Medium computational cost
            "accuracy": 0.8,                             # High accuracy
            "interpretability": 0.9,                     # High interpretability
        },
        InterpretabilityMethod.GRADIENT_ATTRIBUTION: {
            "priority": 2,                               # Medium priority
            "computational_cost": 0.4,                    # Low computational cost
            "accuracy": 0.7,                             # Medium accuracy
            "interpretability": 0.7,                     # Medium interpretability
        },
        InterpretabilityMethod.ATTENTION_ANALYSIS: {
            "priority": 1,                               # High priority
            "computational_cost": 0.3,                    # Low computational cost
            "accuracy": 0.8,                             # High accuracy
            "interpretability": 0.8,                     # High interpretability
        },
        InterpretabilityMethod.FEATURE_IMPORTANCE: {
            "priority": 2,                               # Medium priority
            "computational_cost": 0.5,                    # Medium computational cost
            "accuracy": 0.7,                             # Medium accuracy
            "interpretability": 0.8,                     # High interpretability
        },
        InterpretabilityMethod.DECISION_BOUNDARIES: {
            "priority": 3,                               # Lower priority
            "computational_cost": 0.7,                    # High computational cost
            "accuracy": 0.6,                             # Medium accuracy
            "interpretability": 0.6,                     # Medium interpretability
        },
        InterpretabilityMethod.STRATEGY_CLUSTERING: {
            "priority": 2,                               # Medium priority
            "computational_cost": 0.6,                    # Medium computational cost
            "accuracy": 0.7,                             # Medium accuracy
            "interpretability": 0.7,                     # Medium interpretability
        },
        InterpretabilityMethod.TEMPORAL_PATTERNS: {
            "priority": 2,                               # Medium priority
            "computational_cost": 0.5,                    # Medium computational cost
            "accuracy": 0.8,                             # High accuracy
            "interpretability": 0.8,                     # High interpretability
        },
    },
    
    # Performance metrics and their properties
    "performance_metrics": {
        PerformanceMetric.WIN_RATE: {
            "weight": 0.2,                               # High weight
            "importance": 1.0,                           # Very important
            "update_frequency": 1,                       # Update every step
        },
        PerformanceMetric.SURVIVAL_RATE: {
            "weight": 0.15,                              # High weight
            "importance": 0.9,                            # Very important
            "update_frequency": 1,                       # Update every step
        },
        PerformanceMetric.EFFICIENCY_SCORE: {
            "weight": 0.15,                              # High weight
            "importance": 0.9,                            # Very important
            "update_frequency": 5,                       # Update every 5 steps
        },
        PerformanceMetric.COORDINATION_SCORE: {
            "weight": 0.1,                               # Medium weight
            "importance": 0.8,                           # Important
            "update_frequency": 5,                       # Update every 5 steps
        },
        PerformanceMetric.RESOURCE_UTILIZATION: {
            "weight": 0.1,                               # Medium weight
            "importance": 0.8,                           # Important
            "update_frequency": 10,                      # Update every 10 steps
        },
        PerformanceMetric.STRATEGY_SUCCESS_RATE: {
            "weight": 0.1,                               # Medium weight
            "importance": 0.8,                           # Important
            "update_frequency": 10,                      # Update every 10 steps
        },
        PerformanceMetric.META_LEARNING_PROGRESS: {
            "weight": 0.1,                               # Medium weight
            "importance": 0.7,                           # Important
            "update_frequency": 10,                      # Update every 10 steps
        },
        PerformanceMetric.COMMUNICATION_EFFECTIVENESS: {
            "weight": 0.05,                              # Low weight
            "importance": 0.6,                           # Somewhat important
            "update_frequency": 15,                      # Update every 15 steps
        },
        PerformanceMetric.EXPERIENCE_SHARING_RATE: {
            "weight": 0.05,                              # Low weight
            "importance": 0.6,                           # Somewhat important
            "update_frequency": 15,                      # Update every 15 steps
        },
        PerformanceMetric.STRATEGY_DIVERSITY: {
            "weight": 0.05,                              # Low weight
            "importance": 0.5,                           # Somewhat important
            "update_frequency": 20,                      # Update every 20 steps
        },
    },
    
    # Visualization objectives
    "visualization_objectives": {
        "strategy_interpretability": 0.3,                # Weight for strategy interpretability
        "performance_tracking": 0.25,                    # Weight for performance tracking
        "decision_transparency": 0.2,                     # Weight for decision transparency
        "learning_progress": 0.15,                       # Weight for learning progress
        "system_understanding": 0.1,                     # Weight for system understanding
    },
    
    # Visualization rewards
    "visualization_rewards": {
        "successful_interpretation": 0.2,                 # Reward for successful interpretation
        "performance_insight": 0.15,                      # Reward for performance insights
        "decision_clarity": 0.1,                          # Reward for decision clarity
        "learning_visualization": 0.1,                    # Reward for learning visualization
        "system_understanding": 0.1,                      # Reward for system understanding
    },
}

# Visualization state components
VISUALIZATION_STATE_COMPONENTS = {
    "performance_history": 20,                           # Performance history
    "strategy_metrics": 15,                               # Strategy metrics
    "interpretability_scores": 10,                        # Interpretability scores
    "visualization_quality": 5,                           # Visualization quality
    "learning_progress": 8,                              # Learning progress
    "system_metrics": 12,                                # System metrics
}

# Total visualization state size
VISUALIZATION_STATE_SIZE = sum(VISUALIZATION_STATE_COMPONENTS.values())

HQ_STRATEGY_PARAMETERS = {
    # Binary parameters (will be interpreted as probabilities > 0.5)
    "target_role": {
        "type": "binary",  # 0 = gatherer, 1 = peacekeeper
        "description": "Which role to target for recruit/swap actions",
    },
    "priority_resource": {
        "type": "binary",  # 0 = gold, 1 = food
        "description": "Which resource type to prioritize",
    },
    "use_mission_system": {
        "type": "binary",  # 0 = old task system, 1 = new mission system
        "description": "Whether to use mission-oriented task assignment",
    },
    
    # Continuous parameters (0.0 to 1.0)
    "aggression_level": {
        "type": "continuous",
        "range": [0.0, 1.0],
        "description": "0=defensive, 1=offensive (affects threat engagement distance)",
    },
    "resource_threshold": {
        "type": "continuous",
        "range": [0.0, 1.0],
        "description": "Resource availability threshold to trigger action (0=execute always, 1=only when abundant)",
    },
    "urgency": {
        "type": "continuous",
        "range": [0.0, 1.0],
        "description": "How urgent is this strategy (affects task priority assignment)",
    },
    "mission_autonomy": {
        "type": "continuous",
        "range": [0.0, 1.0],
        "description": "How much autonomy to give agents (0=micro-manage, 1=full autonomy)",
    },
    "coordination_preference": {
        "type": "continuous",
        "range": [0.0, 1.0],
        "description": "How much agents should coordinate (0=independent, 1=highly coordinated)",
    },
    "agent_adaptability": {
        "type": "continuous",
        "range": [0.0, 1.0],
        "description": "How adaptive agents should be (0=rigid, 1=highly adaptive)",
    },
    "failure_tolerance": {
        "type": "continuous",
        "range": [0.0, 1.0],
        "description": "How many failures agents should tolerate (0=give up quickly, 1=persist)",
    },
    
    # Discrete parameters (will be rounded)
    "agent_count_target": {
        "type": "discrete",
        "range": [1, 10],  # Min 1, max 10 agents
        "description": "Target number of agents for this strategy",
    },
    "mission_complexity": {
        "type": "discrete",
        "range": [1, 4],  # 1=beginner, 2=intermediate, 3=advanced, 4=expert
        "description": "Complexity level of missions to assign",
    },
}
"""
Defines learnable parameters for parametric strategy execution.
Each parameter is output by the HQ network and interpreted by the faction.
"""

# Number of parameter outputs from HQ network
HQ_NUM_PARAMETERS = len(HQ_STRATEGY_PARAMETERS)  # 12 parameters


ROLE_ACTIONS_MAP = {
    "gatherer": [
        "move_up",
        "move_down",
        "move_left",
        "move_right",
        "mine_gold",
        "forage_apple",
        "heal_with_apple",
        "explore",
        "plant_tree",
        "plant_gold_vein",
    ],
    "peacekeeper": [
        "move_up",
        "move_down",
        "move_left",
        "move_right",
        "patrol",
        "heal_with_apple",
        "eliminate_threat",
        "explore",
        "block",
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
        "resource_count",  # Total resources
        "threat_count",  # Total threats
    ],
    "local_perception": [
        "position_x",
        "position_y",
        "health",
        "nearby_resource_count",  # Count of nearby resources
        "nearby_threat_count",  # Count of nearby threats
    ],
    "task_features": [
        "task_type",  # Encoded task type
        "task_target_x",  # Target X position
        "task_target_y",  # Target Y position
        "current_action",  # Current action
    ],
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
Peacekeeper_PNG = "RENDER\IMAGES\peacekeeper.png"  # Path to peacekeeper image
Gatherer_PNG = "RENDER\IMAGES\gatherer.png"  # Path to gatherer image

Gold_Cost_for_Agent = 10  # Gold cost for an agent
Gold_Cost_for_Agent_Swap = (
    5  # Gold cost for swapping an existing agent to a different role
)


DEF_AGENT_STATE_SIZE = 26 + len(
    TASK_TYPE_MAPPING
)  # Updated to support enhanced state: 8 core + 2 role + N task one-hot + 6 task info + 2 context + 8 terrain awareness

"""
State size breakdown:
- Core state (8): pos_x, pos_y, health, threat_proximity, threat_distance, resource_proximity, resource_distance, hq_proximity
- Role vector (2): gatherer_onehot, peacekeeper_onehot
- Task one-hot: len(TASK_TYPE_MAPPING)
- Task info (6): target_x, target_y, action_norm, norm_dist, task_urgency, task_progress
- Context (2): threat_count_norm, resource_count_norm
- Terrain awareness (8): N, S, E, W, NE, NW, SE, SW traversability (1.0 = land, 0.0 = water)
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
FACTON_COUNT = 3  # Number of factons # Customise as needed!
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
        "state": TaskState.PENDING,  # Track the task's lifecycle from assignment
    }
    if task_id:
        task["id"] = task_id

    return task

def create_mission(mission_type, parameters=None, mission_id=None, priority=MissionPriority.MEDIUM):
    """
    Create a mission-oriented task object for autonomous agent execution.
    
    Args:
        mission_type (MissionType): The type of mission to assign.
        parameters (dict, optional): Mission-specific parameters for agent interpretation.
        mission_id (str, optional): Unique identifier for the mission.
        priority (MissionPriority): Priority level for the mission.
        
    Returns:
        dict: A mission object with flexible parameters for agent interpretation.
    """
    mission = {
        "type": "mission",  # Distinguish from old task system
        "mission_type": mission_type.value if isinstance(mission_type, MissionType) else mission_type,
        "parameters": parameters or {},
        "priority": priority.value if isinstance(priority, MissionPriority) else priority,
        "state": MissionState.ASSIGNED,
        "assigned_step": None,  # Will be set when assigned to agent
        "start_step": None,     # Will be set when agent starts mission
        "progress": 0.0,        # Mission completion progress (0.0 to 1.0)
        "subtasks": [],         # Subtasks the agent creates to fulfill the mission
        "success_criteria_met": False,
        "failure_reason": None,
    }
    
    if mission_id:
        mission["id"] = mission_id
    
    return mission


# ===== PERSISTENT SETTINGS HELPERS =====


def has_run_installer():
    """Check if the startup installer has been run"""
    if _load_persistent_settings and settings_manager:
        return settings_manager.has_run_installer()
    return False


def is_first_run():
    """Check if this is the first run"""
    if _load_persistent_settings and settings_manager:
        return settings_manager.is_first_run()
    return True


def save_headless_mode(value):
    """Save headless mode setting"""
    if _load_persistent_settings and settings_manager:
        settings_manager.set_headless_mode(value)


def get_persistent_episodes():
    """Get last used episode count"""
    if _load_persistent_settings and settings_manager:
        return settings_manager.get_last_episodes()
    return EPISODES_LIMIT


def save_persistent_episodes(value):
    """Save last used episode count"""
    if _load_persistent_settings and settings_manager:
        settings_manager.set_last_episodes(value)


def get_persistent_steps():
    """Get last used steps per episode"""
    if _load_persistent_settings and settings_manager:
        return settings_manager.get_last_steps()
    return STEPS_PER_EPISODE


def save_persistent_steps(value):
    """Save last used steps per episode"""
    if _load_persistent_settings and settings_manager:
        settings_manager.set_last_steps(value)


# ===== LOAD PERSISTENT CONFIG SETTINGS AT MODULE LEVEL =====
# Call the load function after all constants are defined
load_persistent_config()
