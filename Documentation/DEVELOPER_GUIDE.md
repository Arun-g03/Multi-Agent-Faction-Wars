# Developer Guide

## Extending the System

This guide explains how to add new features, modify existing systems, and contribute to the project.

## Table of Contents

1. [Adding New Agent Roles](#adding-new-agent-roles)
2. [Adding New Resources](#adding-new-resources)
3. [Adding HQ Strategies](#adding-hq-strategies)
4. [Modifying Rewards](#modifying-rewards)
5. [Custom Behaviors](#custom-behaviors)
6. [Neural Network Modifications](#neural-network-modifications)
7. [Rendering New Elements](#rendering-new-elements)
8. [Performance Optimization](#performance-optimization)
9. [Testing](#testing)
10. [Code Style Guidelines](#code-style-guidelines)

## Adding New Agent Roles

### Step 1: Define the Role

Add your new role to `UTILITIES/utils_config.py`:

```python
ROLE_ACTIONS_MAP = {
    "peacekeeper": ["move", "attack", "eliminate_threat", "patrol", "block"],
    "gatherer": ["move", "gather", "mine", "explore", "plant_tree", "plant_gold_vein"],
    "new_role": ["move", "custom_action", "explore"],  # Add here
}
```

### Step 2: Create Behavior File

Create `AGENT/Agent_Behaviours/new_role_behaviours.py`:

```python
"""File Specific Imports"""
import UTILITIES.utils_config as utils_config
from SHARED.core_imports import *

logger = Logger(log_file="behavior_log.txt", log_level=logging.DEBUG)


class NewRoleBehavioursMixin:
    """Custom behaviors for new role."""
    
    def __init__(self):
        pass
    
    def custom_action(self):
        """
        Implement your custom action logic.
        Returns SUCCESS, FAILURE, or ONGOING.
        """
        # Your implementation here
        return utils_config.TaskState.SUCCESS
```

### Step 3: Add to Agent Base

In `AGENT/Agent_Types/new_role.py` (or extend existing file):

```python
from AGENT.agent_base import BaseAgent
from AGENT.Agent_Behaviours.new_role_behaviours import NewRoleBehavioursMixin

class NewRole(BaseAgent, NewRoleBehavioursMixin):
    def __init__(self, ...):
        super().__init__(...)
        # Add role-specific initialization
```

### Step 4: Update Rewards

In `AGENT/agent_behaviours.py`:

```python
def assign_reward(self, ...):
    # Add reward logic for your role's actions
    if agent.role == "new_role":
        if action == "custom_action":
            reward += 0.5  # Reward for custom action
```

## Adding New Resources

### Step 1: Create Resource Class

Create `ENVIRONMENT/Resources/new_resource.py`:

```python
"""Common Imports"""
from SHARED.core_imports import *
import UTILITIES.utils_config as utils_config


class NewResource:
    def __init__(self, x, y, grid_x, grid_y, terrain, resource_manager):
        self.x = x
        self.y = y
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.terrain = terrain
        self.resource_manager = resource_manager
        self.quantity = 5  # Your resource quantity
        # ... additional initialization
    
    def collect(self, amount):
        """Agent collects resource."""
        if self.quantity > 0:
            collected = min(amount, self.quantity)
            self.quantity -= collected
            return collected
        return 0
    
    def is_depleted(self):
        return self.quantity == 0
    
    def render(self, screen, camera):
        """Render the resource."""
        if utils_config.HEADLESS_MODE:
            return
        # Your rendering logic
```

### Step 2: Add to Resource Manager

In `ENVIRONMENT/env_resources.py`:

```python
from ENVIRONMENT.Resources.new_resource import NewResource

class ResourceManager:
    def __init__(self, ...):
        self.new_resource_count = 0
        
    def generate_resources(self, ...):
        # Add resource generation logic
        if random.random() < spawn_probability:
            new_resource = NewResource(...)
            self.resources.append(new_resource)
            self.new_resource_count += 1
```

### Step 3: Add to Resource Tracking

In `AGENT/agent_factions.py`:

```python
def aggregate_faction_state(self):
    # Track your resource in global state
    new_resource_count = len([
        r for r in self.global_state.get("resources", [])
        if r["type"] == "new_resource"
    ])
    self.global_state["new_resource_count"] = new_resource_count
```

## Adding HQ Strategies

### Step 1: Add Strategy to Options

In `UTILITIES/utils_config.py`:

```python
HQ_STRATEGY_OPTIONS = [
    "RECRUIT_PEACEKEEPER",
    "RECRUIT_GATHERER",
    "SWAP_TO_GATHERER",
    "SWAP_TO_PEACEKEEPER",
    "ATTACK_THREATS",
    "DEFEND_HQ",
    "COLLECT_GOLD",
    "COLLECT_FOOD",
    "PLANT_TREES",
    "PLANT_GOLD_VEINS",
    "NEW_STRATEGY",  # Add here
]
```

### Step 2: Implement Strategy

In `AGENT/agent_factions.py`, add to `perform_HQ_Strategy()`:

```python
def perform_HQ_Strategy(self):
    action = self.current_strategy
    
    # ... existing strategies ...
    
    elif action == "NEW_STRATEGY":
        # Check prerequisites
        if not prerequisites_met:
            self.hq_step_rewards.append(-0.5)
            self.current_strategy = None
            return retest_strategy()
        
        # Execute strategy
        result = execute_strategy_logic()
        
        # Reward based on outcome
        if result:
            self.hq_step_rewards.append(+1.0)
        else:
            self.hq_step_rewards.append(0.0)
```

## Modifying Rewards

### Agent Rewards

Edit `AGENT/agent_behaviours.py`:

```python
def assign_reward(self, state, action, new_state, reward, done, info):
    # Base reward
    reward_base = reward
    
    # Add role-specific bonuses
    if agent.role == "peacekeeper":
        # Defense bonus for eliminating threats near HQ
        if action == "eliminate_threat":
            hq_distance = distance_to_hq()
            bonus = max(0.0, 0.3 - 0.001 * hq_distance)
            reward += bonus
    
    # Add context-aware bonuses
    if health_ratio < 0.3:
        survival_bonus = (0.3 - health_ratio) * 0.2
        reward += survival_bonus
    
    return reward
```

### HQ Rewards

Edit `AGENT/agent_factions.py` in `perform_HQ_Strategy()`:

```python
# Reward based on strategic value and actual execution
if strategy_executed and goal_achieved:
    self.hq_step_rewards.append(+1.5)  # High reward for success
elif strategy_executed:
    self.hq_step_rewards.append(+0.5)  # Medium reward for attempt
else:
    self.hq_step_rewards.append(-0.5)  # Penalty for failure
```

## Custom Behaviors

### Creating New Behaviors

In your behavior file:

```python
class CustomBehaviorMixin:
    def custom_behavior(self):
        """
        Custom behavior implementation.
        
        Returns:
            TaskState.SUCCESS: Behavior completed
            TaskState.FAILURE: Behavior failed
            TaskState.ONGOING: Behavior in progress
        """
        # Check preconditions
        if not self.can_perform_action():
            return utils_config.TaskState.FAILURE
        
        # Perform action
        result = self.execute_action()
        
        if result:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"{self.agent.role} completed custom action.",
                    level=logging.INFO,
                )
            return utils_config.TaskState.SUCCESS
        else:
            return utils_config.TaskState.FAILURE
```

### Integrating with Task System

Add task mapping in `agent_behaviours.py`:

```python
def assign_task(self, action):
    if action == "custom_behavior":
        return {
            "type": "custom",
            "status": "pending",
            # ... task data
        }
```

## Neural Network Modifications

### Modifying Agent Network

In `NEURAL_NETWORK/PPO_Agent_Network.py`:

```python
class PPOModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Add new layers
        self.new_layer = nn.Linear(state_dim, hidden_dim)
        
    def forward(self, state):
        # Integrate new layer
        x = self.new_layer(state)
        # ... rest of forward pass
```

### Modifying HQ Network

In `NEURAL_NETWORK/HQ_Network.py`:

```python
def encode_state_parts(self, ...):
    # Add new state information
    new_state_vector = [
        state["new_feature"],
        # ... other features
    ]
    
    # Update state_size if adding features
    if len(new_state_vector) > self.state_size:
        self.state_size = len(new_state_vector)
```

## Rendering New Elements

### Adding New Render Methods

In `RENDER/Game_Renderer.py`:

```python
def render_new_element(self, screen, camera):
    """Render new game element."""
    if utils_config.HEADLESS_MODE:
        return
    
    for element in self.game_manager.elements:
        # Calculate screen position
        screen_x, screen_y = camera.apply((element.x, element.y))
        
        # Render element
        pygame.draw.circle(screen, (255, 0, 0), (screen_x, screen_y), 10)
```

### Adding to Main Render Loop

```python
def render(self, ...):
    # ... existing rendering ...
    
    # Add your new render call
    self.render_new_element(screen, camera)
```

## Performance Optimization

### Profiling

Use the built-in profiling system:

```python
from UTILITIES.utils_helpers import profile_function

@profile_function
def my_function():
    # Your code
    pass
```

### Caching

Implement caching for expensive operations:

```python
# In __init__:
self.cache = {}

# In method:
if not hasattr(self, '_cache_key'):
    result = expensive_operation()
    self.cache['_cache_key'] = result
else:
    result = self.cache['_cache_key']
```

### Batch Operations

Process data in batches:

```python
# Instead of:
for item in items:
    process(item)

# Do:
batch_size = 100
for i in range(0, len(items), batch_size):
    batch = items[i:i+batch_size]
    process_batch(batch)
```

## Testing

### Unit Tests

Create test files in a `tests/` directory:

```python
import unittest
from AGENT.agent_base import BaseAgent

class TestAgent(unittest.TestCase):
    def test_agent_creation(self):
        agent = BaseAgent(...)
        self.assertIsNotNone(agent)
```

### Integration Tests

Test system interactions:

```python
def test_faction_resources():
    faction = Faction(...)
    faction.food_balance = 10
    
    assert faction.can_afford_recruit("gatherer") == True
```

## Code Style Guidelines

### Imports
```python
"""Common Imports"""
from SHARED.core_imports import *

"""File Specific Imports"""
import UTILITIES.utils_config as utils_config
from MODULE.specific import SpecificClass
```

### Logging
```python
if utils_config.ENABLE_LOGGING:
    logger.log_msg(
        "Descriptive message with context",
        level=logging.INFO,  # or DEBUG, WARNING, ERROR
    )
```

### Constants
```python
# Use utils_config.py for global constants
# Use local constants with descriptive names
RESOURCE_SPAWN_PROBABILITY = 0.15
MAX_RESOURCE_COUNT = 50
```

### Comments
```python
# Explain WHY, not WHAT
# Use docstrings for functions
def my_function(param):
    """
    Brief description.
    
    :param param: Parameter description
    :return: Return description
    """
    pass
```

### Error Handling
```python
try:
    result = risky_operation()
except SpecificError as e:
    if utils_config.ENABLE_LOGGING:
        logger.log_msg(f"Error: {e}", level=logging.ERROR)
    # Handle error appropriately
    return None
```

## Common Patterns

### Singleton Pattern
```python
class ResourceManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

### Observer Pattern
```python
class EventManager:
    def __init__(self):
        self.listeners = []
    
    def subscribe(self, listener):
        self.listeners.append(listener)
    
    def notify(self, event):
        for listener in self.listeners:
            listener.on_event(event)
```

### Strategy Pattern
```python
class ActionStrategy:
    def execute(self, agent):
        raise NotImplementedError

class MoveStrategy(ActionStrategy):
    def execute(self, agent):
        # Move logic
        pass
```

## Debugging Tips

1. **Enable Logging**: Set `ENABLE_LOGGING = True` in `utils_config.py`
2. **Check Logs**: Look in `RUNTIME_LOGS/` for error messages
3. **TensorBoard**: Monitor training metrics
4. **Print Statements**: Use for quick debugging (remove before committing)
5. **Breakpoints**: Use debugger for complex issues

## Contributing

1. Follow code style guidelines
2. Add comments for complex logic
3. Update CHANGELOG.md with changes
4. Test thoroughly before committing
5. Update documentation as needed

## Resources

- **Architecture**: See `ARCHITECTURE.md`
- **Examples**: Check existing code for patterns
- **Configuration**: See `UTILITIES/utils_config.py`
- **Logs**: Check `RUNTIME_LOGS/` for runtime information

