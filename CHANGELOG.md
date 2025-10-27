# Changelog

All notable changes to the Multi-Agent Faction Wars project will be documented in this file.

## [Unreleased]

### Added
- **Structured Logging System**
  - Separate log files for `ERROR` and `CRITICAL` level messages
  - Organized log directories: `RUNTIME_LOGS/General_Logs/` and `RUNTIME_LOGS/Error_Logs/`
  - Dedicated error log file that is cleared on each application start

- **Enhanced Agent State Representation**
  - Replaced raw coordinates with proximity-based signals (threat, resource, HQ)
  - Added normalized distance calculations using inverse distance with sigmoid scaling
  - Implemented role-specific one-hot encoding for gatherer/peacekeeper roles
  - Task urgency and task progress features for better task prioritization
  - Environmental context signals (threat count, resource count)
  - Local terrain awareness (8-directional traversability checking)

- **Enhanced HQ State Perception**
  - Added territory ownership tracking (`territory_count`)
  - Implemented threat report tracking with timestamp-based freshness metrics
  - Added agent count signals (friendly, gatherer, peacekeeper)
  - Introduced affordability indicators (`can_afford_recruit`, `can_afford_swap`)
  - Territory delta tracking for observing changes in control
  - Threat report frequency and freshness awareness

- **Code Organization Improvements**
  - Created `AGENT/Agent_Types/` directory for role-specific agent classes
    - `peacekeeper.py` - Peacekeeper class implementation
    - `gatherer.py` - Gatherer class implementation
  - Created `AGENT/Agent_Behaviours/` directory for behavior mixins
    - `core_actions.py` - CoreActionsMixin with movement and healing
    - `gatherer_behaviours.py` - GathererBehavioursMixin with resource gathering
    - `peacekeeper_behaviours.py` - PeacekeeperBehavioursMixin with combat and patrolling

- **Configuration Flags**
  - `ENABLE_PLOTS` flag to toggle plot and CSV generation
  - Integrated into settings UI for runtime configuration
  - Added to `RENDER/Settings_Renderer.py` debugging category

- **World Ownership Metric**
  - Added territory tracking system (`territory_count` per faction)
  - Calculated during terrain generation (`max_traversable_tiles`)
  - Displayed as percentage in HQ hover tooltip
  - Integrated into faction metrics and plotting

- **Mini-Batch Training**
  - Implemented mini-batch training during episodes (every 1000 steps)
  - Added `train_agents_mini()` for agent networks
  - Added `train_hq_networks_mini()` for HQ networks
  - Memory clearing after episode completion to prevent stale data

- **Neural Network Stability Improvements**
  - FIFO memory buffer management (max 20,000 samples)
  - Logit clipping to [-10, 10] to prevent numerical instability
  - Value clipping to [-100, 100] for value estimates
  - Faction-specific random seed initialization for HQ networks

### Changed
- **Execution Order Fix**
  - Moved `Startup_installer.py` execution to the very top of `main.py`
  - Ensures dependencies are installed before any imports
  - Fixed `ModuleNotFoundError` on startup

- **Agent Movement Logic**
  - Fixed coordinate system mismatch in `handle_move_to_task()`
  - Implemented consistent grid-based coordinate conversion
  - Corrected explore task target handling (grid vs world coordinates)

- **Task Assignment Improvements**
  - Refined `assign_high_level_tasks()` to prevent excessive reassignment
  - Only assign tasks to truly idle agents (no ONGOING/PENDING tasks)
  - Added check to prevent reassigning DefendHQ if agent is already at HQ
  - Fixed gold deduction logic in `recruit_agent()` to only deduct on success

- **HQ Strategy Selection Logic**
  - Added faction-specific random seed initialization (`42 + faction_id`)
  - Increased exploration noise from 0.5 to 2.0 for untrained networks
  - Reduced `strategy_update_interval` from 100 to 50 steps
  - Implemented deterministic selection for trained networks
  - Enhanced state signals with proximity and distance observations

- **PPO Agent Training**
  - Fixed tensor conversion in `train_batched()` for state stacking
  - Added explicit type checking before tensor operations
  - Improved error handling for memory buffer operations

- **Memory Management**
  - Implemented FIFO eviction for memory buffers exceeding max size
  - Clear agent memories after episode training
  - Prevent memory overflow by trimming oldest samples

- **Terrain Generation**
  - Added `get_traversable_tile_count()` method to calculate total land tiles
  - Store maximum traversable tiles for ownership percentage calculations

- **Coordinate System Consistency**
  - Explore tasks store targets in grid coordinates
  - Defend HQ strategy converts HQ position to grid coordinates
  - Movement calculations consistently use grid-based coordinates
  - Reverted pixel-to-grid conversions that caused regressions

### Fixed
- **ModuleNotFoundError on Startup**
  - Fixed import order to ensure dependencies are installed first
  - Uncommented `Startup_installer.py` execution at the beginning of `main.py`

- **Peacekeepers Stuck in Pending State**
  - Fixed coordinate system mismatch in movement logic
  - Corrected grid vs pixel coordinate comparisons

- **Excessive Task Switching**
  - Implemented proper task state checking before reassignment
  - Only assign to agents without ONGOING/PENDING tasks

- **TypeError in Logging**
  - Removed accidental `print()` statement in `eliminate_threat()` logger call
  - Fixed `log_msg()` argument passing

- **TypeError in Plot Generation**
  - Added fallback logic for `save_dir` when `self.image_dir` is None
  - Default to `VISUALS/PLOTS` directory if tensorboard path unavailable

- **Double Gold Deduction**
  - Modified `recruit_agent()` to only deduct gold after successful agent creation
  - Return `None` on failure to prevent gold loss

- **HQ Network Initialization**
  - Fixed `global_state_size` parameter handling when loading from checkpoint
  - Proper state size calculation for loaded networks

- **Agent Training TypeError**
  - Fixed state tensor conversion in `train_batched()`
  - Added type checking before stacking operations

- **Circular Import Errors**
  - Resolved circular imports between `AgentBehaviour`, `Gatherer`, `Peacekeeper`
  - Removed unnecessary imports in `agent_base.py`
  - Restructured mixin inheritance pattern

- **World Ownership Not Updating**
  - Fixed type mismatch in `calculate_territory()` (string vs integer faction ID comparison)
  - Initialize `territory_count` to 0 in faction `__init__`

- **All HQs Picking Same Strategy**
  - Implemented faction-specific random seeds for network initialization
  - Increased exploration noise magnitude to 2.0
  - Added deterministic selection for trained networks
  - Log network state (memory size, update count, is_trained status)

### Removed
- Reverted epsilon-greedy exploration based on user feedback
- Removed excessive debug logging to reduce log file sizes
- Removed coordinate conversion logic that caused regressions

### Technical Details

#### Agent State Vector Breakdown (26 + len(TASK_TYPE_MAPPING))
- **Core State (8)**: pos_x, pos_y, health, threat_proximity, threat_distance, resource_proximity, resource_distance, hq_proximity
- **Role Vector (2)**: gatherer_onehot, peacekeeper_onehot
- **Task One-Hot**: len(TASK_TYPE_MAPPING)
- **Task Info (6)**: target_x, target_y, action_norm, norm_dist, task_urgency, task_progress
- **Context (2)**: threat_count_norm, resource_count_norm
- **Terrain Awareness (8)**: N, S, E, W, NE, NW, SE, SW traversability (1.0 = land, 0.0 = water)

#### HQ Network Enhancements
- Faction-specific initialization with deterministic seeds
- Enhanced global state encoding with proximity signals
- Territory change observation signals
- Threat report freshness and frequency tracking
- Agent count and affordability indicators

#### Memory Buffer Management
- MAX_MEMORY_SIZE = 20,000 samples (FIFO eviction)
- Mini-batch training every 1000 steps during episodes
- Clear memories after episode completion

---

## Latest Fixes (2025-10-27)

### Fixed: All HQs Picking Same Strategy in First Episode

**Problem**: All HQs were maintaining the `DEFEND_HQ` strategy throughout the entire first episode, showing no diversity in strategy selection.

**Root Cause**: All HQ networks were being initialized with identical random weights, causing them to produce identical logits for the same input state, leading to the same strategy being selected.

**Solution**:
1. **Faction-Specific Random Seeds**: Implemented deterministic seed initialization using `torch.manual_seed(42 + faction_id)` before network layer initialization.
2. **Increased Exploration Noise**: Increased noise magnitude from `0.5` to `2.0` for untrained networks to encourage diverse strategy selection.
3. **Network State Logging**: Added detailed logging of memory size, update count, and `is_trained` status for debugging.
4. **Seed Reset**: Reset the random seed after network initialization to avoid affecting other random operations.

**Files Modified**:
- `NEURAL_NETWORK/HQ_Network.py`: Added `faction_id` parameter and seed-based initialization
- `AGENT/agent_factions.py`: Updated `initialise_network()` to pass `faction_id` to HQ_Network

**Expected Behavior**: Each HQ now starts with unique network weights, leading to different initial strategy selections and more diverse exploration during the first episode.

---

## Previous Changes

### Execution Order Fix
**Commit Message**: `fix: ensure startup installer runs before imports to prevent ModuleNotFoundError`

Moved the `Startup_installer.py` execution to the very beginning of `main.py`, before any project-specific imports. This ensures that all dependencies are installed before the application attempts to import any modules.

### Enhanced State Perception
**Commit Message**: `feat: enhance agent and HQ state perception with proximity-based signals and role awareness`

Replaced raw coordinate values with proximity-based signals for both agents and HQs. Added role awareness, task urgency tracking, environmental context, and terrain awareness to enable better decision-making by the neural networks.

### Code Refactoring
**Commit Message**: `refactor: extract agent types and behaviors into separate modules using mixin pattern`

Created `AGENT/Agent_Types/` and `AGENT/Agent_Behaviours/` directories to improve code organization. Implemented mixin pattern for behaviors to allow role-specific actions while maintaining a unified interface.

### Mini-Batch Training Implementation
**Commit Message**: `feat: implement mini-batch training for agents and HQs during episodes`

Added `train_agents_mini()` and `train_hq_networks_mini()` functions that perform training every 1000 steps during an episode, allowing for more frequent weight updates and better learning.

### Neural Network Stability
**Commit Message**: `fix: add logit/value clipping and FIFO memory buffer management to prevent numerical instability`

Implemented logit clipping to [-10, 10] and value clipping to [-100, 100] to prevent NaN/inf values. Added FIFO eviction for memory buffers to cap at 20,000 samples and prevent overflow.

### World Ownership Metric
**Commit Message**: `feat: add territory tracking and display as ownership percentage in HQ tooltip`

Implemented `territory_count` tracking for each faction, calculated during terrain generation. Added percentage display in the renderer when hovering over HQs. Integrated into faction metrics and plotting.

### HQ Exploration and Diversity
**Commit Message**: `fix: ensure HQs start with unique weights and explore different strategies through noise injection`

Fixed the issue where all HQs were picking the same strategy by implementing faction-specific random seeds for network initialization and increasing exploration noise for untrained networks.

---

## Commit Summary

```
fix: resolve HQ strategy diversity issue in first episode

Problem:
- All HQs were maintaining the same DEFEND_HQ strategy throughout
  the entire first episode, showing no diversity in strategy selection
- Behavior persisted even after implementing mini-batch training
  and adjusting the strategy update interval

Root Cause:
- All HQ networks were initialized with identical random weights
- This caused identical logits for the same input state
- Deterministic argmax selection always chose the same strategy

Solution:
1. Implemented faction-specific random seed initialization
   - Use torch.manual_seed(42 + faction_id) before network creation
   - Ensures each HQ starts with unique weights
   - Reset seed after initialization to avoid side effects

2. Increased exploration noise magnitude
   - Changed from 0.5 to 2.0 for untrained networks
   - Encourages diverse strategy selection during initial exploration
   - Only applied when len(hq_memory) == 0 and total_updates == 0

3. Added comprehensive logging
   - Memory size, update count, and is_trained status
   - Logit values before and after noise injection
   - Strategy selection reasoning for debugging

4. Maintained deterministic selection for trained networks
   - No epsilon-greedy exploration (as per user feedback)
   - Pure argmax for experienced networks

Files Modified:
- NEURAL_NETWORK/HQ_Network.py
  - Added faction_id parameter to __init__
  - Implemented seed-based initialization (lines 56-59, 115-116)
  - Enhanced predict_strategy() with detailed logging (lines 457-476)
  - Increased noise magnitude to 2.0 (line 466)

- AGENT/agent_factions.py
  - Updated initialise_network() to pass faction_id
  - Added faction_id parameter to HQ_Network instantiation

Expected Behavior:
- Each HQ now starts with unique network weights
- More diverse initial strategy selections
- Better exploration during first episode
- Deterministic behavior for trained networks

Testing:
- Verified with logs showing different strategies
- Confirmed memory size and update tracking
- Noise injection only for untrained networks
- Seed reset preserves global random state
```

This fix ensures that during the first episode, HQs will explore different strategies rather than all converging to DEFEND_HQ, providing a more dynamic and varied gameplay experience while maintaining deterministic behavior once networks are trained.

