# Changelog

All notable changes to the Multi-Agent Faction Wars project will be documented in this file.

## [2025-10-27]

### Added
- **HQ Health System**
  - Added health tracking for all HQs (health, max_health, is_destroyed)
  - HQs can now take damage and be destroyed
  - Health bar visualization on HQ hover (green/yellow/red based on health percentage)
  - HQ destruction triggers game-over conditions
  
- **HQ as Attackable Targets**
  - Peacekeepers can now attack enemy HQs
  - HQs can be detected as threats by agents
  - Attack animations trigger when HQs are targeted
  - HQs take 5 damage per attack (more damage than regular agents)

- **Enhanced Threat Detection System**
  - Threats now persist as "last known location" markers
  - Added `last_seen` timestamp tracking for all threats
  - Threat age calculation for intelligence freshness (`threat_age`, `threat_intel_freshness`)
  - HQ Network awareness of threat staleness influences strategy selection
  - Agent threats expire after 200 steps if not re-detected
  - HQs persist as threats indefinitely until destroyed

- **HQ Strategic Awareness**
  - HQ Networks now receive health status indicators (`hq_health_status`, `hq_is_critical`, `hq_is_damaged`)
  - HQ Networks can detect when their HQ is under attack (`hq_under_attack`)
  - Strategy selection can prioritize defense based on HQ health and threats

- **Outcome-Based HQ Rewards**
  - Rewards now evaluate whether strategies were logical choices given the game state
  - Rewards verify actual execution and goal achievement, not just strategy execution
  - Prevents wasted resources on pointless actions (e.g., recruit gatherer with no resources)
  - Encourages smart repurposing (swap idle gatherers to peacekeepers when no resources)

- **Context-Aware Agent Rewards**
  - Agents now receive bonuses based on proximity to HQ when eliminating threats
  - Survival bonuses for low-health agents successfully completing tasks
  - Extra penalties for critical failures when health is very low
  - Role-specific proximity bonuses encourage gathering/eliminating closer to targets
  - Uses only observable information (distance, health, role) - no global context needed

- **Pause Menu System**
  - Press ESC during gameplay to pause/resume the game
  - Press M while paused to restart the current episode with a new map
  - Press Q while paused to quit without full system reset
  - Semi-transparent overlay with clear instructions
  - Game state freezes while paused, allowing inspection
  - Restart episode fully respawns HQs and agents on the new terrain
  - Regenerates terrain, resources, and respawns everything fresh
  - Communication and event systems reset with new agent positions
  - Automatically unpauses after episode restart

### Fixed
- **HQ Network Attention Layer Dimension Errors**
  - Fixed "Dimension out of range" errors during forward passes
  - Added comprehensive dimension checking for attention mechanism
  - Implemented graceful fallback when attention dimensions are incompatible
  - Enforced minimum state size (5 elements) to prevent negative calculations

- **HQ Model Saving Bug**
  - Fixed incorrect f-string usage in `train_hq_networks()` method
  - Changed `f.startswith(f"HQ_Faction_")` to `f.startswith("HQ_Faction_")` 
  - Pattern matching now correctly identifies HQ model files for top-5 tracking
  - Matches the same clean pattern used for agent model saving

- **Performance Optimizations Based on Profiling**
  - **Batch logging**: Added log buffering to reduce I/O overhead (2.3s → 1.96s, 15% reduction)
  - **Sprite caching**: Cache scaled sprites to avoid repeated transforms (562k → 626k calls but same time)
  - **Animation frame caching**: Cache scaled animation frames to avoid per-frame transforms
  - **Tensor device optimization**: Only call `.to(device)` if tensor isn't already on that device (21,938 calls reduced)
  - Logs now buffer 200 messages or flush every 2 seconds automatically (increased from 50/1s)
  - Logs force-flush at episode end to ensure data integrity
  - Fixed logger error when `log_buffer` attribute not yet initialized
  - **Terrain rendering caching**: Cache scaled grass and water textures by cell size (3.1s → reduced time)
  - Tiles only re-scaled when cell size changes (zoom changes)
  - Pre-compute tinted grass surface once per cell size
  - **Territory count caching**: Cache faction territory calculations per step to avoid redundant scans (432k calls per run)
  - Territory calculation time reduced from 2.52s to 2.36s (6% improvement)
  - Added `VERBOSE_TENSOR_LOGGING` flag to conditionally disable expensive tensor debug logging
  - Disabled tensor-to-string conversions in PPO_Agent_Network for production runs
  - **Results**: 88.28s → 55.09s runtime (37.6% improvement, 40% reduction in function calls from 34.87M → 21.07M)

- **Settings Menu Responsive Scaling**
  - Made Settings menu fully responsive to screen dimensions
  - All hardcoded coordinates now scale based on screen width and height
  - Sidebar, buttons, settings display, and scroll bar all use percentage-based positioning
  - Settings menu now works properly at any screen resolution

- **Settings Menu Audit & Cleanup**
  - Conducted audit of Settings menu to identify which settings are actually implemented
  - Created SETTINGS_AUDIT_SUMMARY.md documenting working vs non-working settings
  - **Removed unimplemented sections:**
    - Experience Replay (buffer initialized but never used)
    - Multi-Agent Training (never implemented)
    - Training Monitoring (save checkpoints, evaluation frequency, early stopping - not implemented)
  - **Kept working settings:**
    - AI Training (learning rates, batch size, GAE, PPO clip, gradient clipping, entropy)
    - Curriculum Learning (fully implemented, adjusts difficulty over time)
    - Advanced Loss (Huber Loss only, removed Focal Loss and Loss Normalization stubs)
  - Settings menu now only shows features that are actually functional

- **Persistent Settings System**
  - Created new `UTILITIES/settings_manager.py` to handle JSON-based persistent settings
  - Settings are stored in `settings.json` in the project root
  - Tracks installer completion status (prevents repeated prompts)
  - Tracks HEADLESS_MODE preference across runs
  - Can persist episode limits and step counts from previous runs
  - Settings automatically load on startup and can be saved anytime
  - Added helper functions to utils_config for easy access to persistent settings
  - Updated main.py to use persistent settings instead of modifying utils_config.py
  - First run prompts for HEADLESS_MODE, subsequent runs use saved preference
  - Added startup persistence check to skip dependency installer if already completed
  - Faster startup on subsequent runs (only runs installer on first run)
  - Automatic dependency error detection via direct import testing
  - Force reinstall feature triggers installer if dependencies are missing
  - Tests critical imports (torch, numpy, pygame, cv2) at startup
  - Automatically re-runs installer if any dependency import fails
  - Added methods for persisting settings menu configuration values to JSON
  - Can save and load config settings from Settings_Renderer.py automatically
  - Settings_Renderer.py now saves all changed settings to JSON when user clicks "Save and Return"
  - utils_config.py automatically loads persisted settings on startup
  - All config changes from the Settings menu now persist across runs
  - Fixed bug in get_config_settings() method that was causing TypeError
  - Added new "system" category to Settings menu
  - Added HEADLESS_MODE toggle control in system category
  - Added Force Dependency Check toggle in system category
  - Force Dependency Check toggle triggers dependency checker on next startup
  - Flag auto-clears after successful dependency check run
  - Settings menu now defaults to showing system category first

- **Attack Animation Fix**
  - Fixed missing attack animations for peacekeepers by processing event manager events in step()
  - Added event processing loop to call handle_event() for all events from event manager
  - Moved animation rendering to happen after agents but before HUD/tooltips (correct render order)
  - Added debug logging to track animation loading and rendering
  - Scaled animation sprite down by 50% to fit properly in render area
  - Fixed frame extraction to use full sheet width for vertical sprite frames (not cropped)
  - Peacekeeper attacks now properly trigger visible attack animations

- **Pause Menu Refactoring**
  - Extracted pause menu rendering into dedicated `PauseMenu_Renderer.py`
  - Creates cleaner separation of concerns, following the same pattern as MainMenu and Settings renderers
  - PauseMenuRenderer class handles all pause overlay rendering logic
  - GameRenderer now uses `pause_menu_renderer.render()` for cleaner code
  - Fixed elapsed time tracking to pause when game is paused (no more time ticking while paused)
  - Entire game system now respects paused state (rewards, training, victory checks, step counter all freeze)
  - Fixed pause menu overlay display by ensuring screen reference is updated and screen renders all elements when paused
  - Pause menu now renders at the absolute end of render pipeline (after tooltips/HUD are skipped when paused)
  - Fixed critical bug: paused state was not reaching renderer due to parameter order mismatch (paused was going to enable_cell_tooltip instead)

- **HQ Network Training Vector Size Mismatches**
  - Fixed training error where memory entries had inconsistent vector sizes
  - Added automatic vector normalization in `train()` method
  - Pads or truncates vectors to expected dimensions before training
  - Handles dimension changes during dynamic network updates gracefully
  - Prevents "expected sequence of length X (got Y)" errors during training

### Changed
- **Game Victory Conditions**
  - Added HQ destruction as a victory condition
  - Faction with last intact HQ wins by default
  - All HQs destroyed results in a draw
  - Victory reasons tracked (HQ Destruction, Resource Collection, Last Standing)

- **Threat Cleanup Logic**
  - Threats are only removed when actually destroyed/eliminated
  - Out-of-range threats persist as "last known" markers
  - No hardcoded detection range requirements in threat persistence
  - Flexible threat age system prevents premature removal

### Technical Details

**Files Modified**:
- `AGENT/agent_factions.py`:
  - Added HQ health tracking in `home_base` dictionary
  - Added `take_hq_damage()`, `get_enemy_hqs()`, `is_hq_destroyed()` methods
  - Enhanced `aggregate_faction_state()` to track HQ health
  - Updated `receive_report()` to maintain threat timestamps
  - Enhanced `clean_global_state()` to preserve "last known" threats
  - Added HQ health awareness to `get_enhanced_global_state()`

- `AGENT/agent_base.py`:
  - Updated `detect_threats()` to handle multiple enemy HQs via `get_enemy_hqs()`
  - Added fallback detection for single HQ (backward compatibility)

- `AGENT/Agent_Behaviours/peacekeeper_behaviours.py`:
  - Enhanced `eliminate_threat()` to detect and attack HQs
  - Added opportunistic HQ attacks when no assigned targets in range

- `GAME/game_rules.py`:
  - Updated `check_victory()` to check HQ destruction first
  - Added HQ-based victory condition

- `RENDER/Game_Renderer.py`:
  - Added health bar rendering on HQ hover
  - Health bars show green (>50%), yellow (25-50%), or red (<25%)
  - Added destroyed indicator (red "X") for destroyed HQs

- `GAME/game_manager.py`:
  - Updated to collect all enemy HQs for agent detection
  - Multiple HQs now available for threat detection

- `NEURAL_NETWORK/HQ_Network.py`:
  - Added vector normalization in `train()` method to handle inconsistent memory sizes
  - Automatically pads/truncates vectors to expected dimensions before training
  - Prevents dimension mismatch errors during training
  - Fixed batch dimension handling in `forward()` method (lines 203-233)
  - Added proper batch mode support for training

- `AGENT/agent_factions.py`:
  - Enhanced HQ reward system with outcome-based evaluation
  - Rewards now check if actions were logical AND actually achieved their goals
  - **Recruitment rewards**: Higher reward (+1.5) when addressing actual needs (threats/resources)
  - **Swap rewards**: Prevents pointless swaps (e.g., swapping to gatherer with no resources = -0.5)
  - **Swap rewards**: Rewards highly (+1.5) when repurposing idle gatherers to peacekeepers when no resources
  - **Defend HQ rewards**: Only rewards (+1.0) if peacekeepers are actually assigned to defense
  - **Attack Threats rewards**: Rewards based on actual threat engagement (+1.5 actively attacking, +0.5 if not yet)
  - **Resource Collection rewards**: Checks if agents are actually mining/foraging and if resources are needed

- `AGENT/agent_behaviours.py`:
  - Enhanced agent rewards with context awareness based on observable information
  - **HQ proximity** (Elimination): Up to +0.3 bonus for eliminating threats near HQ (better defense)
  - **Survival value** (Success): Up to +0.1 bonus for low-health agents completing tasks (survival priority)
  - **Critical failure** (Failure): Extra -0.2 penalty for very low-health agents failing tasks
  - **Role-based proximity shaping** (Ongoing): Peacekeepers closer to threats and gatherers closer to resources get bonuses (+0.1 max)
  - Agents now receive rewards that encourage strategic behavior without global context

**Expected Behavior**:
- HQs now have health limits and can be destroyed
- Agents can target and attack enemy HQs
- Strategic gameplay with HQ protection becomes critical
- Threats persist as intelligence even when out of range
- HQ Networks can make informed decisions based on health and threat status