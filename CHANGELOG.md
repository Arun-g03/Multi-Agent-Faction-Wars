# Multi-Agent Faction Wars - Changelog

## Recent Improvements (Latest Session)

### 1. Training Improvements
**Problem:** Episodes lasted 20,000 steps with training only at episode end, leading to inefficient learning and memory overflow issues.

**Changes Made:**
- **Mini-batch training during episodes** (`GAME/game_manager.py` lines 816-818)
  - Train every 1,000 steps during episodes
  - Use recent samples without clearing memory
  - Provides more frequent learning updates

- **Memory clearing after episode** (lines 884-887)
  - Clear agent memories after full-batch training at episode end
  - Prevents stale data from previous episodes
  - Starts fresh for next episode

- **Improved training logic** (lines 1178-1222)
  - Only trains agents with sufficient memory (>= 128 samples)
  - Skips agents with insufficient data
  - Logs training statistics for debugging

**Impact:** ~20x more frequent training updates, better sample efficiency, reduced memory growth.

---

### 2. Neural Network Bug Fixes
**Problem:** Memory overflow and numerical instability causing crashes during training.

**Changes Made:**
- **Memory buffer overflow protection** (`NEURAL_NETWORK/PPO_Agent_Network.py` lines 248-259)
  - Added FIFO buffer with max size of 20,000 (one episode)
  - Prevents unbounded memory growth
  - Automatically removes oldest samples when full

- **Numerical stability improvements** (lines 183, 364-367)
  - Logit clipping to [-10, 10] prevents extreme values in softmax
  - Value clipping to [-100, 100] prevents overflow in training
  - Added at forward pass and training time

**Impact:** Eliminates crashes from memory overflow and NaN/inf values in training.

---

### 3. HQ Recruitment for Zero-Agent Case
**Problem:** HQs with gold but no agents weren't recruiting because the network wasn't informed about the agent count state.

**Changes Made:**

#### A. Enhanced State Information (`AGENT/agent_factions.py` lines 2050-2060)
- Always include agent count data regardless of agent presence:
  ```python
  enhanced_state["friendly_agent_count"] = len(self.agents)
  enhanced_state["gatherer_count"] = len([a for a in self.agents if a.role == "gatherer"])
  enhanced_state["peacekeeper_count"] = len([a for a in self.agents if a.role == "peacekeeper"])
  enhanced_state["has_agents"] = 1.0 if len(self.agents) > 0 else 0.0
  ```

- Always include affordability signals:
  ```python
  enhanced_state["can_afford_recruit"] = 1.0 if self.gold_balance >= utils_config.Gold_Cost_for_Agent else 0.0
  enhanced_state["can_afford_swap"] = 1.0 if self.gold_balance >= utils_config.Gold_Cost_for_Agent_Swap else 0.0
  enhanced_state["gold_balance_norm"] = min(self.gold_balance / 1000.0, 2.0)
  ```

- Handle zero-agent case explicitly (lines 2085-2092):
  ```python
  # No agents: set default priorities
  enhanced_state["unit_balance"] = 0.0
  enhanced_state["resource_priority"] = 0.5
  enhanced_state["defense_priority"] = 0.5
  enhanced_state["exploration_priority"] = 0.0
  ```

#### B. Network Encoding Updates (`NEURAL_NETWORK/HQ_Network.py` lines 463, 471-472)
- Added neutral agent presence signal:
  ```python
  global_vector.append(1.0 if friendly_count > 0 else 0.0)  # has_agents (binary)
  ```

- Network receives afford signals (when `global_state_size >= 7`):
  - Feature 8: `can_afford_recruit`
  - Feature 9: `can_afford_swap`

**Impact:** Network can learn to recruit when it has 0 agents and sufficient gold, without hardcoded "emergency" logic.

---

## Earlier Improvements (Previous Sessions)

### Code Refactoring
- **Agent Types** (`AGENT/Agent_Types/`)
  - Extracted `Peacekeeper` and `Gatherer` into separate files
  - Improved modularity and organization

- **Agent Behaviors** (`AGENT/Agent_Behaviours/`)
  - Created mixin-based behavior system
  - `CoreActionsMixin`: Basic movement and healing
  - `GathererBehavioursMixin`: Mining and foraging
  - `PeacekeeperBehavioursMixin`: Combat and patrolling
  - Eliminated circular import issues

### State Perception Improvements
- **Agent state** (`AGENT/agent_base.py` `get_state()` method)
  - Replaced raw coordinates with proximity signals
  - Added role awareness (one-hot encoding)
  - Added task urgency and progress tracking
  - Added environmental context (threat/resource counts)
  
- **HQ state** (`AGENT/agent_factions.py` `get_enhanced_global_state()` method)
  - Added territory delta (gaining/losing ground)
  - Added nearest resource/threat distances
  - Added composition analysis (gatherer/peacekeeper ratios)
  - Added affordability signals

### Network Architecture Improvements
- **State encoding** (`NEURAL_NETWORK/HQ_Network.py`)
  - Proximity-based signals instead of raw distances
  - Count-based normalization
  - Territory change signals
  - Recruitment affordability signals

### Configuration & Infrastructure
- **Logging** (`UTILITIES/utils_logger.py`)
  - Separate error logs in `RUNTIME_LOGS/Error_Logs/`
  - General logs in `RUNTIME_LOGS/General_Logs/`
  - Better debugging capabilities

- **Plots & CSVs** (`UTILITIES/utils_config.py`, `RENDER/Settings_Renderer.py`)
  - Added `ENABLE_PLOTS` flag
  - Can be toggled from UI settings
  - Prevents crashes when disabled

- **Dependency installation** (`main.py`)
  - Startup installer runs first before imports
  - Ensures dependencies are installed before use

---

## Technical Details

### Training Flow Improvements
**Old Flow:**
1. Episode runs (up to 20,000 steps)
2. Training at end only
3. Memory carries over, leading to stale data

**New Flow:**
1. Episode runs (up to 20,000 steps)
2. Mini-batch training every 1,000 steps (on recent samples)
3. Full batch training at episode end
4. Memory cleared for next episode
5. Fresh start with clean slate

**Benefits:**
- ~20x more training opportunities
- Reduced memory footprint
- Better sample efficiency
- Faster convergence

### Memory Management
- **Buffer size**: Limited to 20,000 samples (one episode)
- **Eviction**: FIFO when buffer overflows
- **Clearing**: After episode-end training

### State Vector Updates
- **Agent state size**: `18 + len(TASK_TYPE_MAPPING)`
- **HQ global state**: Now includes agent counts, affordability, and has_agents signal
- **Network encoding**: Properly handles zero-agent case

---

## Files Modified

### Training & Game Management
- `GAME/game_manager.py` - Added mini-batch training, memory clearing

### Neural Networks
- `NEURAL_NETWORK/PPO_Agent_Network.py` - Memory overflow protection, numerical stability
- `NEURAL_NETWORK/HQ_Network.py` - Enhanced state encoding, zero-agent signals

### Agent Logic
- `AGENT/agent_factions.py` - Enhanced state with agent counts and affordability
- `AGENT/agent_base.py` - Improved state perception (previous session)

### Infrastructure
- `UTILITIES/utils_logger.py` - Separate error logs (previous session)
- `UTILITIES/utils_config.py` - Added ENABLE_PLOTS flag (previous session)
- `RENDER/Settings_Renderer.py` - Plot toggle in UI (previous session)
- `main.py` - Fixed startup order (previous session)

### New Files Created
- `AGENT/Agent_Types/peacekeeper.py` (previous session)
- `AGENT/Agent_Types/gatherer.py` (previous session)
- `AGENT/Agent_Behaviours/core_actions.py` (previous session)
- `AGENT/Agent_Behaviours/gatherer_behaviours.py` (previous session)
- `AGENT/Agent_Behaviours/peacekeeper_behaviours.py` (previous session)

---

## Testing Recommendations

1. **Train for 30 episodes** and observe:
   - HQs with gold and zero agents should recruit
   - Memory should not exceed 20,000 samples
   - No NaN/inf errors in training logs

2. **Monitor logs** in `RUNTIME_LOGS/`:
   - General logs for agent activity
   - Error logs for any issues

3. **Check TensorBoard** for:
   - Training loss convergence
   - Reward trends
   - Strategy selections

---

## Future Improvements

1. **Curriculum learning**: Already implemented but could be refined
2. **Adaptive episode length**: Shorten episodes for faster experimentation
3. **Better reward shaping**: Encourage zero-agent → recruit transitions
4. **Multi-faction learning**: Scale to more factions
5. **Advanced strategies**: More complex HQ decision-making

---

## Commit Summary

### feat: Improve training efficiency and fix critical bugs

**Training improvements:**
- Add mini-batch training every 1000 steps during episodes (~20x more updates)
- Implement FIFO memory buffer (max 20k samples) to prevent overflow
- Clear memory after episode-end training for fresh starts
- Only train agents with sufficient memory (>=128 samples)

**Bug fixes:**
- Fix memory overflow causing unbounded growth in PPO network
- Add logit and value clipping to prevent NaN/inf in training
- Fix HQ recruitment when agents=0 but has gold

**State perception enhancements:**
- Always provide agent count info to HQ (friendly_agent_count, gatherer_count, peacekeeper_count)
- Add affordability signals (can_afford_recruit, can_afford_swap)
- Add has_agents binary signal for zero-agent detection
- Handle zero-agent case in enhanced global state

**Network encoding:**
- Add agent presence indicator to global state vector
- Include affordability signals when global_state_size >= 7
- Enable network to learn recruitment when no agents + has gold

**Files changed:**
- GAME/game_manager.py: +88 lines (mini-batch training, memory clearing)
- NEURAL_NETWORK/PPO_Agent_Network.py: +30 lines (overflow protection, clipping)
- NEURAL_NETWORK/HQ_Network.py: +15 lines (zero-agent signals)
- AGENT/agent_factions.py: +45 lines (enhanced state info)

**Testing:**
- Run 30 episodes to verify HQ recruits with 0 agents + gold
- Check memory stays under 20k samples
- Verify no NaN/inf errors in training
- Monitor TensorBoard for convergence

