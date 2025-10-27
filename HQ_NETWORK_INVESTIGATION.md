# HQ Network Investigation Report

## Overview
Investigation into the HQ Network's ability to pick strategies, focusing on dimension mismatch errors and state encoding issues.

## Issues Found

### 1. **Dimension Out of Range Error** 🔴
**Location:** `NEURAL_NETWORK/HQ_Network.py` forward method

**Problem:** 
- Error message: "Dimension out of range (expected to be in range of [-1, 0], but got -2)"
- The network was attempting to update dynamically but using incorrect tensor reshapes
- The forward pass would fail when the network tried to apply attention to improperly shaped tensors

**Root Cause:**
- The attention mechanism was not handling all possible input tensor dimensions correctly
- Single dimension tensors were not being properly reshaped before attention

### 2. **State Vector Size Mismatch** 🟡
**Location:** `NEURAL_NETWORK/HQ_Network.py` encode_state_parts method

**Problem:**
- Network expected inputs of size 62 but received varying sizes (57, 52, 47, 42)
- The `local_state_vector` was hardcoded to 5 elements but should match `self.local_state_size` (which was 10)
- This caused the network to constantly update its architecture

**Root Cause:**
- The `encode_state_parts()` method created a `local_vector` with only 5 hardcoded elements
- The configuration (`agent_faction_manager.py`) set `local_state_size=10`
- No padding was applied to match the expected size

### 3. **Dynamic Network Update Issues** 🟡
**Location:** `NEURAL_NETWORK/HQ_Network.py` update_network method

**Problem:**
- The network update logic attempted to copy layers by index which could fail with different architectures
- Fragile indexing that could break if layer count changed

**Root Cause:**
- Using `old_feature_extractor[4], old_feature_extractor[5]...` pattern
- No bounds checking or flexible layer handling

## Fixes Applied

### Fix 1: Enhanced Local Vector Padding ✅
**File:** `NEURAL_NETWORK/HQ_Network.py` lines 575-592

**Changes:**
- Added proper padding/truncation for `local_vector` to match `self.local_state_size`
- Ensures consistent input dimensions regardless of configuration
- Now pads with zeros if too short, or truncates if too long

```python
# Build local vector with fixed number of elements
local_vector_base = [
    resource_proximity,
    threat_proximity,
    resource_count_norm,
    threat_count_norm,
    min(max(g.get("agent_density", 0.0) / 10.0, 0.0), 1.0),
]

# Pad or truncate to match expected size
if len(local_vector_base) < self.local_state_size:
    local_vector = local_vector_base + [0.0] * (self.local_state_size - len(local_vector_base))
elif len(local_vector_base) > self.local_state_size:
    local_vector = local_vector_base[:self.local_state_size]
else:
    local_vector = local_vector_base
```

### Fix 2: Improved Network Update Logic ✅
**File:** `NEURAL_NETWORK/HQ_Network.py` lines 156-187

**Changes:**
- Replaced fragile indexing with flexible layer copying
- Added bounds checking before accessing layers
- Uses a more robust approach to preserve network structure

```python
# Get old feature extractor layers (skip the first linear layer as we'll replace it)
old_layers = list(self.feature_extractor.children())

# Build new feature extractor with first layer updated
new_layers = [
    nn.Linear(new_input_size, self.hidden_size),
    nn.LayerNorm(self.hidden_size),
    nn.ReLU(),
    nn.Dropout(0.1) if self.use_dropout else nn.Identity(),
]

# Add the remaining layers from the old feature extractor
if len(old_layers) > 4:
    for i in range(4, len(old_layers)):
        new_layers.append(old_layers[i])
```

### Fix 3: Enhanced Attention Layer Input Handling ✅
**File:** `NEURAL_NETWORK/HQ_Network.py` lines 221-234

**Changes:**
- Added handling for 1-dimensional tensors
- Improved dimension checking before attention
- Better dimension restoration after attention

```python
# Ensure features has the right dimensions
if features.dim() == 1:
    features = features.unsqueeze(0).unsqueeze(0)  # Add batch and seq dimensions
elif features.dim() == 2:
    features = features.unsqueeze(1)  # Add seq dimension
features = self.attention(features)
# Restore original dimensions
if features.dim() == 3:
    features = features.squeeze(1)
elif features.dim() == 1:
    features = features.unsqueeze(0)
```

## Strategy Selection Analysis

### Current Strategy Selection Process
1. **State Encoding** (encode_state_parts):
   - Extracts HQ health, resources, threats
   - Encodes agent proximity and density
   - Adds global faction state information

2. **Strategy Prediction** (predict_strategy):
   - Encodes state into 4 vectors: state, role, local_state, global_state
   - Passes through neural network for logits
   - Adds exploration noise for untrained networks
   - Selects strategy via argmax

3. **Exploration Mechanism**:
   - **For untrained networks**: Adds significant noise (2.0 scale) to encourage exploration
   - **For trained networks**: Uses deterministic selection
   - This ensures different HQs pick different strategies initially

### Strategy Options
The HQ can choose from 9 strategies:
- `DEFEND_HQ`: Protect headquarters
- `ATTACK_THREATS`: Engage enemy agents
- `COLLECT_GOLD`: Prioritize gold resources
- `COLLECT_FOOD`: Prioritize food resources
- `RECRUIT_GATHERER`: Spawn gatherer agents
- `RECRUIT_PEACEKEEPER`: Spawn peacekeeper agents
- `SWAP_TO_GATHERER`: Convert agents to gatherers
- `SWAP_TO_PEACEKEEPER`: Convert agents to peacekeepers
- `NO_PRIORITY`: No specific strategy (baseline)

### Network Architecture
- **Input Size**: Variable (state_size + role_size + local_state_size + global_state_size)
- **Architecture**: Enhanced with attention mechanism
- **Features**: 
  - Feature extraction layers with LayerNorm and Dropout
  - Attention mechanism for strategic decision making
  - Separate policy and value heads
  - AdamW optimizer with learning rate scheduling

## Recommendations

### 1. **Simplify Dynamic Network Updates**
Consider using a fixed input size at initialization rather than dynamic updates. The current approach is fragile and can cause training instability.

**Suggested Approach:**
```python
# Calculate expected input size at initialization
expected_input_size = state_size + role_size + local_state_size + global_state_size
# Use this as a fixed size, pad/truncate inputs as needed
```

### 2. **Add Strategy Selection Logging**
The network logs strategy selection, but consider adding:
- Confidence scores for each strategy
- Strategy performance tracking over time
- Strategy transition statistics

### 3. **Improve State Encoding**
Consider adding more meaningful features:
- Agent health distribution
- Resource availability metrics
- Threat proximity and severity
- Economic indicators (gold flow rate, etc.)

### 4. **Enhanced Training**
- Add curriculum learning for strategy complexity
- Implement strategy-specific rewards
- Track strategy success rates

## Testing Recommendations

1. **Run the simulation** and check logs for:
   - No more "Dimension out of range" errors
   - Consistent input sizes
   - Proper strategy selection
   - Network updates working smoothly

2. **Monitor strategy diversity**:
   - Are different HQs selecting different strategies?
   - Does the selection make tactical sense?

3. **Check performance**:
   - Are strategies being executed?
   - Are faction outcomes improving with better strategies?

## Conclusion

The main issues were:
1. ✅ **Fixed**: Dimension handling in attention layer
2. ✅ **Fixed**: Local state vector padding to match expected size
3. ✅ **Fixed**: Network update logic robustness

The HQ Network should now be able to:
- Properly encode game state
- Pass through network without dimension errors
- Select strategies reliably
- Update network architecture if needed

---

**Date:** 2025-10-27
**Status:** Fixes Applied - Ready for Testing

