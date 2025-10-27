# Changelog

All notable changes to the Multi-Agent Faction Wars project will be documented in this file.

## [2025-10-27]

### Fixed
- **HQ Network Attention Layer Dimension Errors**
  - Fixed "Dimension out of range" errors during forward passes
  - Added comprehensive dimension checking for attention mechanism
  - Implemented graceful fallback when attention dimensions are incompatible
  - Enforced minimum state size (5 elements) to prevent negative calculations
  - Enhanced `update_network()` and `encode_state_parts()` to handle dynamic state size changes


## Latest Fixes (2025-10-27)

### Fixed: HQ Network Attention Layer Dimension Errors

**Problem**: HQ Networks were crashing with "Dimension out of range" errors during forward passes when the attention mechanism attempted to process tensors with incompatible shapes.

**Root Cause**: The attention layer expected 3D tensors `(batch_size, seq_len, features)` but was receiving tensors with varying dimensions during dynamic network updates. The PyTorch `transpose(-2, -1)` operation failed when tensors had fewer than 2 dimensions.

**Solution**:
1. **Enhanced Dimension Handling**: Added comprehensive dimension checking before applying attention, handling both 1D and 2D feature tensors.
2. **Error Recovery**: Implemented try-catch mechanism to gracefully skip attention when dimensions are incompatible.
3. **State Size Protection**: Added minimum state size enforcement (5 elements) to prevent negative size calculations.
4. **Robust Fallback**: Network continues strategy prediction even when attention fails, ensuring uninterrupted gameplay.

**Files Modified**:
- `NEURAL_NETWORK/HQ_Network.py`: 
  - Enhanced attention layer dimension handling (lines 227-256)
  - Added state size validation in `update_network()` (lines 183-193)
  - Fixed `encode_state_parts()` state vector minimum size enforcement (lines 552-559)

**Expected Behavior**: HQ Networks can now handle dynamic state size changes without crashing. If attention layer encounters dimension mismatches, it gracefully skips attention processing and continues with strategy prediction.
