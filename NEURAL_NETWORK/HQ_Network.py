"""Common Imports"""
from SHARED.core_imports import *

"""File Specific Imports"""
from NEURAL_NETWORK.Common import Training_device, save_checkpoint, load_checkpoint
from NEURAL_NETWORK.AttentionLayer import AttentionLayer
import torch.nn.functional as F
import UTILITIES.utils_config as utils_config


logger = Logger(log_file="HQ_Network.txt", log_level=logging.DEBUG)

#    _   _  ___    _   _ _____ _______        _____  ____  _  __
#   | | | |/ _ \  | \ | | ____|_   _\ \      / / _ \|  _ \| |/ /
#   | |_| | | | | |  \| |  _|   | |  \ \ /\ / / | | | |_) | ' /
#   |  _  | |_| | | |\  | |___  | |   \ V  V /| |_| |  _ <| . \
#   |_| |_|\__\_\ |_| \_|_____| |_|    \_/\_/  \___/|_| \_\_|\_\
#
class HQ_Network(nn.Module):
    def __init__(
            self,
            state_size=29,
            action_size = len(utils_config.HQ_STRATEGY_OPTIONS),
            role_size=5,
            local_state_size=5,
            global_state_size=5,
            device=None,
            global_state=None,
            use_attention=True,
            use_dropout=True,
            hidden_size=256,
            AgentID=None):
        super().__init__()
        # Store AgentID for consistency with other network models
        self.AgentID = AgentID
        
        # Initialise the device to use (CPU or GPU) default to Training_device if none
        if device is None:
           device = Training_device
        self.device = device
        
        # Store configuration
        self.state_size = state_size
        self.role_size = role_size
        self.local_state_size = local_state_size
        self.global_state_size = global_state_size
        self.use_attention = use_attention
        self.use_dropout = use_dropout
        self.hidden_size = hidden_size
        
        # Compute total input size dynamically
        total_input_size = state_size + role_size + local_state_size + global_state_size
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[DEBUG] Enhanced HQ_Network expected input size: {total_input_size}",
                level=logging.INFO)
        self.strategy_labels = utils_config.HQ_STRATEGY_OPTIONS

        if utils_config.ENABLE_LOGGING:
            print(
                f"[DEBUG] Enhanced HQ_Network initialised with input size: {total_input_size}")

        # Enhanced neural network architecture
        self.feature_extractor = nn.Sequential(
            nn.Linear(total_input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1) if use_dropout else nn.Identity(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1) if use_dropout else nn.Identity()
        )
        
        # Attention mechanism for strategic decision making
        if use_attention:
            self.attention = AttentionLayer(hidden_size, hidden_size)
            self.attention_output_size = hidden_size
        else:
            self.attention = None
            self.attention_output_size = hidden_size
        
        # Separate policy and value heads
        self.policy_head = nn.Sequential(
            nn.Linear(self.attention_output_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.05) if use_dropout else nn.Identity(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(self.attention_output_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.05) if use_dropout else nn.Identity(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)
        )

        # Enhanced optimizer with configuration-based parameters
        self.optimizer = optim.AdamW(
            self.parameters(), 
            lr=utils_config.INITIAL_LEARNING_RATE_HQ, 
            weight_decay=1e-5
        )
        
        # Advanced learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=utils_config.LEARNING_RATE_STEP_SIZE, 
            gamma=utils_config.LEARNING_RATE_DECAY
        )

        # Enhanced memory system
        self.hq_memory = []
        self.training_history = {
            "losses": [],
            "policy_losses": [],
            "value_losses": [],
            "entropies": [],
            "learning_rates": []
        }
        
        # Performance tracking
        self.total_updates = 0
        self.best_reward = float('-inf')
        
        self.global_state = global_state
        self.to(self.device)
        
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"Enhanced HQ_Network initialised successfully with attention={use_attention}, "
                f"dropout={use_dropout}, hidden_size={hidden_size}", 
                level=logging.INFO)

    def update_network(self, new_input_size):
        """
        Update the network structure dynamically when the input size changes.
        """
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[INFO] Updating HQ_Network input size from {self.feature_extractor[0].in_features} to {new_input_size}")
        
        # Update the first layer of feature extractor
        old_feature_extractor = self.feature_extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(new_input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1) if self.use_dropout else nn.Identity(),
            old_feature_extractor[4],  # Second linear layer
            old_feature_extractor[5],  # Second LayerNorm
            old_feature_extractor[6],  # Second ReLU
            old_feature_extractor[7]   # Second Dropout/Identity
        ).to(self.device)
        
        # Update stored sizes
        self.state_size = new_input_size - (self.role_size + self.local_state_size + self.global_state_size)

    def forward(self, state, role, local_state, global_state):
        """
        Enhanced forward pass through the HQ network with attention.
        """
        device = self.device

        # Ensure all inputs are on the correct device and properly shaped
        state = state.to(device).view(-1)
        role = role.to(device).view(-1)
        local_state = local_state.to(device).view(-1)
        global_state = global_state.to(device).view(-1)

        input_size_check = state.shape[0] + role.shape[0] + \
            local_state.shape[0] + global_state.shape[0]

        if input_size_check != self.feature_extractor[0].in_features:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[INFO] Updating HQ_Network input size from {self.feature_extractor[0].in_features} to {input_size_check}")
            self.update_network(input_size_check)

        # Concatenate all inputs
        x = torch.cat([state, role, local_state, global_state], dim=-1)

        # Feature extraction
        features = self.feature_extractor(x)
        
        # Apply attention if enabled
        if self.attention is not None:
            # Reshape for attention (batch_size, seq_len, features)
            if features.dim() == 2:
                features = features.unsqueeze(1)
            features = self.attention(features)
            if features.dim() == 3:
                features = features.squeeze(1)
        
        # Policy and value outputs
        policy_logits = self.policy_head(features)
        value = self.value_head(features)

        return policy_logits, value
        

    def add_memory(self, state: list, role: list, local_state: list, global_state: list, action: int, reward: float = 0.0):
        """
        Store a full HQ memory entry for reinforcement learning.
        Each part should be a list of floats representing the input state.

        :param state: Shared/central state vector
        :param role: One-hot or numerical role vector
        :param local_state: Local (agent/HQ-specific) state vector
        :param global_state: Global game state vector
        :param action: Chosen action index
        :param reward: Reward received (can be updated later)
        """
        # Validate inputs
        if not isinstance(action, int) or action < 0 or action >= len(self.strategy_labels):
            logger.log_msg(f"[ERROR] Invalid action index: {action}", level=logging.ERROR)
            return
        
        if not isinstance(reward, (int, float)):
            logger.log_msg(f"[ERROR] Invalid reward type: {type(reward)}", level=logging.ERROR)
            return

        memory_entry = {
            "state": state,
            "role": role,
            "local_state": local_state,
            "global_state": global_state,
            "action": action,
            "reward": reward,
            "timestamp": len(self.hq_memory)  # Track order
        }

        self.hq_memory.append(memory_entry)

        # Update best reward if applicable
        self.update_best_reward(reward)

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[MEMORY] Added HQ experience: action={action} ({self.strategy_labels[action]}), "
                f"reward={reward:.2f}, memory_size={len(self.hq_memory)}",
                level=logging.DEBUG
            )

    def update_memory_rewards(self, total_reward: float):
        """Update the reward for each memory entry."""
        if not isinstance(total_reward, (int, float)):
            logger.log_msg(f"[ERROR] Invalid reward type: {type(total_reward)}", level=logging.ERROR)
            return
            
        for m in self.hq_memory:
            m["reward"] = total_reward
        
        # Update best reward
        self.update_best_reward(total_reward)
        
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(f"[MEMORY] Updated all HQ memory rewards to {total_reward:.2f}", level=logging.DEBUG)

    def clear_memory(self):
        """Clear the HQ memory and reset performance tracking."""
        self.hq_memory = []
        self.best_reward = float('-inf')
        
        if utils_config.ENABLE_LOGGING:
            logger.log_msg("[MEMORY] HQ memory cleared", level=logging.INFO)

    def get_memory_efficiency(self):
        """
        Analyze memory usage and efficiency.
        """
        if not self.hq_memory:
            return {"efficiency": 0.0, "status": "empty", "details": {}}
        
        # Check memory consistency
        memory_sizes = {}
        for entry in self.hq_memory:
            for key in ["state", "role", "local_state", "global_state"]:
                if key not in memory_sizes:
                    memory_sizes[key] = []
                memory_sizes[key].append(len(entry.get(key, [])))
        
        # Calculate efficiency based on consistency
        all_lengths = []
        for key, lengths in memory_sizes.items():
            all_lengths.extend(lengths)
        
        if not all_lengths:
            return {"efficiency": 0.0, "status": "empty", "details": memory_sizes}
        
        min_len = min(all_lengths)
        max_len = max(all_lengths)
        efficiency = min_len / max_len if max_len > 0 else 0.0
        
        status = "optimal" if efficiency > 0.95 else "inconsistent"
        
        return {
            "efficiency": efficiency,
            "status": status,
            "total_entries": len(self.hq_memory),
            "details": memory_sizes,
            "recommendation": "clear_memory()" if status == "inconsistent" else "memory_ok"
        }

    def predict_strategy(self, global_state: dict) -> str:
        """
        Enhanced strategy prediction with better error handling and performance tracking.
        """
        if hasattr(self, 'predicting'):
            logger.log_msg("[WARNING] Recursive prediction call detected, returning default strategy", level=logging.WARNING)
            return self.strategy_labels[0]
        
        self.predicting = True

        try:
            # Validate global state
            if not isinstance(global_state, dict):
                logger.log_msg(f"[ERROR] Invalid global_state type: {type(global_state)}", level=logging.ERROR)
                return self.strategy_labels[0]
            
            # Check if global_state has required keys
            required_keys = ["HQ_health", "gold_balance", "food_balance", "resource_count", "threat_count"]
            missing_keys = [key for key in required_keys if key not in global_state]
            if missing_keys:
                logger.log_msg(f"[WARNING] Missing keys in global_state: {missing_keys}", level=logging.WARNING)
                # Fill missing keys with defaults
                for key in missing_keys:
                    global_state[key] = 0.0

            # Store the enhanced global_state temporarily and use it for encoding
            old_global_state = self.global_state
            self.global_state = global_state  # Use the enhanced state passed in

            try:
                # 1. Extract structured input parts
                state, role, local_state, global_state_vec = self.encode_state_parts()
            finally:
                # Restore original global_state
                self.global_state = old_global_state

            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[INFO] Predicting strategy for global state: {global_state}",
                    level=logging.DEBUG)
                logger.log_msg(
                    f"[DEBUG] Encoded state vector: {state}",
                    level=logging.DEBUG)

            # 2. Convert to tensors and move to correct device
            device = self.device
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            role_tensor = torch.tensor(role, dtype=torch.float32, device=device).unsqueeze(0)
            local_tensor = torch.tensor(local_state, dtype=torch.float32, device=device).unsqueeze(0)
            global_tensor = torch.tensor(global_state_vec, dtype=torch.float32, device=device).unsqueeze(0)

            # 3. Forward pass with error handling
            with torch.no_grad():
                try:
                    logits, value = self.forward(state_tensor, role_tensor, local_tensor, global_tensor)
                except Exception as e:
                    logger.log_msg(f"[ERROR] Forward pass failed: {e}", level=logging.ERROR)
                    return self.strategy_labels[0]

            # 4. Log raw logits and value
            logits_list = logits.squeeze(0).tolist()
            for i, value_logit in enumerate(logits_list):
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[LOGITS] {self.strategy_labels[i]}: {value_logit:.4f}",
                        level=logging.INFO)

            # 5. Select strategy with highest probability
            action_index = torch.argmax(logits).item()
            selected_strategy = self.strategy_labels[action_index]

            # 6. Store memory for training
            self.add_memory(state, role, local_state, global_state_vec, action_index)

            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[HQ STRATEGY] Selected: {selected_strategy} (index: {action_index})",
                    level=logging.INFO)

            return selected_strategy
            
        except Exception as e:
            logger.log_msg(f"[ERROR] Strategy prediction failed: {e}", level=logging.ERROR)
            # Return default strategy on error
            return self.strategy_labels[0]
        finally:
            delattr(self, 'predicting')



    def encode_state_parts(self):
        """
        Enhanced state encoding with better error handling and normalization.
        Convert the global_state dictionary into structured input vectors
        for the HQ neural network: [state, role, local_state, global_state].
        """
        try:
            g = self.global_state
            if g is None:
                logger.log_msg("[WARNING] Global state is None, using default values", level=logging.WARNING)
                g = {}

            # Shared central state vector (used for main state input)
            # Normalize values to prevent extreme inputs
            state_vector = [
                min(max(g.get("HQ_health", 100.0) / 100.0, 0.0), 1.0),  # Clamp to [0, 1]
                min(max(g.get("gold_balance", 0.0) / 1000.0, 0.0), 2.0),  # Allow up to 2x normal
                min(max(g.get("food_balance", 0.0) / 1000.0, 0.0), 2.0),  # Allow up to 2x normal
                min(max(g.get("resource_count", 0.0) / 100.0, 0.0), 2.0),  # Allow up to 2x normal
                min(max(g.get("threat_count", 0.0) / 10.0, 0.0), 2.0)   # Allow up to 2x normal
            ]

            # Role vector (placeholder: 1-hot HQ, or make dynamic later)
            # Ensure role vector has correct size
            role_vector = [1.0] + [0.0] * (self.role_size - 1)  # [HQ, 0, 0, ...]
            if len(role_vector) > self.role_size:
                role_vector = role_vector[:self.role_size]
            elif len(role_vector) < self.role_size:
                role_vector.extend([0.0] * (self.role_size - len(role_vector)))

            # Local state (near HQ) - use proximity instead of raw coordinates
            nearest_resource_dist = g.get("nearest_resource", {}).get("distance", float('inf'))
            nearest_threat_dist = g.get("nearest_threat", {}).get("distance", float('inf'))
            
            # Normalize distances: closer = higher value (inverse distance with sigmoid)
            # Network will learn what proximity values correlate with good outcomes
            resource_proximity = 1.0 / (1.0 + nearest_resource_dist / 200.0) if nearest_resource_dist < float('inf') else 0.0
            threat_proximity = 1.0 / (1.0 + nearest_threat_dist / 150.0) if nearest_threat_dist < float('inf') else 0.0
            
            # Provide counts as additional context
            resource_count_norm = min(g.get("resource_count", 0) / 20.0, 1.0)
            threat_count_norm = min(g.get("threat_count", 0) / 5.0, 1.0)
            
            local_vector = [
                resource_proximity,          # How close is nearest resource (0-1, higher=closer)
                threat_proximity,            # How close is nearest threat (0-1, higher=closer)
                resource_count_norm,         # How many resources known total
                threat_count_norm,           # How many threats known total
                min(max(g.get("agent_density", 0.0) / 10.0, 0.0), 1.0)  # Agent concentration near HQ
            ]

            # Global map-wide state with enhanced features  
            # Use enhanced features from get_enhanced_global_state() if available
            friendly_count = g.get("friendly_agent_count", 0.0)
            global_vector = [
                min(max(friendly_count / 10.0, 0.0), 2.0),  # Feature 0: number of friendly agents
                min(max(g.get("enemy_agent_count", 0.0) / 10.0, 0.0), 2.0),     # Feature 1
                min(max(g.get("total_agents", 0.0) / 10.0, 0.0), 2.0),          # Feature 2
                min(max(g.get("gatherer_count", 0.0) / 10.0, 0.0), 2.0),         # Feature 3: gatherer_count (enhanced)
                min(max(g.get("peacekeeper_count", 0.0) / 10.0, 0.0), 2.0),      # Feature 4: peacekeeper_count (enhanced)
            ]
            
            # Add agent presence indicator (neutral observation, not a directive)
            # This gives the network information about agent availability without forcing recruitment
            global_vector.append(1.0 if friendly_count > 0 else 0.0)  # Feature 5: has_agents (binary)
            
            # If global_state_size allows more features, add swap benefit signals and affordability
            if self.global_state_size >= 7:
                # Note: we already added no_agents_emergency, so this shifts indices
                global_vector.extend([
                    min(max(g.get("swap_to_gatherer_benefit", 0.0), 0.0), 1.0),   # Feature 6: swap to gatherer benefit
                    min(max(g.get("swap_to_peacekeeper_benefit", 0.0), 0.0), 1.0), # Feature 7: swap to peacekeeper benefit
                    g.get("can_afford_recruit", 0.0),  # Feature 8: can afford new recruitment (binary)
                    g.get("can_afford_swap", 0.0)       # Feature 9: can afford swap (binary)
                ])
            elif self.global_state_size < 5:
                global_vector = global_vector[:self.global_state_size]
            
            # Ensure global vector has correct size
            if len(global_vector) > self.global_state_size:
                global_vector = global_vector[:self.global_state_size]
            elif len(global_vector) < self.global_state_size:
                global_vector.extend([0.0] * (self.global_state_size - len(global_vector)))

            # Validate vector sizes
            if len(state_vector) != self.state_size:
                logger.log_msg(f"[WARNING] State vector size mismatch: expected {self.state_size}, got {len(state_vector)}", level=logging.WARNING)
                # Pad or truncate as needed
                if len(state_vector) < self.state_size:
                    state_vector.extend([0.0] * (self.state_size - len(state_vector)))
                else:
                    state_vector = state_vector[:self.state_size]

            if utils_config.ENABLE_LOGGING:
                logger.log_msg(f"[ENCODE] State parts encoded successfully: state={len(state_vector)}, "
                             f"role={len(role_vector)}, local={len(local_vector)}, global={len(global_vector)}", 
                             level=logging.DEBUG)

            return state_vector, role_vector, local_vector, global_vector
            
        except Exception as e:
            logger.log_msg(f"[ERROR] State encoding failed: {e}", level=logging.ERROR)
            # Return safe default values
            default_state = [0.5] * self.state_size
            default_role = [1.0] + [0.0] * (self.role_size - 1)
            default_local = [0.0] * self.local_state_size
            default_global = [0.0] * self.global_state_size
            
            return default_state, default_role, default_local, default_global


    def train(self, memory, optimizer=None, gamma=0.99, entropy_coeff=None):
        """
        Enhanced training for the HQ strategy model using policy gradient.
        :param memory: A list of dicts with keys: 'state', 'role', 'local_state', 'global_state', 'action', 'reward'
        :param optimizer: Torch optimizer (uses self.optimizer if None)
        :param gamma: Discount factor for future rewards
        :param entropy_coeff: Entropy coefficient for exploration (uses config if None)
        """
        if not memory:
            logger.log_msg("[HQ TRAIN] No memory to train on.", level=logging.WARNING)
            return

        # Use configuration-based parameters if none provided
        if entropy_coeff is None:
            entropy_coeff = utils_config.ENTROPY_COEFF

        # Use internal optimizer if none provided
        if optimizer is None:
            optimizer = self.optimizer

        # Prepare batches from structured memory
        device = self.device
        states = torch.tensor([m['state'] for m in memory], dtype=torch.float32, device=device)
        roles = torch.tensor([m['role'] for m in memory], dtype=torch.float32, device=device)
        locals_ = torch.tensor([m['local_state'] for m in memory], dtype=torch.float32, device=device)
        globals_ = torch.tensor([m['global_state'] for m in memory], dtype=torch.float32, device=device)
        actions = torch.tensor([m['action'] for m in memory], dtype=torch.long, device=device)
        rewards = [m['reward'] for m in memory]

        # Compute discounted returns with better numerical stability
        returns = []
        G = 0 
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        # Normalise returns with clipping for stability
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        returns = torch.clamp(returns, -10.0, 10.0)

        # Forward pass with full input
        logits, values = self.forward(states, roles, locals_, globals_)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)

        # Enhanced loss computation with configuration-based parameters
        advantage = returns - values.squeeze(-1)
        
        # Policy loss with clipping for stability
        policy_loss = -(log_probs * advantage.detach()).mean()
        
        # Value loss with clipping and optional Huber loss
        if utils_config.USE_HUBER_LOSS:
            value_loss = F.huber_loss(values.squeeze(-1), returns, reduction='mean')
        else:
            value_loss = advantage.pow(2).mean()
        
        # Entropy bonus for exploration
        entropy = dist.entropy().mean()
        
        # Combined loss with configurable coefficients
        loss = policy_loss + utils_config.VALUE_LOSS_COEFF * value_loss - entropy_coeff * entropy
        
        # Backpropagation with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), utils_config.GRADIENT_CLIP_NORM)
        optimizer.step()
        
        # Update learning rate
        if hasattr(self, 'scheduler'):
            self.scheduler.step()
        
        # Track training metrics
        self.total_updates += 1
        self.training_history["losses"].append(loss.item())
        self.training_history["policy_losses"].append(policy_loss.item())
        self.training_history["value_losses"].append(value_loss.item())
        self.training_history["entropies"].append(entropy.item())
        if hasattr(self, 'scheduler'):
            self.training_history["learning_rates"].append(self.scheduler.get_last_lr()[0])

        # Logging
        logger.log_msg(
            f"[HQ TRAIN] Loss: {loss.item():.4f}, Policy: {policy_loss.item():.4f}, "
            f"Value: {value_loss.item():.4f}, Entropy: {entropy.item():.4f}, "
            f"LR: {self.scheduler.get_last_lr()[0]:.6f}",
            level=logging.INFO
        )

    def save_model(self, path):
        save_checkpoint(self, path)

    def get_training_stats(self):
        """
        Get comprehensive training statistics for monitoring and debugging.
        """
        if not self.training_history["losses"]:
            return {
                "total_updates": self.total_updates,
                "current_lr": self.scheduler.get_last_lr()[0] if hasattr(self, 'scheduler') else 0.0,
                "memory_size": len(self.hq_memory),
                "best_reward": self.best_reward,
                "avg_loss": 0.0,
                "avg_policy_loss": 0.0,
                "avg_value_loss": 0.0,
                "avg_entropy": 0.0,
                "loss_trend": "stable"
            }
        
        recent_losses = self.training_history["losses"][-10:]  # Last 10 updates
        if len(recent_losses) >= 2:
            loss_trend = "improving" if recent_losses[-1] < recent_losses[0] else "degrading"
        else:
            loss_trend = "stable"
        
        return {
            "total_updates": self.total_updates,
            "current_lr": self.scheduler.get_last_lr()[0] if hasattr(self, 'scheduler') else 0.0,
            "memory_size": len(self.hq_memory),
            "best_reward": self.best_reward,
            "avg_loss": sum(self.training_history["losses"]) / len(self.training_history["losses"]),
            "avg_policy_loss": sum(self.training_history["policy_losses"]) / len(self.training_history["policy_losses"]),
            "avg_value_loss": sum(self.training_history["value_losses"]) / len(self.training_history["value_losses"]),
            "avg_entropy": sum(self.training_history["entropies"]) / len(self.training_history["entropies"]),
            "loss_trend": loss_trend,
            "recent_losses": recent_losses
        }

    def reset_model(self, keep_memory=False):
        """
        Reset the model weights while optionally preserving memory.
        Useful for starting fresh training runs.
        """
        if not keep_memory:
            self.clear_memory()
        
        # Reset optimizer and scheduler
        self.optimizer = optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.95)
        
        # Reset counters
        self.total_updates = 0
        self.best_reward = float('-inf')
        
        # Clear training history
        for key in self.training_history.keys():
            self.training_history[key] = []
        
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(f"[MODEL RESET] HQ Network reset successfully. Memory preserved: {keep_memory}", level=logging.INFO)

    def get_model_complexity(self):
        """
        Get information about model complexity and parameters.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "attention_layers": 1 if self.attention else 0,
            "dropout_enabled": self.use_dropout,
            "hidden_size": self.hidden_size,
            "architecture": "Enhanced HQ Network with Attention" if self.attention else "Enhanced HQ Network"
        }

    def set_learning_rate(self, new_lr):
        """
        Dynamically adjust learning rate during training.
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(f"[LR UPDATE] HQ Network learning rate updated to {new_lr:.6f}", level=logging.INFO)

    def get_learning_rate(self):
        """
        Get current learning rate.
        """
        return self.optimizer.param_groups[0]['lr']

    def freeze_layers(self, layer_names=None):
        """
        Freeze specific layers to prevent them from being updated during training.
        Useful for transfer learning or fine-tuning.
        """
        if layer_names is None:
            # Freeze feature extractor by default
            layer_names = ['feature_extractor']
        
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(f"[FREEZE] Frozen HQ layer: {name}", level=logging.DEBUG)
        
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(f"[FREEZE] Frozen HQ layers: {layer_names}", level=logging.INFO)

    def unfreeze_layers(self, layer_names=None):
        """
        Unfreeze previously frozen layers.
        """
        if layer_names is None:
            # Unfreeze all layers
            for param in self.parameters():
                param.requires_grad = True
        else:
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = True
                    if utils_config.ENABLE_LOGGING:
                        logger.log_msg(f"[UNFREEZE] Unfrozen HQ layer: {name}", level=logging.DEBUG)
        
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(f"[UNFREEZE] Unfrozen HQ layers: {layer_names if layer_names else 'all'}", level=logging.INFO)

    def update_best_reward(self, reward):
        """
        Update the best reward achieved and track performance.
        """
        if reward > self.best_reward:
            self.best_reward = reward
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(f"[BEST REWARD] New best HQ reward: {reward:.2f}", level=logging.INFO)
            return True
        return False

    def get_strategy_confidence(self, strategy_index):
        """
        Get confidence level for a specific strategy.
        """
        if not self.hq_memory:
            return 0.0
        
        # Count how often this strategy was chosen
        strategy_count = sum(1 for m in self.hq_memory if m['action'] == strategy_index)
        total_choices = len(self.hq_memory)
        
        return strategy_count / total_choices if total_choices > 0 else 0.0

    def get_strategy_performance(self):
        """
        Analyze performance of different strategies.
        """
        if not self.hq_memory:
            return {}
        
        strategy_performance = {}
        for i, strategy_name in enumerate(self.strategy_labels):
            strategy_memories = [m for m in self.hq_memory if m['action'] == i]
            if strategy_memories:
                avg_reward = sum(m['reward'] for m in strategy_memories) / len(strategy_memories)
                usage_count = len(strategy_memories)
                strategy_performance[strategy_name] = {
                    "average_reward": avg_reward,
                    "usage_count": usage_count,
                    "confidence": usage_count / len(self.hq_memory)
                }
            else:
                strategy_performance[strategy_name] = {
                    "average_reward": 0.0,
                    "usage_count": 0,
                    "confidence": 0.0
                }
        
        return strategy_performance 