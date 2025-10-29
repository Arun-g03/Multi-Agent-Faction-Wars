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
        action_size=len(utils_config.HQ_STRATEGY_OPTIONS),
        role_size=5,
        local_state_size=5,
        global_state_size=5,
        device=None,
        global_state=None,
        use_attention=True,
        use_dropout=True,
        hidden_size=256,
        AgentID=None,
        faction_id=None,
    ):
        super().__init__()
        # Store AgentID for consistency with other network models
        self.AgentID = AgentID
        self.faction_id = faction_id

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

        # Use faction_id as a seed offset to ensure different initial weights for different HQs
        if faction_id is not None:
            # Set a deterministic seed based on faction ID
            torch.manual_seed(42 + faction_id)  # Base seed 42 + faction_id offset

        # Compute total input size dynamically
        total_input_size = state_size + role_size + local_state_size + global_state_size
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[DEBUG] Enhanced HQ_Network expected input size: {total_input_size}",
                level=logging.INFO,
            )
        self.strategy_labels = utils_config.HQ_STRATEGY_OPTIONS

        if utils_config.ENABLE_LOGGING:
            print(
                f"[DEBUG] Enhanced HQ_Network initialised with input size: {total_input_size}"
            )

        # Enhanced neural network architecture
        self.feature_extractor = nn.Sequential(
            nn.Linear(total_input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1) if use_dropout else nn.Identity(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1) if use_dropout else nn.Identity(),
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
            nn.Linear(hidden_size // 2, action_size),
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.attention_output_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.05) if use_dropout else nn.Identity(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
        )

        # ============================================================================
        # PARAMETRIC STRATEGY HEADS
        # ============================================================================

        # Binary parameter heads (sigmoid output, interpreted as probabilities)
        self.binary_param_head = nn.Sequential(
            nn.Linear(self.attention_output_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.05) if use_dropout else nn.Identity(),
            nn.Linear(
                hidden_size // 2, 3
            ),  # target_role, priority_resource, use_mission_system
            nn.Sigmoid(),  # Output probabilities for binary decisions
        )

        # Continuous parameter heads (tanh output, scaled to [0,1])
        self.continuous_param_head = nn.Sequential(
            nn.Linear(self.attention_output_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.05) if use_dropout else nn.Identity(),
            nn.Linear(
                hidden_size // 2, 7
            ),  # aggression_level, resource_threshold, urgency, mission_autonomy, coordination_preference, agent_adaptability, failure_tolerance
            nn.Tanh(),  # Output [-1,1], will be scaled to [0,1]
        )

        # Discrete parameter head (softmax output, interpreted as distribution)
        self.discrete_param_head = nn.Sequential(
            nn.Linear(self.attention_output_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.05) if use_dropout else nn.Identity(),
            nn.Linear(
                hidden_size // 2, 14
            ),  # agent_count_target (1-10) + mission_complexity (1-4)
            nn.Softmax(dim=-1),  # Output probability distribution over discrete choices
        )

        # Reset seed back to original after initialization to avoid affecting other HQs
        if faction_id is not None:
            torch.manual_seed(torch.initial_seed())

        # Enhanced optimizer with configuration-based parameters
        self.optimizer = optim.AdamW(
            self.parameters(),
            lr=utils_config.INITIAL_LEARNING_RATE_HQ,
            weight_decay=1e-5,
        )

        # Advanced learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=utils_config.LEARNING_RATE_STEP_SIZE,
            gamma=utils_config.LEARNING_RATE_DECAY,
        )

        # Enhanced memory system
        self.hq_memory = []
        self.training_history = {
            "losses": [],
            "policy_losses": [],
            "value_losses": [],
            "entropies": [],
            "learning_rates": [],
        }

        # Performance tracking
        self.total_updates = 0
        self.best_reward = float("-inf")

        self.global_state = global_state
        self.to(self.device)

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"Enhanced HQ_Network initialised successfully with attention={use_attention}, "
                f"dropout={use_dropout}, hidden_size={hidden_size}",
                level=logging.INFO,
            )

    def update_network(self, new_input_size):
        """
        Update the network structure dynamically when the input size changes.
        """
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[INFO] Updating HQ_Network input size from {self.feature_extractor[0].in_features} to {new_input_size}"
            )

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
            old_feature_extractor[7],  # Second Dropout/Identity
        ).to(self.device)

        # Update stored sizes - prevent negative state size
        calculated_state_size = new_input_size - (
            self.role_size + self.local_state_size + self.global_state_size
        )

        # Minimum state size is 5 (base state vector has 5 elements)
        MIN_STATE_SIZE = 5

        # Prevent invalid state size - if calculation goes negative or below minimum, keep original
        if calculated_state_size >= MIN_STATE_SIZE:
            self.state_size = calculated_state_size
        else:
            logger.log_msg(
                f"[WARNING] Calculated state_size={calculated_state_size} is invalid (min={MIN_STATE_SIZE}), keeping original size {self.state_size}",
                level=logging.WARNING,
            )

    def forward(self, state, role, local_state, global_state):
        """
        Enhanced forward pass through the HQ network with attention.
        """
        device = self.device

        # Ensure all inputs are on the correct device and properly shaped
        # Keep batch dimension if present (for training), otherwise flatten
        # Note: Tensors are created with device=device, so .to() is only needed if not already on device
        if state.device != device:
            state = state.to(device)
        if role.device != device:
            role = role.to(device)
        if local_state.device != device:
            local_state = local_state.to(device)
        if global_state.device != device:
            global_state = global_state.to(device)

        if state.dim() > 1:
            # Batch mode: state is (batch_size, state_size), etc.
            # Reshape to (batch_size, -1) to preserve batch dimension
            state = state.reshape(state.shape[0], -1)
            role = role.reshape(role.shape[0], -1)
            local_state = local_state.reshape(local_state.shape[0], -1)
            global_state = global_state.reshape(global_state.shape[0], -1)
        else:
            # Single sample: flatten to 1D
            state = state.view(-1)
            role = role.view(-1)
            local_state = local_state.view(-1)
            global_state = global_state.view(-1)

        # Calculate total input size (handle both batch and single sample modes)
        if state.dim() > 1:
            # Batch mode: use last dimension
            input_size_check = (
                state.shape[-1]
                + role.shape[-1]
                + local_state.shape[-1]
                + global_state.shape[-1]
            )
        else:
            # Single sample mode: use first dimension
            input_size_check = (
                state.shape[0]
                + role.shape[0]
                + local_state.shape[0]
                + global_state.shape[0]
            )

        if input_size_check != self.feature_extractor[0].in_features:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[INFO] Updating HQ_Network input size from {self.feature_extractor[0].in_features} to {input_size_check}"
                )
            self.update_network(input_size_check)

        # Concatenate all inputs
        x = torch.cat([state, role, local_state, global_state], dim=-1)

        # Feature extraction
        features = self.feature_extractor(x)

        # Apply attention if enabled
        if self.attention is not None:
            # Reshape for attention (batch_size, seq_len, features)
            # Ensure features has the right dimensions
            if features.dim() == 1:
                features = features.unsqueeze(0).unsqueeze(
                    0
                )  # Add batch and seq dimensions
            elif features.dim() == 2:
                features = features.unsqueeze(1)  # Add seq dimension
            # At this point features should be (batch_size, seq_len, features) = (?, ?, hidden_size)

            # Only apply attention if we have valid dimensions
            if features.dim() >= 2:
                try:
                    features = self.attention(features)
                except RuntimeError as e:
                    if "Dimension out of range" in str(e):
                        # Skip attention if dimensions are invalid
                        if utils_config.ENABLE_LOGGING:
                            logger.log_msg(
                                f"[WARNING] Skipping attention due to dimension error: {e}",
                                level=logging.WARNING,
                            )
                    else:
                        raise

            # Restore original dimensions
            if features.dim() == 3:
                features = features.squeeze(1)  # Remove seq dimension if present
            # Note: Don't squeeze batch dimension - we need it for batch training

        # Policy and value outputs
        policy_logits = self.policy_head(features)
        value = self.value_head(features)

        # ============================================================================
        # PARAMETRIC STRATEGY OUTPUTS
        # ============================================================================

        # Binary parameters (sigmoid output)
        binary_params = self.binary_param_head(features)

        # Continuous parameters (tanh output, scaled to [0,1])
        continuous_params_raw = self.continuous_param_head(features)
        continuous_params = (
            continuous_params_raw + 1.0
        ) / 2.0  # Scale from [-1,1] to [0,1]

        # Discrete parameters (softmax output)
        discrete_params = self.discrete_param_head(features)

        return policy_logits, value, binary_params, continuous_params, discrete_params

    def add_memory(
        self,
        state: list,
        role: list,
        local_state: list,
        global_state: list,
        action: int,
        reward: float = 0.0,
    ):
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
        if (
            not isinstance(action, int)
            or action < 0
            or action >= len(self.strategy_labels)
        ):
            logger.log_msg(
                f"[ERROR] Invalid action index: {action}", level=logging.ERROR
            )
            return

        if not isinstance(reward, (int, float)):
            logger.log_msg(
                f"[ERROR] Invalid reward type: {type(reward)}", level=logging.ERROR
            )
            return

        memory_entry = {
            "state": state,
            "role": role,
            "local_state": local_state,
            "global_state": global_state,
            "action": action,
            "reward": reward,
            "timestamp": len(self.hq_memory),  # Track order
        }

        self.hq_memory.append(memory_entry)

        # Update best reward if applicable
        self.update_best_reward(reward)

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[MEMORY] Added HQ experience: action={action} ({self.strategy_labels[action]}), "
                f"reward={reward:.2f}, memory_size={len(self.hq_memory)}",
                level=logging.DEBUG,
            )

    def add_memory_with_parameters(
        self,
        state: list,
        role: list,
        local_state: list,
        global_state: list,
        action: int,
        parameters: dict,
        reward: float = 0.0,
    ):
        """
        Store a full HQ memory entry with parameters for parametric strategy learning.
        """
        # Validate inputs
        if (
            not isinstance(action, int)
            or action < 0
            or action >= len(self.strategy_labels)
        ):
            logger.log_msg(
                f"[ERROR] Invalid action index: {action}", level=logging.ERROR
            )
            return

        if not isinstance(reward, (int, float)):
            logger.log_msg(
                f"[ERROR] Invalid reward type: {type(reward)}", level=logging.ERROR
            )
            return

        memory_entry = {
            "state": state,
            "role": role,
            "local_state": local_state,
            "global_state": global_state,
            "action": action,
            "parameters": parameters,  # Store learned parameters
            "reward": reward,
            "timestamp": len(self.hq_memory),  # Track order
        }

        self.hq_memory.append(memory_entry)

        # Update best reward if applicable
        self.update_best_reward(reward)

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[MEMORY] Added HQ parametric experience: action={action} ({self.strategy_labels[action]}), "
                f"parameters={parameters}, reward={reward:.2f}, memory_size={len(self.hq_memory)}",
                level=logging.DEBUG,
            )

    def update_memory_rewards(self, total_reward: float):
        """Update the reward for each memory entry."""
        if not isinstance(total_reward, (int, float)):
            logger.log_msg(
                f"[ERROR] Invalid reward type: {type(total_reward)}",
                level=logging.ERROR,
            )
            return

        for m in self.hq_memory:
            m["reward"] = total_reward

        # Update best reward
        self.update_best_reward(total_reward)

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[MEMORY] Updated all HQ memory rewards to {total_reward:.2f}",
                level=logging.DEBUG,
            )

    def clear_memory(self):
        """Clear the HQ memory and reset performance tracking."""
        self.hq_memory = []
        self.best_reward = float("-inf")

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
            "recommendation": (
                "clear_memory()" if status == "inconsistent" else "memory_ok"
            ),
        }

    def predict_strategy(self, global_state: dict) -> str:
        """
        Enhanced strategy prediction with better error handling and performance tracking.
        """
        if hasattr(self, "predicting"):
            logger.log_msg(
                "[WARNING] Recursive prediction call detected, returning default strategy",
                level=logging.WARNING,
            )
            return self.strategy_labels[0]

        self.predicting = True

        try:
            # Validate global state
            if not isinstance(global_state, dict):
                logger.log_msg(
                    f"[ERROR] Invalid global_state type: {type(global_state)}",
                    level=logging.ERROR,
                )
                return self.strategy_labels[0]

            # Check if global_state has required keys
            required_keys = [
                "HQ_health",
                "gold_balance",
                "food_balance",
                "resource_count",
                "threat_count",
            ]
            missing_keys = [key for key in required_keys if key not in global_state]
            if missing_keys:
                logger.log_msg(
                    f"[WARNING] Missing keys in global_state: {missing_keys}",
                    level=logging.WARNING,
                )
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
                    level=logging.DEBUG,
                )
                logger.log_msg(
                    f"[DEBUG] Encoded state vector: {state}", level=logging.DEBUG
                )

            # 2. Convert to tensors and move to correct device
            device = self.device
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            role_tensor = torch.tensor(
                role, dtype=torch.float32, device=device
            ).unsqueeze(0)
            local_tensor = torch.tensor(
                local_state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            global_tensor = torch.tensor(
                global_state_vec, dtype=torch.float32, device=device
            ).unsqueeze(0)

            # 3. Forward pass with error handling
            with torch.no_grad():
                try:
                    logits, value, binary_params, continuous_params, discrete_params = (
                        self.forward(
                            state_tensor, role_tensor, local_tensor, global_tensor
                        )
                    )
                except Exception as e:
                    logger.log_msg(
                        f"[ERROR] Forward pass failed: {e}", level=logging.ERROR
                    )
                    return self.strategy_labels[0]

            # 4. Log raw logits and value
            logits_list = logits.squeeze(0).tolist()
            for i, value_logit in enumerate(logits_list):
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[LOGITS] {self.strategy_labels[i]}: {value_logit:.4f}",
                        level=logging.INFO,
                    )

            # 5. Select strategy - add exploration for untrained networks
            # Check if network is trained (has memory entries or total_updates > 0)
            is_trained = len(self.hq_memory) > 0 or self.total_updates > 0

            logger.log_msg(
                f"[STRATEGY_SELECT] Memory size: {len(self.hq_memory)}, Total updates: {self.total_updates}, is_trained: {is_trained}",
                level=logging.INFO,
            )

            # Add noise to logits if untrained to prevent all HQs picking same strategy
            if not is_trained:
                # Add significant random noise to encourage exploration during initial training
                # Use a higher noise scale to ensure different HQs pick different strategies
                noise = torch.randn_like(logits) * 2.0  # Increased from 0.5 to 2.0
                logits = logits + noise
                logger.log_msg(
                    f"[EXPLORATION] Adding significant noise. Logits before: {logits_list[:3]}... After: {logits.squeeze(0).tolist()[:3]}...",
                    level=logging.INFO,
                )
            else:
                logger.log_msg(
                    "[STRATEGY_SELECT] Network is trained, using deterministic selection",
                    level=logging.INFO,
                )

            # Select with deterministic argmax for trained, with noise for untrained
            action_index = torch.argmax(logits).item()
            selected_strategy = self.strategy_labels[action_index]

            # ============================================================================
            # INTERPRET PARAMETERS
            # ============================================================================

            # Extract parameter values
            binary_values = binary_params.squeeze(0).tolist()
            continuous_values = continuous_params.squeeze(0).tolist()
            discrete_values = discrete_params.squeeze(0).tolist()

        except Exception as e:
            logger.log_msg(
                f"[ERROR] Strategy prediction failed: {e}", level=logging.ERROR
            )
            # Return default strategy on error
            return self.strategy_labels[0]
        finally:
            delattr(self, "predicting")

    def predict_strategy_parametric(self, global_state: dict) -> tuple:
        """
        Enhanced strategy prediction that returns both strategy and parameters.
        Returns: (strategy_name, parameters_dict)
        """
        if hasattr(self, "predicting"):
            logger.log_msg(
                "[WARNING] Recursive prediction call detected, returning default strategy",
                level=logging.WARNING,
            )
            return self.strategy_labels[0], {}

        self.predicting = True

        try:
            # Validate global state
            if not isinstance(global_state, dict):
                logger.log_msg(
                    f"[ERROR] Invalid global_state type: {type(global_state)}",
                    level=logging.ERROR,
                )
                return self.strategy_labels[0], {}

            # Check if global_state has required keys
            required_keys = [
                "HQ_health",
                "gold_balance",
                "food_balance",
                "resource_count",
                "threat_count",
            ]
            missing_keys = [key for key in required_keys if key not in global_state]
            if missing_keys:
                logger.log_msg(
                    f"[WARNING] Missing keys in global_state: {missing_keys}",
                    level=logging.WARNING,
                )
                # Fill missing keys with defaults
                for key in missing_keys:
                    global_state[key] = 0.0

            # Store the enhanced global_state temporarily and use it for encoding
            old_global_state = self.global_state
            self.global_state = global_state

            try:
                # 1. Extract structured input parts
                state, role, local_state, global_state_vec = self.encode_state_parts()
            finally:
                # Restore original global_state
                self.global_state = old_global_state

            # 2. Convert to tensors and move to correct device
            device = self.device
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            role_tensor = torch.tensor(
                role, dtype=torch.float32, device=device
            ).unsqueeze(0)
            local_tensor = torch.tensor(
                local_state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            global_tensor = torch.tensor(
                global_state_vec, dtype=torch.float32, device=device
            ).unsqueeze(0)

            # 3. Forward pass with error handling
            with torch.no_grad():
                try:
                    logits, value, binary_params, continuous_params, discrete_params = (
                        self.forward(
                            state_tensor, role_tensor, local_tensor, global_tensor
                        )
                    )
                except Exception as e:
                    logger.log_msg(
                        f"[ERROR] Forward pass failed: {e}", level=logging.ERROR
                    )
                    return self.strategy_labels[0], {}

            # 4. Select strategy
            action_index = torch.argmax(logits).item()
            selected_strategy = self.strategy_labels[action_index]

            # 5. Interpret parameters
            binary_values = binary_params.squeeze(0).tolist()
            continuous_values = continuous_params.squeeze(0).tolist()
            discrete_values = discrete_params.squeeze(0).tolist()

            agent_count_idx = torch.argmax(
                discrete_params[:, :10]
            ).item()  # First 10 for agent count
            mission_complexity_idx = torch.argmax(
                discrete_params[:, 10:]
            ).item()  # Last 4 for complexity

            parameters = {
                "target_role": "peacekeeper" if binary_values[0] > 0.5 else "gatherer",
                "priority_resource": "food" if binary_values[1] > 0.5 else "gold",
                "use_mission_system": binary_values[2] > 0.5,
                "aggression_level": continuous_values[0],
                "resource_threshold": continuous_values[1],
                "urgency": continuous_values[2],
                "mission_autonomy": continuous_values[3],
                "coordination_preference": continuous_values[4],
                "agent_adaptability": continuous_values[5],
                "failure_tolerance": continuous_values[6],
                "agent_count_target": agent_count_idx + 1,  # Convert 0-9 to 1-10
                "mission_complexity": mission_complexity_idx + 1,  # Convert 0-3 to 1-4
            }

            # 6. Store memory for training
            self.add_memory_with_parameters(
                state, role, local_state, global_state_vec, action_index, parameters
            )

            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[HQ PARAMETRIC] Selected: {selected_strategy} with parameters: {parameters}",
                    level=logging.INFO,
                )

            return selected_strategy, parameters

        except Exception as e:
            logger.log_msg(
                f"[ERROR] Parametric strategy prediction failed: {e}",
                level=logging.ERROR,
            )
            return self.strategy_labels[0], {}
        finally:
            delattr(self, "predicting")

    def encode_state_parts(self):
        """
        Enhanced state encoding with better error handling and normalization.
        Convert the global_state dictionary into structured input vectors
        for the HQ neural network: [state, role, local_state, global_state].
        """
        try:
            g = self.global_state
            if g is None:
                logger.log_msg(
                    "[WARNING] Global state is None, using default values",
                    level=logging.WARNING,
                )
                g = {}

            # Shared central state vector (used for main state input)
            # Normalize values to prevent extreme inputs
            # Always create 5 base elements
            state_vector = [
                min(
                    max(g.get("HQ_health", 100.0) / 100.0, 0.0), 1.0
                ),  # Clamp to [0, 1]
                min(
                    max(g.get("gold_balance", 0.0) / 1000.0, 0.0), 2.0
                ),  # Allow up to 2x normal
                min(
                    max(g.get("food_balance", 0.0) / 1000.0, 0.0), 2.0
                ),  # Allow up to 2x normal
                min(
                    max(g.get("resource_count", 0.0) / 100.0, 0.0), 2.0
                ),  # Allow up to 2x normal
                min(
                    max(g.get("threat_count", 0.0) / 10.0, 0.0), 2.0
                ),  # Allow up to 2x normal
            ]

            # If state_size is smaller than the minimum required size, fix it
            MIN_STATE_SIZE = len(state_vector)  # Ensure at least 5 elements
            if self.state_size < MIN_STATE_SIZE:
                logger.log_msg(
                    f"[WARNING] state_size={self.state_size} is too small, setting to minimum {MIN_STATE_SIZE}",
                    level=logging.WARNING,
                )
                self.state_size = MIN_STATE_SIZE

            # Role vector (placeholder: 1-hot HQ, or make dynamic later)
            # Ensure role vector has correct size
            role_vector = [1.0] + [0.0] * (self.role_size - 1)  # [HQ, 0, 0, ...]
            if len(role_vector) > self.role_size:
                role_vector = role_vector[: self.role_size]
            elif len(role_vector) < self.role_size:
                role_vector.extend([0.0] * (self.role_size - len(role_vector)))

            # Local state (near HQ) - use proximity instead of raw coordinates
            nearest_resource_dist = g.get("nearest_resource", {}).get(
                "distance", float("inf")
            )
            nearest_threat_dist = g.get("nearest_threat", {}).get(
                "distance", float("inf")
            )

            # Normalize distances: closer = higher value (inverse distance with sigmoid)
            # Network will learn what proximity values correlate with good outcomes
            resource_proximity = (
                1.0 / (1.0 + nearest_resource_dist / 200.0)
                if nearest_resource_dist < float("inf")
                else 0.0
            )
            threat_proximity = (
                1.0 / (1.0 + nearest_threat_dist / 150.0)
                if nearest_threat_dist < float("inf")
                else 0.0
            )

            # Provide counts as additional context
            resource_count_norm = min(g.get("resource_count", 0) / 20.0, 1.0)
            threat_count_norm = min(g.get("threat_count", 0) / 5.0, 1.0)

            local_vector = [
                resource_proximity,  # How close is nearest resource (0-1, higher=closer)
                threat_proximity,  # How close is nearest threat (0-1, higher=closer)
                resource_count_norm,  # How many resources known total
                threat_count_norm,  # How many threats known total
                min(
                    max(g.get("agent_density", 0.0) / 10.0, 0.0), 1.0
                ),  # Agent concentration near HQ
            ]

            # Global map-wide state with enhanced features
            # Use enhanced features from get_enhanced_global_state() if available
            friendly_count = g.get("friendly_agent_count", 0.0)
            global_vector = [
                min(
                    max(friendly_count / 10.0, 0.0), 2.0
                ),  # Feature 0: number of friendly agents
                min(max(g.get("enemy_agent_count", 0.0) / 10.0, 0.0), 2.0),  # Feature 1
                min(max(g.get("total_agents", 0.0) / 10.0, 0.0), 2.0),  # Feature 2
                min(
                    max(g.get("gatherer_count", 0.0) / 10.0, 0.0), 2.0
                ),  # Feature 3: gatherer_count (enhanced)
                min(
                    max(g.get("peacekeeper_count", 0.0) / 10.0, 0.0), 2.0
                ),  # Feature 4: peacekeeper_count (enhanced)
            ]

            # Add agent presence indicator (neutral observation, not a directive)
            # This gives the network information about agent availability without forcing recruitment
            global_vector.append(
                1.0 if friendly_count > 0 else 0.0
            )  # Feature 5: has_agents (binary)

            # If global_state_size allows more features, add swap benefit signals and affordability
            if self.global_state_size >= 7:
                # Note: we already added no_agents_emergency, so this shifts indices
                global_vector.extend(
                    [
                        min(
                            max(g.get("swap_to_gatherer_benefit", 0.0), 0.0), 1.0
                        ),  # Feature 6: swap to gatherer benefit
                        min(
                            max(g.get("swap_to_peacekeeper_benefit", 0.0), 0.0), 1.0
                        ),  # Feature 7: swap to peacekeeper benefit
                        g.get(
                            "can_afford_recruit", 0.0
                        ),  # Feature 8: can afford new recruitment (binary)
                        g.get(
                            "can_afford_swap", 0.0
                        ),  # Feature 9: can afford swap (binary)
                    ]
                )
            elif self.global_state_size < 5:
                global_vector = global_vector[: self.global_state_size]

            # Ensure global vector has correct size
            if len(global_vector) > self.global_state_size:
                global_vector = global_vector[: self.global_state_size]
            elif len(global_vector) < self.global_state_size:
                global_vector.extend(
                    [0.0] * (self.global_state_size - len(global_vector))
                )

            # Validate and fix state_size BEFORE truncation to prevent negative slicing
            if self.state_size <= 0:
                # Prevent negative state_size - use actual encoded size
                logger.log_msg(
                    f"[WARNING] Invalid state_size={self.state_size}, using actual state vector size {len(state_vector)}",
                    level=logging.WARNING,
                )
                self.state_size = len(state_vector)

            # Now safely validate and pad/truncate
            if len(state_vector) != self.state_size:
                logger.log_msg(
                    f"[WARNING] State vector size mismatch: expected {self.state_size}, got {len(state_vector)}",
                    level=logging.WARNING,
                )
                # Pad or truncate as needed
                if len(state_vector) < self.state_size:
                    state_vector.extend([0.0] * (self.state_size - len(state_vector)))
                else:
                    # Only truncate if state_size is positive AND larger than minimum
                    MIN_STATE_SIZE = 5  # Base state vector is always 5 elements
                    if self.state_size > 0 and self.state_size >= MIN_STATE_SIZE:
                        state_vector = state_vector[: self.state_size]
                    elif self.state_size < MIN_STATE_SIZE:
                        # Don't truncate below minimum - just keep what we have
                        logger.log_msg(
                            f"[WARNING] Cannot truncate state vector: size {len(state_vector)} required, but state_size={self.state_size} is below minimum. Keeping full vector.",
                            level=logging.WARNING,
                        )
                        self.state_size = len(state_vector)
                    else:
                        # If somehow still negative, just pad to current size
                        logger.log_msg(
                            f"[ERROR] State size still invalid after fix: {self.state_size}. Keeping original vector.",
                            level=logging.ERROR,
                        )

            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[ENCODE] State parts encoded successfully: state={len(state_vector)}, "
                    f"role={len(role_vector)}, local={len(local_vector)}, global={len(global_vector)}",
                    level=logging.DEBUG,
                )

            return state_vector, role_vector, local_vector, global_vector

        except Exception as e:
            logger.log_msg(f"[ERROR] State encoding failed: {e}", level=logging.ERROR)
            # Return safe default values with positive size
            safe_state_size = max(self.state_size, 5)  # Ensure at least 5 elements
            default_state = [0.5] * safe_state_size
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

        # Validate and normalize state vectors to ensure consistent sizes
        expected_state_size = self.state_size
        expected_role_size = self.role_size
        expected_local_size = self.local_state_size
        expected_global_size = self.global_state_size

        normalized_memory = []
        for m in memory:
            normalized = {}

            # Normalize state vector to expected size
            state = m["state"]
            if len(state) != expected_state_size:
                if len(state) < expected_state_size:
                    # Pad with zeros
                    state = state + [0.0] * (expected_state_size - len(state))
                else:
                    # Truncate to expected size
                    state = state[:expected_state_size]
            normalized["state"] = state

            # Normalize role vector
            role = m["role"]
            if len(role) != expected_role_size:
                if len(role) < expected_role_size:
                    role = role + [0.0] * (expected_role_size - len(role))
                else:
                    role = role[:expected_role_size]
            normalized["role"] = role

            # Normalize local_state vector
            local = m["local_state"]
            if len(local) != expected_local_size:
                if len(local) < expected_local_size:
                    local = local + [0.0] * (expected_local_size - len(local))
                else:
                    local = local[:expected_local_size]
            normalized["local_state"] = local

            # Normalize global_state vector
            global_vec = m["global_state"]
            if len(global_vec) != expected_global_size:
                if len(global_vec) < expected_global_size:
                    global_vec = global_vec + [0.0] * (
                        expected_global_size - len(global_vec)
                    )
                else:
                    global_vec = global_vec[:expected_global_size]
            normalized["global_state"] = global_vec

            normalized["action"] = m["action"]
            normalized["reward"] = m["reward"]
            normalized["timestamp"] = m.get("timestamp", 0)
            normalized_memory.append(normalized)

        states = torch.tensor(
            [m["state"] for m in normalized_memory], dtype=torch.float32, device=device
        )
        roles = torch.tensor(
            [m["role"] for m in normalized_memory], dtype=torch.float32, device=device
        )
        locals_ = torch.tensor(
            [m["local_state"] for m in normalized_memory],
            dtype=torch.float32,
            device=device,
        )
        globals_ = torch.tensor(
            [m["global_state"] for m in normalized_memory],
            dtype=torch.float32,
            device=device,
        )
        actions = torch.tensor(
            [m["action"] for m in normalized_memory], dtype=torch.long, device=device
        )
        rewards = [m["reward"] for m in normalized_memory]

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

        # Forward pass with full input - unpack all 5 return values
        logits, values, binary_params, continuous_params, discrete_params = self.forward(states, roles, locals_, globals_)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)

        # Enhanced loss computation with configuration-based parameters
        advantage = returns - values.squeeze(-1)

        # Policy loss with clipping for stability
        policy_loss = -(log_probs * advantage.detach()).mean()

        # Value loss with clipping and optional Huber loss
        if utils_config.USE_HUBER_LOSS:
            value_loss = F.huber_loss(values.squeeze(-1), returns, reduction="mean")
        else:
            value_loss = advantage.pow(2).mean()

        # Entropy bonus for exploration
        entropy = dist.entropy().mean()

        # Combined loss with configurable coefficients
        loss = (
            policy_loss
            + utils_config.VALUE_LOSS_COEFF * value_loss
            - entropy_coeff * entropy
        )

        # Backpropagation with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.parameters(), utils_config.GRADIENT_CLIP_NORM
        )
        optimizer.step()

        # Update learning rate
        if hasattr(self, "scheduler"):
            self.scheduler.step()

        # Track training metrics
        self.total_updates += 1
        self.training_history["losses"].append(loss.item())
        self.training_history["policy_losses"].append(policy_loss.item())
        self.training_history["value_losses"].append(value_loss.item())
        self.training_history["entropies"].append(entropy.item())
        if hasattr(self, "scheduler"):
            self.training_history["learning_rates"].append(
                self.scheduler.get_last_lr()[0]
            )

        # Logging
        logger.log_msg(
            f"[HQ TRAIN] Loss: {loss.item():.4f}, Policy: {policy_loss.item():.4f}, "
            f"Value: {value_loss.item():.4f}, Entropy: {entropy.item():.4f}, "
            f"LR: {self.scheduler.get_last_lr()[0]:.6f}",
            level=logging.INFO,
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
                "current_lr": (
                    self.scheduler.get_last_lr()[0]
                    if hasattr(self, "scheduler")
                    else 0.0
                ),
                "memory_size": len(self.hq_memory),
                "best_reward": self.best_reward,
                "avg_loss": 0.0,
                "avg_policy_loss": 0.0,
                "avg_value_loss": 0.0,
                "avg_entropy": 0.0,
                "loss_trend": "stable",
            }

        recent_losses = self.training_history["losses"][-10:]  # Last 10 updates
        if len(recent_losses) >= 2:
            loss_trend = (
                "improving" if recent_losses[-1] < recent_losses[0] else "degrading"
            )
        else:
            loss_trend = "stable"

        return {
            "total_updates": self.total_updates,
            "current_lr": (
                self.scheduler.get_last_lr()[0] if hasattr(self, "scheduler") else 0.0
            ),
            "memory_size": len(self.hq_memory),
            "best_reward": self.best_reward,
            "avg_loss": sum(self.training_history["losses"])
            / len(self.training_history["losses"]),
            "avg_policy_loss": sum(self.training_history["policy_losses"])
            / len(self.training_history["policy_losses"]),
            "avg_value_loss": sum(self.training_history["value_losses"])
            / len(self.training_history["value_losses"]),
            "avg_entropy": sum(self.training_history["entropies"])
            / len(self.training_history["entropies"]),
            "loss_trend": loss_trend,
            "recent_losses": recent_losses,
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
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=500, gamma=0.95
        )

        # Reset counters
        self.total_updates = 0
        self.best_reward = float("-inf")

        # Clear training history
        for key in self.training_history.keys():
            self.training_history[key] = []

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[MODEL RESET] HQ Network reset successfully. Memory preserved: {keep_memory}",
                level=logging.INFO,
            )

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
            "architecture": (
                "Enhanced HQ Network with Attention"
                if self.attention
                else "Enhanced HQ Network"
            ),
        }

    def set_learning_rate(self, new_lr):
        """
        Dynamically adjust learning rate during training.
        """
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[LR UPDATE] HQ Network learning rate updated to {new_lr:.6f}",
                level=logging.INFO,
            )

    def get_learning_rate(self):
        """
        Get current learning rate.
        """
        return self.optimizer.param_groups[0]["lr"]

    def freeze_layers(self, layer_names=None):
        """
        Freeze specific layers to prevent them from being updated during training.
        Useful for transfer learning or fine-tuning.
        """
        if layer_names is None:
            # Freeze feature extractor by default
            layer_names = ["feature_extractor"]

        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[FREEZE] Frozen HQ layer: {name}", level=logging.DEBUG
                    )

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[FREEZE] Frozen HQ layers: {layer_names}", level=logging.INFO
            )

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
                        logger.log_msg(
                            f"[UNFREEZE] Unfrozen HQ layer: {name}", level=logging.DEBUG
                        )

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[UNFREEZE] Unfrozen HQ layers: {layer_names if layer_names else 'all'}",
                level=logging.INFO,
            )

    def update_best_reward(self, reward):
        """
        Update the best reward achieved and track performance.
        """
        if reward > self.best_reward:
            self.best_reward = reward
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[BEST REWARD] New best HQ reward: {reward:.2f}",
                    level=logging.INFO,
                )
            return True
        return False

    def get_strategy_confidence(self, strategy_index):
        """
        Get confidence level for a specific strategy.
        """
        if not self.hq_memory:
            return 0.0

        # Count how often this strategy was chosen
        strategy_count = sum(1 for m in self.hq_memory if m["action"] == strategy_index)
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
            strategy_memories = [m for m in self.hq_memory if m["action"] == i]
            if strategy_memories:
                avg_reward = sum(m["reward"] for m in strategy_memories) / len(
                    strategy_memories
                )
                usage_count = len(strategy_memories)
                strategy_performance[strategy_name] = {
                    "average_reward": avg_reward,
                    "usage_count": usage_count,
                    "confidence": usage_count / len(self.hq_memory),
                }
            else:
                strategy_performance[strategy_name] = {
                    "average_reward": 0.0,
                    "usage_count": 0,
                    "confidence": 0.0,
                }

        return strategy_performance
