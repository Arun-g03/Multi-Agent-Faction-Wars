"""Common Imports"""
from SHARED.core_imports import *

"""File Specific Imports"""
from NEURAL_NETWORK.Common import  Training_device, save_checkpoint, load_checkpoint
from NEURAL_NETWORK.AttentionLayer import AttentionLayer
import torch.nn.functional as F

import UTILITIES.utils_config as utils_config

if utils_config.ENABLE_LOGGING:
    logger = Logger(log_file="PPO_Agent_Network.txt", log_level=logging.DEBUG)




#    ____  ____   ___    __  __           _      _
#   |  _ \|  _ \ / _ \  |  \/  | ___   __| | ___| |
#   | |_) | |_) | | | | | |\/| |/ _ \ / _` |/ _ \ |
#   |  __/|  __/| |_| | | |  | | (_) | (_| |  __/ |
#   |_|   |_|    \___/  |_|  |_|\___/ \__,_|\___|_|
#


class PPOModel(nn.Module):
    def __init__(self, state_size, action_size, training_mode="train", device=None, 
                 learning_rate=None, use_attention=True, use_dropout=True, hidden_size=256, AgentID=None):
        super().__init__()
        
        # Store AgentID for identification and logging
        self.AgentID = AgentID
        
        # Use configuration-based learning rate if none provided
        if learning_rate is None:
            learning_rate = utils_config.INITIAL_LEARNING_RATE_PPO
        
        # Initialise the device to use (CPU or GPU) default to Training_device if none
        if device is None:
            device = Training_device
        self.device = device
        
        # Store configuration
        self.state_size = state_size
        self.action_size = action_size
        self.training_mode = training_mode
        self.use_attention = use_attention
        self.use_dropout = use_dropout
        self.hidden_size = hidden_size
        
        # Enhanced neural network architecture
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1) if use_dropout else nn.Identity(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1) if use_dropout else nn.Identity()
        )
        
        # Attention mechanism for complex state understanding
        if use_attention and state_size > 20:  # Use attention for complex states
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
            lr=learning_rate, 
            weight_decay=1e-5
        )
        
        # Advanced learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=utils_config.LEARNING_RATE_STEP_SIZE, 
            gamma=utils_config.LEARNING_RATE_DECAY
        )
        
        # Enhanced memory system with experience replay
        self.memory = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "values": [],
            "global_values": [],
            "dones": []
        }
        
        # Experience replay buffer for better training
        if utils_config.ENABLE_EXPERIENCE_REPLAY:
            self.replay_buffer = []
            self.replay_buffer_size = utils_config.REPLAY_BUFFER_SIZE
        
        # Training history tracking
        self.training_history = {
            "losses": [],
            "policy_losses": [],
            "value_losses": [],
            "entropies": [],
            "learning_rates": [],
            "advantages": [],
            "returns": []
        }
        
        # Performance tracking
        self.total_updates = 0
        self.best_reward = float('-inf')
        self.episode_rewards = []
        
        # Curriculum learning support
        self.curriculum_difficulty = 0.0
        self.curriculum_step = 0
        
        # Move to device
        self.to(device)
        
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"Enhanced PPOAgent initialised successfully with attention={use_attention}, "
                f"dropout={use_dropout}, hidden_size={hidden_size}, lr={learning_rate}", 
                level=logging.INFO)

    def forward(self, state):
        # Enhanced forward pass with attention
        features = self.feature_extractor(state)
        
        if self.attention is not None:
            # Reshape for attention if needed (batch_size, seq_len, features)
            if features.dim() == 2:
                features = features.unsqueeze(1)  # Add sequence dimension
            features = self.attention(features)
            if features.dim() == 3:
                features = features.squeeze(1)  # Remove sequence dimension
        
        # Separate policy and value outputs
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        
        return policy_logits, value

    def choose_action(self, state, valid_indices=None):
        if state is None:
            raise ValueError(f"[CRITICAL] Agent {self.AgentID} received a None state.")
        if any(v is None for v in state):
            raise ValueError(f"[CRITICAL] Agent {self.AgentID} state contains None values: {state}")

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f" AgentID: {self.AgentID}__State: {state}",
                level=logging.DEBUG)

        # 🛠️ KEY: Now call your OWN forward() directly
        logits, value = self.forward(state_tensor)
        
        # Clip logits to prevent numerical instability
        logits = torch.clamp(logits, -10.0, 10.0)

        if valid_indices is not None:
            mask = torch.full_like(logits, float('-inf'))
            mask[0, valid_indices] = logits[0, valid_indices]
            logits = mask
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[CTX FILTER] Valid indices: {valid_indices}, Masked logits: {logits.tolist()}",
                    level=logging.DEBUG)

        probs = torch.softmax(logits, dim=-1)
        if torch.isnan(probs).any().item() or torch.isinf(probs).any().item():
            logger.log_msg("[PROBS ERROR] Detected NaNs/Infs in probs!", level=logging.ERROR)
            raise ValueError("Action probabilities contain NaN or Inf.")

        dist = torch.distributions.Categorical(probs)
        if self.training_mode not in ['train', 'evaluate']:
            raise ValueError(
                f"Invalid mode: {self.training_mode}. Choose either 'train' or 'evaluate'.")

        if self.training_mode == "train":
            action = dist.sample()
        else:
            action = torch.argmax(probs, dim=-1)

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"Action probabilities: {probs.tolist()}, Selected action: {action.item()}, State value: {value.item()}.",
                level=logging.DEBUG)

        return action.item(), dist.log_prob(action), value.item()

        

    


    
    def store_transition(
            self,
            state,
            action,
            log_prob,
            reward,
            local_value,
            global_value,
            done):
        """
        Store a single transition in memory.
        """
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(f"AGENT - Storing transition: state={state}"
                           f"action={action}"
                           f"reward={reward}"
                           f"local_value={local_value}"
                           f"global_value={global_value}"
                           f"done={done}.",
                           level=logging.DEBUG
                           )
        self.memory["states"].append(state)
        self.memory["actions"].append(action)
        self.memory["log_probs"].append(log_prob)
        self.memory["rewards"].append(reward)
        self.memory["values"].append(local_value)
        self.memory["global_values"].append(global_value)
        self.memory["dones"].append(done)
        
        # Prevent memory buffer overflow
        MAX_MEMORY_SIZE = 20000  # Max steps per episode
        if len(self.memory["rewards"]) > MAX_MEMORY_SIZE:
            # Remove oldest transitions (FIFO)
            for key in self.memory.keys():
                self.memory[key] = self.memory[key][-MAX_MEMORY_SIZE:]
            
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[MEMORY OVERFLOW] {self.AgentID} memory exceeded {MAX_MEMORY_SIZE}, truncated to latest",
                    level=logging.WARNING
                )
        
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[MEMORY COUNT] {self.AgentID}"
                f"Transitions Stored: {len(self.memory['rewards'])}",
                level=logging.DEBUG
            )

    def train(self, mode='train', batching=False):
        """
        Enhanced training for the PPO agent using stored experiences.
        In evaluation mode, we don't perform training.

        :param mode: 'train' or 'evaluate'
        """
        try:
            required_keys = ["states", "actions", "log_probs", "rewards", "values", "dones"]

            if mode == 'train':
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg("[AGENT NETWORK] Training Agent...", level=logging.DEBUG)

                # Check if memory is populated
                if not all(len(self.memory[k]) > 0 for k in required_keys):
                    if utils_config.ENABLE_LOGGING:
                        logger.log_msg(
                            "[ERROR] Training :skipped Memory buffer is incomplete or empty. " +
                            f"Memory sizes: {[len(self.memory[k]) for k in required_keys]}",
                            level=logging.ERROR)
                    return

                # Ensure sufficient transitions for training
                min_transitions = utils_config.MIN_MEMORY_SIZE
                if len(self.memory["rewards"]) < min_transitions:
                    if utils_config.ENABLE_LOGGING:
                        logger.log_msg(
                            f"[WARNING] Training skipped: Not enough transitions in memory. "
                            f"Required: {min_transitions}, Available: {len(self.memory['rewards'])}",
                            level=logging.WARNING)
                    return

                # Update curriculum difficulty
                self.update_curriculum_difficulty()

                # === Enhanced Batched training ===
                if batching and len(self.memory["rewards"]) >= utils_config.BATCH_SIZE:
                    self.train_batched()
                else:
                    self.train_individual()

            elif mode == 'evaluate':
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        "[EVALUATE MODE] No training applied.",
                        level=logging.DEBUG)
            else:
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"Invalid mode: {mode}. Choose either 'train' or 'evaluate'.",
                        level=logging.ERROR)

        except Exception as e:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(f"[ERROR] Exception in PPOAgent.train: {e}", level=logging.ERROR)
            print(f"[ERROR] Training failed: {e}")
            traceback.print_exc()

    def train_batched(self):
        """Enhanced batched training with advanced loss computation."""
        device = self.device
        
        # Prepare batched data
        batch_size = utils_config.BATCH_SIZE
        indices = torch.randperm(len(self.memory["rewards"]))[:batch_size]
        
        # Convert states to tensors before stacking
        state_list = [self.memory["states"][i] for i in indices]
        # Check if states are already tensors or need conversion
        if isinstance(state_list[0], torch.Tensor):
            states = torch.stack(state_list).to(device)
        else:
            states = torch.tensor(state_list, dtype=torch.float32, device=device)
        
        actions = torch.tensor([self.memory["actions"][i] for i in indices], dtype=torch.long, device=device)
        old_log_probs = torch.tensor([self.memory["log_probs"][i] for i in indices], dtype=torch.float32, device=device)
        rewards = torch.tensor([self.memory["rewards"][i] for i in indices], dtype=torch.float32, device=device)
        values = torch.tensor([self.memory["values"][i] for i in indices], dtype=torch.float32, device=device)
        dones = torch.tensor([self.memory["dones"][i] for i in indices], dtype=torch.float32, device=device)

        # Compute GAE advantages
        advantages, returns = self.compute_gae(rewards, values, values, dones)
        if advantages is None:
            return

        # Normalize advantages for stability
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Forward pass
        logits, new_values = self.forward(states)
        
        # Clip logits and values to prevent numerical instability
        logits = torch.clamp(logits, -10.0, 10.0)
        new_values = torch.clamp(new_values, -100.0, 100.0)
        
        # Check for NaN values
        if torch.isnan(logits).any():
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(f"[TRAIN LOGITS ERROR] NaNs in logits during training", level=logging.ERROR)
            return

        probs = torch.softmax(logits, dim=-1)
        if torch.isnan(probs).any():
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(f"[TRAIN PROBS ERROR] NaNs in probs during training", level=logging.ERROR)
            return

        # Compute losses with advanced techniques
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions)
        
        # PPO ratio with clipping
        ratio = torch.exp(new_log_probs - old_log_probs)
        ratio = torch.clamp(ratio, 0.0, 10.0)  # Prevent extreme ratios
        
        # Policy loss with PPO clipping
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - utils_config.PPO_CLIP_RATIO, 1 + utils_config.PPO_CLIP_RATIO) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss with Huber loss if enabled
        if utils_config.USE_HUBER_LOSS:
            value_loss = F.huber_loss(new_values.squeeze(-1), returns, reduction='mean')
        else:
            value_loss = F.mse_loss(new_values.squeeze(-1), returns)
        
        # Entropy bonus for exploration
        entropy = dist.entropy().mean()
        
        # Combined loss with configurable coefficients
        loss = policy_loss + utils_config.VALUE_LOSS_COEFF * value_loss - utils_config.ENTROPY_COEFF * entropy
        
        # Backpropagation with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), utils_config.GRADIENT_CLIP_NORM)
        self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step()
        
        # Track training metrics
        self.total_updates += 1
        self.training_history["losses"].append(loss.item())
        self.training_history["policy_losses"].append(policy_loss.item())
        self.training_history["value_losses"].append(value_loss.item())
        self.training_history["entropies"].append(entropy.item())
        self.training_history["learning_rates"].append(self.scheduler.get_last_lr()[0])
        self.training_history["advantages"].append(advantages.mean().item())
        self.training_history["returns"].append(returns.mean().item())

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[TRAIN][BATCHED] Batch training completed. Loss={loss.item():.4f}, "
                f"Policy={policy_loss.item():.4f}, Value={value_loss.item():.4f}, "
                f"Entropy={entropy.item():.4f}, LR={self.scheduler.get_last_lr()[0]:.6f}",
                level=logging.INFO)

    def train_individual(self):
        """Enhanced individual training with advanced loss computation."""
        device = self.device
        
        # Prepare individual data
        states = torch.tensor(self.memory["states"], dtype=torch.float32, device=device)
        actions = torch.tensor(self.memory["actions"], dtype=torch.long, device=device)
        old_log_probs = torch.tensor(self.memory["log_probs"], dtype=torch.float32, device=device)
        rewards = torch.tensor(self.memory["rewards"], dtype=torch.float32, device=device)
        values = torch.tensor(self.memory["values"], dtype=torch.float32, device=device)
        dones = torch.tensor(self.memory["dones"], dtype=torch.float32, device=device)

        # Compute GAE advantages
        advantages, returns = self.compute_gae(rewards, values, values, dones)
        if advantages is None:
            return

        # Normalize advantages for stability
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Forward pass
        logits, new_values = self.forward(states)
        
        # Clip logits and values to prevent numerical instability
        logits = torch.clamp(logits, -10.0, 10.0)
        new_values = torch.clamp(new_values, -100.0, 100.0)
        
        # Check for NaN values
        if torch.isnan(logits).any():
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(f"[TRAIN LOGITS ERROR] NaNs in logits during training", level=logging.ERROR)
            return

        probs = torch.softmax(logits, dim=-1)
        if torch.isnan(probs).any():
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(f"[TRAIN PROBS ERROR] NaNs in probs during training", level=logging.ERROR)
            return

        # Compute losses with advanced techniques
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions)
        
        # PPO ratio with clipping
        ratio = torch.exp(new_log_probs - old_log_probs)
        ratio = torch.clamp(ratio, 0.0, 10.0)  # Prevent extreme ratios
        
        # Policy loss with PPO clipping
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - utils_config.PPO_CLIP_RATIO, 1 + utils_config.PPO_CLIP_RATIO) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss with Huber loss if enabled
        if utils_config.USE_HUBER_LOSS:
            value_loss = F.huber_loss(new_values.squeeze(-1), returns, reduction='mean')
        else:
            value_loss = F.mse_loss(new_values.squeeze(-1), returns)
        
        # Entropy bonus for exploration
        entropy = dist.entropy().mean()
        
        # Combined loss with configurable coefficients
        loss = policy_loss + utils_config.VALUE_LOSS_COEFF * value_loss - utils_config.ENTROPY_COEFF * entropy
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), utils_config.GRADIENT_CLIP_NORM)
        self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step()
        
        # Track training metrics
        self.total_updates += 1
        self.training_history["losses"].append(loss.item())
        self.training_history["policy_losses"].append(policy_loss.item())
        self.training_history["value_losses"].append(value_loss.item())
        self.training_history["entropies"].append(entropy.item())
        self.training_history["learning_rates"].append(self.scheduler.get_last_lr()[0])
        self.training_history["advantages"].append(advantages.mean().item())
        self.training_history["returns"].append(returns.mean().item())

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[TRAIN] Individual training completed. Loss={loss.item():.4f}, "
                f"Policy={policy_loss.item():.4f}, Value={value_loss.item():.4f}, "
                f"Entropy={entropy.item():.4f}, LR={self.scheduler.get_last_lr()[0]:.6f}",
                level=logging.INFO)

    def update_curriculum_difficulty(self):
        """Update curriculum difficulty based on training progress."""
        if not utils_config.ENABLE_CURRICULUM_LEARNING:
            return
            
        # Calculate difficulty based on episode count
        episode_count = len(self.episode_rewards)
        difficulty_steps = utils_config.CURRICULUM_DIFFICULTY_STEPS
        
        for i, step in enumerate(difficulty_steps):
            if episode_count >= step:
                self.curriculum_difficulty = (i + 1) / len(difficulty_steps)
                self.curriculum_step = i + 1
        
        # Clamp difficulty to [0, 1]
        self.curriculum_difficulty = min(max(self.curriculum_difficulty, 0.0), 1.0)
        
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[CURRICULUM] Difficulty updated to {self.curriculum_difficulty:.2f} "
                f"(step {self.curriculum_step}/{len(difficulty_steps)})",
                level=logging.DEBUG
            )

    def get_curriculum_multiplier(self):
        """Get curriculum multiplier for resource spawning and difficulty scaling."""
        if not utils_config.ENABLE_CURRICULUM_LEARNING:
            return 1.0
            
        # Interpolate between initial and final spawn rates
        initial_rate = utils_config.INITIAL_RESOURCE_SPAWN_RATE
        final_rate = utils_config.FINAL_RESOURCE_SPAWN_RATE
        
        return initial_rate + (final_rate - initial_rate) * self.curriculum_difficulty

    def compute_gae(self, rewards, local_values, global_values, dones):
        """
        Compute Generalised Advantage Estimation (GAE) with enhanced robustness.
        """
        if len(rewards) == 0 or len(local_values) == 0 or len(
                global_values) == 0 or len(dones) == 0:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    "[ERROR] GAE input arrays are empty.", level=logging.ERROR)
            print(f"[DEBUG] rewards: {type(rewards)}, len={len(rewards)}")
            print(f"[DEBUG] local_values: {local_values.shape}")
            print(f"[DEBUG] global_values: {global_values.shape}")
            print(f"[DEBUG] dones: {dones.shape}")

            return None, None

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[GAE] Computing GAE for {len(rewards)} transitions", level=logging.DEBUG)

        try:
            # Convert to tensors if needed
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            if not isinstance(local_values, torch.Tensor):
                local_values = torch.tensor(local_values, dtype=torch.float32, device=self.device)
            if not isinstance(global_values, torch.Tensor):
                global_values = torch.tensor(global_values, dtype=torch.float32, device=self.device)
            if not isinstance(dones, torch.Tensor):
                dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

            # Ensure all tensors are on the same device
            rewards = rewards.to(self.device)
            local_values = local_values.to(self.device)
            global_values = global_values.to(self.device)
            dones = dones.to(self.device)

            # Use configuration-based parameters
            gamma = 0.99  # Can be made configurable
            lambda_ = utils_config.GAE_LAMBDA

            # Compute GAE
            advantages = torch.zeros_like(rewards, device=self.device)
            returns = torch.zeros_like(rewards, device=self.device)
            
            gae = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = local_values[t + 1]
                
                delta = rewards[t] + gamma * next_value * (1 - dones[t]) - local_values[t]
                gae = delta + gamma * lambda_ * (1 - dones[t]) * gae
                advantages[t] = gae
                returns[t] = advantages[t] + local_values[t]

            # Normalize advantages for training stability
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Clip extreme values for stability
            advantages = torch.clamp(advantages, -10.0, 10.0)
            returns = torch.clamp(returns, -10.0, 10.0)

            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[GAE] GAE computed successfully. Advantages: mean={advantages.mean():.4f}, "
                    f"std={advantages.std():.4f}, Returns: mean={returns.mean():.4f}, std={returns.std():.4f}",
                    level=logging.DEBUG)

            return advantages, returns

        except Exception as e:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(f"[ERROR] GAE computation failed: {e}", level=logging.ERROR)
            print(f"[ERROR] GAE computation failed: {e}")
            traceback.print_exc()
            return None, None

    def clear_memory(self):
        for key in self.memory.keys():
            self.memory[key] = []
        
        # Also clear training history
        for key in self.training_history.keys():
            self.training_history[key] = []

    def save_model(self, path):
        save_checkpoint(self, path)

    def get_training_stats(self):
        """
        Get comprehensive training statistics for monitoring and debugging.
        """
        if not self.training_history["losses"]:
            return {
                "total_updates": self.total_updates,
                "current_lr": self.scheduler.get_last_lr()[0],
                "memory_size": len(self.memory["rewards"]),
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
            "current_lr": self.scheduler.get_last_lr()[0],
            "memory_size": len(self.memory["rewards"]),
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
        self.optimizer = optim.AdamW(self.parameters(), lr=self.optimizer.param_groups[0]["lr"], weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        
        # Reset counters
        self.total_updates = 0
        
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(f"[MODEL RESET] PPO model reset successfully. Memory preserved: {keep_memory}", level=logging.INFO)

    def get_memory_efficiency(self):
        """
        Analyze memory usage and efficiency.
        """
        memory_sizes = {key: len(value) for key, value in self.memory.items()}
        total_memory = sum(memory_sizes.values())
        
        if total_memory == 0:
            return {"efficiency": 0.0, "status": "empty", "details": memory_sizes}
        
        # Check if all memory arrays have the same length (efficient)
        lengths = set(memory_sizes.values())
        if len(lengths) == 1:
            efficiency = 1.0
            status = "optimal"
        else:
            # Calculate efficiency based on consistency
            min_len = min(lengths)
            max_len = max(lengths)
            efficiency = min_len / max_len if max_len > 0 else 0.0
            status = "inconsistent"
        
        return {
            "efficiency": efficiency,
            "status": status,
            "total_memory": total_memory,
            "details": memory_sizes,
            "recommendation": "clear_memory()" if status == "inconsistent" else "memory_ok"
        }

    def adaptive_entropy_coefficient(self, episode_reward, target_entropy=0.1):
        """
        Dynamically adjust entropy coefficient based on performance.
        Higher entropy encourages exploration when performance is poor.
        """
        if episode_reward < 0:
            # Poor performance - increase exploration
            new_entropy_coeff = min(self.entropy_coeff * 1.1, 0.1)
        elif episode_reward > 10:
            # Good performance - decrease exploration
            new_entropy_coeff = max(self.entropy_coeff * 0.95, 0.001)
        else:
            # Moderate performance - slight adjustment
            new_entropy_coeff = self.entropy_coeff
        
        if abs(new_entropy_coeff - self.entropy_coeff) > 0.001:
            self.entropy_coeff = new_entropy_coeff
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(f"[ADAPTIVE ENTROPY] Adjusted entropy coefficient to {self.entropy_coeff:.4f}", level=logging.DEBUG)
        
        return self.entropy_coeff

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
            "architecture": "Enhanced PPO with Attention" if self.attention else "Enhanced PPO"
        }

    def get_role_vector_size(self):
        """
        Get the role vector size for compatibility with other parts of the system.
        """
        # Default role vector size if not specified
        return 5  # This matches the default used in HQ_Network

    def set_learning_rate(self, new_lr):
        """
        Dynamically adjust learning rate during training.
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(f"[LR UPDATE] Learning rate updated to {new_lr:.6f}", level=logging.INFO)

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
                    logger.log_msg(f"[FREEZE] Frozen layer: {name}", level=logging.DEBUG)
        
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(f"[FREEZE] Frozen layers: {layer_names}", level=logging.INFO)

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
                        logger.log_msg(f"[UNFREEZE] Unfrozen layer: {name}", level=logging.DEBUG)
        
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(f"[UNFREEZE] Unfrozen layers: {layer_names if layer_names else 'all'}", level=logging.INFO)

    
