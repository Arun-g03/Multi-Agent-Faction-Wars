"""Common Imports"""
from SHARED.core_imports import *

"""File Specific Imports"""
from NEURAL_NETWORK.Common import  Training_device, save_checkpoint, load_checkpoint

import UTILITIES.utils_config as utils_config

if utils_config.ENABLE_LOGGING:
    logger = Logger(log_file="PPO_Agent_Network.txt", log_level=logging.DEBUG)

if utils_config.ENABLE_TENSORBOARD:
            tensorboard_logger = TensorBoardLogger()


#    ____  ____   ___    __  __           _      _
#   |  _ \|  _ \ / _ \  |  \/  | ___   __| | ___| |
#   | |_) | |_) | | | | | |\/| |/ _ \ / _` |/ _ \ |
#   |  __/|  __/| |_| | | |  | | (_) | (_| |  __/ |
#   |_|   |_|    \___/  |_|  |_|\___/ \__,_|\___|_|
#


class PPOModel(nn.Module):
    def __init__(
            self,
            AgentID,
            training_mode,
            state_size=utils_config.DEF_AGENT_STATE_SIZE,
            action_size=10,
            learning_rate=1e-4,
            gamma=0.70,
            clip_epsilon=0.1,
            entropy_coeff=0.01,
            lambda_=0.95,
            device=Training_device
            ):
        # Call the parent class constructor first
        super(PPOModel, self).__init__()
        # Initialise the device to use(CPU or GPU)
        self.device = device

        # Ensure `state_size` is dynamically assigned
        if state_size is None:
            self.input_size = utils_config.DEF_AGENT_STATE_SIZE
        self.AgentID = AgentID
        self.training_mode = training_mode

        

        # Define the neural network architecture

        self.shared_fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, action_size)
        self.critic = nn.Linear(128, 1)

        # Optimizer for the WHOLE PPOModel (not ai anymore)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.total_updates = 0  # Track training updates across episodes

        self.lambda_ = lambda_
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff

        self.memory = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "values": [],
            "global_values": [],
            "dones": []
        }
        

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"Agent_Network initialised successfully using PPO model (state_size={state_size}, "
                f"action_size={action_size}, gamma={gamma}, clip_epsilon={clip_epsilon}, "
                f"entropy_coeff={entropy_coeff}).", level=logging.INFO)
        
        self.to(self.device)

        

        




    def forward(self, state):
        shared = self.shared_fc(state)
        logits = self.actor(shared)
        value = self.critic(shared)
        return logits, value

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
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[MEMORY COUNT] {self.AgentID}"
                f"Transitions Stored: {len(self.memory['rewards'])}",
                level=logging.DEBUG
            )

    def train(self, mode='train', batching=False):
        """
        Train the PPO agent using stored experiences.
        In evaluation mode, we don’t perform training.

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
                            ", ".join(f"{k}={len(self.memory[k])}" for k in required_keys),
                            level=logging.ERROR)
                    return

                # Ensure sufficient transitions for training
                if len(self.memory["rewards"]) < 10:
                    if utils_config.ENABLE_LOGGING:
                        logger.log_msg(
                            f"[WARNING] Training skipped: Not enough transitions in memory. "
                            f"Current transitions: {len(self.memory['rewards'])}",
                            level=logging.WARNING)
                    return

                # Convert stored memory to tensors
                states = torch.tensor(self.memory["states"], dtype=torch.float32, device=self.device)
                actions = torch.tensor(self.memory["actions"], dtype=torch.long, device=self.device)
                log_probs_old = torch.stack(self.memory["log_probs"]).detach().to(self.device)
                rewards = self.memory["rewards"]
                local_values = torch.tensor(self.memory["values"], dtype=torch.float32, device=self.device)
                global_values = torch.tensor(self.memory["global_values"], dtype=torch.float32, device=self.device)
                dones = torch.tensor(self.memory["dones"], dtype=torch.float32, device=self.device)

                # Compute returns and advantages using GAE
                try:
                    returns, advantages = self.compute_gae(rewards, local_values, global_values, dones)
                except ValueError as e:
                    if utils_config.ENABLE_LOGGING:
                        logger.log_msg(f"[ERROR] Failed to compute GAE: {e}", level=logging.ERROR)
                    return

                if batching:
                    # === Batched training ===
                    batch_size = 64
                    dataset_size = len(rewards)
                    indices = np.arange(dataset_size)
                    np.random.shuffle(indices)

                    for start in range(0, dataset_size, batch_size):
                        end = start + batch_size
                        batch_idx = torch.tensor(indices[start:end], dtype=torch.long, device=self.device)


                        # Slice batch tensors
                        state = states[batch_idx]
                        action = actions[batch_idx]
                        log_prob_old = log_probs_old[batch_idx]
                        advantage = advantages[batch_idx]
                        return_ = returns[batch_idx]

                        # Forward pass
                        logits, new_values = self.forward(state)
                        probs = torch.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        new_log_probs = dist.log_prob(action)
                        entropy = dist.entropy().mean()

                        ratio = torch.exp(new_log_probs - log_prob_old)
                        surr1 = ratio * advantage
                        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
                        policy_loss = -torch.min(surr1, surr2).mean()
                        value_loss = (return_ - new_values.squeeze(-1)).pow(2).mean()
                        loss = policy_loss + 0.5 * value_loss - self.entropy_coeff * entropy

                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                        self.optimizer.step()
                        self.total_updates += 1

                        if utils_config.ENABLE_LOGGING:
                            logger.log_msg(
                                f"[TRAIN][BATCHED] Loss={loss.item():.4f}, PolicyLoss={policy_loss.item():.4f}, "
                                f"ValueLoss={value_loss.item():.4f}, Entropy={entropy.item():.4f}",
                                level=logging.DEBUG
                            )
                else:

                    # Now process each experience one by one (no batching)
                    for idx in range(len(self.memory["rewards"])):
                        state = states[idx:idx+1]  # single experience
                        action = actions[idx:idx+1]
                        log_prob_old = log_probs_old[idx:idx+1]
                        advantage = advantages[idx:idx+1]
                        return_ = returns[idx:idx+1]

                        # Forward pass through the model
                        logits, new_values = self.forward(state)

                        # Check for NaN or Inf values in the logits or probabilities
                        if torch.isnan(logits).any() or torch.isinf(logits).any():
                            if utils_config.ENABLE_LOGGING:
                                logger.log_msg(f"[TRAIN LOGITS ERROR] NaNs in logits during training", level=logging.ERROR)
                            return

                        probs = torch.softmax(logits, dim=-1)

                        if torch.isnan(probs).any() or torch.isinf(probs).any():
                            if utils_config.ENABLE_LOGGING:
                                logger.log_msg(f"[TRAIN PROBS ERROR] NaNs in probs during training", level=logging.ERROR)
                            return

                        dist = torch.distributions.Categorical(probs)
                        new_log_probs = dist.log_prob(action)
                        entropy = dist.entropy().mean()

                        # Calculate the ratio and surrogates
                        ratio = torch.exp(new_log_probs - log_prob_old)
                        surr1 = ratio * advantage
                        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
                        policy_loss = -torch.min(surr1, surr2).mean()

                        value_loss = (return_ - new_values.squeeze(-1)).pow(2).mean()
                        loss = policy_loss + 0.5 * value_loss - self.entropy_coeff * entropy

                        # Backpropagation
                        self.optimizer.zero_grad()
                        loss.backward()

                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)

                        self.total_updates += 1
                        self.optimizer.step()

                        if utils_config.ENABLE_LOGGING:
                            logger.log_msg(
                                f"[TRAIN] Loss={loss.item():.4f}, PolicyLoss={policy_loss.item():.4f}, "
                                f"ValueLoss={value_loss.item():.4f}, Entropy={entropy.item():.4f}",
                                level=logging.DEBUG)

                if utils_config.ENABLE_LOGGING:
                    logger.log_msg("[TRAIN COMPLETE] PPO update applied.", level=logging.INFO)
                
                if utils_config.ENABLE_TENSORBOARD:
                    try:
                        hparams = {
                            "learning_rate": self.optimizer.param_groups[0]["lr"],
                            "gamma": self.gamma,
                            "clip_epsilon": self.clip_epsilon,
                            "entropy_coeff": self.entropy_coeff,
                            "lambda_gae": self.lambda_
                        }

                        # 🛠 Calculate final reward directly from memory
                        if len(self.memory["rewards"]) > 0:
                            average_reward = sum(self.memory["rewards"]) / len(self.memory["rewards"])
                        else:
                            average_reward = 0.0  # Fallback if somehow no rewards

                        final_metrics = {
                            "final_reward": average_reward
                        }

                        tensorboard_logger.log_hparams(hparams, final_metrics)

                        if utils_config.ENABLE_LOGGING:
                            logger.log_msg(
                                f"[HPARAMS LOGGED] {hparams}, final_reward={average_reward:.2f}",
                                level=logging.INFO
                            )
                    except Exception as e:
                        print(f"[ERROR] Failed to log hparams: {e}")




            


            elif mode == 'evaluate':
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        "[EVALUATE MODE] No training applied.",
                        level=logging.DEBUG)
        
        except Exception as e:
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(f"[ERROR] Exception in PPOAgent.train: {e}", level=logging.ERROR)
                    print(f"[DEBUG] rewards: {type(rewards)}, len={len(rewards)}")
                    traceback.print_exc()
                return
                
        

        

    def compute_gae(self, rewards, local_values, global_values, dones):
        """
        Compute Generalised Advantage Estimation (GAE).
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
                "Computing Generalised Advantage Estimation (GAE).",
                level=logging.INFO)
        returns = []
        advantages = []
        gae = 0
        last_return = 0

        for step in reversed(range(len(rewards))):
            mask = 1 - dones[step]

            if step < len(rewards) - 1:
                next_global = global_values[step + 1]
            else:
                # or use global_values[step] if continuing value
                next_global = torch.tensor(0.0, device=self.device)

            delta = rewards[step] + self.gamma * \
                next_global * mask - local_values[step]

            gae = delta + self.gamma * self.lambda_ * mask * gae

            advantages.insert(0, gae)
            last_return = rewards[step] + self.gamma * last_return * mask
            returns.insert(0, last_return)

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"GAE computed. Returns: {returns}, Advantages: {advantages}.",
                level=logging.DEBUG)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        return returns, (advantages - advantages.mean()) / \
            (advantages.std() + 1e-8)

    def clear_memory(self):
        for key in self.memory.keys():
            self.memory[key] = []

    def save_model(self, path):
        save_checkpoint(self, path)

    def load_model(self, path):
        load_checkpoint(self, path)

    
