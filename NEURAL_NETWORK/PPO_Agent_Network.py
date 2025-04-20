"""Common Imports"""
from SHARED.core_imports import *

"""File Specific Imports"""
from NEURAL_NETWORK.Common import  Training_device, save_checkpoint, load_checkpoint

import UTILITIES.utils_config as utils_config


logger = Logger(log_file="PPO_Agent_Network.txt", log_level=logging.DEBUG)

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
            clip_epsilon=0.2,
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

        

        # Initialise other components of the agent
        self.ai = Agent_Critic(state_size, action_size)
        self.optimizer = optim.Adam(self.ai.parameters(), lr=learning_rate)
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

        logits, value = self.ai.forward(state_tensor)

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
            # Sample from distribution during training
            action = dist.sample()
        else:
            # Use deterministic action during evaluation
            action = torch.argmax(probs, dim=-1)

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"Action probabilities: {probs.tolist()}, Selected action: {action.item()}, State value: {value.item()}.",
                level=logging.DEBUG)

        return action.item(), dist.log_prob(action), value.item()

    
    def get_valid_action_indices(self, resource_manager, agents):
        role_actions = utils_config.ROLE_ACTIONS_MAP[self.agent.role]
        valid_indices = set()

        # Always include movement and exploration
        for i, action in enumerate(role_actions):
            if action.startswith("move") or action == "explore":
                valid_indices.add(i)

        # Light context filtering
        for i, action in enumerate(role_actions):
            if action == "mine_gold":
                resources = self.agent.detect_resources(
                    resource_manager, threshold=5)
                if any(r.__class__.__name__ == "GoldLump" for r in resources):
                    valid_indices.add(i)

            elif action == "forage_apple":
                resources = self.agent.detect_resources(
                    resource_manager, threshold=5)
                if any(r.__class__.__name__ == "AppleTree" for r in resources):
                    valid_indices.add(i)

            elif action == "heal_with_apple":
                if self.agent.Health < 90 and self.agent.faction.food_balance > 0:
                    valid_indices.add(i)

            elif action in ["eliminate_threat", "patrol"]:
                threats = self.agent.detect_threats(
                    agents, enemy_hq={"faction_id": -1})
                if threats:
                    valid_indices.add(i)

        # Mild fallback boost: if only a few are valid, allow full list
        if len(valid_indices) < max(2, len(role_actions) // 2):
            # give more room to try stuff
            return list(range(len(role_actions)))
        else:
            return list(valid_indices)

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

    def train(self, mode='train'):
        """
        Train the PPO agent using stored experiences.
        In evaluation mode, we don’t perform training.

        :param mode: 'train' or 'evaluate'
        """
        required_keys = ["states", "actions",
                         "log_probs", "rewards", "values", "dones"]

        if mode == 'train':
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    "[AGENT NETWORK] Training Agent...", level=logging.DEBUG)

            # Check if memory is populated
            if not all(len(self.memory[k]) > 0 for k in required_keys):
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        "[ERROR] Training :skipped Memory buffer is incomplete or empty. " +
                        ", ".join(
                            f"{k}={len(self.memory[k])}" for k in required_keys),
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
            log_probs_old = torch.stack(self.memory["log_probs"]).detach()
            rewards = self.memory["rewards"]
            local_values  = torch.tensor(self.memory["values"],       dtype=torch.float32, device=self.device)
            global_values = torch.tensor(self.memory["global_values"],dtype=torch.float32, device=self.device)
            dones = torch.tensor(self.memory["dones"], dtype=torch.float32, device=self.device)
            # Compute returns and advantages using GAE
            try:
                returns, advantages = self.compute_gae(
                    rewards, 
                    local_values, 
                    global_values, 
                    dones
                )
            except ValueError as e:
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[ERROR] Failed to compute GAE: {e}",
                        level=logging.ERROR)
                return

            # Determine effective number of training epochs
            dataset_size = len(self.memory["rewards"])
            batch_size = 32
            indices = np.arange(dataset_size)
            ppo_epochs = getattr(self, 'ppo_epochs', 10)
            batches_per_epoch = max(1, int(dataset_size) // int(batch_size))
            effective_epochs = min(ppo_epochs, batches_per_epoch)

            for epoch in range(effective_epochs):
                np.random.shuffle(indices)
                for start in range(0, dataset_size, batch_size):
                    end = start + batch_size
                    batch_idx = indices[start:end]

                    states_batch = states[batch_idx]
                    actions_batch = actions[batch_idx]
                    log_probs_old_batch = log_probs_old[batch_idx]
                    advantages_batch = advantages[batch_idx]
                    returns_batch = returns[batch_idx]

                    #Move all batches to model's device
                    states_batch = states_batch.to(self.device)
                    actions_batch = actions_batch.to(self.device)
                    log_probs_old_batch = log_probs_old_batch.to(self.device)
                    advantages_batch = advantages_batch.to(self.device)
                    returns_batch = returns_batch.to(self.device)

                    logits, new_values = self.ai(states_batch)


                    if torch.isnan(logits).any().item(
                    ) or torch.isinf(logits).any().item():
                        if utils_config.ENABLE_LOGGING:
                            logger.log_msg(
                                f"[TRAIN LOGITS ERROR] NaNs in logits during training epoch {epoch}",
                                level=logging.ERROR)
                        return

                    probs = torch.softmax(logits, dim=-1)
                    if torch.isnan(probs).any().item(
                    ) or torch.isinf(probs).any().item():
                        if utils_config.ENABLE_LOGGING:
                            logger.log_msg(
                                f"[TRAIN PROBS ERROR] NaNs in probs during training epoch {epoch}",
                                level=logging.ERROR)
                        return

                    dist = Categorical(probs)
                    new_log_probs = dist.log_prob(actions_batch)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_log_probs - log_probs_old_batch)
                    surr1 = ratio * advantages_batch
                    surr2 = torch.clamp(
                        ratio,
                        1 - self.clip_epsilon,
                        1 + self.clip_epsilon) * advantages_batch
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = (returns_batch -
                                  new_values.squeeze(-1)).pow(2).mean()
                    loss = policy_loss + 0.5 * value_loss - self.entropy_coeff * entropy

                    self.optimizer.zero_grad()
                    loss.backward()
                    if utils_config.ENABLE_TENSORBOARD:
                        TensorBoardLogger().log_scalar(
                            f"Agent_{self.AgentID}/PolicyLoss", policy_loss.item(), self.total_updates)
                        TensorBoardLogger().log_scalar(
                            f"Agent_{self.AgentID}/ValueLoss", value_loss.item(), self.total_updates)
                        TensorBoardLogger().log_scalar(
                            f"Agent_{self.AgentID}/TotalLoss", loss.item(), self.total_updates)
                        TensorBoardLogger().log_scalar(
                            f"Agent_{self.AgentID}/Entropy", entropy.item(), self.total_updates)
                    
                    
                    torch.nn.utils.clip_grad_norm_(self.ai.parameters(), max_norm=0.5)
                    self.total_updates += 1
                    self.optimizer.step()

                    if utils_config.ENABLE_LOGGING:
                        logger.log_msg(
                            f"[TRAIN] Epoch {epoch + 1}: Loss={loss.item():.4f}, PolicyLoss={policy_loss.item():.4f}, ValueLoss={value_loss.item():.4f}, Entropy={entropy.item():.4f}",
                            level=logging.DEBUG)

            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    "[TRAIN COMPLETE] PPO update applied.", level=logging.INFO)

        elif mode == 'evaluate':
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    "[EVALUATE MODE] No training applied.",
                    level=logging.DEBUG)

        

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

    

#       _    ____ _____ _   _ _____    ____ ____  ___ _____ ___ ____
#      / \  / ___| ____| \ | |_   _|  / ___|  _ \|_ _|_   _|_ _/ ___|
#     / _ \| |  _|  _| |  \| | | |   | |   | |_) || |  | |  | | |
#    / ___ \ |_| | |___| |\  | | |   | |___|  _ < | |  | |  | | |___
#   /_/   \_\____|_____|_| \_| |_|    \____|_| \_\___| |_| |___\____|
#


class Agent_Critic(nn.Module):
    def __init__(self, input_size=100, action_size=10, device=Training_device):
        super(Agent_Critic, self).__init__()
        # Initialise the device to use (CPU or GPU)
        self.device = device
        # Dynamically determine input size
        self.input_size = input_size

        self.shared_fc = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, action_size)
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        assert state.size(1) == self.input_size, \
            f"Input size mismatch: Expected {self.input_size}, got {state.size(1)}"
        shared = self.shared_fc(state)
        logits = self.actor(shared)
        value = self.critic(shared)
        return logits, value
