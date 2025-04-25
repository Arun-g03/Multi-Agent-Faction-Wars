"""Common Imports"""
from SHARED.core_imports import *

"""File Specific Imports"""
from NEURAL_NETWORK.Common import Training_device, save_checkpoint, load_checkpoint
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
            global_state=None):
        super().__init__()
        # Initialise the device to use (CPU or GPU) default to Training_device if none
        if device is None:
           device = Training_device
        self.device = device
        #  Compute total input size dynamically
        total_input_size = state_size + role_size + local_state_size + global_state_size
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[DEBUG] HQ_Network expected input size: {total_input_size}",
                level=logging.INFO)
        self.strategy_labels = utils_config.HQ_STRATEGY_OPTIONS

        if utils_config.ENABLE_LOGGING:
            print(
                f"[DEBUG] HQ_Network initialised with input size: {total_input_size}")

        #  Make input size dynamic
        
        self.fc1 = nn.Linear(total_input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, action_size)
        self.critic = nn.Linear(128, 1)
        self.role_size = role_size
        self.local_state_size = local_state_size
        self.global_state = global_state
        self.global_state_size = global_state_size

        self.hq_memory = []
        self.to(self.device)
        

    def update_network(self, new_input_size):
        """
        Update the network structure dynamically when the input size changes.
        """
        if utils_config.ENABLE_LOGGING:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[INFO] Updating HQ_Network input size from {self.fc1.in_features} to {new_input_size}")
        self.fc1 = nn.Linear(new_input_size, 128).to(self.device)

    def forward(self, state, role, local_state, global_state):
        """
        Forward pass through the HQ network.
        """
        device = self.device

        # Ensure all inputs are on the correct device
        state = state.to(device).view(-1)
        role = role.to(device).view(-1)
        local_state = local_state.to(device).view(-1)
        global_state = global_state.to(device).view(-1)

        input_size_check = state.shape[0] + role.shape[0] + \
            local_state.shape[0] + global_state.shape[0]

        if input_size_check != self.fc1.in_features:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[INFO] Updating HQ_Network input size from {self.fc1.in_features} to {input_size_check}")
            self.update_network(input_size_check)

        x = torch.cat([state, role, local_state, global_state], dim=-1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        task_assignment = self.actor(x)

        return task_assignment, self.critic(x)
        

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
        memory_entry = {
            "state": state,
            "role": role,
            "local_state": local_state,
            "global_state": global_state,
            "action": action,
            "reward": reward
        }

        self.hq_memory.append(memory_entry)

        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[MEMORY] Added HQ experience: action={action}, reward={reward:.2f}, len={len(self.hq_memory)}",
                level=logging.DEBUG
            )



    def update_memory_rewards(self, total_reward: float):
        """Update the reward for each memory entry."""
        for m in self.hq_memory:
            m["reward"] = total_reward

    def clear_memory(self):
        """Clear the HQ memory."""
        self.hq_memory = []

    def has_memory(self):
        """Check if the HQ memory is not empty."""
        return len(self.hq_memory) > 0

    def convert_state_to_tensor(self, aggregated_state):
        """
        Converts faction state data into tensor format for neural network processing.
        Uses batch operations to improve speed.
        """

        #  Convert all values in a single operation
        state_features = torch.tensor([
            aggregated_state.get("HQ_health", 100) / 100,
            aggregated_state.get("gold_balance", 0) / 100,
            aggregated_state.get("food_balance", 0) / 100,
            aggregated_state.get("resource_count", 0) / 10,
            aggregated_state.get("threat_count", 0) / 5,
            aggregated_state.get("friendly_agent_count", 0) / 10,
            aggregated_state.get("enemy_agent_count", 0) / 10,
            aggregated_state.get("agent_density", 0) / 10,
            aggregated_state.get("total_agents", 0) / 20,
        ], dtype=torch.float32)

        #  Process nearest threat and resource locations
        nearest_threat = aggregated_state.get(
            "nearest_threat", {"location": (-1, -1)})
        nearest_resource = aggregated_state.get(
            "nearest_resource", {"location": (-1, -1)})

        threat_features = torch.tensor([
            nearest_threat["location"][0] / 100,
            nearest_threat["location"][1] / 100,
        ], dtype=torch.float32)

        resource_features = torch.tensor([
            nearest_resource["location"][0] / 100,
            nearest_resource["location"][1] / 100,
        ], dtype=torch.float32)

        #  Process agent states efficiently
        MAX_AGENTS = 5
        expected_agent_size = 18 * MAX_AGENTS

        agent_states = aggregated_state.get("agent_states", [])
        limited_agents = agent_states[:MAX_AGENTS]
        flattened_agent_states = [
            feature for agent in limited_agents for feature in agent]

        #  Ensure fixed size
        if len(flattened_agent_states) < expected_agent_size:
            flattened_agent_states += [0] * \
                (expected_agent_size - len(flattened_agent_states))

        agent_states_tensor = torch.tensor(
            flattened_agent_states, dtype=torch.float32)

        #  Move everything to device in one operation (FASTER)
        full_feature_vector = torch.cat(
            [state_features, threat_features, resource_features, agent_states_tensor])

        return full_feature_vector.to(self.device).view(1, -1)

    def predict_strategy(self, global_state: dict) -> str:
        """
        Given the current global state, returns the best strategy label.
        """
        if hasattr(self, 'predicting'):
            return self.strategy_labels[0]
        self.predicting = True

        try:
            # 1. Extract structured input parts
            state, role, local_state, global_state_vec = self.encode_state_parts()

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

            # 3. Forward pass
            with torch.no_grad():
                logits, _ = self.forward(state_tensor, role_tensor, local_tensor, global_tensor)

            # 🔍 Log raw logits
            logits_list = logits.squeeze(0).tolist()
            for i, value in enumerate(logits_list):
                logger.log_msg(
                    f"[LOGITS] {self.strategy_labels[i]}: {value:.4f}",
                    level=logging.INFO)

            # 4. Select strategy with highest probability
            action_index = torch.argmax(logits).item()
            selected_strategy = self.strategy_labels[action_index]

            # 5. Store memory
            self.add_memory(state, role, local_state, global_state_vec, action_index)

            logger.log_msg(
                f"[HQ STRATEGY] Selected: {selected_strategy} (index: {action_index})",
                level=logging.INFO)

            return selected_strategy
        finally:
            delattr(self, 'predicting')



    def encode_state_parts(self):
        """
        Convert the global_state dictionary into structured input vectors
        for the HQ neural network: [state, role, local_state, global_state].
        """

        g = self.global_state

        # Shared central state vector (used for main state input)
        state_vector = [
            g["HQ_health"] / 100.0,
            g["gold_balance"] / 1000.0,
            g["food_balance"] / 1000.0,
            g["resource_count"] / 100.0,
            g["threat_count"] / 10.0
        ]

        # Role vector (placeholder: 1-hot HQ, or make dynamic later)
        role_vector = [1.0, 0.0]  # e.g. [HQ, Agent] — assuming 2 roles

        # Local state (near HQ)
        nearest_resource = g["nearest_resource"]["location"]
        nearest_threat = g["nearest_threat"]["location"]
        local_vector = [
            nearest_resource[0] / 100.0,  # Normalise to map size if known
            nearest_resource[1] / 100.0,
            nearest_threat[0] / 100.0,
            nearest_threat[1] / 100.0,
            g["agent_density"] / 10.0
        ]

        # Global map-wide state
        global_vector = [
            g["friendly_agent_count"] / 10.0,
            g["enemy_agent_count"] / 10.0,
            g["total_agents"] / 10.0,
        ]

        return state_vector, role_vector, local_vector, global_vector


    def train(self, memory, optimizer, gamma=0.99):
        """
        Trains the HQ strategy model using policy gradient.
        :param memory: A list of dicts with keys: 'state', 'role', 'local_state', 'global_state', 'action', 'reward'
        :param optimizer: Torch optimizer
        :param gamma: Discount factor for future rewards
        """
        if not memory:
            logger.log_msg("[HQ TRAIN] No memory to train on.", level=logging.WARNING)
            return

        # Prepare batches from structured memory
        device = self.device
        states = torch.tensor([m['state'] for m in memory], dtype=torch.float32, device=device)
        roles = torch.tensor([m['role'] for m in memory], dtype=torch.float32, device=device)
        locals_ = torch.tensor([m['local_state'] for m in memory], dtype=torch.float32, device=device)
        globals_ = torch.tensor([m['global_state'] for m in memory], dtype=torch.float32, device=device)
        actions = torch.tensor([m['action'] for m in memory], dtype=torch.long, device=device)
        rewards = [m['reward'] for m in memory]

        # Compute discounted returns
        returns = []
        G = 0 
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        # Normalise returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Forward pass with full input
        logits, values = self.forward(states, roles, locals_, globals_)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)

        # Compute loss
        advantage = returns - values.squeeze(-1)
        policy_loss = -(log_probs * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()
        loss = policy_loss + 0.5 * value_loss
        entropy = dist.entropy().mean()
        entropy_coeff = 0.01  # Added entropy coefficient
        loss = policy_loss + 0.5*value_loss - entropy_coeff*entropy
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        # Logging
        logger.log_msg(
            f"[HQ TRAIN] Loss: {loss.item():.4f}, Policy: {policy_loss.item():.4f}, Value: {value_loss.item():.4f}",
            level=logging.INFO
        )
        
    def save_model(self, path):
        save_checkpoint(self, path)

    