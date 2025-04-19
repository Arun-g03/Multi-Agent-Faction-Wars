"""Common Imports"""
from SHARED.core_imports import *

"""File Specific Imports"""
from NEURAL_NETWORK.Common import check_training_device
Training_device = check_training_device()
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
            action_size=10,
            role_size=5,
            local_state_size=5,
            global_state_size=5,
            device=Training_device):
        super().__init__()
        # Initialise the device to use (CPU or GPU)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
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
        total_input_size = state_size + role_size + local_state_size + global_state_size
        self.fc1 = nn.Linear(total_input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, action_size)
        self.critic = nn.Linear(128, 1)

        self.role_size = role_size
        self.local_state_size = local_state_size
        self.global_state_size = global_state_size

        self.hq_memory = []

    def update_network(self, new_input_size):
        """
        Update the network structure dynamically when the input size changes.
        """
        if utils_config.ENABLE_LOGGING:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[INFO] Updating HQ_Network input size from {self.fc1.in_features} to {new_input_size}")
        self.fc1 = nn.Linear(new_input_size, 128)

    def forward(self, state, role, local_state, global_state):
        """
        Forward pass through the HQ network.
        """
        state = state.view(-1)
        role = role.view(-1)
        local_state = local_state.view(-1)
        global_state = global_state.view(-1)

        #  Automatically update input layer if sizes change
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

    def add_memory(self, state: list, action: int, reward: float = 0.0):
        self.hq_memory.append({
            "state": state,
            "action": action,
            "reward": reward
        })

    def update_memory_rewards(self, total_reward: float):
        for m in self.hq_memory:
            m["reward"] = total_reward

    def clear_memory(self):
        self.hq_memory = []

    def has_memory(self):
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

        return full_feature_vector.view(1, -1)

    def predict_strategy(self, global_state: dict) -> str:
        """
        Given the current global state, returns the best strategy label.
        """
        # Prevent reentrant calls
        if hasattr(self, 'predicting'):
            # Return default strategy if already predicting
            return self.strategy_labels[0]
        self.predicting = True

        try:
            # 1. Encode the global state into a fixed-size input vector
            state_vector = self.encode_state(global_state)
            logger.log_msg(
                f"[INFO] Predicting strategy for global state: {global_state}",
                level=logging.DEBUG)
            logger.log_msg(
                f"[DEBUG] Encoded state vector: {state_vector}",
                level=logging.DEBUG)

            # 2. Convert to tensor and run forward pass
            with torch.no_grad():
                input_tensor = torch.tensor(
                    state_vector, dtype=torch.float32).unsqueeze(0)  # Add batch dim
                logits, _ = self.forward(input_tensor, torch.zeros(
                    0), torch.zeros(0), torch.zeros(0))

            # 🔍 Log raw logits
            logits_list = logits.squeeze(0).tolist()
            for i, value in enumerate(logits_list):
                logger.log_msg(
                    f"[LOGITS] {self.strategy_labels[i]}: {value:.4f}",
                    level=logging.INFO)

            # 3. Choose the strategy with highest score
            action_index = torch.argmax(logits).item()
            selected_strategy = self.strategy_labels[action_index]
            self.add_memory(state_vector, action_index)
            logger.log_msg(
                f"[HQ STRATEGY] Selected: {selected_strategy} (index: {action_index})",
                level=logging.INFO)

            return selected_strategy
        finally:
            delattr(self, 'predicting')

    def encode_state(self, global_state: dict) -> list:
        """
        Converts the global state dictionary into a flat list of features
        for input to the HQ strategy network.
        """
        return [
            global_state.get("gold_balance", 0),
            global_state.get("food_balance", 0),
            global_state.get("HQ_health", 100),
            global_state.get("friendly_agent_count", 0),
            global_state.get("enemy_agent_count", 0),
            global_state.get("resource_count", 0),
            global_state.get("threat_count", 0),
            global_state.get("agent_density", 0),
            # Add any other numeric signals you want the HQ to learn from
        ]

    def train(self, memory, optimizer, gamma=0.99):
        """
        Trains the HQ strategy model using policy gradient.
        :param memory: A list of dicts with keys: 'state', 'action', 'reward'
        :param optimizer: Torch optimizer
        :param gamma: Discount factor for future rewards
        """
        if not memory:
            logger.log_msg("[HQ TRAIN] No memory to train on.",
                           level=logging.WARNING)
            return

        # Prepare batches
        states = torch.tensor([m['state'] for m in memory],
                              dtype=torch.float32, device=self.device)
        actions = torch.tensor(
            [m['action'] for m in memory], dtype=torch.long, device=self.device)
        rewards = [m['reward'] for m in memory]

        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(
            returns, dtype=torch.float32, device=self.device)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Create properly-shaped dummy inputs for role/local/global state
        batch_size = states.size(0)
        role = torch.zeros(batch_size, self.role_size,
                           dtype=torch.float32, device=self.device)
        local_state = torch.zeros(
            batch_size,
            self.local_state_size,
            dtype=torch.float32,
            device=self.device)
        global_state = torch.zeros(
            batch_size,
            self.global_state_size,
            dtype=torch.float32,
            device=self.device)

        # Forward pass
        logits, values = self.forward(states, role, local_state, global_state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)

        # Compute loss
        advantage = returns - values.squeeze(-1)
        policy_loss = -(log_probs * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()
        loss = policy_loss + 0.5 * value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        logger.log_msg(
            f"[HQ TRAIN] Loss: {loss.item():.4f}, Policy: {policy_loss.item():.4f}, Value: {value_loss.item():.4f}",
            level=logging.INFO)


#    _   _  ___     ____ ____  ___ _____ ___ ____
#   | | | |/ _ \   / ___|  _ \|_ _|_   _|_ _/ ___|
#   | |_| | | | | | |   | |_) || |  | |  | | |
#   |  _  | |_| | | |___|  _ < | |  | |  | | |___
#   |_| |_|\__\_\  \____|_| \_\___| |_| |___\____|
#


class HQ_Critic(nn.Module):
    """
    HQ Critic for evaluating task assignments. Inherits from nn.Module.
    Evaluates the value of current task assignments and strategies.
    """

    def __init__(
            self,
            input_size=100,
            role_size=5,
            local_state_size=5,
            global_state_size=5,
            device=Training_device):
        super().__init__()
        # Initialise the device to use (CPU or GPU)
        self.device = device
        self.role_size = role_size
        self.local_state_size = local_state_size
        self.global_state_size = global_state_size

        # Shared layers for both Actor and Critic
        self.fc1 = nn.Linear(input_size + self.local_state_size +
                             self.role_size + self.global_state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.critic = nn.Linear(128, 1)

    def forward(self, state, role, local_state, global_state, mode='train'):
        """
        Forward pass through the HQCritic network.
        :param state: The state vector (e.g., environmental state, resources, etc.)
        :param role: The one-hot encoded role vector (e.g., gatherer, peacekeeper)
        :param local_state: The local state vector for the agent (e.g., health, position)
        :param global_state: The global state vector (e.g., faction status, resources)
        :param mode: 'train' or 'evaluate', determines whether the model is being trained or evaluated
        :return: Value estimate from the critic (how good the current task assignment is).
        """
        x = torch.cat([state, role, local_state, global_state], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        if mode == 'train':
            return self.critic(x)  # Value estimate in training mode
        else:
            return None  # No value estimate in evaluation mode

