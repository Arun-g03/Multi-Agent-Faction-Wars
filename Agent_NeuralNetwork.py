import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import logging
from utils_config import STATE_FEATURES_MAP, DEF_AGENT_STATE_SIZE, ENABLE_LOGGING, HQ_STRATEGY_OPTIONS, ENABLE_TENSORBOARD
from utils_logger import Logger
from utils_logger import TensorBoardLogger

logger = Logger(log_file="neural_network_log.txt", log_level=logging.DEBUG)

# Check for GPU
Training_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    if ENABLE_LOGGING: logger.log_msg(f"\033[93mUsing GPU: {torch.cuda.get_device_name(0)}\033[0m")
    else:
        print("\033[93m**Agent Networks**\nNo GPU detected.\nUsing CPU.\nNote training will be slower.\n\033[0m")



#I was gonna break the file up but instead, i like that all the neural network parts are together



#       _   _   _             _   _               _                          
#      / \ | |_| |_ ___ _ __ | |_(_) ___  _ __   | |    __ _ _   _  ___ _ __ 
#     / _ \| __| __/ _ \ '_ \| __| |/ _ \| '_ \  | |   / _` | | | |/ _ \ '__|
#    / ___ \ |_| ||  __/ | | | |_| | (_) | | | | | |__| (_| | |_| |  __/ |   
#   /_/   \_\__|\__\___|_| |_|\__|_|\___/|_| |_| |_____\__,_|\__, |\___|_|   
#                                                            |___/           

class AttentionLayer(nn.Module):
    """
    A simple Self-Attention Layer for focusing on different parts of the input.
    """
    def __init__(self, input_dim, output_dim):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Forward pass for the attention layer.
        :param x: The input tensor (batch_size, seq_len, input_dim)
        :return: The attention output (batch_size, seq_len, output_dim)
        """
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Apply attention weights to value
        attention_output = torch.matmul(attention_weights, value)
        return attention_output



#    _   _  ___    _   _ _____ _______        _____  ____  _  __
#   | | | |/ _ \  | \ | | ____|_   _\ \      / / _ \|  _ \| |/ /
#   | |_| | | | | |  \| |  _|   | |  \ \ /\ / / | | | |_) | ' / 
#   |  _  | |_| | | |\  | |___  | |   \ V  V /| |_| |  _ <| . \ 
#   |_| |_|\__\_\ |_| \_|_____| |_|    \_/\_/  \___/|_| \_\_|\_\
#                                                               



class HQ_Network(nn.Module):
    def __init__(self, state_size=29, action_size=10, role_size=5, local_state_size=5, global_state_size=5, device=Training_device):
        super().__init__()
        # Initialise the device to use (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #  Compute total input size dynamically
        total_input_size = state_size + role_size + local_state_size + global_state_size
        if ENABLE_LOGGING: logger.log_msg(f"[DEBUG] HQ_Network expected input size: {total_input_size}", level=logging.INFO)
        self.strategy_labels = HQ_STRATEGY_OPTIONS


        if ENABLE_LOGGING:
            print(f"[DEBUG] HQ_Network initialised with input size: {total_input_size}")

        #  Make input size dynamic
        total_input_size = state_size + role_size + local_state_size + global_state_size
        self.fc1 = nn.Linear(total_input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, action_size)
        self.critic = nn.Linear(128, 1)


        self.hq_memory = []

    def update_network(self, new_input_size):
        """
        Update the network structure dynamically when the input size changes.
        """
        if ENABLE_LOGGING:
            if ENABLE_LOGGING: logger.log_msg(f"[INFO] Updating HQ_Network input size from {self.fc1.in_features} to {new_input_size}")
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
        input_size_check = state.shape[0] + role.shape[0] + local_state.shape[0] + global_state.shape[0]

        if input_size_check != self.fc1.in_features:
            if ENABLE_LOGGING: logger.log_msg(f"[INFO] Updating HQ_Network input size from {self.fc1.in_features} to {input_size_check}")
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
        nearest_threat = aggregated_state.get("nearest_threat", {"location": (-1, -1)})
        nearest_resource = aggregated_state.get("nearest_resource", {"location": (-1, -1)})

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
        flattened_agent_states = [feature for agent in limited_agents for feature in agent]

        #  Ensure fixed size
        if len(flattened_agent_states) < expected_agent_size:
            flattened_agent_states += [0] * (expected_agent_size - len(flattened_agent_states))

        agent_states_tensor = torch.tensor(flattened_agent_states, dtype=torch.float32)

        #  Move everything to device in one operation (FASTER)
        full_feature_vector = torch.cat([state_features, threat_features, resource_features, agent_states_tensor])

        return full_feature_vector.view(1, -1)
    
    def predict_strategy(self, global_state: dict) -> str:
        """
        Given the current global state, returns the best strategy label.
        """
        # Prevent reentrant calls
        if hasattr(self, 'predicting'):
            return self.strategy_labels[0]  # Return default strategy if already predicting
        self.predicting = True

        try:
            # 1. Encode the global state into a fixed-size input vector
            state_vector = self.encode_state(global_state)
            logger.log_msg(f"[INFO] Predicting strategy for global state: {global_state}", level=logging.DEBUG)
            logger.log_msg(f"[DEBUG] Encoded state vector: {state_vector}", level=logging.DEBUG)

            # 2. Convert to tensor and run forward pass
            with torch.no_grad():
                input_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0)  # Add batch dim
                logits, _ = self.forward(input_tensor, torch.zeros(0), torch.zeros(0), torch.zeros(0))

            # 🔍 Log raw logits
            logits_list = logits.squeeze(0).tolist()
            for i, value in enumerate(logits_list):
                logger.log_msg(f"[LOGITS] {self.strategy_labels[i]}: {value:.4f}", level=logging.INFO)

            # 3. Choose the strategy with highest score
            action_index = torch.argmax(logits).item()
            selected_strategy = self.strategy_labels[action_index]
            self.add_memory(state_vector, action_index)
            logger.log_msg(f"[HQ STRATEGY] Selected: {selected_strategy} (index: {action_index})", level=logging.INFO)

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
            logger.log_msg("[HQ TRAIN] No memory to train on.", level=logging.WARNING)
            return

        # Prepare batches
        states = torch.tensor([m['state'] for m in memory], dtype=torch.float32).to(self.device)
        actions = torch.tensor([m['action'] for m in memory], dtype=torch.long).to(self.device)
        rewards = [m['reward'] for m in memory]

        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # Normalize returns (helps with stability)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Forward pass
        logits, values = self.forward(states, torch.zeros_like(states), torch.zeros_like(states), torch.zeros_like(states))
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
        logger.log_msg(f"[HQ TRAIN] Loss: {loss.item():.4f}, Policy: {policy_loss.item():.4f}, Value: {value_loss.item():.4f}", level=logging.INFO)







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
    def __init__(self, input_size=100, role_size=5, local_state_size=5, global_state_size=5, device=Training_device):
        super().__init__()
        # Initialise the device to use (CPU or GPU)
        self.device = device
        self.role_size = role_size
        self.local_state_size = local_state_size
        self.global_state_size = global_state_size

        # Shared layers for both Actor and Critic
        self.fc1 = nn.Linear(input_size + self.local_state_size + self.role_size + self.global_state_size, 128)
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





#    ____  ____   ___    __  __           _      _ 
#   |  _ \|  _ \ / _ \  |  \/  | ___   __| | ___| |
#   | |_) | |_) | | | | | |\/| |/ _ \ / _` |/ _ \ |
#   |  __/|  __/| |_| | | |  | | (_) | (_| |  __/ |
#   |_|   |_|    \___/  |_|  |_|\___/ \__,_|\___|_|
#                                                  






class PPOModel(nn.Module):
    def __init__(self, AgentID, state_size=DEF_AGENT_STATE_SIZE,  action_size=10, hq_critic=None, learning_rate=1e-4, gamma=0.70, clip_epsilon=0.2, entropy_coeff=0.01, device=Training_device):
        # Call the parent class constructor first
        super(PPOModel, self).__init__()
        # Initialise the device to use(CPU or GPU)
        self.device = device
        
        # Ensure `state_size` is dynamically assigned
        if state_size is None:
                self.input_size = DEF_AGENT_STATE_SIZE
        self.AgentID = AgentID

        # Initialise the critic if not provided
        if hq_critic is None:
            hq_critic = HQ_Critic(input_size=state_size)  # Default critic
        
        # Store critic
        self.hq_critic = hq_critic

        # Initialise other components of the agent
        self.ai = Agent_Critic(state_size, action_size)
        self.optimizer = optim.Adam(self.ai.parameters(), lr=learning_rate)
        self.total_updates = 0  # Track training updates across episodes


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
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, action_size)
        self.critic = nn.Linear(128, 1)


        if ENABLE_LOGGING: logger.log_msg(
            f"Agent_Network initialised successfully using PPO model (state_size={state_size}, " \
            f"action_size={action_size}, gamma={gamma}, clip_epsilon={clip_epsilon}, " \
            f"entropy_coeff={entropy_coeff}).", level=logging.INFO)    
    
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        task_assignment = self.actor(x)  # logits for action probabilities
        task_value = self.critic(x)      # value estimate

        #Check for NaNs/Infs in output logits
        if torch.isnan(task_assignment).any() or torch.isinf(task_assignment).any():
            print("[ERROR] NaNs or Infs detected in network output (task_assignment).")
            print(f"Input state: {state}")
            print(f"Hidden activation: {x}")
            print(f"Logits: {task_assignment}")
            raise ValueError("Network logits contain NaN or Inf.")

        return task_assignment, task_value



    def choose_action(self, state):
        if state is None:
            raise ValueError(f"[CRITICAL] Agent {self.AgentID} received a None state.")

        if any(v is None for v in state):
            raise ValueError(f"[CRITICAL] Agent {self.AgentID} state contains None values: {state}")

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        if ENABLE_LOGGING: logger.log_msg(f" AgentID: {self.AgentID}__State: {state}", level=logging.DEBUG)
        logits, value = self.forward(state_tensor)
        
        if torch.isnan(logits).any().item() or torch.isinf(logits).any().item():
        
            if ENABLE_LOGGING: logger.log_msg(f"[LOGITS ERROR] Detected NaNs/Infs in logits!", level=logging.ERROR)
            if ENABLE_LOGGING: logger.log_msg(f"State: {state}", level=logging.ERROR)
            if ENABLE_LOGGING: logger.log_msg(f"Logits: {logits}", level=logging.ERROR)
            raise ValueError("Logits contain NaN or Inf before softmax.")

        probs = torch.softmax(logits, dim=-1)

        if torch.isnan(probs).any().item() or torch.isinf(probs).any().item():
            if ENABLE_LOGGING: logger.log_msg(f"[PROBS ERROR] Detected NaNs/Infs in probs!", level=logging.ERROR)
            if ENABLE_LOGGING: logger.log_msg(f"Logits: {logits}", level=logging.ERROR)
            if ENABLE_LOGGING: logger.log_msg(f"Softmaxed probs: {probs}", level=logging.ERROR)
            if ENABLE_LOGGING: logger.log_msg(f"State: {state}", level=logging.ERROR)
            raise ValueError("Action probabilities contain NaN or Inf.")

        dist = Categorical(probs)
        action = dist.sample()

        if ENABLE_LOGGING: logger.log_msg(f"Action probabilities: {probs.tolist()}, Selected action: {action.item()}, State value: {value.item()}.", level=logging.DEBUG)

        return action.item(), dist.log_prob(action), value.item()


    def store_transition(self, state, action, log_prob, reward, local_value, global_value, done):
        """
        Store a single transition in memory.
        """
        if ENABLE_LOGGING: logger.log_msg(f"AGENT - Storing transition: state={state}"\
                            f"action={action}"\
                            f"reward={reward}"\
                            f"local_value={local_value}"\
                            f"global_value={global_value}"\
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
        if ENABLE_LOGGING: logger.log_msg(
                        f"[MEMORY COUNT] {self.AgentID}"\
                        f"Transitions Stored: {len(self.memory['rewards'])}",
                        level=logging.DEBUG
                    )



    def train(self, mode='train'):
        """
        Train the PPO agent using stored experiences.
        In evaluation mode, we don’t perform training.
        
        :param mode: 'train' or 'evaluate'
        """
        required_keys = ["states", "actions", "log_probs", "rewards", "values", "dones"]

        if mode == 'train':
            if ENABLE_LOGGING: logger.log_msg("[AGENT NETWORK] Training Agent...", level=logging.DEBUG)

            # Check if memory is populated
            if not all(len(self.memory[k]) > 0 for k in required_keys):
                if ENABLE_LOGGING: logger.log_msg(
                    "[ERROR] Training :skipped Memory buffer is incomplete or empty. " +
                    ", ".join(f"{k}={len(self.memory[k])}" for k in required_keys),
                    level=logging.ERROR
                )
                return

            # Ensure sufficient transitions for training
            if len(self.memory["rewards"]) < 10:
                if ENABLE_LOGGING: logger.log_msg(
                    f"[WARNING] Training skipped: Not enough transitions in memory. "
                    f"Current transitions: {len(self.memory['rewards'])}",
                    level=logging.WARNING
                )
                return

            # Convert stored memory to tensors
            states = torch.tensor(self.memory["states"], dtype=torch.float32)
            actions = torch.tensor(self.memory["actions"], dtype=torch.long)
            log_probs_old = torch.stack(self.memory["log_probs"]).detach()
            rewards = self.memory["rewards"]
            values = torch.tensor(self.memory["values"], dtype=torch.float32)
            dones = torch.tensor(self.memory["dones"], dtype=torch.float32)

            # Compute returns and advantages using GAE
            try:
                returns, advantages = self.compute_gae(rewards, values, values, dones)
            except ValueError as e:
                if ENABLE_LOGGING: logger.log_msg(f"[ERROR] Failed to compute GAE: {e}", level=logging.ERROR)
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

                    logits, new_values = self.ai(states_batch)

                    if torch.isnan(logits).any().item() or torch.isinf(logits).any().item():
                        if ENABLE_LOGGING: logger.log_msg(f"[TRAIN LOGITS ERROR] NaNs in logits during training epoch {epoch}", level=logging.ERROR)
                        return

                    probs = torch.softmax(logits, dim=-1)
                    if torch.isnan(probs).any().item() or torch.isinf(probs).any().item():
                        if ENABLE_LOGGING: logger.log_msg(f"[TRAIN PROBS ERROR] NaNs in probs during training epoch {epoch}", level=logging.ERROR)
                        return

                    dist = Categorical(probs)
                    new_log_probs = dist.log_prob(actions_batch)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_log_probs - log_probs_old_batch)
                    surr1 = ratio * advantages_batch
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_batch
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = (returns_batch - new_values.squeeze(-1)).pow(2).mean()
                    loss = policy_loss + 0.5 * value_loss - self.entropy_coeff * entropy

                    self.optimizer.zero_grad()
                    loss.backward()
                    if ENABLE_TENSORBOARD:
                        TensorBoardLogger().log_scalar(f"Agent_{self.AgentID}/PolicyLoss", policy_loss.item(), self.total_updates)
                        TensorBoardLogger().log_scalar(f"Agent_{self.AgentID}/ValueLoss", value_loss.item(), self.total_updates)
                        TensorBoardLogger().log_scalar(f"Agent_{self.AgentID}/TotalLoss", loss.item(), self.total_updates)
                        TensorBoardLogger().log_scalar(f"Agent_{self.AgentID}/Entropy", entropy.item(), self.total_updates)

                    self.optimizer.step()

                    if ENABLE_LOGGING: logger.log_msg(
                        f"[TRAIN] Epoch {epoch + 1}: Loss={loss.item():.4f}, PolicyLoss={policy_loss.item():.4f}, ValueLoss={value_loss.item():.4f}, Entropy={entropy.item():.4f}",
                        level=logging.DEBUG
                    )

            if ENABLE_LOGGING: logger.log_msg("[TRAIN COMPLETE] PPO update applied.", level=logging.INFO)

        elif mode == 'evaluate':
            if ENABLE_LOGGING: logger.log_msg("[EVALUATE MODE] No training applied.", level=logging.DEBUG)

        self.clear_memory()






    def compute_gae(self, rewards, local_values, global_values, dones):
        """
        Compute Generalised Advantage Estimation (GAE).
        """
        if len(rewards) == 0 or len(local_values) == 0 or len(global_values) == 0 or len(dones) == 0:
            if ENABLE_LOGGING: logger.log_msg("[ERROR] GAE input arrays are empty.", level=logging.ERROR)
            print(f"[DEBUG] rewards: {type(rewards)}, len={len(rewards)}")
            print(f"[DEBUG] local_values: {local_values.shape}")
            print(f"[DEBUG] global_values: {global_values.shape}")
            print(f"[DEBUG] dones: {dones.shape}")

            return None, None

            
        if ENABLE_LOGGING: logger.log_msg("Computing Generalised Advantage Estimation (GAE).", level=logging.INFO)
        returns = []
        advantages = []
        gae = 0
        last_return = 0

        for step in reversed(range(len(rewards))):
            mask = 1 - dones[step]

            if step < len(rewards) - 1:
                next_global = global_values[step + 1]
            else:
                next_global = torch.tensor(0.0)  # or use global_values[step] if continuing value

            delta = rewards[step] + self.gamma * next_global * mask - local_values[step]

            gae = delta + self.gamma * self.clip_epsilon * mask * gae

            advantages.insert(0, gae)
            last_return = rewards[step] + self.gamma * last_return * mask
            returns.insert(0, last_return)

        if ENABLE_LOGGING: logger.log_msg(f"GAE computed. Returns: {returns}, Advantages: {advantages}.", level=logging.DEBUG)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        return returns, (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    def clear_memory(self):
        for key in self.memory.keys():
            self.memory[key] = []

    

    
    
    def calculate_loss(self):
        """
        Calculate the loss for PPO using collected experiences.
        """
        # Retrieve experiences from agents (or memory buffer)
        all_experiences = []
        for agent in self.agents:
            all_experiences.extend(agent.experience_buffer)  # Collect experiences for all agents

        # Extract states, actions, rewards, and next states from experiences
        states = torch.tensor([exp["state"] for exp in all_experiences], dtype=torch.float32)
        actions = torch.tensor([exp["action"] for exp in all_experiences], dtype=torch.long)
        rewards = torch.tensor([exp["reward"] for exp in all_experiences], dtype=torch.float32)
        next_states = torch.tensor([exp["next_state"] for exp in all_experiences], dtype=torch.float32)
        dones = torch.tensor([exp["done"] for exp in all_experiences], dtype=torch.float32)

        # Compute policy logits and values
        logits, values = self.ai(states)
        values = values.squeeze(-1)  # Remove extra dimensions

        # Compute advantages (reward-to-go or GAE)
        with torch.no_grad():
            _, next_values = self.ai(next_states)
            next_values = next_values.squeeze(-1)
            advantages = rewards + self.gamma * next_values * (1 - dones) - values

        # Compute policy loss
        log_probs = torch.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        policy_loss = -(advantages.detach() * selected_log_probs).mean()

        # Compute value loss
        value_loss = (advantages ** 2).mean()

        # Compute entropy loss for exploration regularisation
        entropy_loss = -torch.mean(torch.sum(torch.softmax(logits, dim=-1) * log_probs, dim=-1))

        # Combine losses with coefficients
        loss = policy_loss + self.value_loss_coeff * value_loss - self.entropy_coeff * entropy_loss

        # Log individual losses for debugging
        print(f"Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}, Entropy Loss: {entropy_loss.item()}")

        return loss
    

    def update(self, experiences):
        """
        Perform a training update using the experiences.
        :param experiences: A list of experience dictionaries with keys: state, action, reward, next_state, done.
        """
        # Calculate the loss
        loss = self.calculate_loss(experiences)

        # Backpropagate and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss






#Scalable network architecture for the agents to use. Allows switching between different architectures.









#    ____   ___  _   _   __  __           _      _ 
#   |  _ \ / _ \| \ | | |  \/  | ___   __| | ___| |
#   | | | | | | |  \| | | |\/| |/ _ \ / _` |/ _ \ |
#   | |_| | |_| | |\  | | |  | | (_) | (_| |  __/ |
#   |____/ \__\_\_| \_| |_|  |_|\___/ \__,_|\___|_|
#                                                  









class DQNModel(nn.Module):
    def __init__(self, state_size, action_size, device=Training_device):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.q_values = nn.Linear(128, action_size)

    def forward(self, state):
        """
        Forward pass through the DQN network.
        :param state: The input state
        :return: Q-values for each action
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.q_values(x)
        return q_values

    def choose_action(self, state):
        """
        Given a state, return the action with the highest Q-value.
        :param state: The input state
        :return: action (chosen)
        """
        q_values = self.forward(state)
        action = torch.argmax(q_values, dim=-1)  # Select action with highest Q-value
        return action.item()
