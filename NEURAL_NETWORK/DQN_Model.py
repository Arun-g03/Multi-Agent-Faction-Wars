"""Common Imports"""
from SHARED.core_imports import *

"""File Specific Imports"""
from NEURAL_NETWORK.Common import check_training_device
import UTILITIES.utils_config as utils_config

Training_device = check_training_device()


logger = Logger(log_file="DQN_Network.txt", log_level=logging.DEBUG)



"""
Future works would see the DQN model being used for the Agents.
This would allow variety in active models used by the agents.
"""



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
        # Select action with highest Q-value
        action = torch.argmax(q_values, dim=-1)
        return action.item()
