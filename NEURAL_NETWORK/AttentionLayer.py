"""Common Imports"""
from SHARED.core_imports import *
"""File Specific Imports"""
import UTILITIES.utils_config as utils_config



logger = Logger(log_file="AttentionLayer.txt", log_level=logging.DEBUG)


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

        attention_scores = torch.matmul(
            query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply attention weights to value
        attention_output = torch.matmul(attention_weights, value)
        return attention_output

