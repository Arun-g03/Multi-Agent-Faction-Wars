"""Common Imports"""
from SHARED.core_imports import *


logger = Logger(log_file="neural_network_common_log.txt", log_level=logging.DEBUG)

def check_training_device():
    # Check for GPU
    Training_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"\033[93mUsing GPU: {torch.cuda.get_device_name(0)}\033[0m")
    else:
        print("\033[93m**Agent Networks**\nNo GPU detected.\nUsing CPU.\nNote training will be slower.\n\033[0m")
    
    return Training_device# I was gonna break the file up but instead, i like that all the neural
# network parts are together


Training_device = check_training_device()
