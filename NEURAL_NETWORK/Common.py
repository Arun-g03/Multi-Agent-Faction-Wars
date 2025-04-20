"""Common Imports"""
from SHARED.core_imports import *
import UTILITIES.utils_config as utils_config
import functools



logger = Logger(log_file="neural_network_common_log.txt", log_level=logging.DEBUG)

@functools.lru_cache(maxsize=1)
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



def save_checkpoint(model_obj, path):
    """
    Save model state and optimizer state (if available) to disk.
    """
    try:
        checkpoint = {
            'model_state_dict': model_obj.state_dict(),
            'optimizer_state_dict': model_obj.optimizer.state_dict() if hasattr(model_obj, "optimizer") else None,
            'total_updates': getattr(model_obj, "total_updates", 0)
        }
        torch.save(checkpoint, path)
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(f"[MODEL SAVED] {model_obj.__class__.__name__} saved to {path}")
    except Exception as e:
        raise Exception(f"Failed to save checkpoint for {model_obj.__class__.__name__}: {str(e)}")
def load_checkpoint(model_obj, path, device=None):
    """
    Load model state and optimizer state (if available) from disk.
    """
    checkpoint = torch.load(path, map_location=device or model_obj.device)
    model_obj.load_state_dict(checkpoint['model_state_dict'])
    if hasattr(model_obj, "optimizer") and checkpoint.get("optimizer_state_dict"):
        model_obj.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if hasattr(model_obj, "total_updates"):
        model_obj.total_updates = checkpoint.get("total_updates", 0)
    if utils_config.ENABLE_LOGGING:
        logger.log_msg(f"[MODEL LOADED] {model_obj.__class__.__name__} loaded from {path}")