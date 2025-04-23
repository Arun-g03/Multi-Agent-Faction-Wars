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
    Low level function to save model state and optimizer state (if available) to disk.
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
    Load model state and optimizer state (if available) from disk,
    updating model layers dynamically if input shape mismatches.
    """
    checkpoint = torch.load(path, map_location=device or model_obj.device)
    print(f"\nLoading checkpoint from {path}")

    # Load model state
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']

        # Check for input mismatch and update if needed
        fc1_weights = model_state.get('fc1.weight')
        if fc1_weights is not None:
            saved_input_size = fc1_weights.shape[1]
            current_input_size = model_obj.fc1.in_features
            if saved_input_size != current_input_size:
                print(f"[RESIZE] Updating input size from {current_input_size} to {saved_input_size}")
                model_obj.update_network(saved_input_size)

        # Load weights
        model_obj.load_state_dict(model_state)

        # ✅ Always verify model weights after loading
        sample_weight = next(iter(model_obj.parameters())).flatten()[0].item()
        if utils_config.ENABLE_LOGGING: 
            logger.log_msg(
            f"[VERIFY] Loaded weight sample: {sample_weight:.5f} from {path}",
            level=logging.DEBUG
                                )

    # Optional: Load optimizer
    if hasattr(model_obj, "optimizer") and checkpoint.get("optimizer_state_dict"):
        model_obj.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Update training counter if stored
    if hasattr(model_obj, "total_updates"):
        model_obj.total_updates = checkpoint.get("total_updates", 0)


    if utils_config.ENABLE_LOGGING:
        logger.log_msg(f"[MODEL LOADED] {model_obj.__class__.__name__} loaded from {path}")



def get_hq_input_size_from_checkpoint(path):
    """
    Returns the input layer size (in_features) from a saved HQ checkpoint.
    """
    checkpoint = torch.load(path, map_location='cpu')
    weight = checkpoint['model_state_dict']['fc1.weight']
    return weight.shape[1]  # This is the input dimension
