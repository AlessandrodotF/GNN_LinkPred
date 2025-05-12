import torch
import random
import numpy as np

def set_seed(seed):
    """
    Set the random seed for reproducibility across multiple libraries.

    This function sets the seed for PyTorch, NumPy, and Python's random module.
    It also configures CUDA settings to ensure deterministic behavior in GPU operations.
    
    Parameters
    ----------
    seed : int
        The seed value to be used for initializing the random number generators.
    """
    #print("just to check if it works")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False   