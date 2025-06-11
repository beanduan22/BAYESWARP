import torch

def to_device(tensor, device):
    return tensor.to(device)

def set_seed(seed=seed):
    import random, numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_model(path, device="cuda"):
    state = torch.load(path, map_location=device)

    model = ...  # Instantiate your model
    model.load_state_dict(state)
    return model.to(device)
