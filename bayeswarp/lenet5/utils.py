# bayeswarp/utils.py

import torch

def to_device(tensor, device):
    return tensor.to(device)

def set_seed(seed=42):
    import random, numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_model(path, device="cuda"):
    # 加载模型权重
    state = torch.load(path, map_location=device)

    model = ...  # 实例化你的模型
    model.load_state_dict(state)
    return model.to(device)
