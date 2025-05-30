# metrics/compute_sc.py
import torch
import numpy as np

def compute_sc(adv_samples, orig_samples, threshold=1e-3):

    if isinstance(adv_samples, list):
        adv_samples = torch.stack(adv_samples)
    if isinstance(orig_samples, list):
        orig_samples = torch.stack(orig_samples)
    # 保证在CPU上比对
    adv_np = adv_samples.detach().cpu().numpy()
    orig_np = orig_samples.detach().cpu().numpy()
    covered = 0
    total = orig_np.shape[0]
    for orig, adv in zip(orig_np, adv_np):
        diff = np.abs(orig - adv)
        # 可以选用 L2、L1 或全元素判定
        if np.any(diff > threshold):
            covered += 1
    return covered / total if total > 0 else 0.0

