# metrics/compute_fid.py
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
import torch
import torch.nn.functional as F

def get_activations(samples, model, batch_size=32):

    model.eval()
    act = []
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = torch.stack(samples[i:i+batch_size]).cuda()
            pred = model(batch)[0].view(batch.size(0), -1)
            act.append(pred.cpu().numpy())
    return np.concatenate(act, axis=0)

def compute_fid(adv_samples, orig_samples):

    inception = inception_v3(pretrained=True, transform_input=False).cuda()
    orig_acts = get_activations(orig_samples, inception)
    adv_acts = get_activations(adv_samples, inception)
    mu1, sigma1 = orig_acts.mean(axis=0), np.cov(orig_acts, rowvar=False)
    mu2, sigma2 = adv_acts.mean(axis=0), np.cov(adv_acts, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
