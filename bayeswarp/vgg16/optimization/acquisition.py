import torch
from torch.distributions.normal import Normal

def expected_improvement(mu, sigma, f_best, maximize=True, eps=1e-8):

    mu = mu.clone()
    sigma = sigma.clone()
    if not maximize:
        mu, f_best = -mu, -f_best

    sigma = sigma.clamp_min(eps)
    z = (mu - f_best) / sigma
    normal = Normal(0, 1)
    cdf = normal.cdf(z)
    pdf = torch.exp(normal.log_prob(z))

    ei = (mu - f_best) * cdf + sigma * pdf

    ei[sigma <= eps] = 0.0

    ei = torch.where(torch.isnan(ei), torch.zeros_like(ei), ei)
    return ei


