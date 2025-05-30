import torch
import numpy as np

class SparseVariationalGP:

    def __init__(self, base_input, region_mask, n_inducing=10, device="cuda"):
        self.device = device
        self.n_inducing = n_inducing
        self.base_input = base_input.clone().to(device)
        self.region_mask = region_mask.to(device)
        self.inducing_points = self._init_inducing(self.base_input, self.region_mask)
        self.observations = []
        self.values = []

    def _init_inducing(self, base_input, region_mask):

        mask = region_mask.bool()
        indices = torch.where(mask)
        if indices[0].numel() == 0:
            raise ValueError("Region mask contains no positive region!")
        idx = torch.randperm(indices[0].shape[0])[:min(self.n_inducing, indices[0].shape[0])]
        inducing_points = []
        for i in range(len(idx)):
            x_cand = base_input.clone()

            c, h, w = indices[1][idx[i]], indices[2][idx[i]], indices[3][idx[i]]
            x_cand[0, c, h, w] += torch.randn(1, device=self.device) * 0.1
            inducing_points.append(x_cand)
        return torch.stack(inducing_points, dim=0)

    def propose(self, last_x, region_mask=None, sigma=0.03, strategy="random"):

        region_mask = region_mask if region_mask is not None else self.region_mask
        if strategy == "random" or len(self.observations) == 0:
            perturb = torch.randn_like(last_x) * sigma
            perturb = perturb * region_mask
            proposed_x = last_x + perturb
        elif strategy == "furthest":

            candidates = []
            for _ in range(5):
                perturb = torch.randn_like(last_x) * sigma
                perturb = perturb * region_mask
                cand = last_x + perturb
                dists = [torch.norm(cand - obs.to(cand.device)) for obs in self.observations]
                min_dist = min(dists) if dists else 0
                candidates.append((min_dist, cand))

            proposed_x = max(candidates, key=lambda x: x[0])[1]
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        proposed_x = torch.clamp(proposed_x, 0.0, 1.0)
        return proposed_x

    def update(self, x, y):

        self.observations.append(x.detach().cpu())
        if isinstance(y, torch.Tensor):
            self.values.append(y.detach().cpu().item())
        else:
            self.values.append(float(y))

    def get_gp_kernel_matrix(self):

        if len(self.observations) == 0:
            return None
        obs_mat = torch.stack([obs.view(-1) for obs in self.observations])

        K = torch.cdist(obs_mat, obs_mat, p=2)
        gamma = 1.0
        K = torch.exp(-gamma * K)
        return K

    def get_observation_stats(self):

        if len(self.observations) == 0:
            return None, None
        obs_mat = torch.stack([obs.view(-1) for obs in self.observations])
        return obs_mat.mean(dim=0), obs_mat.std(dim=0)

