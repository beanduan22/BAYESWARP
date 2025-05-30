import torch
import torch.nn.functional as F
from .svgp import SparseVariationalGP
from .acquisition import expected_improvement

class BayesianOptimizer:
    def __init__(
        self, model, input_tensor, region_mask, orig_class,
        device="cuda", max_classes=10, max_iters=30,
        eta=0.1, random_noise=0.02, tol=1e-3
    ):

        self.model = model.to(device)
        self.input_tensor = input_tensor.detach().clone().to(device)
        self.region_mask = region_mask.to(device)
        self.orig_class = orig_class
        self.device = device
        self.max_classes = max_classes
        self.max_iters = max_iters
        self.eta = eta
        self.random_noise = random_noise
        self.tol = tol
        self.svgp = SparseVariationalGP(self.input_tensor, self.region_mask, device=device)
        self.reset_class_status()

    def reset_class_status(self):

        self.tried_classes = set([self.orig_class])
        self.best_adv = self.input_tensor.clone()
        self.adv_found = False

    def adaptive_lambda(self, conf_tg, conf_orig):

        H = lambda p: -p * torch.log(p + 1e-8)
        return H(conf_tg) / (H(conf_tg) + H(conf_orig) + 1e-8)

    def objective(self, x, orig_class, target_class):

        out = self.model(x)
        probs = F.softmax(out, dim=1)
        conf_tg = probs[0, target_class]
        conf_orig = probs[0, orig_class]
        λ = self.adaptive_lambda(conf_tg, conf_orig)
        obj = λ * conf_tg - (1 - λ) * conf_orig
        return obj, conf_tg, conf_orig

    def find_next_target_class(self, x):

        out = self.model(x)
        probs = F.softmax(out, dim=1)[0]
        candidate_probs = probs.clone()
        candidate_probs[list(self.tried_classes)] = -1  # 屏蔽已尝试
        next_class = candidate_probs.argmax().item()
        return next_class

    def optimize(self, return_details=False):

        x0 = self.input_tensor.clone()
        best_adv_list = []
        info_list = []

        for _ in range(self.max_classes - 1):

            tg_class = self.find_next_target_class(x0)
            if tg_class in self.tried_classes:
                continue
            self.tried_classes.add(tg_class)
            x = x0.clone().detach()
            best_score = None
            prev_obj = None
            loss_trajectory = []

            for iter in range(self.max_iters):
                # SVGP生成候选扰动样本
                candidate = self.svgp.propose(x, self.region_mask)
                candidate = candidate.clone().detach().requires_grad_(True)

                obj, conf_tg, conf_orig = self.objective(candidate, self.orig_class, tg_class)
                loss_trajectory.append(float(obj.item()))
                self.svgp.update(candidate, obj)

                if prev_obj is not None and torch.abs(obj - prev_obj) < self.tol:
                    noise = torch.randn_like(candidate) * self.random_noise
                    candidate = torch.clamp(candidate + noise * self.region_mask, 0.0, 1.0)
                prev_obj = obj

                out_cls = self.model(candidate).argmax(dim=1).item()
                if out_cls == tg_class:
                    best_score = obj.item()
                    best_adv_list.append(candidate.detach().cpu())
                    if return_details:
                        info_list.append({
                            "target_class": tg_class,
                            "steps": iter+1,
                            "loss_traj": loss_trajectory,
                            "conf_tg": float(conf_tg.item()),
                            "conf_orig": float(conf_orig.item())
                        })
                    break
                x = candidate.detach()

            if len(self.tried_classes) >= self.max_classes:
                break

        if best_adv_list:
            if return_details:
                return best_adv_list, info_list
            return best_adv_list
        else:
            if return_details:
                return [self.input_tensor.cpu()], []
            return [self.input_tensor.cpu()]

