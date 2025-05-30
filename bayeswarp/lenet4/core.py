import torch
from bayeswarp.interpret.grad_cam import GradCAM
from bayeswarp.interpret.integrated_gradients import IntegratedGradients
from bayeswarp.interpret.smoothgrad import SmoothGrad
from bayeswarp.optimization.bayes_opt import BayesianOptimizer
from bayeswarp.mutation import region_additive_mutation, region_signed_mutation, region_gradient_mutation, region_interpolation

class BayesWarp:
    def __init__(self, model, region_method="gradcam", device="cuda"):
        self.model = model.to(device).eval()
        self.device = device
        if region_method == "gradcam":
            self.region_finder = GradCAM(self.model, device=device)
        elif region_method == "integrated":
            self.region_finder = IntegratedGradients(self.model, device=device)
        elif region_method == "smoothgrad":
            self.region_finder = SmoothGrad(self.model, device=device)
        else:
            self.region_finder = None

    def get_region_mask(self, input_tensor, target_class):
        if self.region_finder is not None:
            region_mask = self.region_finder.get_critical_region(input_tensor, target_class)
        else:
            region_mask = torch.ones_like(input_tensor)[:, :1]
        return region_mask

    def run(self, input_tensor, target_class=None, bayes_iters=30, max_classes=10, region_mode=None, mutation_mode="additive"):
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            orig_class = output.argmax(dim=1).item() if target_class is None else target_class

        region_mask = self.get_region_mask(input_tensor, orig_class)
        optimizer = BayesianOptimizer(
            self.model,
            input_tensor,
            region_mask,
            orig_class,
            device=self.device,
            max_classes=max_classes,
            max_iters=bayes_iters
        )
        adv_list = optimizer.optimize()
        adv_tensors = []
        for adv in adv_list:
            adv = adv.to(self.device)
            if mutation_mode == "additive":
                adv = region_additive_mutation(adv, region_mask, epsilon=0.05)
            elif mutation_mode == "signed":
                adv = region_signed_mutation(adv, region_mask, epsilon=0.03, sign="random")
            elif mutation_mode == "gradient":
                adv.requires_grad = True
                out = self.model(adv)
                score = out[0, orig_class]
                self.model.zero_grad()
                if adv.grad is not None:
                    adv.grad.zero_()
                score.backward()
                adv = region_gradient_mutation(adv, region_mask, adv.grad, epsilon=0.03)
            adv_tensors.append(adv.detach().cpu())
        return adv_tensors
