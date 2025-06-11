import torch

class IntegratedGradients:

    def __init__(self, model, device="cuda", steps=50):
        self.model = model
        self.device = device
        self.steps = steps

    def get_critical_region(self, input_tensor, target_class, baseline=None, threshold=0.5):

        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        if baseline is None:
            baseline = torch.zeros_like(input_tensor).to(self.device)


        scaled_inputs = [baseline + (float(i) / self.steps) * (input_tensor - baseline) for i in range(self.steps + 1)]

        grads = []
        for x in scaled_inputs:
            x = x.clone().detach().requires_grad_(True)

            output = self.model(x)
            if output.dim() == 1:
                output = output.unsqueeze(0)
            score = output[0, target_class]

            self.model.zero_grad()
            if x.grad is not None:
                x.grad.zero_()

            score.backward(retain_graph=True)
            grads.append(x.grad.detach().clone())

        avg_grads = torch.stack(grads, dim=0).mean(dim=0)
        ig = (input_tensor - baseline) * avg_grads

        ig_min, ig_max = ig.min(), ig.max()
        norm = (ig - ig_min) / (ig_max - ig_min + 1e-8)
        region_mask = (norm > threshold).float()

        return region_mask


