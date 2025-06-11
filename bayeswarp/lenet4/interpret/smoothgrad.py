import torch

class SmoothGrad:

    def __init__(self, model, device="cuda", n_samples=30, noise_level=0.1):
        self.model = model
        self.device = device
        self.n_samples = n_samples
        self.noise_level = noise_level

    def get_critical_region(self, input_tensor, target_class, threshold=0.5):

        self.model.eval()
        input_tensor = input_tensor.to(self.device)

        grads = []
        for i in range(self.n_samples):

            noise = torch.randn_like(input_tensor) * self.noise_level
            noisy_input = (input_tensor + noise).clone().detach().requires_grad_(True)

            output = self.model(noisy_input)
            if output.dim() == 1:
                output = output.unsqueeze(0)
            score = output[0, target_class]

            self.model.zero_grad()
            if noisy_input.grad is not None:
                noisy_input.grad.zero_()

            score.backward(retain_graph=True)
            grads.append(noisy_input.grad.detach().clone())

        avg_grad = torch.stack(grads, dim=0).mean(dim=0)  # [1, C, H, W]
        abs_avg_grad = avg_grad.abs().mean(dim=1, keepdim=True)  # [1, 1, H, W]
        norm = (abs_avg_grad - abs_avg_grad.min()) / (abs_avg_grad.max() - abs_avg_grad.min() + 1e-8)
        region_mask = (norm > threshold).float()  # [1, 1, H, W]
        return region_mask

