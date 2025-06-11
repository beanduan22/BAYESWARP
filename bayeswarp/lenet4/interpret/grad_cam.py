import torch
import torch.nn.functional as F

class GradCAM:

    def __init__(self, model, target_layer=None, device="cuda"):
        self.model = model
        self.device = device
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []

    def _find_last_conv(self):

        conv_layers = [m for m in self.model.modules() if isinstance(m, torch.nn.Conv2d)]
        if not conv_layers:
            raise ValueError("Model has no Conv2d layers.")
        return conv_layers[-1]

    def _register_hooks(self):
        if self.target_layer is None:
            self.target_layer = self._find_last_conv()


        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        handle1 = self.target_layer.register_forward_hook(forward_hook)
        handle2 = self.target_layer.register_backward_hook(backward_hook)
        self.hook_handles.extend([handle1, handle2])

    def get_critical_region(self, input_tensor, target_class, threshold=0.5):

        self.model.eval()
        self._register_hooks()

        input_tensor = input_tensor.to(self.device)
        if input_tensor.requires_grad is False:
            input_tensor = input_tensor.clone().detach().requires_grad_(True)

        output = self.model(input_tensor)
        if output.dim() == 1:

            output = output.unsqueeze(0)
        score = output[0, target_class]

        self.model.zero_grad()
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()

        score.backward(retain_graph=True)


        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)

        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        region_mask = (cam > threshold).float() 

        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

        return region_mask

    def __del__(self):
        for h in self.hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self.hook_handles.clear()

