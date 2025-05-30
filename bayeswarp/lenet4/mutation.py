import torch

def region_additive_mutation(input_tensor, region_mask, epsilon=0.05, mode="uniform", clamp=(0.0, 1.0)):

    if region_mask.shape[1] == 1 and input_tensor.shape[1] > 1:
        region_mask = region_mask.repeat(1, input_tensor.shape[1], 1, 1)
    if mode == "uniform":
        noise = (torch.rand_like(input_tensor) - 0.5) * 2 * epsilon
    elif mode == "gaussian":
        noise = torch.randn_like(input_tensor) * epsilon
    else:
        raise ValueError(f"Unknown mode: {mode}")
    mutation = noise * region_mask
    mutated = input_tensor + mutation
    mutated = torch.clamp(mutated, *clamp)
    return mutated

def region_signed_mutation(input_tensor, region_mask, epsilon=0.05, sign="random", clamp=(0.0, 1.0)):

    if region_mask.shape[1] == 1 and input_tensor.shape[1] > 1:
        region_mask = region_mask.repeat(1, input_tensor.shape[1], 1, 1)
    if sign == "random":
        sign_tensor = torch.randint_like(input_tensor, 0, 2, dtype=torch.float32) * 2 - 1
    elif sign == "positive":
        sign_tensor = torch.ones_like(input_tensor)
    elif sign == "negative":
        sign_tensor = -torch.ones_like(input_tensor)
    else:
        raise ValueError(f"Unknown sign: {sign}")
    mutation = epsilon * sign_tensor * region_mask
    mutated = input_tensor + mutation
    mutated = torch.clamp(mutated, *clamp)
    return mutated

def region_gradient_mutation(input_tensor, region_mask, gradient, epsilon=0.03, clamp=(0.0, 1.0)):

    if region_mask.shape[1] == 1 and input_tensor.shape[1] > 1:
        region_mask = region_mask.repeat(1, input_tensor.shape[1], 1, 1)
    grad_sign = gradient.sign()
    mutation = epsilon * grad_sign * region_mask
    mutated = input_tensor + mutation
    mutated = torch.clamp(mutated, *clamp)
    return mutated

def region_interpolation(input_tensor, region_mask, target_tensor, alpha=0.5, clamp=(0.0, 1.0)):

    if region_mask.shape[1] == 1 and input_tensor.shape[1] > 1:
        region_mask = region_mask.repeat(1, input_tensor.shape[1], 1, 1)
    mutation = (target_tensor - input_tensor) * region_mask * alpha
    mutated = input_tensor + mutation
    mutated = torch.clamp(mutated, *clamp)
    return mutated

def region_custom_mutation(input_tensor, region_mask, custom_func, **kwargs):

    noise = custom_func(input_tensor, region_mask, **kwargs)
    mutated = input_tensor + noise * region_mask
    mutated = torch.clamp(mutated, 0.0, 1.0)
    return mutated
