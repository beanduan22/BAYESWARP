import torch

def compute_nof(model, adv_samples, orig_labels, device="cuda"):

    model.eval()
    count = 0
    with torch.no_grad():
        if isinstance(adv_samples, list):
            adv_samples = torch.stack(adv_samples)
        adv_samples = adv_samples.to(device)
        outputs = model(adv_samples)
        preds = outputs.argmax(dim=1).cpu()
        if isinstance(orig_labels, torch.Tensor):
            orig_labels = orig_labels.cpu()
        else:
            orig_labels = torch.tensor(orig_labels)

        for pred, orig_label in zip(preds, orig_labels):
            if pred.item() != orig_label.item():
                count += 1
    return count
