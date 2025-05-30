import torch

def compute_dof(model, adv_samples, orig_labels, device="cuda"):

    model.eval()
    failed_classes = set()
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
                failed_classes.add(pred.item())
    return len(failed_classes)


