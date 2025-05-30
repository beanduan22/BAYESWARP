
import torch
from torchvision import datasets, transforms


IMAGENET_50_CLASSES = [
    'n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475',
    'n01496331', 'n01498041', 'n01514668', 'n01514859', 'n01518878',
    'n01530575', 'n01531178', 'n01532829', 'n01534433', 'n01537544',
    'n01558993', 'n01560419', 'n01582220', 'n01592084', 'n01601694',
    'n01608432', 'n01614925', 'n01616318', 'n01622779', 'n01629819',
    'n01630670', 'n01632777', 'n01641577', 'n01644373', 'n01644900',
    'n01664065', 'n01665541', 'n01667114', 'n01667778', 'n01669191',
    'n01675722', 'n01677366', 'n01682714', 'n01685808', 'n01687978',
    'n01688243', 'n01689811', 'n01692333', 'n01693334', 'n01694178',
    'n01695060', 'n01697457', 'n01698640', 'n01704323', 'n01728572'
]

def get_imagenet50_loader(batch_size=64, train=True, path='./data/imagenet', shuffle=True):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    split = 'train' if train else 'val'
    full_dataset = datasets.ImageFolder(root=f"{path}/{split}", transform=transform)

    class_to_idx = full_dataset.class_to_idx
    selected_class_indices = [class_to_idx[c] for c in IMAGENET_50_CLASSES if c in class_to_idx]

    indices_50 = [i for i, (img, label) in enumerate(full_dataset.samples)
                  if label in selected_class_indices]
    subset = torch.utils.data.Subset(full_dataset, indices_50)
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
    return loader
