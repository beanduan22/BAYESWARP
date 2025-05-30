
import torch
from torchvision import datasets, transforms

def get_cifar10_loader(batch_size=128, train=True, download=True, shuffle=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset = datasets.CIFAR10(root='./data', train=train, download=download, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
