import torch
import torch.nn as nn
import torch.optim as optim
from data.imagenet import get_imagenet50_loader
from models.resnet50 import ResNet50

def train(model, train_loader, test_loader, device="cuda", epochs=90, lr=0.1):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
        scheduler.step()
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}, Test Acc: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "resnet50_best.pth")

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return 100. * correct / total

if __name__ == "__main__":
    train_loader = get_imagenet50_loader(batch_size=64, train=True)
    test_loader = get_imagenet50_loader(batch_size=256, train=False)
    model = ResNet50(num_classes=50)
    train(model, train_loader, test_loader)
