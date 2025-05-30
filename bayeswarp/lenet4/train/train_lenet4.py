import torch
import torch.nn as nn
import torch.optim as optim
from data.mnist import get_mnist_loader
from models.lenet4 import LeNet4

def train(model, train_loader, test_loader, device="cuda", epochs=50, lr=0.01):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
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
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}, Test Acc: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "lenet4_best.pth")

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
    train_loader = get_mnist_loader(batch_size=64, train=True)
    test_loader = get_mnist_loader(batch_size=256, train=False)
    model = LeNet4(num_classes=10)
    train(model, train_loader, test_loader)
