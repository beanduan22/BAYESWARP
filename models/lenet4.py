import torch.nn as nn
import torch.nn.functional as F

class LeNet4(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet4, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, 1, 0)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    def forward(self, x):
        x = F.relu(self.conv1(x))      # [B,6,28,28]
        x = F.max_pool2d(x, 2, 2)      # [B,6,14,14]
        x = F.relu(self.conv2(x))      # [B,16,10,10]
        x = F.max_pool2d(x, 2, 2)      # [B,16,5,5]
        x = x.view(x.size(0), -1)      # [B,16*5*5]
        x = F.relu(self.fc1(x))        # [B,120]
        x = F.relu(self.fc2(x))        # [B,84]
        x = self.fc3(x)                # [B,num_classes]
        return x
