import torchvision.models as models
import torch.nn as nn

def ResNet50(num_classes=50, pretrained=False):
    model = models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
