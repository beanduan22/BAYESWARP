import torch.nn as nn
import torchvision.models as models

def VGG19(num_classes=50, pretrained=False):
    model = models.vgg19(pretrained=pretrained)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model
