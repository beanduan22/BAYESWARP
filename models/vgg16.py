import torchvision.models as models
import torch.nn as nn

def VGG16(num_classes=10, pretrained=False):
    model = models.vgg16(pretrained=pretrained)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model
