import torch.nn as nn
import torchvision.models as models


def build_resnet18_regressor(pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)

    return model


def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True

    return model