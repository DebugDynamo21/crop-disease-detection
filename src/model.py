# import torch.nn as nn
# from torchvision import models


# def create_model(num_classes):
#     model = models.resnet18(weights="DEFAULT")

#     model.fc = nn.Linear(
#         model.fc.in_features,
#         num_classes
#     )

#     return model

import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes, pretrained=True):

    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet18(
        weights=weights
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model


def load_model(
    model_path,
    num_classes,
    device
):

    model = create_model(
        num_classes=num_classes,
        pretrained=False
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)


    model.to(device)

    model.eval()

    return model