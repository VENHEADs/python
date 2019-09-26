from torchvision import models
import torch
from config import _get_default_config
import torch.nn as nn

config = _get_default_config()


if config.model == 'resnet18':
    classes = 1108
    model_resnet_18 = models.resnet18(pretrained=True)

    num_ftrs = model_resnet_18.fc.in_features
    model_resnet_18.fc = torch.nn.Linear(num_ftrs, classes)

    # let's make our model work with 6 channels
    trained_kernel = model_resnet_18.conv1.weight
    new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv.weight[:, :] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)
    model_resnet_18.conv1 = new_conv

if config.model == 'densenet121':
    classes = 1108
    model_resnet_18 = models.densenet121(pretrained=True)

    num_ftrs = model_resnet_18.classifier.in_features
    model_resnet_18.classifier = torch.nn.Linear(num_ftrs, classes)
    #
    # # let's make our model work with 6 channels
    trained_kernel = model_resnet_18.features.conv0.weight
    new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv.weight[:, :] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)
    model_resnet_18.features.conv0 = new_conv

