from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn


class AMLResnet50(torch.Module):

    def __init__(self, out_dim:int):
        # New weights with accuracy 80.858%
        self.net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Take the input of the fully connected layer of effnet
        in_dim = self.net.classifier[-1].in_features

        # Disable efficient net b7 classifier
        self.net.classifier = nn.Identity()

        # Declare the fully connected layer
        self.classifier = nn.Linear(in_dim, out_dim)

        # Definition of multiple dropout
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])

    def freeze(self):
        # Don't compute the gradients for net feature
        for _, param in self.net.named_parameters():
            param.requires_grad = False
