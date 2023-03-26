from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
import torch as torch
import torch.nn as nn

class AMLResnet50_V0(nn.Module):

    def __init__(self, out_dim:int):

        super().__init__()

        # New weights with accuracy 80.858%
        self.net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Take the input of the fully connected layer of effnet
        in_dim = self.net.fc.in_features

        # Noop operation
        self.net.fc = nn.Identity()

        # Freeze layers
        self.__freeze_layers()

        # Declare the fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim))


        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

    def __freeze_layers(self):
        # Don't compute the gradients for net feature
        for _, param in self.net.named_parameters():
            param.requires_grad = False


    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)
        return x


class AMLResnet50_V1(nn.Module):
    """This AMLRestnet50 emulates fastai architecture"""

    def __init__(self, out_dim:int):

        super().__init__()

        # New weights with accuracy 80.858%
        self.net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Take the input of the fully connected layer of effnet
        in_dim = self.net.fc.in_features

        # Disable efficient net b7 classifier
        self.net.fc = nn.Identity()

        # Declare the fully connected layer
        self.fc = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Dropout(0.5),
            nn.Linear(2048,512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512,out_dim),
        )

        # Freeze layers
        self.__freeze_layers()

        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

    def __freeze_layers(self):
        # Don't compute the gradients for net feature
        for _, param in self.net.named_parameters():
            param.requires_grad = False


    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)
        return x


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)

class AMLResnet50_fastAI(nn.Module):

    def __init__(self, out_dim:int):

        super().__init__()

        # New weights with accuracy 80.858%
        self.net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Take the input of the fully connected layer of effnet
        in_dim = self.net.fc.in_features

        # Disable efficient net b7 classifier
        self.net.fc = nn.Identity()

        self.fc = nn.Sequential(
            #AdaptiveConcatPool2d((32, 4096)),
            #nn.Flatten(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.5),
            nn.Linear(2048,512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512,10),
        )

        # Disable efficient net b7 classifier
        #self.net.fc = nn.Identity()

        # Declare the fully connected layer
        # self.fc = nn.Linear(in_dim, out_dim)

        # Definition of multiple dropout
        # self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])

        # Freeze layers
        self.freeze()

        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

    def freeze(self):
        # Don't compute the gradients for net feature
        for _, param in self.net.named_parameters():
            param.requires_grad = False


    def forward(self, x):

        x = self.net(x)

        x = self.fc(x)

        return x
