from torchvision.models import (resnet50, ResNet50_Weights,
                                resnet101, ResNet101_Weights,
                                efficientnet_b4, EfficientNet_B4_Weights,
                                efficientnet_v2_s, EfficientNet_V2_S_Weights)



import torchvision.transforms as transforms
import torch as torch
import torch.nn as nn


class AMLResnet50(nn.Module):
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
        self.freeze_base()

        self.transforms = transforms.Compose([
            transforms.Resize(232),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

    def freeze_base(self):
        # Don't compute the gradients for net feature
        for _, param in self.net.named_parameters():
            param.requires_grad = False

    def unfreeze_base(self):
        # Compute the gradients for net feature
        for _, param in self.net.named_parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)
        return x


class AdaptiveConcatPool2d(nn.Module):
    """Technique that involve 2 type of pooling,
        in this case we use AvgPool and MaxPool"""

    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)

        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x):
        # x = torch.reshape(input=x, shape=(32, 2048, 8, 8))
        #print(x.shape)
        #x = self.ap(x)
        #print(x.shape)
        #x = self.mp(x)
        #print(x.shape)
        #return x
        x = torch.cat([self.mp(x), self.ap(x)], 1)
        #print(x.shape)
        return x


class AMLResnet50_FastAI(nn.Module):

    """
    We emulate the FASTAI Resnet 50
    """

    def __init__(self, out_dim:int):

        super().__init__()

        # New weights with accuracy 80.858%
        self.net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        self.freeze_base()

        # Take the input of the fully connected layer of effnet
        in_dim = self.net.fc.in_features

        # Disable efficient net b7 classifier
        fc = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.5),
            nn.Linear(4096,512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512,out_dim),
        )

        self.net.fc = fc
        self.net.avgpool = AdaptiveConcatPool2d(1)


        self.transforms = transforms.Compose([
            transforms.Resize(232),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])


    def freeze_base(self):
        # Don't compute the gradients for net feature
        for _, param in self.net.named_parameters():
            param.requires_grad = False


    def forward(self, x):
        #print('forward', x.shape)
        x = self.net(x)
        #print('forward', x.shape)
        return x


class AMLResnet101(nn.Module):
    """
    Base on Resnet101
    url: https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet101.html#torchvision.models.ResNet101_Weights
    """

    def __init__(self, out_dim:int):

        super().__init__()

        self.net = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)

        # Take the input of the fully connected layer of effnet
        in_dim = self.net.fc.in_features

        # Noop operation
        self.net.fc = nn.Identity()

        self.freeze_base()

        # Disable efficient net b7 classifier
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(in_dim),
            nn.Dropout(0.5),
            nn.Linear(in_dim,512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512,out_dim),
        )

        self.transforms = transforms.Compose([
            transforms.Resize(232),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])


    def freeze_base(self):
        # Don't compute the gradients for net feature
        for _, param in self.net.named_parameters():
            param.requires_grad = False


    def unfreeze_base(self):
        # Don't compute the gradients for net feature
        for _, param in self.net.named_parameters():
            param.requires_grad = True


    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)
        return x


class AMLEfficientNetB4(nn.Module):
    """Base on efficientnet_b4"""

    def __init__(self, out_dim: int):

        super().__init__()

        self.net = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)

        # Take the input of the fully connected layer of effnet
        in_dim = self.net.classifier[1].in_features

        # Noop operation
        self.net.classifier = nn.Identity()

        # Freeze layers
        self.freeze_base()

        self.classifier = nn.Sequential(nn.Dropout(p=0.4, inplace=True),
                                        nn.Linear(in_features=in_dim, out_features=out_dim))


        self.transforms = transforms.Compose([
            transforms.Resize(384),
            transforms.CenterCrop(380),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        x = self.net(x)
        x = self.classifier(x)
        return x

    def freeze_base(self):
        # Don't compute the gradients for net feature
        for _, param in self.net.named_parameters():
            param.requires_grad = False


    def unfreeze_base(self):
        # Don't compute the gradients for net feature
        for _, param in self.net.named_parameters():
            param.requires_grad = True


class AMLEfficientNet_V2_S(nn.Module):
    """
    Base model: https://pytorch.org/vision/master/models/generated/torchvision.models.efficientnet_v2_s.html#torchvision.models.EfficientNet_V2_S_Weights
    """

    def __init__(self, out_dim: int):

        super().__init__()

        self.net = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights)

        # Take the input of the fully connected layer of effnet
        in_dim = self.net.classifier[1].in_features

        # Noop operation
        self.net.classifier = nn.Identity()

        # Freeze layers
        self.freeze_base()

        self.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(in_features=in_dim, out_features=out_dim))


        self.transforms = EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()

    def forward(self, x):
        x = self.net(x)
        x = self.classifier(x)
        return x

    def freeze_base(self):
        # Don't compute the gradients for net feature
        for _, param in self.net.named_parameters():
            param.requires_grad = False


    def unfreeze_base(self):
        # Don't compute the gradients for net feature
        for _, param in self.net.named_parameters():
            param.requires_grad = True
