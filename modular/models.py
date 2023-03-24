from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
import torch as torch
import torch.nn as nn

class AMLResnet50(nn.Module):

    def __init__(self, out_dim:int):

        super().__init__()

        # New weights with accuracy 80.858%
        self.net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Take the input of the fully connected layer of effnet
        in_dim = self.net.fc.in_features

        # Disable efficient net b7 classifier
        self.net.fc = nn.Identity()

        # Declare the fully connected layer
        self.fc = nn.Linear(in_dim, out_dim)

        # Definition of multiple dropout
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])

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
        # Apply multiple dropouts

        x = self.net(x).squeeze(-1).squeeze(-1)

        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.fc(dropout(x))
            else:
                out += self.fc(dropout(x))

        # Compute the average of dropouts
        out /= len(self.dropouts)

        return out

class AMLResnet50_V2(nn.Module):

    def __init__(self, out_dim:int):

        super().__init__()

        # New weights with accuracy 80.858%
        self.net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Take the input of the fully connected layer of effnet
        in_dim = self.net.fc.in_features

        #intermediate dimensions
        dim_range = in_dim - out_dim 
        dim_75 = int(out_dim + (dim_range * 0.75))
        dim_50 = int(out_dim + (dim_range * 0.5))
        dim_25 = int(out_dim + (dim_range * 0.25))

        # Disable efficient net b7 classifier
        self.net.fc = nn.Identity()

        # Declare the fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(in_dim,dim_75),
            nn.Linear(dim_75,dim_50),
            nn.Linear(dim_50,dim_25),
            nn.Linear(dim_25,out_dim),
        )

        # Definition of multiple dropout
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])

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
        # Apply multiple dropouts

        x = self.net(x).squeeze(-1).squeeze(-1)

        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.fc(dropout(x))
            else:
                out += self.fc(dropout(x))

        # Compute the average of dropouts
        out /= len(self.dropouts)

        return out

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

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
            AdaptiveConcatPool2d((32, 2048)),
            nn.Flatten(),
            nn.BatchNorm1d(60000),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.BatchNorm1d(60000),
            nn.Dropout(0.5),
            nn.Linear(60000,out_dim),
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

        x = self.net(x).squeeze(-1).squeeze(-1)

        x = self.fc(x)

        return x
    
