from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

def get_img_size(model_id: str):

    models = {
        'amlresnet50': 232
    }
    
    return models[model_id]

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
