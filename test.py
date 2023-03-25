from modular.models import AMLResnet50
from torchvision.models import resnet50, ResNet50_Weights

net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

print(net.fc)
