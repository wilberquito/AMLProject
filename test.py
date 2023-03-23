from modular.models import AMLResnet50
from torchvision.models import resnet50, ResNet50_Weights

model = AMLResnet50(8)

print(model.transforms)
