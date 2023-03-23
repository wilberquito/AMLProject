from modular.models import AMLResnet50, get_img_size
from torchvision.models import resnet50, ResNet50_Weights

image_size = get_img_size('amlresnet50')
model = AMLResnet50(image_size)

weights = ResNet50_Weights.DEFAULT
print(weights.transforms())
