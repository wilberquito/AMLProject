import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader

def get_transforms(img_size:int):

    # the training transforms
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        # transforms.RandomRotation(degrees=(30, 70)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    # the validation transforms
    valid_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    return train_transform, valid_transform

def get_dataloader(folder_root, transformer, batch_size=32, suffle=False):

    dataset = torchvision.datasets.ImageFolder(root=folder_root,transform=transformer)

    # training data loaders
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=suffle)

    return dataloader
