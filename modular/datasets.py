import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader

def get_dataloader(folder_root, transformer, batch_size=32, suffle=False):

    dataset = torchvision.datasets.ImageFolder(root=folder_root,transform=transformer)

    # training data loaders
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=suffle)

    return dataloader
