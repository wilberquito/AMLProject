import torchvision
from torch.utils.data import DataLoader
import os
from PIL import Image
from torch.utils.data import Dataset


def get_dataloader(folder_root, transformer, batch_size=32, suffle=False):
    dataset = torchvision.datasets.ImageFolder(root=folder_root,
                                               transform=transformer)

    # training data loaders
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=suffle)

    return dataloader


class TestDataset(Dataset):
    """
    Dataset for testing porpouse. It just read the images
    into PIL format and returns them
    """

    def __init__(self, root, transform):
        self.root = root
        self.file_names = sorted(os.listdir(root))
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # load the image and label
        img_path = os.path.join(self.root, self.file_names[idx])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img
