import cv2
from torch.utils.data import Dataset
import torch


class CassavaDataset(Dataset):
    def __init__(self, images_filepaths, targets, transform=None):
        self.images_filepaths = images_filepaths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        image = image[:, :, ::-1]

        if self.transform is not None:
            image = self.transform(image=image)['image']

        label = torch.tensor(self.targets[idx])
        return image, label


class CassavaTestDataset(Dataset):
    def __init__(self, images_filepaths, transform=None):
        self.images_filepaths = images_filepaths
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        image = image[:, :, ::-1]

        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image, image_filepath
