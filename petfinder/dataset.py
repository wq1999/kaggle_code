from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch


class CuteDataset(Dataset):
    def __init__(self, images_filepaths, targets, transform=None):
        self.images_filepaths = images_filepaths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]

        with open(image_filepath, 'rb') as f:
            image = Image.open(f)
            image_rgb = image.convert('RGB')
        image = np.array(image_rgb)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        image = image / 255  # convert to 0-1
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        image = torch.tensor(image, dtype = torch.float)
        label = torch.tensor(self.targets[idx]).float()
        return image, label
