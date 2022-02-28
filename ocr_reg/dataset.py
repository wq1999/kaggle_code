import torch
from torch.utils.data import Dataset
import cv2


def img_loader(img_path):
    image = cv2.imread(img_path)
    image = image[:, :, ::-1]
    return image


def make_dataset(df, alphabet, num_class, num_char):
    img_names = df['filename'].values.tolist()
    labels = df['label'].values.tolist()
    samples = []
    for i in range(len(img_names)):
        img_path = img_names[i]
        target_str = labels[i]
        assert len(target_str) == num_char
        target = []
        for char in target_str:
            vec = [0] * num_class
            vec[alphabet.find(char)] = 1
            target += vec
        samples.append((img_path, target))
    return samples


class CaptchaData(Dataset):
    def __init__(self, data_path, num_class=62, num_char=4,
                 transform=None, target_transform=None, alphabet=None):
        super(Dataset, self).__init__()
        self.data_path = data_path
        self.num_class = num_class
        self.num_char = num_char
        self.transform = transform
        self.target_transform = target_transform
        self.alphabet = alphabet
        self.samples = make_dataset(self.data_path, self.alphabet,
                                    self.num_class, self.num_char)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, target = self.samples[index]
        img = img_loader(img_path)
        if self.transform is not None:
            img = self.transform(image=img)['image']
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, torch.Tensor(target)


class TestData(Dataset):
    def __init__(self, df, transform=None):
        super(Dataset, self).__init__()
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.df['file'][index]
        img = img_loader(img_path)
        if self.transform is not None:
            img = self.transform(image=img)['image']
        return img, img_path
