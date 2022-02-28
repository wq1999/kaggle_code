import torch.nn as nn
import timm


class Net(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained, num_classes=n_class)

    def forward(self, x):
        x = self.model(x)
        return x
