import torch
import torch.nn as nn
import timm


class PetNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        model_name = params['model']
        out_features = params['out_features']
        inp_channels = params['inp_channels']
        pretrained = params['pretrained']
        self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=inp_channels, num_classes=out_features)

    def forward(self, image):
        output = self.model(image)
        return output
