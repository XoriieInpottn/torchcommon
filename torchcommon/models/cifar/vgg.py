#!/usr/bin/env python3

import torch.nn as nn

__all__ = [
    'vgg11',
    'vgg13',
    'vgg16',
    'vgg19'
]

CFG = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes):
        super(VGG, self).__init__()
        self.features = VGG._make_layers(CFG[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)

        # revise the original "flatten" to "mean" to adapt to different image size
        # out = out.view(out.size(0), -1)
        out = out.mean((2, 3))

        out = self.classifier(out)
        return out

    @staticmethod
    def _make_layers(cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, (3, 3), (1, 1), (1, 1)),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def vgg11(num_classes):
    return VGG('VGG11', num_classes)


def vgg13(num_classes):
    return VGG('VGG13', num_classes)


def vgg16(num_classes):
    return VGG('VGG16', num_classes)


def vgg19(num_classes):
    return VGG('VGG19', num_classes)
