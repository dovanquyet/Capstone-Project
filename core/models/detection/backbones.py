'''
Implementation of RAPiD's components
Source: "RAPiD: Rotation-Aware People Detection in Overhead Fisheye Images"
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

#################################################################################
# Modules
#################################################################################
class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DarkBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels):
        super(DarkBlock, self).__init__()
        self.conv0 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.conv1 = Conv(hidden_channels, in_channels, kernel_size=3, stride=1)

    def forward(self, x):
        residual = x
        x = self.conv0(x)
        x = self.conv1(x)
        return x + residual


class Darknet53(nn.Module):

    def __init__(self):
        super(Darknet53, self).__init__()
        self.layers = nn.ModuleList()
        self._create_layers()

    def _create_layers(self):
        # Conv 1
        self.layers.append(Conv(3, 32, kernel_size=3, stride=1))
        for n_blocks, in_channels, out_channels in [
            [1, 32, 64],    # Downsampled by 2 + Residual Blocks
            [2, 64, 128],   # Downsampled by 4 + Residual Blocks
            [8, 128, 256],  # Downsampled by 8 + Residual Blocks
            [8, 256, 512],  # Downsampled by 16 + Residual Blocks
            [4, 512, 1024], # Downsampled by 32 + Residual Blocks
        ]:
            self.layers.append(Conv(in_channels, out_channels, kernel_size=3, stride=2))
            for _ in range(n_blocks):
                self.layers.append(DarkBlock(out_channels, in_channels))

    def forward(self, x):
        for i in range(0, 15):
            x = self.layers[i](x)
        small = x
        for i in range(15, 24):
            x = self.layers[i](x)
        medium = x
        for i in range(24, 29):
            x = self.layers[i](x)
        large = x

        return small, medium, large

class YOLOBranch(nn.Module):

    def __init__(self, in_channels, out_channels, prev_channels=None):
        super(YOLOBranch, self).__init__()
        if prev_channels:
            self.preprocess = Conv(prev_channels[0], prev_channels[1], kernel_size=1, stride=1)
            in_channels_cat = in_channels + prev_channels[1]
        else:
            self.preprocess = None
            in_channels_cat = in_channels

        self.conv0 = Conv(in_channels_cat, in_channels//2, kernel_size=1, stride=1)
        self.conv1 = Conv(in_channels//2, in_channels, kernel_size=3, stride=1)

        self.conv2 = Conv(in_channels, in_channels//2, kernel_size=1, stride=1)
        self.conv3 = Conv(in_channels//2, in_channels, kernel_size=3, stride=1)

        self.conv4 = Conv(in_channels, in_channels//2, kernel_size=1, stride=1)
        self.conv5 = Conv(in_channels//2, in_channels, kernel_size=3, stride=1)

        self.to_box = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        
    def forward(self, x, prev_feature=None):
        if prev_feature is not None:
            prev_feature = self.preprocess(prev_feature)
            prev_feature = F.interpolate(prev_feature, scale_factor=2, mode='nearest')
            x = torch.cat((prev_feature, x), dim=1)
        
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        feature = self.conv4(x)
        x = self.conv5(feature)
        detection = self.to_box(x)
        return detection, feature
