'''
EfficientNets general architecture
Source: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
'''

import cv2
import numpy as np
import torch
from math import ceil
from PIL.Image import Image
from torch import nn

#################################################################################
# Modules
#################################################################################
class Conv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        super(Conv, self).__init__()
        self.padding = nn.ZeroPad2d(self._get_padding(kernel_size, stride))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, groups=groups, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out")
        self.bn = nn.BatchNorm2d(out_channels, eps=1.e-3, momentum=1.e-2)
        self.act = nn.SiLU()

    def _get_padding(self, kernel_size, stride):
        p = max(kernel_size - stride, 0)
        return [p // 2, p - p // 2, p // 2, p - p // 2]

    def forward(self, x):
        return self.act(self.bn(self.conv(self.padding(x))))

class SqueezeExcitation(nn.Module):

    def __init__(self, in_channels, se_ratio=0.25):
        super(SqueezeExcitation, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        # Reduce channels by sqeeuze excitation ratio
        reduced_channels = max(1, int(in_channels * se_ratio))
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y

class DropConnect(nn.Module):

    def __init__(self, drop_rate=0.2):
        super(DropConnect, self).__init__()
        # Probability of keeping the sample
        self.survival_probability = 1 - drop_rate

    def forward(self, x):
        if not self.training or self.survival_probability == 1: return x
        random_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_probability
        return torch.div(x, self.survival_probability) * random_tensor

class MBConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBConv, self).__init__()
        # Boolean to determine whether to use skip connections
        self.residual = in_channels == out_channels and stride == 1
        expanded_channels = in_channels * expand_ratio
        self.expand = in_channels != expanded_channels
        if self.expand:
            self.pointwise_expand = Conv(in_channels, expanded_channels, kernel_size=1)
        self.depthwise = Conv(
            expanded_channels,
         	expanded_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=expanded_channels
        )
        self.squeeze_excitation = SqueezeExcitation(expanded_channels)
        self.pointwise_reduce = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1.e-3, momentum=1.e-2)
        )
        self.drop_connect = DropConnect()

    def forward(self, x):
        residual = x
        if self.expand:
            x = self.pointwise_expand(x)
        x = self.depthwise(x)
        x = self.squeeze_excitation(x)
        x = self.pointwise_reduce(x)
        if self.residual:
            x = self.drop_connect(x)
            x = x + residual
        return x

class FusedMBConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(FusedMBConv, self).__init__()
        # Boolean to determine whether to use skip connections
        self.residual = in_channels == out_channels and stride == 1
        expanded_channels = in_channels * expand_ratio
        self.expansion = Conv(
            in_channels,
            expanded_channels,
            kernel_size=kernel_size,
            stride=stride
        )
        self.squeeze_excitation = SqueezeExcitation(expanded_channels)
        self.pointwise_reduce = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1.e-3, momentum=1.e-2)
        )
        self.drop_connect = DropConnect()

    def forward(self, x):
        residual = x
        x = self.expansion(x)
        x = self.squeeze_excitation(x)
        x = self.pointwise_reduce(x)
        if self.residual:
            x = self.drop_connect(x)
            x = x + residual
        return x

####################################################################################################
# Binary Image Classifier (V1)
####################################################################################################
class EfficientNet(nn.Module):

    def __init__(self, n_classes=2):
        super(EfficientNet, self).__init__()
        self.n_classes = n_classes

        # Compute width and depth factor from phi
        self.alpha = 1.2
        self.beta = 1.1
        depth_factor = self.alpha ** self.config["phi"]
        width_factor = self.beta ** self.config["phi"]

        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        # Number of output channels of the last hidden layer
        last_out_channels = self._rescale_channels(1280, width_factor)
        self._create_layers(depth_factor, width_factor, last_out_channels)
        self.head = nn.Sequential(
            nn.Dropout(self.config["dropout_rate"]),
            nn.Linear(last_out_channels, self.n_classes)
        )

    def _create_layers(self, depth_factor, width_factor, last_out_channels):
        # Stage 1
        in_channels = self._rescale_channels(32, width_factor)
        layers = [Conv(3, in_channels, kernel_size=3, stride=2)]
        for expand_ratio, kernel_size, stride, n_channels, n_layers in [
            [1, 3, 1, 16, 1],  # Stage 2
            [6, 3, 2, 24, 2],  # Stage 3
            [6, 5, 2, 40, 2],  # Stage 4
            [6, 3, 2, 80, 3],  # Stage 5
            [6, 5, 1, 112, 3], # Stage 6
            [6, 5, 2, 192, 4], # Stage 7
            [6, 3, 1, 320, 1]  # Stage 8
        ]:
            # Force out to be multiple of 4 for squeeze excitation
            out_channels = self._rescale_channels(n_channels, width_factor)
            n_layers = int(ceil(n_layers * depth_factor))
            for i in range(n_layers):
                stride = stride if i == 0 else 1
                layers.append(
                    MBConv(
                        in_channels, out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        expand_ratio=expand_ratio
                ))
                in_channels = out_channels
        # Stage 9
        layers.append(Conv(in_channels, last_out_channels, kernel_size=1))
        self.layers = nn.Sequential(*layers)

    def _rescale_channels(self, channels, width_factor):
        # Depth divisor: 8, ensure channels is divisible by 4 for squeeze excitation
        new_channels = channels * width_factor
        new_channels = int(new_channels + 4) // 8 * 8
        new_channels = max(new_channels, 8)
        if new_channels < 0.9 * channels * width_factor:
            new_channels += 8
        return int(new_channels)

    def forward(self, x):
        output = self.layers(x)
        output = self.adaptive_pool(output)
        output = output.view(output.shape[0], -1)
        output = self.head(output)
        return output

class EfficientNetVariation(EfficientNet):

    phis = [0, 0.5, 2, 2, 3, 4, 5, 6]
    image_sizes = [224, 240, 260, 300, 380, 456, 528, 600]
    dropout_rates = [0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5]

    def __init__(self, type=None, variation=0, inference=False):
        super(EfficientNetVariation, self).__init__()
        self.config = {
            "name": f"EfficientNetB{variation}",
            "phi": phis[variation],
            "image_size": image_sizes[varation],
            "dropout_rate": dropout_rates[variation]
        }
        if inference:
            self.session = onnxruntime.InferenceSession(f"./pretrained/{self.config['name']}_{type}.onnx")
            self.classes = ["no", "yes"]

    ####################################################################################################
    # Inference methods
    ####################################################################################################
    def infer(self, image):
        input = self._preprocess(image)
        predictions = self._inference(input)
        return self._postprocess(predictions)

    def _preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.config["image_size"], self.config["image_size"]))
        image = image.astype(np.float32)
        image = image / 255
        image = image - np.array([0.485, 0.456, 0.406])
        image = image / np.array([0.229, 0.224, 0.225])
        image = np.expand_dims(image.transpose(2, 0, 1), axis=0)
        return image

    def _inference(self, image):
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        predictions = self.session.run([output_name], {input_name: image})[0]
        return predictions

    def _postprocess(self, predictions):
        predictions = predictions[0]
        return self.classes[predictions.argmax()]

####################################################################################################
# Binary Image Classifier (V2)
####################################################################################################
class EfficientNetV2S(nn.Module):

    def __init__(self, n_classes=2, width_factor=1., depth_factor=1., inference=False, type=None):
        super(EfficientNetV2S, self).__init__()

        self.config = {
            "name": "EfficientNetV2S",
            "image_size": 224,
            "dropout_rate": 0.2
        }

        if inference:
            self.session = onnxruntime.InferenceSession(f"./pretrained/{self.config['name']}_{type}.onnx")
            self.classes = ["no", "yes"]
        else:
            self.n_classes = n_classes
    
            self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
            # Number of output channels of the last hidden layer
            last_out_channels = self._rescale_channels(1280, width_factor)
            self._create_layers(depth_factor, width_factor, last_out_channels)
            self.head = nn.Sequential(
                nn.Dropout(self.config["dropout_rate"]),
                nn.Linear(last_out_channels, self.n_classes)
            )

    def _create_layers(self, depth_factor, width_factor, last_out_channels):
        # Stage 0
        in_channels = self._rescale_channels(24, width_factor)
        layers = [Conv(3, in_channels, kernel_size=3, stride=2)]
        # Stage 1-6
        for expand_ratio, kernel_size, stride, n_channels, n_layers, is_fused in [
            [1, 3, 1,  24,  2, 1],  # Stage 1
            [4, 3, 2,  48,  4, 1],  # Stage 2
            [4, 3, 2,  64,  4, 1],  # Stage 3
            [4, 3, 2, 128,  6, 0],  # Stage 4
            [6, 3, 1, 160,  9, 0],  # Stage 5
            [6, 3, 2, 256, 15, 0]   # Stage 6
        ]:
            # Force out to be multiple of 4 for squeeze excitation
            out_channels = self._rescale_channels(n_channels, width_factor)
            n_layers = int(ceil(n_layers * depth_factor))
            block = FusedMBConv if is_fused else MBConv
            for i in range(n_layers):
                stride = stride if i == 0 else 1
                layers.append(
                    block(
                        in_channels, out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        expand_ratio=expand_ratio
                    )
                )
                in_channels = out_channels
        # Stage 7
        layers.append(Conv(in_channels, last_out_channels, kernel_size=1))
        self.layers = nn.Sequential(*layers)

    def _rescale_channels(self, channels, width_factor):
        # Depth divisor: 8, ensure channels is divisible by 4 for squeeze excitation
        new_channels = channels * width_factor
        new_channels = int(new_channels + 4) // 8 * 8
        new_channels = max(new_channels, 8)
        if new_channels < 0.9 * channels * width_factor:
            new_channels += 8
        return int(new_channels)

    def forward(self, x):
        output = self.layers(x)
        output = self.adaptive_pool(output)
        output = output.view(output.shape[0], -1)
        output = self.head(output)
        return output

    ####################################################################################################
    # Inference methods
    ####################################################################################################
    def infer(self, image):
        input = self._preprocess(image)
        predictions = self._inference(input)
        return self._postprocess(predictions)

    def _preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.config["image_size"], self.config["image_size"]))
        image = image.astype(np.float32)
        image = image / 255
        image = image - np.array([0.485, 0.456, 0.406])
        image = image / np.array([0.229, 0.224, 0.225])
        image = np.expand_dims(image.transpose(2, 0, 1), axis=0)
        return image

    def _inference(self, image):
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        predictions = self.session.run([output_name], {input_name: image})[0]
        return predictions

    def _postprocess(self, predictions):
        predictions = predictions[0]
        return self.classes[predictions.argmax()]
