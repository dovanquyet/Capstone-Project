'''
Customized augmentation methods for preprocessing dataset used in training the detector.
Current methods include:
    - Random blur
    - Random change HSV values
    - Resize
    - Convert to tensor
Note: Each sample passed to these methods consists of an image and information about all of its groundtruth bounding boxes
'''
import cv2
import itertools
import numpy as np
import torch
from random import random

# Random blur with 3 blurring options available for sample's image
class RandomBlur(object):

    def __init__(self, blur_type="normal", kernel_size=(3,3), p=0.5):
        self.p = p
        self.kernel_size = kernel_size
        self.blur_type = blur_type

    def __call__(self, sample):
        if random() < self.p:
            image, targets = sample["image"], sample["targets"]
            switcher = {
                "normal": cv2.blur(image, self.kernel_size),
                "gaussian": cv2.GaussianBlur(image, self.kernel_size, 0),
                "median": cv2.medianBlur(image, self.kernel_size[0]),
            }
            image = switcher[self.blur_type]
            return { "image": image, "targets": targets }
        else:
            return sample

# Random change HSV values of sample's images
class RandomHSV(object):
    def __init__(self, hgain, sgain, vgain, p=0.5):
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
        self.p = p

    def __call__(self, sample):
        if random() < self.p:
            image, targets = sample["image"], sample["targets"]

            # Convert BGR to HSV
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            image[:, :, 0] = (self.hgain * image[:, :, 0]).astype(int) # Changes the H value
            image[:, :, 1] = (self.sgain * image[:, :, 1]).astype(int) # Changes the S value
            image[:, :, 2] = (self.vgain * image[:, :, 2]).astype(int) # Changes the V value

            # Convert HSV to BGR
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
            return { "image": image, "targets": targets }
        else:
            return sample

# Resize sample's image
class Resize(object):

    def __init__(self, image_dim):
        self.image_dim = image_dim

    def __call__(self, sample):
        image, targets = sample["image"], sample["targets"]
        image = cv2.resize(image, self.image_dim)
        return { "image": image, "targets": targets }

# Convert sample (image and its bounding boxes) to tensor    
class ToTensor(object):

    def __call__(self, sample):
        image, targets = sample["image"], sample["targets"]
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        targets = torch.from_numpy(targets).float()
        targets_copy = targets.clone()
        targets = torch.zeros(50, 5)
        targets[:targets_copy.shape[0], :] = targets_copy
        return { "image": image, "targets": targets }
