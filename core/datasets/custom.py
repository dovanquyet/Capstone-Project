'''
Customized dataset reader used to pass dataset into training process of the detector
'''
import cv2
import os
import torch
import json
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict

# Normalize the bounding boxes
class TargetTransform(object):

    def __call__(self, sample):
        image, targets = sample["image"], sample["targets"]
        width, height = image.size
        # Normalized x, y, w, h
        bboxs = targets[:, :4].copy()
        bboxs[:, 0::2] /= width
        bboxs[:, 1::2] /= height
        targets[:, :4] = bboxs
        return { "image": image, "targets": targets }

# Customized dataset reader
class CustomizedDataset(Dataset):

    def __init__(self, json_path, transform=None, target_transform=TargetTransform()):
        self.root_dir = os.path.dirname(json_path)
        self.json_path = json_path # path to annotations file
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = json.load(open(json_path, "r"))
        self.classes = ["person"]
        self._create_index()

    def __len__(self):
        return len(self.image_paths)
    
    # get sample (image and its bounding boxes) by index
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if image.mode == 'L':
            image = np.array(image)
            image = np.repeat(np.expand_dims(image, 2), 3, axis=2)
            image = Image.fromarray(image)
        
        bboxs = self.annotations[idx]

        targets = np.zeros((0, 5))
        if len(bboxs) != 0:
            for bbox in bboxs:
                target = np.zeros((1, 5))
                target[0, 0] = bbox[0] # x
                target[0, 1] = bbox[1] # y
                target[0, 2] = bbox[2] # w
                target[0, 3] = bbox[3] # h
                target[0, 4] = bbox[4] # angle
                targets = np.append(targets, target, axis=0)

        sample = { "image": image, "targets": targets }
        
        # apply transformation to bounding boxes
        if self.target_transform is not None:
            sample = self.target_transform(sample)
        
        # apply transformation to image
        if self.transform is not None:
            image = np.array(image)
            sample["image"] = image
            sample = self.transform(sample)
        return sample
    
    # stack sample in given batch
    @staticmethod
    def collate_fn(batch):
        targets_list = []
        images = []
        for i, sample in enumerate(batch):
            image, targets = sample["image"], sample["targets"]
            images.append(image)
            targets_list.append(targets)
        return (torch.stack(images, 0), torch.stack(targets_list, 0))

    def _create_index(self):
        self.image_paths = {}
        self.annotations = defaultdict(list)
        
        k2k = {}
        for idx, image in enumerate(self.dataset["images"]):
            image_path = os.path.join(self.root_dir, "images", image["file_name"])
            self.image_paths[idx] = image_path
            k2k[image["id"]] = idx

        for annotation in self.dataset["annotations"]:
            x, y, w, h, angle = annotation["bbox"]
            if w == h: h += 1 # force w < h
            if angle == 90: angle = -90 
            bbox = [x, y, w, h, angle]
            self.annotations[k2k[annotation["image_id"]]].append(bbox)

