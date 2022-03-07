'''
RAPiD's full architecture assembling
Source: "RAPiD: Rotation-Aware People Detection in Overhead Fisheye Images"
'''
import cv2
import numpy as np
import torch
import torch.nn as nn
import onnxruntime
import torchvision.transforms.functional as transforms

from PIL import Image
#from detectron2.layers.nms import nms_rotated
from .backbones import Darknet53, YOLOBranch

class RAPiD(nn.Module):

    def __init__(self, inference=False):
        super(RAPiD, self).__init__()

        self.config = {
            "name": "RAPiD",
            "image_size": 1024,
            "conf_thresh": 0.3,
            "nms_thresh": 0.3,
            "top_k": 500
        }

        if inference:
            self.session = onnxruntime.InferenceSession(f"./pretrained/{self.config['name']}.onnx")
        else:
            self.backbone = Darknet53()
            backbone_pretrained_path = './backbones_weights/dark53_imgnet_checkpoint.pth'
            pretrained = torch.load(backbone_pretrained_path)
            self.load_state_dict(pretrained)
    
            channel_S, channel_M, channel_L = 256, 512, 1024
    
            self.branch_L = YOLOBranch(channel_L, 18)
            self.branch_M = YOLOBranch(channel_M, 18, prev_channels=(channel_L//2, channel_M//2))
            self.branch_S = YOLOBranch(channel_S, 18, prev_channels=(channel_M//2, channel_S//2))

    def forward(self, x):
        small, medium, large = self.backbone(x)
        detection_L, feature_L = self.branch_L(large, prev_feature=None)
        detection_M, feature_M = self.branch_M(medium, prev_feature=feature_L)
        detection_S, _ = self.branch_S(small, prev_feature=feature_M)

        return (detection_L, detection_M, detection_S)

    ####################################################################################################
    # Inference methods
    ####################################################################################################    
    def infer(self, image):
        input = self._preprocess(image)
        predictions = self._inference(input)
        return self._postprocess(predictions)

    def _preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self._rect2square(image)
        image = transforms.to_tensor(image)
        image = image.unsqueeze(0)
        return image

    def _inference(self, image):
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        predictions = self.session.run([output_name], {input_name: image.numpy()})[0]
        return predictions

    def _postprocess(self, predictions):
        predictions = torch.from_numpy(predictions).squeeze()
        predictions = predictions[predictions[:, 5] >= self.config["conf_thresh"]]
        if len(predictions) > self.config["top_k"]:
            _, idx = torch.topk(predictions[:, 5], k=self.config["top_k"])
            predictions = predictions[idx, :]

        predictions = self._nms(predictions)
        predictions = self._detection2original(predictions)
        labels = ["workers"] * len(predictions)
        return predictions, labels

    def _rect2square(self, image):
        original_h, original_w = image.height, image.width
        resize_scale = self.config["image_size"] / max(original_w, original_h)
        unpad_w, unpad_h = int(original_w * resize_scale), int(original_h * resize_scale)
        image = transforms.resize(image, (unpad_h, unpad_w))

        l = (self.config["image_size"] - unpad_w) // 2
        t = (self.config["image_size"] - unpad_h) // 2
        r = self.config["image_size"] - unpad_w - l
        b = self.config["image_size"] - unpad_h - t

        self.pad_info = (original_w, original_h, l, t, unpad_w, unpad_h)
        image = transforms.pad(image, padding=(l, t, r, b), fill=0)

        return image

    def _nms(self, predictions):
        valid = nms_rotated(predictions[:, :-1], predictions[:, -1], self.config["nms_thresh"])
        selected = torch.index_select(predictions, 0, valid)
        return selected

    def _detection2original(self, boxes):
        ori_w, ori_h, tl_x, tl_y, im_w, im_h = self.pad_info
        boxes[:, 0] = (boxes[:, 0] - tl_x) / im_w * ori_w
        boxes[:, 1] = (boxes[:, 1] - tl_y) / im_h * ori_h
        boxes[:, 2] = boxes[:, 2] / im_w * ori_w
        boxes[:, 3] = boxes[:, 3] / im_h * ori_h
        
        return boxes
