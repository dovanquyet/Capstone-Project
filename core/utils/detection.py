'''
Helper functions to calculate Intersection of Unions (IoUs) between bounding boxes
'''

import torch
import os

from pycocotools import mask as maskUtils
from math import pi

# get vertices' coordinate of bounding boxes from their center, size and angle of rotation    
def xywha2vertex(box, stack=True):
    batch = box.shape[0]
    device = box.device

    center = box[:, 0:2]
    w = box[:, 2]
    h = box[:, 3]
    rad = box[:, 4]

    verti = torch.empty((batch, 2), dtype=torch.float32, device=device)
    verti[:, 0] = (h / 2) * torch.sin(rad)
    verti[:, 1] = - (h/2) * torch.cos(rad)

    hori = torch.empty(batch, 2, dtype=torch.float32, device=device)
    hori[:, 0] = (w / 2) * torch.cos(rad)
    hori[:, 1] = (w / 2) * torch.sin(rad)

    tl = center + verti - hori
    tr = center + verti + hori
    br = center - verti + hori
    bl = center - verti - hori

    if not stack:
        return torch.cat([tl, tr, br, bl], dim=1)
    return torch.stack((tl, tr, br, bl), dim=1)

# calculate masks from vertices to help calculate IoUs
def vertex2masks(vertices, mask_size=128):
    device = vertices.device
    batch = vertices.shape[0]
    mh, mw = (mask_size, mask_size) if isinstance(mask_size, int) else mask_size

    gx = torch.linspace(0, 1, steps=mw, device=device).view(1, 1, -1)
    gy = torch.linspace(0, 1, steps=mh, device=device).view(1, -1, 1)

    tl_x = vertices[:, 0, 0].view(-1, 1, 1)
    tl_y = vertices[:, 0, 1].view(-1, 1, 1)
    tr_x = vertices[:, 1, 0].view(-1, 1, 1)
    tr_y = vertices[:, 1, 1].view(-1, 1, 1)
    br_x = vertices[:, 2, 0].view(-1, 1, 1)
    br_y = vertices[:, 2, 1].view(-1, 1, 1)
    bl_x = vertices[:, 3, 0].view(-1, 1, 1)
    bl_y = vertices[:, 3, 1].view(-1, 1, 1)

    mask = (tr_y - tl_y) * gx + (tl_x - tr_x) * gy + tl_y * tr_x - tr_y * tl_x < 0
    mask *= (br_y - tr_y) * gx + (tr_x - br_x) * gy + tr_y * br_x - br_y * tr_x < 0
    mask *= (bl_y - br_y) * gx + (br_x - bl_x) * gy + br_y * bl_x - bl_y * br_x < 0
    mask *= (tl_y - bl_y) * gx + (bl_x - tl_x) * gy + bl_y * tl_x - tl_y * bl_x < 0

    return mask

# use mask method to calculate IoU between corresponding boxes
def iou_pairs_mask(boxes1, boxes2, mask_size=128, is_degree=True):
    device = boxes1.device
    batch = boxes1.shape[0]

    if is_degree:
        boxes1[:, 4] = boxes1[:, 4] * pi / 180
        boxes2[:, 4] = boxes2[:, 4] * pi / 180
    
    # get vertices, [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    vts1 = xywha2vertex(boxes1) 
    vts2 = xywha2vertex(boxes2)

    x = torch.cat((vts1[:, :, 0],vts2[:, :, 0]), dim=1)
    y = torch.cat((vts1[:, :, 1],vts2[:, :, 1]), dim=1)
    
    # select the smallest area that contain two boxes
    xmin, _ = x.min(dim=1)
    xmax, _ = x.max(dim=1)
    ymin, _ = y.min(dim=1)
    ymax, _ = y.max(dim=1)

    area = torch.empty(batch, 1, 2, device=device)
    area[:, 0, 0] = xmax - xmin
    area[:, 0, 1] = ymax - ymin

    topleft = torch.stack((xmin, ymin), dim=1).unsqueeze(dim=1)
    
    # coordinates in original to coordinate in small area
    vts1 = (vts1 - topleft) / area 
    vts2 = (vts2 - topleft) / area

    # calculate two maskes
    mask1 = vertex2masks(vts1, mask_size=mask_size)
    mask2 = vertex2masks(vts2, mask_size=mask_size)

    inter = mask1 * mask2
    union = (mask1 + mask2) > 0

    inter_area = inter.sum(dim=(1, 2), dtype=torch.float)
    union_area = union.sum(dim=(1, 2), dtype=torch.float) + 1e-16

    return  inter_area / union_area

# use mask method to calculate IOU between boxes1 and boxes2
def iou_mask(boxes1, boxes2, mask_size=64, is_degree=True):
    if not (torch.is_tensor(boxes1) and torch.is_tensor(boxes2)):
        boxes1 = torch.from_numpy(boxes1).float()
        boxes2 = torch.from_numpy(boxes2).float()
    
    if boxes1.dim() == 1: boxes1 = boxes1.unsqueeze(0)
    if boxes2.dim() == 1: boxes2 = boxes2.unsqueeze(0)

    num1, num2 = boxes1.shape[0], boxes2.shape[0]
    if num1 == 0 or num2 == 0:
        return torch.Tensor([]).view(num1, num2)

    boxes1 = boxes1.repeat(1, num2).view(-1, 5)
    boxes2 = boxes2.repeat(num1, 1)

    iou_list = iou_pairs_mask(boxes1, boxes2, mask_size=mask_size,
                              is_degree=is_degree)

    iou_matrix = iou_list.view(num1, num2)
    return iou_matrix

# different way of using mask method to calculate IOU between boxes1 and boxes2
def iou_rle(boxes1, boxes2, image_size, is_degree=True):
    if not (torch.is_tensor(boxes1) and torch.is_tensor(boxes2)):
        boxes1 = torch.from_numpy(boxes1).float()
        boxes2 = torch.from_numpy(boxes2).float()

    device = boxes1.device
    boxes1 = boxes1.cpu().clone().detach()
    boxes2 = boxes2.cpu().clone().detach()

    if boxes1.dim() == 1: boxes1 = boxes1.unsqueeze(0)
    if boxes2.dim() == 1: boxes2 = boxes2.unsqueeze(0)
    
    h, w = image_size, image_size

    boxes1[:, 0] *= w
    boxes1[:, 1] *= h
    boxes1[:, 2] *= w
    boxes1[:, 3] *= h

    boxes2[:, 0] *= w
    boxes2[:, 1] *= h
    boxes2[:, 2] *= w
    boxes2[:, 3] *= h

    if is_degree:
        boxes1[:, 4] = boxes1[:, 4] * pi / 180
        boxes2[:, 4] = boxes2[:, 4] * pi / 180

    b1 = xywha2vertex(boxes1, stack=False).tolist()
    b2 = xywha2vertex(boxes2, stack=False).tolist()
    
    # use pycocotools
    b1 = maskUtils.frPyObjects(b1, h, w)
    b2 = maskUtils.frPyObjects(b2, h, w)
    ious = maskUtils.iou(b1, b2, [0 for _ in b2])

    return torch.from_numpy(ious).to(device=device)
