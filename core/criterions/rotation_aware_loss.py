'''
Total loss of our detector for detecting bounding boxes' center, size and angle of rotation
Source: "RAPiD: Rotation-Aware People Detection in Overhead Fisheye Images"
'''
import torch
import numpy as np

from torch import nn

from .angle_losses import PeriodL1, PeriodL2
from ..utils.detection import iou_mask, iou_rle

class RotationAwareLoss(nn.Module):

    def __init__(self, image_size, loss_angle="L1"):
        super(RotationAwareLoss, self).__init__()
        self.image_size = image_size
        self.anchors_all = [
            [18.7807, 33.4659], [28.8912, 61.7536], [48.6849, 68.3897],
            [45.0668, 101.4673], [63.0952, 113.5382], [81.3909, 134.4554],
            [91.7364, 144.9949], [137.5189, 178.4791], [194.4429, 250.7985]
        ]
        self.anchors_all = torch.Tensor(self.anchors_all).float()

        self.indices = [[6,7,8], [3,4,5], [0,1,2]]
        self.indices = [torch.Tensor(i).long() for i in self.indices]
        # all anchors, (0, 0, w, h), used for calculating IoU
        self.anchors_00wha = torch.zeros(len(self.anchors_all), 5)
        self.anchors_00wha[:, 2:4] = self.anchors_all # image space, degree

        self.ignore_threshold = 0.6

        if loss_angle == 'L1':
            self.angle_loss = PeriodL1(reduction='sum')
        elif loss_angle == 'L2':
            self.angle_loss = PeriodL2(reduction='sum')
        else:
            self.angle_loss = None

        self.losses_str = []

    def forward(self, predictions, targets):
        detection_L, detection_M, detection_S = predictions
        # process the boxes, and calculate loss if there is gt
        loss_L = self._calculate_loss(detection_L, self.indices[0], self.image_size, targets)
        loss_M = self._calculate_loss(detection_M, self.indices[1], self.image_size, targets)
        loss_S = self._calculate_loss(detection_S, self.indices[2], self.image_size, targets)
        return loss_L + loss_M + loss_S

    def _calculate_loss(self, prediction, anchor_indices, image_size, labels=None):
        device = prediction.device
        nB, _, nH, nW = prediction.shape[0:4] # batch size and feature resolution 
        nA = len(anchor_indices) # number of anchors
        nCH = 6 # number of channels, (x, y, w, h, angle, conf)

        prediction = prediction.view(nB, nA, nCH, nH, nW)
        # convert to shape(nB, nA, nH, nW, nCH), meaning (nB x nA x nH x nW) objects
        prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()
        
        # sigmoid activation for center: (x, y)
        xy_offset = torch.sigmoid(prediction[..., 0:2]) # x,y
        # linear activation for size: (w, h)
        wh_scale = prediction[..., 2:4]
        # sigmoid activation for angle
        angle = torch.sigmoid(prediction[..., 4])
        # sigmoid activation for objectness confidence
        conf = torch.sigmoid(prediction[..., 5])
        # now xy is the offsets, wh are is scaling factor, and angle is normalized between 0~1.
        
        # calculate pred - xywh obj cls
        x_shift = torch.arange(nW, dtype=torch.float, device=device).view(1, 1, 1, nW)
        y_shift = torch.arange(nH, dtype=torch.float, device=device).view(1, 1, nH, 1)
        
        # anchors are not normalized
        anchors = self.anchors_all[anchor_indices].clone().to(device=device)
        anchor_w = anchors[:, 0].view(1, nA, 1, 1) 
        anchor_h = anchors[:, 1].view(1, nA, 1, 1) 

        prediction_final = torch.empty(nB, nA, nH, nW, 6, device=device)
        prediction_final[..., 0] = (xy_offset[..., 0] + x_shift) / nW # 0-1 space
        prediction_final[..., 1] = (xy_offset[..., 1] + y_shift) / nH # 0-1 space
        prediction_final[..., 2] = torch.exp(wh_scale[..., 0]) * anchor_w # image space
        prediction_final[..., 3] = torch.exp(wh_scale[..., 1]) * anchor_h # image space
        prediction_final[..., 4] = angle * 360 - 180 # degree
        prediction_final[..., 5] = conf
        
        # convert final predictions to be normalized
        prediction_final[..., 2] /= image_size
        prediction_final[..., 3] /= image_size
        # force the normalized w and h to be <= 1
        prediction_final[..., 0:4].clamp_(min=0, max=1)

        pred_boxes = prediction_final[..., :5].detach()
        pred_confs = prediction_final[..., 5].detach()

        # target assignment
        object_mask = torch.zeros(nB, nA, nH, nW, dtype=torch.bool, device=device)
        penalty_mask = torch.ones(nB, nA, nH, nW, dtype=torch.bool, device=device)
        target = torch.zeros(nB, nA, nH, nW, nCH, dtype=torch.float, device=device)

        labels = labels.detach()
        n_labels = (labels[:, :, 0:4].sum(dim=2) > 0).sum(dim=1) # number of objects
        labels = labels.to(device=device)

        tx_all, ty_all = labels[:, :, 0] * nW, labels[:, :, 1] * nH # 0-nW, 0-nH
        tw_all, th_all = labels[:, :, 2], labels[:, :, 3] # normalized 0-1
        ta_all = labels[:, :, 4] # degree, 0-max_angle

        ti_all = tx_all.long()
        tj_all = ty_all.long()

        norm_anchor_size = anchors[:, 0:2] / image_size # normalized
        norm_anchors_00wha = self.anchors_00wha.clone().to(device=device)
        norm_anchors_00wha[:, 2:4] /= image_size # normalized

        # traverse all images in a batch
        valid_gt_num = 0
        for b in range(nB):
            n = int(n_labels[b]) # number of ground truths in b'th image
            # no ground truth
            if n == 0:
                continue
            gt_boxes = torch.zeros(n, 5, device=device)
            gt_boxes[:, 2] = tw_all[b, :n] # normalized 0-1
            gt_boxes[:, 3] = th_all[b, :n] # normalized 0-1
            gt_boxes[:, 4] = 0
            
            # calculate iou between truth and reference anchors
            #anchor_ious = iou_mask(gt_boxes, norm_anchors_00wha, mask_size=64, is_degree=True)
            anchor_ious = iou_rle(gt_boxes, norm_anchors_00wha, image_size=image_size, is_degree=True)

            best_n_all = torch.argmax(anchor_ious, dim=1)
            best_n = best_n_all % len(anchor_indices)

            valid_mask = torch.zeros(n, dtype=torch.bool, device=device)
            for i in anchor_indices.to(device=device):
                valid_mask = (valid_mask | (best_n_all == i))
            if sum(valid_mask) == 0:
                # no anchor is responsible for any ground truth
                continue
            else:
                valid_gt_num += sum(valid_mask)

            best_n = best_n[valid_mask]
            truth_i = ti_all[b, :n][valid_mask]
            truth_j = tj_all[b, :n][valid_mask]
            
            gt_boxes[:, 0] = tx_all[b, :n] / nW # normalized 0-1
            gt_boxes[:, 1] = ty_all[b, :n] / nH # normalized 0-1
            gt_boxes[:, 4] = ta_all[b, :n] # degree
            
            # gt_boxes e.g. shape(11,4)
            selected_idx = pred_confs[b] > 0.001
            selected = pred_boxes[b][selected_idx]
            if len(selected) < 2000 and len(selected) > 0:
                # ignore some predicted boxes who have high overlap with any groundtruth
                pred_ious = iou_rle(selected.view(-1, 5), gt_boxes, image_size=image_size, is_degree=True)
                #pred_ious = iou_mask(selected.view(-1, 5), gt_boxes, mask_size=32, is_degree=True)
                pred_best_iou, _ = pred_ious.max(dim=1)
                # set mask to zero (ignore) if the pred BB has a large IoU with any gt BB
                penalty_mask[b, selected_idx] = ~(pred_best_iou > self.ignore_threshold)

            penalty_mask[b, best_n, truth_j, truth_i] = 1
            object_mask[b, best_n, truth_j, truth_i] = 1
            target[b, best_n, truth_j, truth_i, 0] = tx_all[b, :n][valid_mask] - tx_all[b, :n][valid_mask].floor()
            target[b, best_n, truth_j, truth_i, 1] = ty_all[b, :n][valid_mask] - ty_all[b, :n][valid_mask].floor()
            target[b, best_n, truth_j, truth_i, 2] = torch.log(tw_all[b, :n][valid_mask]/norm_anchor_size[best_n, 0] + 1e-16)
            target[b, best_n, truth_j, truth_i, 3] = torch.log(th_all[b, :n][valid_mask]/norm_anchor_size[best_n, 1] + 1e-16)
            # use radian when calculating loss
            target[b, best_n, truth_j, truth_i, 4] = gt_boxes[:, 4][valid_mask] / 180 * np.pi
            target[b, best_n, truth_j, truth_i, 5] = 1 # objectness confidence
        
        loss_xy = nn.BCELoss(reduction='sum')(xy_offset[object_mask], target[..., 0:2][object_mask])
        
        wh_pred = wh_scale[object_mask]
        wh_target = target[..., 2:4][object_mask]
        loss_wh = nn.MSELoss(reduction='sum')(wh_pred, wh_target)

        angle_pred = angle[object_mask] * 2 * np.pi - np.pi

        loss_angle = self.angle_loss(angle_pred, target[..., 4][object_mask])
        loss_object = nn.BCELoss(reduction='sum')(conf[penalty_mask], target[..., 5][penalty_mask])

        loss = loss_xy + 0.5 * loss_wh + loss_angle + loss_object
        n_gt = valid_gt_num + 1e-16
        
        self.losses_str.append(f'level_{nH}x{nW} total {int(n_gt)} objects: ' \
                               f'xy/gt {loss_xy/n_gt:.3f}, wh/gt {loss_wh/n_gt:.3f}' \
                               f', angle/gt {loss_angle/n_gt:.3f}, conf {loss_object:.3f}')

        return loss
