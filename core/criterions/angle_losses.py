'''
Periodic L1 and L2 losses for angle prediction
Source: "RAPiD: Rotation-Aware People Detection in Overhead Fisheye Images"
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PeriodL1(nn.Module):
    def __init__(self, reduction='sum'):
        super(PeriodL1, self).__init__()
        self.reduction = reduction

    def forward(self, theta_pred, theta_gt):
        dt = theta_pred - theta_gt
        dt = torch.abs(torch.remainder(dt-np.pi/2, np.pi) - np.pi/2)
        assert (dt >= 0).all()

        if self.reduction == 'sum':
            loss = dt.sum()
        elif self.reduction == 'mean':
            loss = dt.mean()
        elif self.reduction == 'none':
            loss = dt

        return loss


class PeriodL2(nn.Module):
    def __init__(self, reduction='sum'):
        super(PeriodL2, self).__init__()
        self.reduction = reduction

    def forward(self, theta_pred, theta_gt):
        dt = theta_pred - theta_gt
        dt = (torch.remainder(dt-np.pi/2, np.pi) - np.pi/2) ** 2
        assert (dt >= 0).all()

        if self.reduction == 'sum':
            loss = dt.sum()
        elif self.reduction == 'mean':
            loss = dt.mean()
        elif self.reduction == 'none':
            loss = dt

        return loss
