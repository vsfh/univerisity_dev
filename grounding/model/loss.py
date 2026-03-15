# -*- coding:utf8 -*-

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import bbox_iou, xyxy2xywh

def adjust_learning_rate(args, optimizer, i_iter):

    lr = args.lr*((0.1)**(i_iter//10))

    print(("lr", lr))
    for param_idx, param in enumerate(optimizer.param_groups):
        param['lr'] = lr

# the shape of the target is (batch_size, anchor_count, 5, grid_wh, grid_wh)
def yolo_loss(predictions, gt_bboxes, anchors_full, best_anchor_gi_gj, image_wh):
    batch_size, grid_stride = predictions.shape[0], image_wh // predictions.shape[3]
    best_anchor, gi, gj = best_anchor_gi_gj[:, 0], best_anchor_gi_gj[:, 1], best_anchor_gi_gj[:, 2]
    scaled_anchors = anchors_full / grid_stride
    mseloss = torch.nn.MSELoss(reduction='mean')
    celoss_confidence = torch.nn.CrossEntropyLoss(reduction='mean')
    #celoss_cls = torch.nn.CrossEntropyLoss(size_average=True)

    selected_predictions = predictions[range(batch_size), best_anchor, :, gj, gi]

    #---bbox loss---
    pred_bboxes = torch.zeros_like(gt_bboxes)
    pred_bboxes[:, 0:2] = selected_predictions[:, 0:2].sigmoid()
    pred_bboxes[:, 2:4] = selected_predictions[:, 2:4]
    
    loss_x = mseloss(pred_bboxes[:,0], gt_bboxes[:,0])
    loss_y = mseloss(pred_bboxes[:,1], gt_bboxes[:,1])
    loss_w = mseloss(pred_bboxes[:,2], gt_bboxes[:,2])
    loss_h = mseloss(pred_bboxes[:,3], gt_bboxes[:,3])

    loss_bbox = loss_x + loss_y + loss_w + loss_h

    #---confidence loss---
    pred_confidences = predictions[:,:,4,:,:]
    gt_confidences = torch.zeros_like(pred_confidences)
    gt_confidences[range(batch_size), best_anchor, gj, gi] = 1
    pred_confidences, gt_confidences = pred_confidences.reshape(batch_size, -1), \
                    gt_confidences.reshape(batch_size, -1)
    loss_confidence = celoss_confidence(pred_confidences, gt_confidences.max(1)[1])

    return loss_bbox, loss_confidence

#target_coord:batch_size, 5
def build_target(ori_gt_bboxes, anchors_full, image_wh, grid_wh):
    #the default value of coord_dim is 5
    batch_size, coord_dim, grid_stride, anchor_count = ori_gt_bboxes.shape[0], ori_gt_bboxes.shape[1], image_wh//grid_wh, anchors_full.shape[0]
    
    gt_bboxes = xyxy2xywh(ori_gt_bboxes)
    gt_bboxes = (gt_bboxes/image_wh) * grid_wh
    scaled_anchors = anchors_full/grid_stride

    gxy = gt_bboxes[:, 0:2]
    gwh = gt_bboxes[:, 2:4]
    gij = gxy.long()

    #get the best anchor for each target bbox
    gt_bboxes_tmp, scaled_anchors_tmp = torch.zeros_like(gt_bboxes), torch.zeros((anchor_count, coord_dim), device=gt_bboxes.device)
    gt_bboxes_tmp[:, 2:4] = gwh
    gt_bboxes_tmp = gt_bboxes_tmp.unsqueeze(1).repeat(1, anchor_count, 1).view(-1, coord_dim)
    scaled_anchors_tmp[:, 2:4] = scaled_anchors
    scaled_anchors_tmp = scaled_anchors_tmp.unsqueeze(0).repeat(batch_size, 1, 1).view(-1, coord_dim)
    anchor_ious = bbox_iou(gt_bboxes_tmp, scaled_anchors_tmp).view(batch_size, -1)
    best_anchor=torch.argmax(anchor_ious, dim=1)
    
    twh = torch.log(gwh / scaled_anchors[best_anchor] + 1e-16)
    #print((gxy.dtype, gij.dtype, twh.dtype, gwh.dtype, scaled_anchors.dtype, 'inner'))
    #print((gxy.shape, gij.shape, twh.shape, gwh.shape), flush=True)
    #print(('gxy,gij,twh', gxy, gij, twh), flush=True)
    return torch.cat((gxy - gij, twh), 1), torch.cat((best_anchor.unsqueeze(1), gij), 1)

class MaskDiceLoss(nn.Module):
    def __init__(self):
        super(MaskDiceLoss, self).__init__()

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1)) # b h w -> b 1 h w
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, net_output, target, weight=None, sigmoid=False):
        if sigmoid:
            net_output = torch.sigmoid(net_output) # b 1 h w
        assert net_output.size() == target.size(), 'predict {} & target {} shape do not match'.format(net_output.size(), target.size())
        dice_loss = self._dice_loss(net_output[:, 0], target[:, 0])
        return dice_loss

class Mask_DC_and_BCE_lossV2(nn.Module):
    def __init__(self, pos_weight, dice_weight=0.8):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!
        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(Mask_DC_and_BCE_lossV2, self).__init__()

        self.ce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dc = MaskDiceLoss()
        self.dice_weight = dice_weight

    def forward(self, net_output, target):
        low_res_logits = net_output
        if len(target.shape) == 5:
            target = target.view(-1, target.shape[2], target.shape[3], target.shape[4])
            low_res_logits = low_res_logits.view(-1, low_res_logits.shape[2], low_res_logits.shape[3], low_res_logits.shape[4])
        loss_ce = self.ce(low_res_logits, target)
        loss_dice = self.dc(low_res_logits, target, sigmoid=True)
        loss = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice
        return loss