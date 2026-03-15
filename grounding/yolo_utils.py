import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict
from torch import einsum
from einops import rearrange, repeat


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * 4)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0.0, context_dim=None):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )
        self.ff = FeedForward(dim, dropout=dropout)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    def __init__(
        self, in_channels, n_heads, d_head, depth=1, dropout=0.0, context_dim=None
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(
            in_channels, inner_dim, kernel_size=1, stride=1, padding=0
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim
                )
                for _ in range(depth)
            ]
        )

        self.proj_out = zero_module(
            nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        x = self.proj_out(x)
        return x + x_in


IMG_SIZE = 640
ANCHORS = "37,41, 78,84, 96,215, 129,129, 194,82, 198,179, 246,280, 395,342, 550,573"


def get_anchors():
    anchors_full = np.array([float(x) for x in ANCHORS.split(",")])
    anchors_full = anchors_full.reshape(-1, 2)[::-1].copy()
    return torch.tensor(anchors_full, dtype=torch.float32)


def get_tensor_anchors(device):
    anchors_full = np.array([float(x) for x in ANCHORS.split(",")])
    anchors_full = anchors_full.reshape(-1, 2)[::-1].copy()
    return torch.tensor(anchors_full, dtype=torch.float32).to(device)


def generate_anchors_by_feature_size(predict_feature_size, img_size=IMG_SIZE):
    """Generate anchors for a specific feature size."""
    stride = img_size // predict_feature_size
    anchors = get_anchors()
    anchors = anchors / stride
    return anchors

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    # print(box1, box1.shape)
    # print(box2, box2.shape)
    return inter_area / (b1_area + b2_area - inter_area + 1e-16)

def xyxy2xywh(x):  # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    #y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y = torch.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y

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

def xywh2xyxy(x):  # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    #y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y = torch.zeros_like(x)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)
    return y
    

def eval_iou_acc(pred_anchor, target_bbox, anchors_full, target_gi, target_gj, image_wh, iou_threshold_list=[0.5]):
    #print(pred_anchor)

    batch_size, grid_stride = target_bbox.shape[0], image_wh // pred_anchor.shape[3]
    #batch_size, anchor_count, xywh+confidence, grid_height, grid_width
    assert(len(pred_anchor.shape) == 5)
    assert(pred_anchor.shape[3] == pred_anchor.shape[4])
    
    ## eval: convert center+offset to box prediction
    ## calculate at rescaled image during validation for speed-up
    pred_confidence = pred_anchor[:,:,4,:,:]
    scaled_anchors = anchors_full / grid_stride
    
    pred_gi, pred_gj = torch.zeros_like(target_gi), torch.zeros_like(target_gj)
    pred_bbox = torch.zeros_like(target_bbox)
    for batch_idx in range(batch_size):
        best_n, gj, gi = torch.where(pred_confidence[batch_idx].max() == pred_confidence[batch_idx])
        best_n, gj, gi = best_n[0], gj[0], gi[0]
        pred_gj[batch_idx], pred_gi[batch_idx] = gj, gi
        #print((best_n, gi, gj))
        
        pred_bbox[batch_idx, 0] = pred_anchor[batch_idx, best_n, 0, gj, gi].sigmoid() + gi
        pred_bbox[batch_idx, 1] = pred_anchor[batch_idx, best_n, 1, gj, gi].sigmoid() + gj
        pred_bbox[batch_idx, 2] = torch.exp(pred_anchor[batch_idx, best_n, 2, gj, gi]) * scaled_anchors[best_n][0]
        pred_bbox[batch_idx, 3] = torch.exp(pred_anchor[batch_idx, best_n, 3, gj, gi]) * scaled_anchors[best_n][1]
    pred_bbox = pred_bbox * grid_stride
    pred_bbox = xywh2xyxy(pred_bbox)
    
    ## box iou
    iou = bbox_iou(pred_bbox, target_bbox, x1y1x2y2=True)
    each_acc50 = iou>0.5
    accu_list, each_acc_list=list(), list()
    for threshold in iou_threshold_list:
        each_acc = iou>threshold
        accu = torch.sum(each_acc)/batch_size
        accu_list.append(accu)
        each_acc_list.append(each_acc)
    accu_center = torch.sum((target_gi == pred_gi) * (target_gj == pred_gj))/batch_size
    iou = torch.sum(iou)/batch_size

    return accu_list, accu_center, iou, each_acc_list, pred_bbox, target_bbox