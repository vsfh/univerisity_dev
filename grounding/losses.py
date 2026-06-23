import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from bbox.yolo_utils import bbox_iou, build_target, xywh2xyxy, yolo_loss


@dataclass
class GroundingLoss:
    total: torch.Tensor
    bbox: torch.Tensor
    geo: torch.Tensor
    cls: torch.Tensor
    heatmap: torch.Tensor


def build_geo_features(batch: Dict[str, Any], device: torch.device) -> Optional[torch.Tensor]:
    angle = batch.get("angle")
    height = batch.get("height")
    if angle is None or height is None:
        return None
    angle = angle.to(device=device, dtype=torch.float32).view(-1)
    height = height.to(device=device, dtype=torch.float32).view(-1) / 300.0
    angle_rad = torch.deg2rad(angle)
    return torch.stack([torch.cos(angle_rad), torch.sin(angle_rad), height], dim=1)


def build_heatmap_target(
    target_bbox: torch.Tensor,
    heatmap_hw: Tuple[int, int],
    image_wh: Tuple[int, int],
    edge_value: float = 0.2,
    log_scale: float = 9.0,
) -> torch.Tensor:
    grid_h, grid_w = int(heatmap_hw[0]), int(heatmap_hw[1])
    image_w, image_h = float(image_wh[0]), float(image_wh[1])
    device = target_bbox.device
    dtype = target_bbox.dtype

    x1 = torch.minimum(target_bbox[:, 0], target_bbox[:, 2]).clamp(0.0, image_w)
    y1 = torch.minimum(target_bbox[:, 1], target_bbox[:, 3]).clamp(0.0, image_h)
    x2 = torch.maximum(target_bbox[:, 0], target_bbox[:, 2]).clamp(0.0, image_w)
    y2 = torch.maximum(target_bbox[:, 1], target_bbox[:, 3]).clamp(0.0, image_h)

    center_x = ((x1 + x2) * 0.5 / max(image_w, 1.0) * grid_w).clamp(0.0, grid_w - 1e-6)
    center_y = ((y1 + y2) * 0.5 / max(image_h, 1.0) * grid_h).clamp(0.0, grid_h - 1e-6)

    ys = torch.arange(grid_h, device=device, dtype=dtype).view(1, grid_h, 1)
    xs = torch.arange(grid_w, device=device, dtype=dtype).view(1, 1, grid_w)
    x_img = (xs + 0.5) / max(float(grid_w), 1.0) * image_w
    y_img = (ys + 0.5) / max(float(grid_h), 1.0) * image_h

    inside_x = (x_img >= x1.view(-1, 1, 1)) & (x_img <= x2.view(-1, 1, 1))
    inside_y = (y_img >= y1.view(-1, 1, 1)) & (y_img <= y2.view(-1, 1, 1))
    inside = inside_x & inside_y

    center_img_x = (x1 + x2).view(-1, 1, 1) * 0.5
    center_img_y = (y1 + y2).view(-1, 1, 1) * 0.5
    half_w = ((x2 - x1) * 0.5).view(-1, 1, 1).clamp_min(1e-6)
    half_h = ((y2 - y1) * 0.5).view(-1, 1, 1).clamp_min(1e-6)

    norm_radius = torch.maximum(
        (x_img - center_img_x).abs() / half_w,
        (y_img - center_img_y).abs() / half_h,
    ).clamp(0.0, 1.0)
    log_scale = max(float(log_scale), 1e-6)
    decay = torch.log1p(norm_radius * log_scale) / np.log1p(log_scale)
    target = 1.0 - (1.0 - float(edge_value)) * decay
    target = target * inside.to(dtype=dtype)

    center_x_idx = torch.floor(center_x).long().clamp(0, grid_w - 1)
    center_y_idx = torch.floor(center_y).long().clamp(0, grid_h - 1)
    batch_idx = torch.arange(target.shape[0], device=device)
    target[batch_idx, center_y_idx, center_x_idx] = 1.0
    return target.unsqueeze(1).clamp(0.0, 1.0)


def heatmap_loss_fn(
    heatmap: torch.Tensor,
    target_bbox: torch.Tensor,
    image_wh: Tuple[int, int],
    cfg: Dict[str, Any],
) -> torch.Tensor:
    """Compute loss for the post-spatial-softmax heatmap probability map.

    This matches unified_siglip_supp.py: callers pass the model-produced
    probability map, then both prediction and target are normalized again here.
    """
    heatmap_target = build_heatmap_target(
        target_bbox=target_bbox,
        heatmap_hw=heatmap.shape[-2:],
        image_wh=image_wh,
        edge_value=float(cfg["loss"]["heatmap_bbox_center_edge_value"]),
        log_scale=float(cfg["loss"]["heatmap_bbox_center_log_scale"]),
    )
    pred_prob_map = heatmap.float().flatten(1)
    target = heatmap_target.float().flatten(1)
    target_prob = target / target.sum(dim=1, keepdim=True).clamp_min(1e-6)
    pred_prob = pred_prob_map / pred_prob_map.sum(dim=1, keepdim=True).clamp_min(1e-6)

    loss = heatmap.new_zeros(())
    loss_types = cfg["loss"]["heatmap_loss_type"]
    if "mse" in loss_types:
        loss = loss + F.mse_loss(pred_prob, target_prob, reduction="mean") * pred_prob.shape[1]
    if "cross_entropy" in loss_types:
        loss = loss + -(target_prob * pred_prob.clamp_min(1e-8).log()).sum(dim=1).mean()
    return loss


def build_smgeo_heatmap_target(
    target_bbox: torch.Tensor,
    feature_hw: Tuple[int, int],
    image_wh: Tuple[int, int],
    sigma: float,
) -> torch.Tensor:
    heatmap, _, _ = build_smgeo_anchor_free_targets(target_bbox, feature_hw, image_wh, sigma)
    return heatmap


def build_smgeo_anchor_free_targets(
    target_bbox: torch.Tensor,
    feature_hw: Tuple[int, int],
    image_wh: Tuple[int, int],
    sigma: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = target_bbox.shape[0]
    feat_h, feat_w = int(feature_hw[0]), int(feature_hw[1])
    image_w, image_h = float(image_wh[0]), float(image_wh[1])
    center_x = ((target_bbox[:, 0] + target_bbox[:, 2]) * 0.5 / image_w) * feat_w
    center_y = ((target_bbox[:, 1] + target_bbox[:, 3]) * 0.5 / image_h) * feat_h
    center_ix = center_x.floor().long().clamp(0, feat_w - 1)
    center_iy = center_y.floor().long().clamp(0, feat_h - 1)

    grid_y = torch.arange(feat_h, device=target_bbox.device, dtype=target_bbox.dtype).view(1, feat_h, 1)
    grid_x = torch.arange(feat_w, device=target_bbox.device, dtype=target_bbox.dtype).view(1, 1, feat_w)
    offset_x = grid_x - center_ix.to(dtype=target_bbox.dtype).view(batch_size, 1, 1)
    offset_y = grid_y - center_iy.to(dtype=target_bbox.dtype).view(batch_size, 1, 1)
    radius = max(0, int(math.ceil(float(sigma) * 2.0)))
    mask = (offset_x.abs() <= radius) & (offset_y.abs() <= radius)
    heatmap = torch.exp(
        -0.5
        * (
            offset_x.square()
            + offset_y.square()
        )
        / max(float(sigma) ** 2, 1e-6)
    )
    heatmap = heatmap.masked_fill(~mask, 0.0).unsqueeze(1).clamp(0.0, 1.0)

    box_w = ((target_bbox[:, 2] - target_bbox[:, 0]).clamp_min(1.0) / image_w) * feat_w
    box_h = ((target_bbox[:, 3] - target_bbox[:, 1]).clamp_min(1.0) / image_h) * feat_h
    bbox_target = torch.zeros((batch_size, 4, feat_h, feat_w), device=target_bbox.device, dtype=target_bbox.dtype)
    bbox_target[:, 0] = center_x.view(batch_size, 1, 1) - grid_x
    bbox_target[:, 1] = center_y.view(batch_size, 1, 1) - grid_y
    bbox_target[:, 2] = box_w.view(batch_size, 1, 1)
    bbox_target[:, 3] = box_h.view(batch_size, 1, 1)
    bbox_target = bbox_target * mask.unsqueeze(1).to(dtype=target_bbox.dtype)
    return heatmap, bbox_target, mask.unsqueeze(1).to(dtype=target_bbox.dtype)


def build_smgeo_bbox_target(
    target_bbox: torch.Tensor,
    feature_hw: Tuple[int, int],
    image_wh: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feat_h, feat_w = int(feature_hw[0]), int(feature_hw[1])
    image_w, image_h = float(image_wh[0]), float(image_wh[1])

    cx = ((target_bbox[:, 0] + target_bbox[:, 2]) * 0.5 / image_w) * feat_w
    cy = ((target_bbox[:, 1] + target_bbox[:, 3]) * 0.5 / image_h) * feat_h
    gi = cx.floor().long().clamp(0, feat_w - 1)
    gj = cy.floor().long().clamp(0, feat_h - 1)
    frac_x = cx - gi.to(dtype=target_bbox.dtype)
    frac_y = cy - gj.to(dtype=target_bbox.dtype)
    box_w = (target_bbox[:, 2] - target_bbox[:, 0]).clamp_min(1.0) / image_w
    box_h = (target_bbox[:, 3] - target_bbox[:, 1]).clamp_min(1.0) / image_h
    return torch.stack([frac_x, frac_y, box_w, box_h], dim=1), gi, gj


def smgeo_anchor_free_loss(
    heatmap_logits: torch.Tensor,
    bbox_raw: torch.Tensor,
    target_bbox: torch.Tensor,
    image_wh: Tuple[int, int],
    heatmap_sigma: float = 1.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, _, feat_h, feat_w = heatmap_logits.shape
    heatmap_target, bbox_target, mask = build_smgeo_anchor_free_targets(
        target_bbox,
        feature_hw=(feat_h, feat_w),
        image_wh=image_wh,
        sigma=heatmap_sigma,
    )

    pred_prob = torch.sigmoid(heatmap_logits).clamp(1e-4, 1.0 - 1e-4)
    pos_mask = heatmap_target.eq(1.0)
    neg_mask = heatmap_target.lt(1.0)
    gamma = 2.0
    alpha = 0.25
    beta = 0.75

    if pos_mask.any():
        pos_loss = -torch.log(pred_prob[pos_mask]) * (1.0 - pred_prob[pos_mask]).pow(gamma) * alpha
        pos_loss = pos_loss.mean()
    else:
        pos_loss = heatmap_logits.new_zeros(())
    if neg_mask.any():
        neg_loss = -torch.log(1.0 - pred_prob[neg_mask]) * pred_prob[neg_mask].pow(gamma) * beta
        neg_loss = neg_loss.mean()
    else:
        neg_loss = heatmap_logits.new_zeros(())
    heatmap_loss = torch.clamp(pos_loss + neg_loss, 0.0, 10.0) * 2.0

    center_mask = mask.expand_as(bbox_raw[:, :2])
    wh_mask = mask.expand_as(bbox_raw[:, 2:])
    if center_mask.sum() > 0:
        center_loss = F.l1_loss(
            bbox_raw[:, :2] * center_mask,
            bbox_target[:, :2] * center_mask,
            reduction="sum",
        ) / (center_mask.sum() + 1e-4)
        center_loss = torch.clamp(center_loss, 0.0, 5.0)
    else:
        center_loss = heatmap_logits.new_zeros(())
    if wh_mask.sum() > 0:
        wh_loss = F.l1_loss(
            bbox_raw[:, 2:] * wh_mask,
            bbox_target[:, 2:] * wh_mask,
            reduction="sum",
        ) / (wh_mask.sum() + 1e-4)
        wh_loss = torch.clamp(wh_loss, 0.0, 5.0)
    else:
        wh_loss = heatmap_logits.new_zeros(())

    bbox_loss = (0.7 * center_loss + 0.3 * wh_loss) * 3.0
    return heatmap_loss, bbox_loss


def add_heatmap_to_confidence(
    pred_anchor: torch.Tensor,
    heatmap: Optional[torch.Tensor],
    confidence_weight: float,
) -> torch.Tensor:
    if heatmap is None or confidence_weight <= 0:
        return pred_anchor
    if heatmap.shape[-2:] != pred_anchor.shape[-2:]:
        heatmap = F.interpolate(heatmap, size=pred_anchor.shape[-2:], mode="bilinear", align_corners=False)
    heat_confidence = float(confidence_weight) * heatmap.detach().to(dtype=pred_anchor.dtype).unsqueeze(1)
    return torch.cat([pred_anchor[:, :, :4, :, :], pred_anchor[:, :, 4:5, :, :] + heat_confidence], dim=2)


def compute_grounding_loss(output: Any, batch: Dict[str, Any], anchors_full: torch.Tensor, cfg: Dict[str, Any]) -> GroundingLoss:
    target_bbox = batch["bbox"].to(output.device)
    image_wh = output.image_wh
    pred_anchor = output.pred_anchor

    if cfg["model"]["type"] == "smgeo" and output.heatmap is not None and output.bbox_raw is not None:
        heatmap_loss, bbox_loss = smgeo_anchor_free_loss(
            output.heatmap,
            output.bbox_raw,
            target_bbox,
            image_wh,
            heatmap_sigma=float(cfg["loss"].get("smgeo_heatmap_sigma", 1.5)),
        )
        moe_entropy = output.moe_entropy
        moe_loss = heatmap_loss.new_zeros(()) if moe_entropy is None else -moe_entropy.to(heatmap_loss.device).mean()
        total = (
            float(cfg["loss"].get("heatmap_weight", 0.7)) * heatmap_loss
            + float(cfg["loss"].get("bbox_weight", 0.3)) * bbox_loss
            + float(cfg["loss"].get("moe_entropy_weight", 0.0)) * moe_loss
        )
        zero = bbox_loss.new_zeros(())
        return GroundingLoss(total=total, bbox=bbox_loss, geo=bbox_loss, cls=zero, heatmap=heatmap_loss)

    if pred_anchor is None:
        pred_bbox = output.pred_bbox
        iou = bbox_iou(pred_bbox, target_bbox, x1y1x2y2=True)
        bbox_loss = F.l1_loss(pred_bbox, target_bbox) + (1.0 - iou).mean()
        zero = bbox_loss.new_zeros(())
        total = float(cfg["loss"]["bbox_weight"]) * bbox_loss
        return GroundingLoss(total=total, bbox=bbox_loss, geo=bbox_loss, cls=zero, heatmap=zero)

    if pred_anchor.ndim == 4:
        pred_anchor = pred_anchor.view(pred_anchor.shape[0], 9, 5, pred_anchor.shape[2], pred_anchor.shape[3])
    pred_anchor = add_heatmap_to_confidence(
        pred_anchor,
        output.heatmap,
        float(cfg["loss"]["heatmap_confidence_weight"]),
    )
    grid_wh = (pred_anchor.shape[4], pred_anchor.shape[3])
    new_gt_bbox, best_anchor_gi_gj = build_target(target_bbox, anchors_full, image_wh, grid_wh)
    loss_geo, loss_cls = yolo_loss(
        pred_anchor,
        new_gt_bbox,
        anchors_full,
        best_anchor_gi_gj,
        image_wh,
        confidence_loss_type=str(cfg["loss"].get("anchor_confidence_loss_type", "balanced_bce")),
    )
    bbox_loss = loss_geo + loss_cls
    heatmap_loss = pred_anchor.new_zeros(())
    if output.heatmap is not None and cfg["model"]["use_heatmap"]:
        heatmap_loss = heatmap_loss_fn(output.heatmap, target_bbox, image_wh, cfg)
    total = float(cfg["loss"]["bbox_weight"]) * bbox_loss + float(cfg["loss"]["heatmap_weight"]) * heatmap_loss
    return GroundingLoss(total=total, bbox=bbox_loss, geo=loss_geo, cls=loss_cls, heatmap=heatmap_loss)


def decode_anchor_prediction(pred_anchor: torch.Tensor, anchors_full: torch.Tensor, image_wh: Tuple[int, int]) -> torch.Tensor:
    if pred_anchor.ndim == 4:
        pred_anchor = pred_anchor.view(pred_anchor.shape[0], 9, 5, pred_anchor.shape[2], pred_anchor.shape[3])
    batch_size = pred_anchor.shape[0]
    flat_conf = pred_anchor[:, :, 4, :, :].reshape(batch_size, -1)
    flat_idx = flat_conf.argmax(dim=1)
    anchor_idx = flat_idx // (pred_anchor.shape[3] * pred_anchor.shape[4])
    rem = flat_idx % (pred_anchor.shape[3] * pred_anchor.shape[4])
    gj = rem // pred_anchor.shape[4]
    gi = rem % pred_anchor.shape[4]
    selected = pred_anchor[torch.arange(batch_size, device=pred_anchor.device), anchor_idx, :, gj, gi]

    image_w, image_h = float(image_wh[0]), float(image_wh[1])
    grid_h, grid_w = pred_anchor.shape[3], pred_anchor.shape[4]
    stride_w = image_w / max(float(grid_w), 1.0)
    stride_h = image_h / max(float(grid_h), 1.0)
    anchors = anchors_full.to(pred_anchor.device)
    cx = (selected[:, 0].sigmoid() + gi.float()) * stride_w
    cy = (selected[:, 1].sigmoid() + gj.float()) * stride_h
    bw = torch.exp(selected[:, 2]).clamp(max=1e4) * anchors[anchor_idx, 0]
    bh = torch.exp(selected[:, 3]).clamp(max=1e4) * anchors[anchor_idx, 1]
    xywh = torch.stack([cx, cy, bw, bh], dim=1)
    xyxy = xywh2xyxy(xywh)
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0.0, image_w)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0.0, image_h)
    return xyxy


def center_distance(pred_xyxy: torch.Tensor, target_xyxy: torch.Tensor) -> torch.Tensor:
    pred_center = torch.stack(
        [(pred_xyxy[:, 0] + pred_xyxy[:, 2]) * 0.5, (pred_xyxy[:, 1] + pred_xyxy[:, 3]) * 0.5],
        dim=1,
    )
    target_center = torch.stack(
        [(target_xyxy[:, 0] + target_xyxy[:, 2]) * 0.5, (target_xyxy[:, 1] + target_xyxy[:, 3]) * 0.5],
        dim=1,
    )
    return torch.linalg.norm(pred_center - target_center, dim=1)


def compute_iou_metrics(pred_xyxy: torch.Tensor, target_xyxy: torch.Tensor) -> Dict[str, float]:
    iou = bbox_iou(pred_xyxy, target_xyxy, x1y1x2y2=True).detach().float().cpu()
    distance = center_distance(pred_xyxy.detach().float().cpu(), target_xyxy.detach().float().cpu())
    return {
        "mean_iou": float(iou.mean().item()),
        "iou_at_0_5": float((iou >= 0.5).float().mean().item()),
        "iou_at_0_25": float((iou >= 0.25).float().mean().item()),
        "mean_center_distance": float(distance.mean().item()),
    }
