import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


class RPNHead(nn.Module):
    def __init__(
        self,
        in_dim: int = 768,
        num_anchors: int = 9,
        feat_channels: int = 256,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.num_anchors = num_anchors
        self.feat_channels = feat_channels

        self.conv = nn.Conv2d(in_dim, feat_channels, 3, padding=1)
        self.cls_logits = nn.Conv2d(feat_channels, num_anchors * 2, 1)
        self.bbox_pred = nn.Conv2d(feat_channels, num_anchors * 4, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv(x))
        cls_logits = self.cls_logits(x)
        bbox_pred = self.bbox_pred(x)

        bs, _, h, w = cls_logits.shape
        cls_logits = cls_logits.view(bs, 2, self.num_anchors, h, w)
        cls_logits = cls_logits.permute(0, 1, 3, 4, 2).contiguous()
        cls_logits = cls_logits.view(bs, 2, h * w * self.num_anchors)

        bbox_pred = bbox_pred.view(bs, 4, self.num_anchors, h, w)
        bbox_pred = bbox_pred.permute(0, 1, 3, 4, 2).contiguous()
        bbox_pred = bbox_pred.view(bs, 4, h * w * self.num_anchors)

        return cls_logits, bbox_pred

    def compute_loss(
        self,
        rpn_cls_logits: torch.Tensor,
        rpn_bbox_pred: torch.Tensor,
        anchors: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = rpn_cls_logits.device
        total_loss = torch.tensor(0.0, device=device)
        loss_items = {
            "rpn_cls_loss": torch.tensor(0.0, device=device),
            "rpn_bbox_loss": torch.tensor(0.0, device=device),
        }

        for b in range(rpn_cls_logits.shape[0]):
            if b >= len(targets):
                continue

            tgt = targets[b]
            if tgt is None or "boxes" not in tgt:
                continue

            tgt_boxes = tgt["boxes"]

            gt_labels = torch.zeros(len(anchors[b]), dtype=torch.long, device=device)
            gt_deltas = torch.zeros(len(anchors[b]), 4, device=device)

            for gt_idx, gt_box in enumerate(tgt_boxes):
                ious = self._compute_ious(anchors[b], gt_box)

                best_anchor_iou, best_anchor_idx = ious.max(dim=0)
                best_gt_iou, best_gt_idx = ious.max(dim=1)

                positive_mask = ious > 0.5
                gt_labels[positive_mask] = 1

                gt_labels[best_anchor_idx] = 1
                gt_deltas[best_anchor_idx] = self._bbox_transform(
                    anchors[b][best_anchor_idx], gt_box
                )

                if len(positive_mask.sum(dim=0).nonzero()) > 0:
                    for i in positive_mask.sum(dim=0).nonzero().squeeze(1):
                        gt_labels[i] = 1
                        gt_deltas[i] = self._bbox_transform(anchors[b][i], gt_box)

            cls_loss = F.cross_entropy(rpn_cls_logits[b].t(), gt_labels)
            bbox_loss = F.smooth_l1_loss(
                rpn_bbox_pred[b].t()[gt_labels > 0],
                gt_deltas[gt_labels > 0],
                reduction="sum",
            ) / max(1, gt_labels.sum())

            loss_items["rpn_cls_loss"] += cls_loss
            loss_items["rpn_bbox_loss"] += bbox_loss

        total_loss = loss_items["rpn_cls_loss"] + loss_items["rpn_bbox_loss"]
        return total_loss, {k: v.item() for k, v in loss_items.items()}

    def _compute_ious(
        self, anchors: torch.Tensor, gt_box: torch.Tensor
    ) -> torch.Tensor:
        anchor_areas = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
        gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

        inter_x1 = torch.max(anchors[:, 0], gt_box[0])
        inter_y1 = torch.max(anchors[:, 1], gt_box[1])
        inter_x2 = torch.min(anchors[:, 2], gt_box[2])
        inter_y2 = torch.min(anchors[:, 3], gt_box[3])

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
            inter_y2 - inter_y1, min=0
        )

        union_area = anchor_areas + gt_area - inter_area + 1e-6
        ious = inter_area / union_area

        return ious

    def _bbox_transform(
        self, anchor: torch.Tensor, gt_box: torch.Tensor
    ) -> torch.Tensor:
        ex_widths = anchor[:, 2] - anchor[:, 0]
        ex_heights = anchor[:, 3] - anchor[:, 1]
        ex_ctr_x = anchor[:, 0] + 0.5 * ex_widths
        ex_ctr_y = anchor[:, 1] + 0.5 * ex_heights

        gt_widths = gt_box[2] - gt_box[0]
        gt_heights = gt_box[3] - gt_box[1]
        gt_ctr_x = gt_box[0] + 0.5 * gt_widths
        gt_ctr_y = gt_box[1] + 0.5 * gt_heights

        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths)
        targets_dh = torch.log(gt_heights / ex_heights)

        return torch.stack([targets_dx, targets_dy, targets_dw, targets_dh], dim=-1)

    def generate_anchors(
        self,
        featmap_sizes: List[Tuple[int, int]],
        strides: List[int],
        anchor_scales: List[List[float]] = None,
        anchor_ratios: List[List[float]] = None,
    ) -> Dict[str, List[torch.Tensor]]:
        if anchor_scales is None:
            anchor_scales = [[32], [64], [128], [256], [512]]
        if anchor_ratios is None:
            anchor_ratios = [[0.5, 1.0, 2.0]]

        anchors = {}

        for feat_size, stride, scales, ratios in zip(
            featmap_sizes, strides, anchor_scales, anchor_ratios
        ):
            h, w = feat_size
            anchor_list = []

            for scale in scales:
                for ratio in ratios:
                    anchor_w = scale * ratio[0] ** 0.5
                    anchor_h = scale / ratio[0] ** 0.5

                    shift_x = torch.arange(w, dtype=torch.float32) * stride
                    shift_y = torch.arange(h, dtype=torch.float32) * stride
                    shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")

                    anchor = torch.stack(
                        [
                            shift_x - anchor_w / 2,
                            shift_y - anchor_h / 2,
                            shift_x + anchor_w / 2,
                            shift_y + anchor_h / 2,
                        ],
                        dim=-1,
                    ).view(-1, 4)

                    anchor_list.append(anchor)

            anchors[f"stride_{stride}"] = torch.cat(anchor_list, dim=0)

        return anchors


class FastRCNNHead(nn.Module):
    def __init__(
        self,
        in_dim: int = 768,
        representation_dim: int = 1024,
        num_classes: int = 2,
    ):
        super().__init__()

        self.fc6 = nn.Linear(in_dim, representation_dim)
        self.fc7 = nn.Linear(representation_dim, representation_dim)
        self.cls_score = nn.Linear(representation_dim, num_classes)
        self.bbox_pred = nn.Linear(representation_dim, (num_classes - 1) * 4)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        cls_score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)

        return cls_score, bbox_pred

    def compute_loss(
        self,
        cls_score: torch.Tensor,
        bbox_pred: torch.Tensor,
        labels: torch.Tensor,
        bbox_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = cls_score.device

        loss_cls = F.cross_entropy(cls_score, labels)

        pos_mask = labels > 0
        if pos_mask.sum() > 0:
            loss_bbox = F.smooth_l1_loss(
                bbox_pred[pos_mask],
                bbox_targets[pos_mask],
                reduction="sum",
            ) / max(1, pos_mask.sum())
        else:
            loss_bbox = torch.tensor(0.0, device=device)

        total_loss = loss_cls + loss_bbox
        return total_loss, {
            "rcnn_cls_loss": loss_cls.item(),
            "rcnn_bbox_loss": loss_bbox.item(),
        }
