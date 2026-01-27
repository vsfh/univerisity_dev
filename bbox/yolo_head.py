import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


class YOLOHead(nn.Module):
    def __init__(
        self,
        in_dim: int = 768,
        num_classes: int = 1,
        stride: List[int] = [8, 16, 32],
        anchor_grids: List[List[List[float]]] = None,
        anchor_w: List[float] = None,
        anchor_h: List[float] = None,
        img_size: int = 640,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.num_classes = num_classes
        self.stride = stride
        self.img_size = img_size

        if anchor_grids is None:
            anchor_grids = [
                [[10, 13], [16, 30], [33, 23]],
                [[30, 61], [62, 45], [59, 119]],
                [[116, 90], [156, 198], [373, 326]],
            ]
        self.anchor_grids = anchor_grids

        if anchor_w is None:
            anchor_w = [10, 16, 33, 30, 62, 116, 59, 156, 373]
        if anchor_h is None:
            anchor_h = [13, 30, 23, 61, 45, 90, 119, 198, 326]

        self.register_buffer("anchors_w", torch.tensor(anchor_w), persistent=False)
        self.register_buffer("anchors_h", torch.tensor(anchor_h), persistent=False)

        self.num_anchors = sum(len(grid) for grid in anchor_grids)

        self.conv = nn.ModuleList()
        for i in range(len(stride)):
            conv = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_dim, self.num_anchors * (5 + num_classes), 1),
            )
            self.conv.append(conv)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self, features: Dict[str, torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        outputs = []
        origin_shapes = []

        for i, stride in enumerate(self.stride):
            x = features[f"p{stride}"]
            bs, _, h, w = x.shape
            origin_shapes.append((h, w))

            x = self.conv[i](x)
            x = x.view(bs, self.num_anchors, 5 + self.num_classes, h, w)
            x = x.permute(0, 1, 3, 4, 2).contiguous()

            outputs.append(x)

        return outputs, origin_shapes

    def compute_loss(
        self,
        outputs: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        origin_shapes: List[Tuple[int, int]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = outputs[0].device
        stride_tensor = torch.tensor(self.stride, device=device)

        total_loss = torch.tensor(0.0, device=device)
        loss_items = {
            "box_loss": torch.tensor(0.0, device=device),
            "obj_loss": torch.tensor(0.0, device=device),
            "cls_loss": torch.tensor(0.0, device=device),
        }

        for i, (output, (h, w)) in enumerate(zip(outputs, origin_shapes)):
            bs, na, _, _, _ = output.shape
            stride = self.stride[i]

            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing="ij",
            )
            grid = torch.stack([grid_x, grid_y], dim=-1).float()
            grid = grid.view(1, 1, h, w, 2).expand(bs, na, -1, -1, -1)

            anchor_w = self.anchors_w[na * i : na * (i + 1)].view(1, na, 1, 1, 2)[
                ..., 0:1
            ]
            anchor_h = self.anchors_h[na * i : na * (i + 1)].view(1, na, 1, 1, 2)[
                ..., 1:2
            ]

            tx = output[..., 4:5]
            ty = output[..., 5:6]
            tw = output[..., 6:7]
            th = output[..., 7:8]
            tobj = output[..., 0:1]
            tcls = output[..., 8:]

            pred_x = (torch.sigmoid(tx) + grid_x) * stride
            pred_y = (torch.sigmoid(ty) + grid_y) * stride
            pred_w = torch.exp(tw) * anchor_w
            pred_h = torch.exp(th) * anchor_h

            target_boxes = []
            target_scores = []
            target_obj = []

            for b in range(bs):
                if b >= len(targets):
                    break
                tgt = targets[b]
                if tgt is None:
                    continue

                tgt_boxes = tgt["boxes"]
                tgt_labels = tgt["labels"]

                for box, label in zip(tgt_boxes, tgt_labels):
                    cx = (box[0] + box[2]) / 2
                    cy = (box[1] + box[3]) / 2
                    bw = box[2] - box[0]
                    bh = box[3] - box[1]

                    cx_grid = cx / stride
                    cy_grid = cy / stride

                    grid_x_idx = int(cx_grid % w)
                    grid_y_idx = int(cy_grid % h)

                    iou = self._calculate_iou(
                        pred_x[b, :, grid_y_idx, grid_x_idx],
                        pred_y[b, :, grid_y_idx, grid_x_idx],
                        pred_w[b, :, grid_y_idx, grid_x_idx],
                        pred_h[b, :, grid_y_idx, grid_x_idx],
                        cx,
                        cy,
                        bw,
                        bh,
                    )
                    best_anchor = iou.argmax()

                    target_boxes.append([cx_grid, cy_grid, bw / stride, bh / stride])
                    target_scores.append(F.one_hot(label, self.num_classes).float())
                    target_obj.append(torch.zeros(na, h, w, 1, device=device))
                    target_obj[-1][0, best_anchor, grid_y_idx, grid_x_idx, 0] = 1.0

            if target_boxes:
                target_boxes = torch.tensor(target_boxes, device=device)
                target_scores = (
                    torch.stack(target_scores)
                    if target_scores
                    else torch.zeros(0, self.num_classes, device=device)
                )
                target_obj = (
                    torch.cat(target_obj)
                    if target_obj
                    else torch.zeros(bs, na, h, w, 1, device=device)
                )

                box_loss = F.mse_loss(
                    torch.cat(
                        [
                            output[..., 4:5],
                            output[..., 5:6],
                            output[..., 6:7],
                            output[..., 7:8],
                        ],
                        dim=-1,
                    )[target_obj.bool()],
                    target_boxes[..., :4][target_obj.bool()].view(-1, 4),
                )
                obj_loss = F.binary_cross_entropy_with_logits(
                    output[..., 0:1][target_obj.bool()],
                    target_obj.float()[target_obj.bool()],
                )
                cls_loss = F.binary_cross_entropy_with_logits(
                    output[..., 8:][target_obj.bool().expand_as(output[..., 8:])],
                    target_scores.expand_as(output[..., 8:])[
                        target_obj.bool().expand_as(output[..., 8:])
                    ],
                )

                loss_items["box_loss"] += box_loss
                loss_items["obj_loss"] += obj_loss
                loss_items["cls_loss"] += cls_loss

        total_loss = (
            loss_items["box_loss"] * 0.05
            + loss_items["obj_loss"] * 1.0
            + loss_items["cls_loss"] * 0.5
        )

        return total_loss, {k: v.item() for k, v in loss_items.items()}

    def _calculate_iou(
        self,
        pred_x: torch.Tensor,
        pred_y: torch.Tensor,
        pred_w: torch.Tensor,
        pred_h: torch.Tensor,
        tgt_cx: float,
        tgt_cy: float,
        tgt_w: float,
        tgt_h: float,
    ) -> torch.Tensor:
        pred_x1 = pred_x - pred_w / 2
        pred_y1 = pred_y - pred_h / 2
        pred_x2 = pred_x + pred_w / 2
        pred_y2 = pred_y + pred_h / 2

        tgt_x1 = tgt_cx - tgt_w / 2
        tgt_y1 = tgt_cy - tgt_h / 2
        tgt_x2 = tgt_cx + tgt_w / 2
        tgt_y2 = tgt_cy + tgt_h / 2

        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
            inter_y2 - inter_y1, min=0
        )
        pred_area = pred_w * pred_h
        tgt_area = tgt_w * tgt_h

        union_area = pred_area + tgt_area - inter_area + 1e-6
        iou = inter_area / union_area

        return iou

    def postprocess(
        self,
        outputs: List[torch.Tensor],
        origin_shapes: List[Tuple[int, int]],
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ) -> List[Dict[str, torch.Tensor]]:
        results = []

        for i, (output, (h, w)) in enumerate(zip(outputs, origin_shapes)):
            bs, na, _, _, _ = output.shape
            stride = self.stride[i]

            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=output.device),
                torch.arange(w, device=output.device),
                indexing="ij",
            )

            anchor_w = self.anchors_w[na * i : na * (i + 1)].view(1, na, 1, 1)
            anchor_h = self.anchors_h[na * i : na * (i + 1)].view(1, na, 1, 1)

            x = output.clone()
            x = x.view(bs, na, -1, h, w)
            x = x.permute(0, 1, 3, 4, 2).contiguous()

            obj = torch.sigmoid(x[..., 0:1])
            cls = torch.sigmoid(x[..., 8:])
            box = x[..., 4:8]

            pred_x = (torch.sigmoid(box[..., 0:1]) + grid_x) * stride
            pred_y = (torch.sigmoid(box[..., 1:2]) + grid_y) * stride
            pred_w = torch.exp(box[..., 2:3]) * anchor_w
            pred_h = torch.exp(box[..., 3:4]) * anchor_h

            conf = obj * cls.max(dim=-1, keepdim=True)[0]
            conf_mask = conf > conf_thres

            for b in range(bs):
                valid_mask = conf_mask[b].view(-1)
                if valid_mask.sum() == 0:
                    results.append(
                        {
                            "boxes": torch.zeros(0, 4, device=output.device),
                            "scores": torch.zeros(0, device=output.device),
                            "labels": torch.zeros(
                                0, dtype=torch.long, device=output.device
                            ),
                        }
                    )
                    continue

                boxes = torch.cat(
                    [pred_x[b], pred_y[b], pred_w[b], pred_h[b]], dim=-1
                ).view(-1, 4)
                scores = conf[b].view(-1)
                classes = cls[b].argmax(dim=-1).view(-1)

                boxes = boxes[valid_mask]
                scores = scores[valid_mask]
                classes = classes[valid_mask]

                indices = self._nms(boxes, scores, iou_thres)
                results.append(
                    {
                        "boxes": boxes[indices],
                        "scores": scores[indices],
                        "labels": classes[indices],
                    }
                )

        return results

    def _nms(
        self, boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float
    ) -> torch.Tensor:
        keep_indices = []
        sorted_scores, sorted_idx = scores.sort(descending=True)

        while len(sorted_idx) > 0:
            best_idx = sorted_idx[0]
            keep_indices.append(best_idx.item())

            if len(sorted_idx) == 1:
                break

            remaining_idx = sorted_idx[1:]
            ious = self._iou_box(boxes[best_idx].unsqueeze(0), boxes[remaining_idx])
            mask = ious < iou_thres
            sorted_idx = remaining_idx[mask]
            sorted_scores = sorted_scores[1:][mask]

        return torch.tensor(keep_indices, device=boxes.device)

    def _iou_box(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        b1_x1, b1_y1, b1_w, b1_h = box1.unbind(-1)
        b2_x1, b2_y1, b2_w, b2_h = box2.unbind(-1)

        b1_x2 = b1_x1 + b1_w
        b1_y2 = b1_y1 + b1_h
        b2_x2 = b2_x1 + b2_w
        b2_y2 = b2_y1 + b2_h

        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
            inter_y2 - inter_y1, min=0
        )
        b1_area = b1_w * b1_h
        b2_area = b2_w * b2_h

        union_area = b1_area + b2_area - inter_area + 1e-6
        return inter_area / union_area
