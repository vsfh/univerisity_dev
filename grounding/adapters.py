from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from grounding.losses import build_geo_features, decode_anchor_prediction
from grounding.legacy.train_sm import decode_anchor_free


@dataclass
class GroundingOutput:
    device: torch.device
    image_wh: Tuple[int, int]
    pred_anchor: Optional[torch.Tensor] = None
    pred_bbox: Optional[torch.Tensor] = None
    heatmap: Optional[torch.Tensor] = None


class BaseAdapter:
    def __init__(self, model: torch.nn.Module, cfg: Dict[str, Any]):
        self.model = model
        self.cfg = cfg

    def forward(self, batch: Dict[str, Any], device: torch.device) -> GroundingOutput:
        raise NotImplementedError

    def decode(self, output: GroundingOutput, batch: Dict[str, Any], anchors_full: torch.Tensor) -> torch.Tensor:
        if output.pred_bbox is not None:
            return output.pred_bbox
        return decode_anchor_prediction(output.pred_anchor, anchors_full, output.image_wh)


class SiglipTupleAdapter(BaseAdapter):
    def forward(self, batch: Dict[str, Any], device: torch.device) -> GroundingOutput:
        target = batch["target_pixel_values"].to(device, non_blocking=True)
        search = batch["search_pixel_values"].to(device, non_blocking=True)
        geo = build_geo_features(batch, device) if self.cfg["model"]["use_angle"] else None
        outputs = self.model(target, search, angle=geo)
        pred_anchor = outputs[0]
        heatmap = outputs[6] if len(outputs) > 6 else None
        return GroundingOutput(
            device=target.device,
            image_wh=(search.shape[-1], search.shape[-2]),
            pred_anchor=pred_anchor,
            heatmap=heatmap,
        )


class LegacyAnchorAdapter(BaseAdapter):
    def forward(self, batch: Dict[str, Any], device: torch.device) -> GroundingOutput:
        target = batch["target_pixel_values"].to(device, non_blocking=True)
        search = batch["search_pixel_values"].to(device, non_blocking=True)
        geo = build_geo_features(batch, device) if self.cfg["model"]["use_angle"] else None
        if geo is None:
            outputs = self.model(target, search)
        else:
            try:
                outputs = self.model(target, search, geo=geo)
            except TypeError:
                outputs = self.model(target, search)
        pred_anchor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        return GroundingOutput(
            device=target.device,
            image_wh=(search.shape[-1], search.shape[-2]),
            pred_anchor=pred_anchor,
        )


class SMGeoAdapter(BaseAdapter):
    def forward(self, batch: Dict[str, Any], device: torch.device) -> GroundingOutput:
        target = batch["target_pixel_values"].to(device, non_blocking=True)
        search = batch["search_pixel_values"].to(device, non_blocking=True)
        heatmap_logits, bbox_raw, _ = self.model(target, search)
        pred_bbox = decode_anchor_free(
            heatmap_logits,
            bbox_raw,
            (search.shape[-1], search.shape[-2]),
        )
        return GroundingOutput(
            device=target.device,
            image_wh=(search.shape[-1], search.shape[-2]),
            pred_bbox=pred_bbox,
        )


class DirectBboxAdapter(BaseAdapter):
    def forward(self, batch: Dict[str, Any], device: torch.device) -> GroundingOutput:
        target = batch["target_pixel_values"].to(device, non_blocking=True)
        search = batch["search_pixel_values"].to(device, non_blocking=True)
        geo = build_geo_features(batch, device) if self.cfg["model"]["use_angle"] else None
        outputs = self.model(target, search, geo) if geo is not None else self.model(target, search)
        pred_bbox = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        return GroundingOutput(
            device=target.device,
            image_wh=(search.shape[-1], search.shape[-2]),
            pred_bbox=pred_bbox,
        )
