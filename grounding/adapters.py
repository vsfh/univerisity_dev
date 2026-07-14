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
    bbox_raw: Optional[torch.Tensor] = None
    moe_entropy: Optional[torch.Tensor] = None
    query_embedding: Optional[torch.Tensor] = None
    search_local_features: Optional[torch.Tensor] = None
    search_grid_size: Optional[Tuple[int, int]] = None
    matcher_logits: Optional[torch.Tensor] = None
    paper_aux_losses: Optional[Dict[str, torch.Tensor]] = None


def _mapping_to_grounding_output(
    outputs: Dict[str, Any],
    target: torch.Tensor,
    search: torch.Tensor,
) -> GroundingOutput:
    local_features = outputs.get("search_local_features")
    grid_size = outputs.get("search_grid_size")
    if grid_size is None and isinstance(local_features, torch.Tensor) and local_features.ndim == 4:
        grid_size = (int(local_features.shape[-2]), int(local_features.shape[-1]))
    return GroundingOutput(
        device=target.device,
        image_wh=(search.shape[-1], search.shape[-2]),
        pred_anchor=outputs.get("pred_anchor"),
        pred_bbox=outputs.get("pred_bbox"),
        heatmap=outputs.get("heatmap", outputs.get("heatmap_logits")),
        bbox_raw=outputs.get("bbox_raw"),
        moe_entropy=outputs.get("moe_entropy"),
        query_embedding=outputs.get("query_embedding"),
        search_local_features=local_features,
        search_grid_size=grid_size,
        matcher_logits=outputs.get("matcher_logits"),
        paper_aux_losses=outputs.get("paper_aux_losses"),
    )


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
        query_click = batch.get("query_click")
        if query_click is not None:
            query_click = query_click.to(device, non_blocking=True)
        geo = build_geo_features(batch, device) if self.cfg["model"]["use_angle"] else None
        outputs = self.model(target, search, query_click=query_click, geo=geo)
        if isinstance(outputs, dict):
            return _mapping_to_grounding_output(outputs, target, search)
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
        query_click = batch.get("query_click")
        if query_click is not None:
            query_click = query_click.to(device, non_blocking=True)
        geo = build_geo_features(batch, device) if self.cfg["model"].get("use_angle", False) else None
        outputs = self.model(target, search, query_click=query_click, geo=geo)
        if isinstance(outputs, dict):
            heatmap_logits = outputs["heatmap_logits"]
            bbox_raw = outputs["bbox_raw"]
            pred_bbox = decode_anchor_free(
                heatmap_logits,
                bbox_raw,
                (search.shape[-1], search.shape[-2]),
            )
            outputs = dict(outputs)
            outputs["pred_bbox"] = pred_bbox
            return _mapping_to_grounding_output(outputs, target, search)
        heatmap_logits, bbox_raw, moe_entropy = outputs
        pred_bbox = decode_anchor_free(
            heatmap_logits,
            bbox_raw,
            (search.shape[-1], search.shape[-2]),
        )
        return GroundingOutput(
            device=target.device,
            image_wh=(search.shape[-1], search.shape[-2]),
            pred_bbox=pred_bbox,
            heatmap=heatmap_logits,
            bbox_raw=bbox_raw,
            moe_entropy=moe_entropy,
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
