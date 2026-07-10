from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class FeaturePayload:
    query_feats: torch.Tensor
    candidate_feats: torch.Tensor
    text_feats: Optional[torch.Tensor] = None


class BaseRetrievalAdapter:
    def __init__(self, model: torch.nn.Module, cfg: Dict[str, Any]):
        self.model = model
        self.cfg = cfg

    def forward(self, batch: Dict[str, Any], device: torch.device) -> FeaturePayload:
        raise NotImplementedError


class ForwardMethodAdapter(BaseRetrievalAdapter):
    def forward(self, batch: Dict[str, Any], device: torch.device) -> FeaturePayload:
        query_imgs = batch["target_pixel_values"].to(device, non_blocking=True)
        search_imgs = batch["search_pixel_values"].to(device, non_blocking=True)
        query_feats = self.model.query_forward(query_imgs)
        candidate_feats = self.model.ref_forward(search_imgs)

        text_feats = None
        if bool(self.cfg.get("loss", {}).get("use_text_loss", False)) and hasattr(self.model, "text_forward"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch.get("attention_mask")
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.to(device, non_blocking=True)
            try:
                text_feats = self.model.text_forward(input_ids, attention_mask)
            except TypeError:
                text_feats = self.model.text_forward(input_ids)

        return FeaturePayload(query_feats, candidate_feats, text_feats)


class SampleRetrievalAdapter(BaseRetrievalAdapter):
    def forward(self, batch: Dict[str, Any], device: torch.device) -> FeaturePayload:
        query_imgs = batch["target_pixel_values"].to(device, non_blocking=True)
        search_imgs = batch["search_pixel_values"].to(device, non_blocking=True)
        if hasattr(self.model, "retrieval_forward"):
            query_feats, candidate_feats = self.model.retrieval_forward(query_imgs, search_imgs)
        else:
            query_feats, candidate_feats = self.model(query_imgs, search_imgs)
        return FeaturePayload(query_feats, candidate_feats)
