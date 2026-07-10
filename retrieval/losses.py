from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from unified_siglip_supp import build_retrieval_soft_targets


@dataclass
class RetrievalLossResult:
    total: torch.Tensor
    image_loss: torch.Tensor
    text_loss: Optional[torch.Tensor]
    granularity: str


def infer_granularity(candidate_feats: torch.Tensor, requested: str = "auto") -> str:
    if requested not in {"auto", "grid", "global"}:
        raise ValueError(f"Unsupported retrieval granularity: {requested}")
    if requested == "grid":
        if candidate_feats.ndim != 3 or int(candidate_feats.shape[1]) != 9:
            raise ValueError("grid retrieval requires candidate_feats with shape [B, 9, D].")
        return "grid"
    if requested == "global":
        if candidate_feats.ndim != 2:
            raise ValueError("global retrieval requires candidate_feats with shape [B, D].")
        return "global"
    if candidate_feats.ndim == 2:
        return "global"
    if candidate_feats.ndim == 3 and int(candidate_feats.shape[1]) == 9:
        return "grid"
    if candidate_feats.ndim == 3:
        raise ValueError(
            "auto retrieval expected 9 grid locations for [B, L, D] candidates; "
            f"got L={int(candidate_feats.shape[1])}."
        )
    raise ValueError(f"Unsupported candidate_feats shape: {tuple(candidate_feats.shape)}")


def prepare_candidate_targets(
    candidate_feats: torch.Tensor,
    batch: Dict[str, Any],
    requested_granularity: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    granularity = infer_granularity(candidate_feats, requested_granularity)
    if granularity == "grid":
        local_indices = batch["index"].to(candidate_feats.device).long()
        satellite_ids = batch.get("satellite_id")
        if isinstance(satellite_ids, torch.Tensor):
            satellite_ids = satellite_ids.to(candidate_feats.device)
        targets = build_retrieval_soft_targets(local_indices, satellite_ids)
        return candidate_feats.reshape(-1, candidate_feats.shape[-1]), targets, granularity

    labels = torch.arange(candidate_feats.shape[0], device=candidate_feats.device)
    return candidate_feats, labels, granularity


def contrastive_loss(
    query_feats: torch.Tensor,
    candidate_feats: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    query_feats = F.normalize(query_feats, p=2, dim=1)
    candidate_feats = F.normalize(candidate_feats, p=2, dim=1)
    logits = torch.matmul(query_feats, candidate_feats.T) / float(temperature)
    if targets.ndim == 2:
        return -(targets.to(logits.dtype) * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    return F.cross_entropy(logits, targets.long())


def compute_retrieval_loss(
    query_feats: torch.Tensor,
    candidate_feats: torch.Tensor,
    batch: Dict[str, Any],
    temperature: float = 0.07,
    requested_granularity: str = "auto",
    text_feats: Optional[torch.Tensor] = None,
    use_text_loss: bool = False,
) -> RetrievalLossResult:
    flat_candidates, targets, granularity = prepare_candidate_targets(
        candidate_feats,
        batch,
        requested_granularity=requested_granularity,
    )
    image_loss = contrastive_loss(query_feats, flat_candidates, targets, temperature)
    text_loss = None
    total = image_loss
    if use_text_loss and text_feats is not None:
        text_loss = contrastive_loss(text_feats, flat_candidates, targets, temperature)
        total = total + text_loss
    return RetrievalLossResult(total, image_loss, text_loss, granularity)
