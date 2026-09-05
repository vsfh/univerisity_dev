from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def build_image_retrieval_candidate_mask(
    satellite_ids: torch.Tensor,
    num_locations: int,
) -> torch.Tensor:
    """Ignore other batch rows that contain the same satellite.

    Each query keeps all regions from its paired row. Regions from another row
    with the same satellite ID are removed from the loss denominator so they
    cannot act as false negatives or contribute gradients for that query.
    """
    batch_size = satellite_ids.shape[0]
    device = satellite_ids.device
    candidate_rows = torch.arange(device=device, end=batch_size).repeat_interleave(
        num_locations
    )
    query_rows = torch.arange(device=device, end=batch_size).unsqueeze(1)
    same_satellite = satellite_ids.unsqueeze(1).eq(
        satellite_ids[candidate_rows].unsqueeze(0)
    )
    same_row = query_rows.eq(candidate_rows.unsqueeze(0))
    return ~(same_satellite & ~same_row)


def info_nce_loss(
    query_feats: torch.Tensor,
    candidate_feats: torch.Tensor,
    positive_indices: torch.Tensor,
    temperature: float = 0.07,
    candidate_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Keep this loss in FP32 even when the training loop uses autocast.
    with torch.autocast(device_type=query_feats.device.type, enabled=False):
        query_feats = F.normalize(query_feats.float(), p=2, dim=1)
        candidate_feats = F.normalize(candidate_feats.float(), p=2, dim=1)
        logits = query_feats @ candidate_feats.T / temperature

        if candidate_mask is None:
            return F.cross_entropy(logits, positive_indices)

        candidate_mask = candidate_mask.to(device=logits.device, dtype=torch.bool)
        log_normalizer = torch.logsumexp(
            logits.masked_fill(~candidate_mask, -torch.inf), dim=1
        )

        if positive_indices.ndim == 1:
            positive_logits = logits.gather(1, positive_indices[:, None]).squeeze(1)
            return (log_normalizer - positive_logits).mean()

        targets = positive_indices.float()
        target_scores = (targets * logits).sum(dim=1)
        return (targets.sum(dim=1) * log_normalizer - target_scores).mean()
