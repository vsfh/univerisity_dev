from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DenseMatcherOutput:
    gated_search: torch.Tensor
    query_projected: torch.Tensor
    search_projected: torch.Tensor
    spatial_gate: torch.Tensor
    local_logits: torch.Tensor
    all_logits: torch.Tensor


class DenseQueryMatcher(nn.Module):
    def __init__(
        self,
        query_dim: int,
        search_dim: int,
        projection_dim: int = 256,
        temperature: float = 0.07,
    ):
        super().__init__()
        if query_dim <= 0 or search_dim <= 0 or projection_dim <= 0:
            raise ValueError("query_dim, search_dim and projection_dim must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.query_projection = nn.Linear(query_dim, projection_dim, bias=False)
        self.search_projection = nn.Conv2d(search_dim, projection_dim, kernel_size=1, bias=False)
        self.temperature = float(temperature)

    def forward(
        self,
        query_embedding: torch.Tensor,
        search_features: torch.Tensor,
    ) -> DenseMatcherOutput:
        if query_embedding.ndim != 2:
            raise ValueError(
                f"Expected query_embedding shape (B, C), got {tuple(query_embedding.shape)}."
            )
        if search_features.ndim != 4:
            raise ValueError(
                f"Expected search_features shape (B, C, H, W), got {tuple(search_features.shape)}."
            )
        if query_embedding.shape[0] != search_features.shape[0]:
            raise ValueError(
                "Query/search batch sizes differ: "
                f"{query_embedding.shape[0]} != {search_features.shape[0]}."
            )
        if not torch.isfinite(query_embedding).all() or not torch.isfinite(search_features).all():
            raise ValueError("DenseQueryMatcher inputs must be finite.")

        query_projected = F.normalize(self.query_projection(query_embedding), p=2, dim=1)
        search_projected = F.normalize(self.search_projection(search_features), p=2, dim=1)
        local_logits = torch.einsum(
            "bd,bdhw->bhw",
            query_projected,
            search_projected,
        ) / self.temperature

        batch_size, _, grid_h, grid_w = search_projected.shape
        num_locations = grid_h * grid_w
        spatial_gate = F.softmax(local_logits.flatten(1), dim=1)
        spatial_gate = spatial_gate.view(batch_size, 1, grid_h, grid_w) * num_locations
        gated_search = search_features * spatial_gate

        search_candidates = search_projected.permute(0, 2, 3, 1).reshape(
            batch_size * num_locations,
            -1,
        )
        all_logits = torch.matmul(query_projected, search_candidates.transpose(0, 1))
        all_logits = all_logits / self.temperature

        return DenseMatcherOutput(
            gated_search=gated_search,
            query_projected=query_projected,
            search_projected=search_projected,
            spatial_gate=spatial_gate,
            local_logits=local_logits,
            all_logits=all_logits,
        )


def build_gaussian_click_map(
    query_imgs: torch.Tensor,
    query_click: torch.Tensor,
    sigma: float = 0.075,
) -> torch.Tensor:
    if query_click.ndim != 2 or query_click.shape != (query_imgs.shape[0], 2):
        raise ValueError(
            f"Expected query_click shape {(query_imgs.shape[0], 2)}, got {tuple(query_click.shape)}."
        )
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    height, width = query_imgs.shape[-2:]
    y_coords = torch.linspace(0.0, 1.0, height, device=query_imgs.device, dtype=query_imgs.dtype)
    x_coords = torch.linspace(0.0, 1.0, width, device=query_imgs.device, dtype=query_imgs.dtype)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    click = query_click.to(device=query_imgs.device, dtype=query_imgs.dtype).clamp(0.0, 1.0)
    delta_x = grid_x.unsqueeze(0) - click[:, 0].view(-1, 1, 1)
    delta_y = grid_y.unsqueeze(0) - click[:, 1].view(-1, 1, 1)
    return torch.exp(-0.5 * (delta_x.square() + delta_y.square()) / (sigma * sigma))


def _single_bbox_distribution(
    bbox: torch.Tensor,
    image_wh: Tuple[int, int],
    grid_hw: Tuple[int, int],
    center_weight: float,
) -> torch.Tensor:
    image_w, image_h = int(image_wh[0]), int(image_wh[1])
    grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
    if image_w <= 0 or image_h <= 0 or grid_h <= 0 or grid_w <= 0:
        raise ValueError("image_wh and grid_hw values must be positive.")

    x1 = torch.minimum(bbox[0], bbox[2]).clamp(0.0, float(image_w))
    y1 = torch.minimum(bbox[1], bbox[3]).clamp(0.0, float(image_h))
    x2 = torch.maximum(bbox[0], bbox[2]).clamp(0.0, float(image_w))
    y2 = torch.maximum(bbox[1], bbox[3]).clamp(0.0, float(image_h))

    x_edges = torch.linspace(0.0, float(image_w), grid_w + 1, device=bbox.device, dtype=bbox.dtype)
    y_edges = torch.linspace(0.0, float(image_h), grid_h + 1, device=bbox.device, dtype=bbox.dtype)
    cell_x1 = x_edges[:-1].view(1, grid_w)
    cell_x2 = x_edges[1:].view(1, grid_w)
    cell_y1 = y_edges[:-1].view(grid_h, 1)
    cell_y2 = y_edges[1:].view(grid_h, 1)

    intersection_w = (torch.minimum(cell_x2, x2) - torch.maximum(cell_x1, x1)).clamp_min(0.0)
    intersection_h = (torch.minimum(cell_y2, y2) - torch.maximum(cell_y1, y1)).clamp_min(0.0)
    overlap = intersection_h * intersection_w

    center_x = ((x1 + x2) * 0.5 / float(image_w) * grid_w).floor().long().clamp(0, grid_w - 1)
    center_y = ((y1 + y2) * 0.5 / float(image_h) * grid_h).floor().long().clamp(0, grid_h - 1)
    center = torch.zeros((grid_h, grid_w), device=bbox.device, dtype=bbox.dtype)
    center[center_y, center_x] = 1.0

    overlap_sum = overlap.sum()
    if float(overlap_sum.detach().item()) > 0.0:
        overlap = overlap / overlap_sum
        distribution = center_weight * center + (1.0 - center_weight) * overlap
    else:
        distribution = center
    return distribution.flatten()


def build_dense_bbox_targets(
    bboxes: torch.Tensor,
    image_wh: Tuple[int, int],
    grid_hw: Tuple[int, int],
    identities: torch.Tensor,
    center_weight: float = 0.7,
) -> torch.Tensor:
    if bboxes.ndim != 2 or bboxes.shape[1] != 4:
        raise ValueError(f"Expected bboxes shape (B, 4), got {tuple(bboxes.shape)}.")
    if identities.ndim != 1 or identities.shape[0] != bboxes.shape[0]:
        raise ValueError(
            f"Expected identities shape ({bboxes.shape[0]},), got {tuple(identities.shape)}."
        )
    if not 0.0 <= center_weight <= 1.0:
        raise ValueError("center_weight must be in [0, 1].")
    if not torch.isfinite(bboxes).all():
        raise ValueError("bboxes must be finite.")

    identities = identities.to(device=bboxes.device)
    batch_size = bboxes.shape[0]
    num_locations = int(grid_hw[0]) * int(grid_hw[1])
    per_reference = torch.stack(
        [
            _single_bbox_distribution(bboxes[row], image_wh, grid_hw, center_weight)
            for row in range(batch_size)
        ],
        dim=0,
    )
    targets = bboxes.new_zeros((batch_size, batch_size * num_locations))

    for query_row in range(batch_size):
        positive_rows = torch.where(identities == identities[query_row])[0]
        mass_per_row = 1.0 / max(int(positive_rows.numel()), 1)
        for reference_row in positive_rows.tolist():
            start = reference_row * num_locations
            targets[query_row, start : start + num_locations] = (
                per_reference[reference_row] * mass_per_row
            )

    return targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-12)


def dense_local_contrastive_loss(
    all_logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    if all_logits.ndim != 2 or targets.shape != all_logits.shape:
        raise ValueError(
            f"Expected matching 2D logits/targets, got {tuple(all_logits.shape)} and {tuple(targets.shape)}."
        )
    if not torch.isfinite(all_logits).all() or not torch.isfinite(targets).all():
        raise ValueError("Dense contrastive logits and targets must be finite.")
    return -(targets * F.log_softmax(all_logits, dim=1)).sum(dim=1).mean()


def deranged_score_gap(
    all_logits: torch.Tensor,
    identities: torch.Tensor,
    num_locations: int,
) -> torch.Tensor:
    if all_logits.ndim != 2 or num_locations <= 0:
        raise ValueError("all_logits must be 2D and num_locations must be positive.")
    batch_size = identities.numel()
    if identities.ndim != 1 or all_logits.shape != (batch_size, batch_size * num_locations):
        raise ValueError("Logit shape is inconsistent with identities and num_locations.")

    block_scores = all_logits.view(batch_size, batch_size, num_locations).amax(dim=2)
    same_identity = identities.view(-1, 1) == identities.view(1, -1)
    different_identity = ~same_identity
    valid_rows = different_identity.any(dim=1)
    if not valid_rows.any():
        return all_logits.new_zeros(())

    positive_scores = block_scores.masked_fill(~same_identity, float("-inf")).amax(dim=1)
    deranged_scores = block_scores.masked_fill(~different_identity, float("-inf")).amax(dim=1)
    return (positive_scores[valid_rows] - deranged_scores[valid_rows]).mean()
