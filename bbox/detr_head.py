import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class DETRHead(nn.Module):
    def __init__(
        self,
        in_dim: int = 768,
        num_queries: int = 100,
        num_classes: int = 1,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.num_queries = num_queries
        self.num_classes = num_classes

        self.query_embed = nn.Embedding(num_queries, in_dim)

        self.input_proj = nn.Conv1d(in_dim, in_dim, kernel_size=1)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=in_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        self.class_embed = nn.Linear(in_dim, num_classes + 1)
        self.bbox_embed = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 4),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, visual_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = visual_features.shape[0]

        visual_features = self.input_proj(visual_features.permute(0, 2, 1))

        queries = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

        tgt = queries
        memory = visual_features.permute(2, 0, 1)

        hs = self.decoder(tgt, memory)
        hs = hs.transpose(1, 2)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs)
        outputs_coord = outputs_coord.sigmoid()

        return outputs_class, outputs_coord


class DETRMatcher(nn.Module):
    def __init__(
        self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    def forward(
        self,
        outputs_class: torch.Tensor,
        outputs_coord: torch.Tensor,
        target_classes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bs, num_queries, num_classes_plus_1 = outputs_class.shape
        _, _, num_coords = outputs_coord.shape

        out_prob = outputs_class.flatten(1).softmax(-1)
        out_bbox = outputs_coord.flatten(1)

        tgt_ids = torch.full((bs,), 0, device=target_classes.device, dtype=torch.long)
        tgt_bbox = target_boxes

        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        C = self.cost_class * cost_class + self.cost_bbox * cost_bbox
        C = C.view(bs, num_queries, -1)

        sizes = [1] * bs
        indices = self._hungarian_match(C, sizes)

        matched_indices = torch.zeros(
            (bs, num_queries), dtype=torch.long, device=outputs_class.device
        )
        src_idx = (
            torch.arange(num_queries, device=outputs_class.device)
            .unsqueeze(0)
            .expand(bs, -1)
        )
        tgt_idx = torch.zeros_like(src_idx)

        for batch_idx, (i, j) in enumerate(indices):
            matched_indices[batch_idx, i] = j
            tgt_idx[batch_idx, i] = j

        return matched_indices, tgt_idx, out_bbox, tgt_bbox


def build_detr_matcher(args):
    return DETRMatcher(
        cost_class=args.cost_class,
        cost_bbox=args.cost_bbox,
        cost_giou=args.cost_giou,
    )
