"""
Enhanced Encoder with GeM pooling, gated fusion, and residual connections.
Based on unified_siglip_test.py architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPooling
import math
import argparse
from typing import Optional
from bbox.yolo_utils import SpatialTransformer,CrossAttention
from transformers.models.siglip.modeling_siglip import SiglipVisionConfig, SiglipMLP
MODEL_NAME = "google/siglip-base-patch16-224"
DINO_MODEL_NAME = "nvidia/C-RADIOv4-H"
CACHE_DIR = "/data/feihong/hf_cache"
PROJECTION_DIM = 768
DINO_PROJECTION_DIM = 768


def _infer_patch_grid(token_count, image_h, image_w):
    """Infer patch grid (H, W) from token count and image aspect ratio."""
    if token_count <= 0:
        raise ValueError(f"token_count must be positive, got {token_count}.")
    if image_h <= 0 or image_w <= 0:
        raise ValueError(f"Invalid image shape: {(image_h, image_w)}.")

    target_ratio = image_h / image_w
    best_h, best_w = None, None
    best_error = float("inf")

    for h in range(1, int(token_count**0.5) + 1):
        if token_count % h != 0:
            continue
        w = token_count // h

        if h > 1 and w > 1:
            ratio_error = abs((h / w) - target_ratio)
            if ratio_error < best_error:
                best_h, best_w = h, w
                best_error = ratio_error

            swapped_ratio_error = abs((w / h) - target_ratio)
            if swapped_ratio_error < best_error:
                best_h, best_w = w, h
                best_error = swapped_ratio_error

    if best_h is None:
        raise ValueError(
            f"Cannot infer 2D patch grid for token_count={token_count} with image shape {(image_h, image_w)}."
        )

    return best_h, best_w


class GeM(nn.Module):
    """Generalized Mean pooling with learnable p-norm.

    Args:
        dim: Feature dimension
        p: Initial p value for generalized mean (default: 3.0)
        eps: Small constant for numerical stability
    """

    def __init__(self, dim=768, p=3.0, eps=1e-6, output_size=(3, 3)):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), self.output_size)
        return x


class ResidualBlock(nn.Module):
    """Standard residual block with skip connection.

    Args:
        channels: Number of input/output channels
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.conv(x)
    
class CoordConv(nn.Module):
    """
    Optimized CoordConv for a fixed sequence shape of [B, 196, 768].
    Pre-computes the 14x14 grid to save processing time.
    """
    def __init__(self, feature_dim=768, grid_size=14):
        super().__init__()
        
        # Linear layer to project [B, 196, 768 + 2] back down to [B, 196, 768]
        self.proj = nn.Linear(feature_dim + 2, feature_dim)
        
        # --- Pre-compute the static 14x14 coordinate grid ---
        coords = torch.linspace(-1, 1, grid_size)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
        
        # Stack into [14, 14, 2] and flatten to sequence [1, 196, 2]
        grid_flat = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 2)
        
        # Register as a buffer so it moves to GPU automatically but doesn't require gradients
        self.register_buffer("base_grid", grid_flat)

    def _grid_for_length(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Build a coordinate grid with exactly seq_len tokens."""
        if seq_len == self.base_grid.shape[1]:
            return self.base_grid.to(device=device, dtype=dtype)

        if seq_len == 1:
            return torch.zeros(1, 1, 2, device=device, dtype=dtype)

        # Choose factors closest to square for stable spatial ordering.
        h = int(seq_len ** 0.5)
        while h > 1 and (seq_len % h != 0):
            h -= 1
        w = seq_len // h

        coords_y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        coords_x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(coords_y, coords_x, indexing='ij')
        return torch.stack([grid_x, grid_y], dim=-1).view(1, seq_len, 2)

    def forward(self, x, angle=None):
        B, L, _ = x.shape
        
        # Build a coordinate grid that matches the current token length L.
        base_grid = self._grid_for_length(L, x.device, x.dtype)
        grid = base_grid.expand(B, -1, -1)
        
        if angle is not None:
            # Ensure angle is a tensor and convert to radians
            if not isinstance(angle, torch.Tensor):
                angle = torch.tensor(angle, device=x.device)
            angle_rad = torch.deg2rad(angle).view(-1, 1)
            
            cos_a = torch.cos(angle_rad)
            sin_a = torch.sin(angle_rad)
            
            # grid[..., 0] is X, grid[..., 1] is Y
            rot_x = grid[..., 0] * cos_a - grid[..., 1] * sin_a
            rot_y = grid[..., 0] * sin_a + grid[..., 1] * cos_a
            
            # Re-stack the rotated coordinates
            grid = torch.stack([rot_x, rot_y], dim=-1)
            
        # Concatenate features (C) with coordinates (2) -> [B, L, C+2]
        x_with_coords = torch.cat([x, grid], dim=-1)
        
        # Project back to feature dim -> [B, L, C]
        return self.proj(x_with_coords)
class PoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, SiglipPoolingHead, config: SiglipVisionConfig):
        super().__init__()

        # self.probe = nn.Parameter(torch.randn(1, 9, config.hidden_size))
        self.probe = SiglipPoolingHead.probe
        self.head = SiglipPoolingHead

    def forward(self, hidden_state, dim=None):
        batch_size = hidden_state.shape[0]
        # probe = self.probe.repeat(batch_size, 9, 1)
        probe = self.probe.repeat(batch_size, dim, 1)

        hidden_state = self.head.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.head.layernorm(hidden_state)
        hidden_state = residual + self.head.mlp(hidden_state)

        return hidden_state
    
class PatchEmbedAndPos(nn.Module):
    def __init__(self, in_channels=3, embed_dim=32, patch_size=16, img_size=256):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        if isinstance(img_size, (tuple, list)):
            grid_h = int(img_size[0]) // patch_size
            grid_w = int(img_size[1]) // patch_size
        else:
            grid_h = int(img_size) // patch_size
            grid_w = int(img_size) // patch_size
        self.base_grid_size = (grid_h, grid_w)
        num_patches = grid_h * grid_w
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

    def forward(self, x):
        # Support variable input sizes by interpolating positional embeddings
        # from the base grid to the current projected patch grid.
        x = self.proj(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)

        if tokens.shape[1] == self.pos_embed.shape[1]:
            pos_embed = self.pos_embed
        else:
            base_h, base_w = self.base_grid_size
            pos_embed_2d = self.pos_embed.reshape(1, base_h, base_w, c).permute(
                0, 3, 1, 2
            )
            pos_embed_2d = F.interpolate(
                pos_embed_2d,
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )
            pos_embed = pos_embed_2d.permute(0, 2, 3, 1).reshape(1, h * w, c)

        return tokens + pos_embed

# --- 辅助模块 2: 序列的全局平均池化 ---
class GlobalAvgPool1D(nn.Module):
    def forward(self, x):
        return x.mean(dim=1)

class Encoder_abla(nn.Module):
    """Enhanced encoder with GeM pooling, gated fusion, and residual connections.

    Improvements over Encoder_attn:
    - GeM pooling for learnable feature aggregation
    - Gated fusion for adaptive text+image combination
    - Residual connections in bbox decoder heads
    """

    def __init__(self, model_name=MODEL_NAME, proj_dim=PROJECTION_DIM, usesg=False, useap=False):
        super().__init__()

        try:
            model = AutoModel.from_pretrained(model_name, cache_dir=CACHE_DIR)
            self.vision_model = model.vision_model
            # self.ground_drone_vision_model = self.model.vision_model
            self.text_model = model.text_model

            self.feature_dim = self.vision_model.config.hidden_size
            self.text_feature_dim = self.text_model.config.hidden_size
        except Exception as e:
            print(f"Error loading SIGLIP model: {e}")
            raise

        self.usesg = usesg
        self.useap = useap

        self.attnPooling = PoolingHead(self.vision_model.head, SiglipVisionConfig())
        self.gate_fc = nn.Linear(proj_dim * 2, 1)

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_feature_dim, self.text_feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.text_feature_dim * 2, proj_dim),
        )
        self.bbox_transformer = SpatialTransformer(
            in_channels=self.feature_dim,
            n_heads=8,
            d_head=64,
            depth=1,
            context_dim=self.feature_dim,
        )
        self.bbox_fcn_out = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.feature_dim,
                out_channels=self.feature_dim // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            ResidualBlock(self.feature_dim // 2),
            nn.Conv2d(self.feature_dim // 2, 9 * 5, kernel_size=1),
        )
        self.bbox_adapter = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        # self.bbox_heading = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         in_channels=self.feature_dim,
        #         out_channels=self.feature_dim // 2,
        #         kernel_size=4,
        #         stride=2,
        #         padding=1,
        #     ),
        #     nn.ReLU(inplace=True),
        #     ResidualBlock(self.feature_dim // 2),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
        #     nn.Linear(self.feature_dim // 2, 2),
        # )
        # self.scorer = ScoreHead(embed_dim=self.feature_dim, num_heads=12)

    def text_forward(self, input_ids, attention_mask=None):
        text_outputs = self.text_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        pooler_output = text_outputs.pooler_output
        proj_feature = self.text_projector(pooler_output)
        return proj_feature

    def gated_fusion(self, text_feats, anchor_feats):
        combined = torch.cat([text_feats, anchor_feats], dim=-1)
        gate = 0.2 * torch.sigmoid(self.gate_fc(combined))
        fused = gate * text_feats + (1 - gate) * anchor_feats
        return fused

    def forward(self, anchor_pixel_values, search_pixel_values, input_ids=None, angle=None):
        B = anchor_pixel_values.shape[0]
        # with torch.no_grad():
        anchor_output = self.vision_model(anchor_pixel_values)
        anchor_feats = anchor_output.last_hidden_state
    
        anchor_pooler = anchor_output.pooler_output
        text_feats = None
        if input_ids is not None:
            text_outputs = self.text_model(input_ids=input_ids)
            text_feats = text_outputs.pooler_output
            # fused_feats = self.gated_fusion(text_feats, anchor_pooler)
            fused_feats = None

        sat_output = self.vision_model(
            search_pixel_values, interpolate_pos_encoding=True
        )
        sat_feats = sat_output.last_hidden_state
            
        N = sat_feats.shape[1]
        image_h, image_w = search_pixel_values.shape[-2:]
        H, W = _infer_patch_grid(N, image_h, image_w)
        sat_features_2d = sat_feats.permute(0, 2, 1).reshape(B, self.feature_dim, H, W)

        if self.usesg:
            anchor_context = (
                self.bbox_adapter(anchor_feats.detach()) + anchor_feats.detach()
            )
        else:
            anchor_context = anchor_feats

        fused_features = self.bbox_transformer(
            x=sat_features_2d, context=anchor_context
        )

        pred_anchor = self.bbox_fcn_out(fused_features)
        # pred_heading = torch.sigmoid(self.bbox_heading(fused_features))

        # if self.useap:
        #     sat_feature_2d_pool = (
        #         self.pool_info_sat(sat_features_2d)
        #         .reshape(B, self.feature_dim, -1)
        #         .permute(0, 2, 1)
        #     )
        # else:
        sat_feature_2d_pool = self.attnPooling(sat_feats, 9)



        return (
            pred_anchor,
            None,
            text_feats,
            anchor_pooler,
            sat_feature_2d_pool,
            fused_feats if input_ids is not None else None,
        )

    def ref_forward(self, search_pixel_values, input_ids=None):
        sat_output = self.vision_model(
            search_pixel_values, interpolate_pos_encoding=True
        )
        sat_feats = sat_output.last_hidden_state
        B = sat_feats.shape[0]
        N = sat_feats.shape[1]
        image_h, image_w = search_pixel_values.shape[-2:]
        H, W = _infer_patch_grid(N, image_h, image_w)
        # sat_features_2d = sat_feats.permute(0, 2, 1).reshape(B, self.feature_dim, H, W)

        # sat_feature_2d_pool = (
        #     self.pool_info_sat(sat_features_2d)
        #     .reshape(B, self.feature_dim, -1)
        #     .permute(0, 2, 1)
        # )
        sat_feature_2d_pool = self.attnPooling(sat_feats)
        return sat_feature_2d_pool


class Encoder_text_angle(nn.Module):
    """Enhanced encoder with GeM pooling, gated fusion, and residual connections.

    Improvements over Encoder_attn:
    - GeM pooling for learnable feature aggregation
    - Gated fusion for adaptive text+image combination
    - Residual connections in bbox decoder heads
    """

    def __init__(self, model_name=MODEL_NAME, proj_dim=PROJECTION_DIM, usesg=False, useap=False):
        super().__init__()

        try:
            model = AutoModel.from_pretrained(model_name, cache_dir=CACHE_DIR)
            self.vision_model = model.vision_model
            # self.ground_drone_vision_model = self.model.vision_model
            self.text_model = model.text_model

            self.feature_dim = self.vision_model.config.hidden_size
            self.text_feature_dim = self.text_model.config.hidden_size
        except Exception as e:
            print(f"Error loading SIGLIP model: {e}")
            raise

        self.usesg = usesg
        self.useap = useap

        self.attnPooling = PoolingHead(self.vision_model.head, SiglipVisionConfig())
        self.gate_fc = nn.Linear(proj_dim * 2, 1)

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_feature_dim, self.text_feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.text_feature_dim * 2, proj_dim),
        )
        self.bbox_transformer = SpatialTransformer(
            in_channels=self.feature_dim,
            n_heads=8,
            d_head=64,
            depth=1,
            context_dim=self.feature_dim,
        )
        self.bbox_fcn_out = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.feature_dim,
                out_channels=self.feature_dim // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            ResidualBlock(self.feature_dim // 2),
            nn.Conv2d(self.feature_dim // 2, 9 * 5, kernel_size=1),
        )
        self.bbox_adapter = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.text_attn = CrossAttention(
            query_dim=self.feature_dim,
            context_dim=self.feature_dim,
            heads=8,
            dim_head=64,
            dropout=0.0,
        )

        self.angle_mlp = nn.Sequential(
            nn.Linear(2, self.feature_dim*2),
        )
        self.text_weight = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def vision_forward(
        self,
        pixel_values,
        angle_feat_1, 
        angle_feat_2,
        interpolate_pos_encoding: bool = False,
    ):
        """Run vision encoder manually and inject hidden state offset at one layer."""

        hidden_states = self.vision_model.embeddings(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        layers = self.vision_model.encoder.layers
        mid_idx = len(layers) // 2

        for idx, encoder_layer in enumerate(layers):
            hidden_states = encoder_layer(hidden_states, attention_mask=None)
            if idx == mid_idx:
                hidden_states = hidden_states * angle_feat_1 + angle_feat_2

        last_hidden_state = self.vision_model.post_layernorm(hidden_states)
        pooler_output = None
        if getattr(self.vision_model, "use_head", True):
            pooler_output = self.vision_model.head(last_hidden_state)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
        )

    def text_forward(self, input_ids, attention_mask=None):
        text_outputs = self.text_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        pooler_output = text_outputs.pooler_output
        proj_feature = self.text_projector(pooler_output)
        return proj_feature

    def gated_fusion(self, text_feats, anchor_feats):
        combined = torch.cat([text_feats, anchor_feats], dim=-1)
        gate = 0.2 * torch.sigmoid(self.gate_fc(combined))
        fused = gate * text_feats + (1 - gate) * anchor_feats
        return fused

    def _build_angle_features(self, angle, batch_size, device, dtype):
        """Build two angle features with shape (B, 1, self.feature_dim) from input angle (B)."""
        angle = angle.reshape(-1)

        angle_rad = torch.deg2rad(angle)
        angle_input = torch.stack([torch.sin(angle_rad), torch.cos(angle_rad)], dim=-1)
        angle_proj = self.angle_mlp(angle_input).to(dtype=dtype)
        angle_feat_1, angle_feat_2 = torch.chunk(angle_proj, chunks=2, dim=-1)
        return angle_feat_1.unsqueeze(1), angle_feat_2.unsqueeze(1)

    def _build_text_embedding(self, text_feats, batch_size, device, dtype):


        return text_feats

    def forward(self, anchor_pixel_values, search_pixel_values, input_ids=None, angle=None):
        B = anchor_pixel_values.shape[0]
        # with torch.no_grad():
        if angle is not None:
            angle_feat_1, angle_feat_2 = self._build_angle_features(
                angle,
                batch_size=B,
                device=anchor_pixel_values.device,
                dtype=anchor_pixel_values.dtype,
            )
            anchor_output = self.vision_forward(anchor_pixel_values, angle_feat_1, angle_feat_2)
        else:
            anchor_output = self.vision_model(anchor_pixel_values)
        anchor_feats = anchor_output.last_hidden_state


        anchor_pooler = self.attnPooling(anchor_feats,1)[:,0,:]
        text_feats = torch.ones(B, self.feature_dim, device=anchor_feats.device, dtype=anchor_feats.dtype)
        
        if input_ids is not None:
            text_outputs = self.text_model(input_ids=input_ids)
            text_feats = self.text_attn(
                text_outputs.pooler_output.unsqueeze(1),
                context=anchor_feats.detach(),
            )[:,0,:]
            anchor_pooler = anchor_pooler + self.text_weight * text_feats

        fused_feats = None


        sat_output = self.vision_model(
            search_pixel_values, interpolate_pos_encoding=True
        )
        sat_feats = sat_output.last_hidden_state
        

        sat_feature_2d_pool = self.attnPooling(sat_feats, 9)
           
        N = sat_feats.shape[1]
        image_h, image_w = search_pixel_values.shape[-2:]
        H, W = _infer_patch_grid(N, image_h, image_w)
        sat_features_2d = sat_feats.permute(0, 2, 1).reshape(B, self.feature_dim, H, W)

        anchor_context = (
            self.bbox_adapter(anchor_feats.detach()) + anchor_feats.detach()
        )

        fused_features = self.bbox_transformer(
            x=sat_features_2d, context=anchor_context
        )

        pred_anchor = self.bbox_fcn_out(fused_features)




        return (
            pred_anchor,
            None,
            text_feats,
            anchor_pooler,
            sat_feature_2d_pool,
            fused_feats,
        )

    def ref_forward(self, search_pixel_values, input_ids=None):
        sat_output = self.vision_model(
            search_pixel_values, interpolate_pos_encoding=True
        )
        sat_feats = sat_output.last_hidden_state
        B = sat_feats.shape[0]
        N = sat_feats.shape[1]
        image_h, image_w = search_pixel_values.shape[-2:]
        H, W = _infer_patch_grid(N, image_h, image_w)
        # sat_features_2d = sat_feats.permute(0, 2, 1).reshape(B, self.feature_dim, H, W)

        # sat_feature_2d_pool = (
        #     self.pool_info_sat(sat_features_2d)
        #     .reshape(B, self.feature_dim, -1)
        #     .permute(0, 2, 1)
        # )
        sat_feature_2d_pool = self.attnPooling(sat_feats)
        return sat_feature_2d_pool


class Encoder_dino(nn.Module):
    """C-RADIOv4-H vision encoder with optional text fallback.

    This class keeps the same output contract as `Encoder_abla` so it can be
    used as a drop-in replacement in retrieval/grounding pipelines.
    """

    def __init__(
        self,
        model_name=DINO_MODEL_NAME,
        text_model_name=DINO_MODEL_NAME,
        proj_dim=DINO_PROJECTION_DIM,
        usesg=False,
        useap=True,
        freeze_vision_blocks=15,
        freeze_patch_generator=True,
        freeze_cls_token=True,
    ):
        super().__init__()
        _ = text_model_name

        self.vision_model = AutoModel.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )
        self.radio_model = self.vision_model.radio_model
        self.vision_core = self.radio_model.model
        self.summary_pool = DinoSummaryPoolingHead(self.radio_model.summary_idxs)

        self.text_model = None
        self.feature_dim = int(self.vision_core.embed_dim)
        self.text_feature_dim = proj_dim

        self.usesg = usesg
        self.useap = useap

        self.pool_info_sat = nn.AdaptiveAvgPool2d((3, 3))
        self.gate_fc = nn.Linear(proj_dim * 2, 1)

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_feature_dim, self.text_feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.text_feature_dim * 2, proj_dim),
        )

        self.anchor_projector = nn.Identity()
        if self.feature_dim != proj_dim:
            self.anchor_projector = nn.Linear(self.feature_dim, proj_dim)

        self.bbox_transformer = SpatialTransformer(
            in_channels=self.feature_dim,
            n_heads=8,
            d_head=64,
            depth=1,
            context_dim=self.feature_dim,
        )

        self.bbox_fcn_out = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.feature_dim,
                out_channels=self.feature_dim // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            ResidualBlock(self.feature_dim // 2),
            nn.Conv2d(self.feature_dim // 2, 9 * 5, kernel_size=1),
        )

        self.bbox_adapter = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )

        self.freeze_first_vision_blocks(
            num_blocks=freeze_vision_blocks,
            freeze_patch_generator=freeze_patch_generator,
            freeze_cls_token=freeze_cls_token,
        )

    def freeze_first_vision_blocks(
        self,
        num_blocks,
        freeze_patch_generator=True,
        freeze_cls_token=True,
    ):
        if num_blocks <= 0:
            return

        core = self.vision_core
        blocks = core.blocks
        n = min(int(num_blocks), len(blocks))

        for i in range(n):
            for p in blocks[i].parameters():
                p.requires_grad = False

        if freeze_patch_generator and hasattr(core, "patch_generator"):
            for p in core.patch_generator.parameters():
                p.requires_grad = False

        if freeze_cls_token and hasattr(core, "patch_generator"):
            cls_tok = getattr(core.patch_generator, "cls_token", None)
            if cls_tok is not None:
                for p in cls_tok.parameters():
                    p.requires_grad = False

    def _vision_forward(self, pixel_values):
        conditioned = self.radio_model.input_conditioner(pixel_values)
        tokens = self.vision_core.forward_features(conditioned)
        patch_generator = getattr(self.vision_core, "patch_generator", None)
        if patch_generator is None:
            raise RuntimeError("C-RADIO model missing patch_generator; cannot pool summary tokens.")
        return self.summary_pool(
            tokens=tokens,
            num_cls_tokens=patch_generator.num_cls_tokens,
            num_skip=patch_generator.num_skip,
            feature_normalizer=self.radio_model.feature_normalizer,
        )

    def _split_tokens(self, hidden_state, image_h=None, image_w=None):
        # DINO-style outputs commonly include a CLS token. We support both
        # [CLS + patches] and [patches only] layouts.
        n = hidden_state.shape[1]
        n_patch = n - 1

        if image_h is not None and image_w is not None:
            try:
                _infer_patch_grid(n_patch, image_h, image_w)
                cls_token = hidden_state[:, 0]
                patch_tokens = hidden_state[:, 1:]
                return cls_token, patch_tokens
            except ValueError:
                pass

            try:
                _infer_patch_grid(n, image_h, image_w)
                patch_tokens = hidden_state
                cls_token = patch_tokens.mean(dim=1)
                return cls_token, patch_tokens
            except ValueError:
                pass

        if int(n_patch**0.5) ** 2 == n_patch:
            cls_token = hidden_state[:, 0]
            patch_tokens = hidden_state[:, 1:]
            return cls_token, patch_tokens

        if int(n**0.5) ** 2 == n:
            patch_tokens = hidden_state
            cls_token = patch_tokens.mean(dim=1)
            return cls_token, patch_tokens

        raise ValueError(
            f"Token sequence length {n} cannot be mapped to square patch grid."
        )

    def text_forward(self, input_ids, attention_mask=None):
        if self.text_model is None:
            batch_size = input_ids.shape[0]
            return torch.zeros(
                batch_size,
                self.text_projector[-1].out_features,
                device=input_ids.device,
                dtype=torch.float32,
            )

        text_outputs = self.text_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        pooler_output = text_outputs.pooler_output
        proj_feature = self.text_projector(pooler_output)
        return proj_feature

    def gated_fusion(self, text_feats, anchor_feats):
        combined = torch.cat([text_feats, anchor_feats], dim=-1)
        gate = 0.2 * torch.sigmoid(self.gate_fc(combined))
        fused = gate * text_feats + (1 - gate) * anchor_feats
        return fused

    def forward(self, anchor_pixel_values, search_pixel_values, input_ids=None, angle=None):
        B = anchor_pixel_values.shape[0]

        anchor_output = self._vision_forward(anchor_pixel_values)
        anchor_pooler = anchor_output.summary[:,:self.feature_dim]
        anchor_patch_tokens = anchor_output.features

        sat_output = self._vision_forward(search_pixel_values)
        sat_hidden = sat_output.features
        image_h, image_w = search_pixel_values.shape[-2:]

        N = sat_hidden.shape[1]
        H, W = _infer_patch_grid(N, image_h, image_w)
        sat_features_2d = sat_hidden.permute(0, 2, 1).reshape(
            B, self.feature_dim, H, W
        )

        if self.usesg:
            anchor_context = (
                self.bbox_adapter(anchor_patch_tokens.detach())
                + anchor_patch_tokens.detach()
            )
        else:
            anchor_context = anchor_patch_tokens

        fused_features = self.bbox_transformer(
            x=sat_features_2d,
            context=anchor_context,
        )

        pred_anchor = self.bbox_fcn_out(fused_features)

        sat_feature_2d_pool = (
            self.pool_info_sat(sat_features_2d)
            .reshape(B, self.feature_dim, -1)
            .permute(0, 2, 1)
        )

        text_feats = None
        if input_ids is not None:
            if self.text_model is None:
                text_feats = anchor_pooler
                fused_feats = anchor_pooler
            else:
                text_outputs = self.text_model(input_ids=input_ids)
                pooler_output = text_outputs.pooler_output
                text_feats = self.text_projector(pooler_output)
                fused_feats = self.gated_fusion(text_feats, anchor_pooler)

        return (
            pred_anchor,
            None,
            text_feats,
            anchor_pooler,
            sat_feature_2d_pool,
            fused_feats if input_ids is not None else None,
        )

    def ref_forward(self, search_pixel_values, input_ids=None):
        sat_output = self._vision_forward(search_pixel_values)
        sat_hidden = sat_output.features
        image_h, image_w = search_pixel_values.shape[-2:]

        B = sat_hidden.shape[0]
        N = sat_hidden.shape[1]
        H, W = _infer_patch_grid(N, image_h, image_w)
        sat_features_2d = sat_hidden.permute(0, 2, 1).reshape(
            B, self.feature_dim, H, W
        )

        sat_feature_2d_pool = (
            self.pool_info_sat(sat_features_2d)
            .reshape(B, self.feature_dim, -1)
            .permute(0, 2, 1)
        )

        return sat_feature_2d_pool


class DinoVisionOutput:
    def __init__(self, summary, features):
        self.summary = summary
        self.features = features


class DinoSummaryPoolingHead(nn.Module):
    """Reproduces C-RADIO summary/features extraction for VisionTransformer outputs."""

    def __init__(self, summary_idxs):
        super().__init__()
        if summary_idxs is None:
            self.register_buffer("summary_idxs", None, persistent=False)
        else:
            self.register_buffer("summary_idxs", summary_idxs.clone().to(torch.int64), persistent=False)

    def forward(self, tokens, num_cls_tokens, num_skip, feature_normalizer):
        all_summary = tokens[:, :num_cls_tokens]
        if self.summary_idxs is not None and all_summary.shape[1] > 1:
            summary_tokens = all_summary[:, self.summary_idxs]
        else:
            summary_tokens = all_summary

        summary = summary_tokens.flatten(1)
        features = tokens[:, num_skip:]
        features = feature_normalizer(features)
        return DinoVisionOutput(summary=summary, features=features)


def _unit_test_dino_pooling_head():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[unit-test] device={device}")

    model = Encoder_dino(proj_dim=1280, freeze_vision_blocks=0).to(device).eval()
    x = torch.randn(2, 3, 432, 768, device=device)

    with torch.no_grad():
        native = model.vision_model(x)
        custom = model._vision_forward(x)

    summary_diff = (native.summary - custom.summary).abs().max().item()
    feature_diff = (native.features - custom.features).abs().max().item()

    print(f"[unit-test] native.summary shape={tuple(native.summary.shape)}")
    print(f"[unit-test] custom.summary shape={tuple(custom.summary.shape)}")
    print(f"[unit-test] native.features shape={tuple(native.features.shape)}")
    print(f"[unit-test] custom.features shape={tuple(custom.features.shape)}")
    print(f"[unit-test] max_abs_summary_diff={summary_diff:.8f}")
    print(f"[unit-test] max_abs_feature_diff={feature_diff:.8f}")

    assert summary_diff < 1e-6, f"summary mismatch: {summary_diff}"
    assert feature_diff < 1e-6, f"feature mismatch: {feature_diff}"
    print("[unit-test] PASS")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-test-dino-pooling", action="store_true")
    args = parser.parse_args()

    if args.unit_test_dino_pooling:
        _unit_test_dino_pooling_head()
    else:
        model = Encoder_gem()
        print("Model loaded successfully")
        print(f"Feature dim: {model.feature_dim}")
        print(f"Text feature dim: {model.text_feature_dim}")

        dummy_feat = torch.randn(2, 768, 20, 20)
        gem_out = model.pool_info_sat(dummy_feat)
        print(f"Pool output shape: {gem_out.shape}")

        text = torch.randn(2, 768)
        anchor = torch.randn(2, 768)
        fused = model.gated_fusion(text, anchor)
        print(f"Fused output shape: {fused.shape}")
