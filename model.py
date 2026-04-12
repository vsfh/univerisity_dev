"""
Enhanced Encoder with GeM pooling, gated fusion, and residual connections.
Based on unified_siglip_test.py architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer

from bbox.yolo_utils import SpatialTransformer
from transformers.models.siglip.modeling_siglip import SiglipVisionConfig, SiglipMLP
MODEL_NAME = "google/siglip-base-patch16-224"
DINO_MODEL_NAME = "nvidia/C-RADIOv4-H"
CACHE_DIR = "/data/feihong/hf_cache"
PROJECTION_DIM = 768


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

class PoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, SiglipPoolingHead, config: SiglipVisionConfig):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 9, config.hidden_size))
        self.head = SiglipPoolingHead
        # self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        # self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.mlp = SiglipMLP(config)

    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.head.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.head.layernorm(hidden_state)
        hidden_state = residual + self.head.mlp(hidden_state)

        return hidden_state
    

class Encoder_gem(nn.Module):
    """Enhanced encoder with GeM pooling, gated fusion, and residual connections.

    Improvements over Encoder_attn:
    - GeM pooling for learnable feature aggregation
    - Gated fusion for adaptive text+image combination
    - Residual connections in bbox decoder heads
    """

    def __init__(self, model_name=MODEL_NAME, proj_dim=PROJECTION_DIM):
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

        self.pool_info_sat = nn.AdaptiveAvgPool2d((3, 3))

        self.gate_fc = nn.Linear(proj_dim * 2, 1)
        # self.fuse_fc = nn.Linear(proj_dim, proj_dim)

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
        # self.bbox_coords_out = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         in_channels=self.feature_dim,
        #         out_channels=self.feature_dim // 2,
        #         kernel_size=4,
        #         stride=2,
        #         padding=1,
        #     ),
        #     nn.ReLU(inplace=True),
        #     ResidualBlock(self.feature_dim // 2),
        #     nn.Conv2d(self.feature_dim // 2, 1, kernel_size=1),
        # )

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

    def forward(self, anchor_pixel_values, search_pixel_values, input_ids=None):
        B = anchor_pixel_values.shape[0]

        anchor_output = self.vision_model(anchor_pixel_values)
        anchor_feats = anchor_output.last_hidden_state
        anchor_pooler = anchor_output.pooler_output

        # ground_drone_output = self.ground_drone_vision_model(anchor_pixel_values)
        # anchor_feats = ground_drone_output.last_hidden_state

        sat_output = self.vision_model(
            search_pixel_values, interpolate_pos_encoding=True
        )
        sat_feats = sat_output.last_hidden_state
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
        # pred_coords = self.bbox_coords_out(fused_features)

        sat_feature_2d_pool = (
            self.pool_info_sat(sat_features_2d)
            .reshape(B, self.feature_dim, -1)
            .permute(0, 2, 1)
        )

        text_feats = None
        if input_ids is not None:
            text_outputs = self.text_model(input_ids=input_ids)
            pooler_output = text_outputs.pooler_output
            text_feats = self.text_projector(pooler_output)
            fused_feats = self.gated_fusion(text_feats, anchor_pooler)

        return (
            pred_anchor,
            text_feats,
            anchor_pooler,
            sat_feature_2d_pool,
            fused_feats if input_ids is not None else None,
        )

    def preprocess(self, query_image, search_image):
        if not isinstance(getattr(self, "processor", None)):
            self.processor = AutoImageProcessor.from_pretrained(
                MODEL_NAME, cache_dir=CACHE_DIR
            )
            self.processor_sat = AutoImageProcessor.from_pretrained(
                MODEL_NAME, cache_dir=CACHE_DIR, size={"height": 640, "width": 640}
            )
        query_inputs = self.processor(images=query_image, return_tensors="pt")
        search_inputs = self.processor_sat(images=search_image, return_tensors="pt")
        return query_inputs["pixel_values"][0], search_inputs["pixel_values"][0]

    def ref_forward(self, search_pixel_values, input_ids=None):


        sat_output = self.vision_model(
            search_pixel_values, interpolate_pos_encoding=True
        )
        sat_feats = sat_output.last_hidden_state
        N = sat_feats.shape[1]
        B = sat_feats.shape[0]
        image_h, image_w = search_pixel_values.shape[-2:]
        H, W = _infer_patch_grid(N, image_h, image_w)
        sat_features_2d = sat_feats.permute(0, 2, 1).reshape(B, self.feature_dim, H, W)

        sat_feature_2d_pool = (
            self.pool_info_sat(sat_features_2d)
            .reshape(B, self.feature_dim, -1)
            .permute(0, 2, 1)
        )

        return sat_feature_2d_pool

class Encoder_attn(nn.Module):
    def __init__(self, model_name=MODEL_NAME, proj_dim=768):
        super().__init__()

        try:
            self.model = AutoModel.from_pretrained(model_name, cache_dir=CACHE_DIR)
            self.vision_model = self.model.vision_model
            self.text_model = self.model.text_model

            self.feature_dim = self.vision_model.config.hidden_size
            self.text_feature_dim = self.text_model.config.hidden_size
        except Exception as e:
            print(f"Error loading SIGLIP model: {e}")
            raise

        self.pool_sat = nn.AdaptiveAvgPool2d((20, 20))
        self.pool_dro = nn.AdaptiveAvgPool2d((8, 8))
        self.pool_info_sat = nn.AdaptiveAvgPool2d((3, 3))
        # self.position_embedding = nn.Embedding(400, PROJECTION_DIM)
        # self.register_buffer(
        #     "position_ids", torch.arange(400).expand((1, -1)), persistent=False
        # )

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
            nn.Conv2d(self.feature_dim // 2, 9 * 5, kernel_size=1),
        )

        self.bbox_coords_out = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.feature_dim,
                out_channels=self.feature_dim // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_dim // 2, 1, kernel_size=1),
        )

    def text_forward(self, input_ids, attention_mask=None):
        text_outputs = self.text_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        pooler_output = text_outputs.pooler_output
        proj_feature = self.text_projector(pooler_output)
        return proj_feature

    def forward(self, anchor_pixel_values, search_pixel_values, input_ids=None):
        B = anchor_pixel_values.shape[0]

        anchor_output = self.vision_model(anchor_pixel_values)
        anchor_feats = anchor_output.last_hidden_state
        anchor_pooler = anchor_output.pooler_output

        sat_output = self.vision_model(
            search_pixel_values, interpolate_pos_encoding=True
        )
        sat_feats = sat_output.last_hidden_state
        N = sat_feats.shape[1]
        image_h, image_w = search_pixel_values.shape[-2:]
        H, W = _infer_patch_grid(N, image_h, image_w)
        sat_features_2d = sat_feats.permute(0, 2, 1).reshape(B, self.feature_dim, H, W)

        fused_features = self.bbox_transformer(x=sat_features_2d, context=anchor_feats)

        pred_anchor = self.bbox_fcn_out(fused_features)
        pred_coords = self.bbox_coords_out(fused_features)

        sat_feature_2d_pool = (
            self.pool_info_sat(sat_features_2d)
            .reshape(B, self.feature_dim, -1)
            .permute(0, 2, 1)
        )

        text_feats = None
        if input_ids is not None:
            text_outputs = self.text_model(input_ids=input_ids)
            pooler_output = text_outputs.pooler_output
            text_feats = self.text_projector(pooler_output)

        return (
            pred_anchor,
            pred_coords,
            anchor_pooler,
            sat_feature_2d_pool,
            text_feats,
        )

    def _pool_grid_features(self, vision_features, layer):
        B, N, D = vision_features.shape
        H = W = int(N**0.5)
        patch_tokens = vision_features.permute(0, 2, 1).reshape(B, D, H, W)
        pooled_features = layer(patch_tokens)
        return pooled_features

    def preprocess(self, query_image, search_image):
        if not isinstance(getattr(self, "processor", None)):
            self.processor = AutoImageProcessor.from_pretrained(
                MODEL_NAME, cache_dir=CACHE_DIR
            )
            self.processor_sat = AutoImageProcessor.from_pretrained(
                MODEL_NAME, cache_dir=CACHE_DIR, size={"height": 640, "width": 640}
            )
        query_inputs = self.processor(images=query_image, return_tensors="pt")
        search_inputs = self.processor_sat(images=search_image, return_tensors="pt")
        return query_inputs["pixel_values"][0], search_inputs["pixel_values"][0]


class Encoder_heading(nn.Module):
    """Enhanced encoder with GeM pooling, gated fusion, and residual connections.

    Improvements over Encoder_attn:
    - GeM pooling for learnable feature aggregation
    - Gated fusion for adaptive text+image combination
    - Residual connections in bbox decoder heads
    """

    def __init__(self, model_name=MODEL_NAME, proj_dim=PROJECTION_DIM):
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

        self.pool_info_sat = nn.AdaptiveAvgPool2d((3, 3))
        self.gate_fc = nn.Linear(proj_dim * 2, 1)
        # self.fuse_fc = nn.Linear(proj_dim, proj_dim)

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
        self.bbox_heading = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.feature_dim,
                out_channels=self.feature_dim // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            ResidualBlock(self.feature_dim // 2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.feature_dim // 2, 2),
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

    def forward(self, anchor_pixel_values, search_pixel_values, input_ids=None):
        B = anchor_pixel_values.shape[0]

        anchor_output = self.vision_model(anchor_pixel_values)
        anchor_feats = anchor_output.last_hidden_state
        anchor_pooler = anchor_output.pooler_output

        # ground_drone_output = self.ground_drone_vision_model(anchor_pixel_values)
        # anchor_feats = ground_drone_output.last_hidden_state

        sat_output = self.vision_model(
            search_pixel_values, interpolate_pos_encoding=True
        )
        sat_feats = sat_output.last_hidden_state
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
        pred_heading = torch.sigmoid(self.bbox_heading(fused_features))

        sat_feature_2d_pool = (
            self.pool_info_sat(sat_features_2d)
            .reshape(B, self.feature_dim, -1)
            .permute(0, 2, 1)
        )

        text_feats = None
        if input_ids is not None:
            text_outputs = self.text_model(input_ids=input_ids)
            pooler_output = text_outputs.pooler_output
            text_feats = self.text_projector(pooler_output)
            fused_feats = self.gated_fusion(text_feats, anchor_pooler)

        return (
            pred_anchor,
            pred_heading,
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
        sat_features_2d = sat_feats.permute(0, 2, 1).reshape(B, self.feature_dim, H, W)

        sat_feature_2d_pool = (
            self.pool_info_sat(sat_features_2d)
            .reshape(B, self.feature_dim, -1)
            .permute(0, 2, 1)
        )

        return sat_feature_2d_pool
        
class Rotator(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        assert feature_dim % 2 == 0, "Feature dimension must be even for rotary operations."
        
        # Learnable frequencies: allows the model to scale the angle differently 
        # for different feature pairs. 
        # Shape updated to (1, 1, feature_dim // 2) for easy broadcasting over N.
        self.freqs = nn.Parameter(torch.randn(1, 1, feature_dim // 2))

    def forward(self, x, angle):
        """
        x: sequence hidden states of shape (B, N, C)
        angle: rotation angle in radians of shape (B,), (B, 1), or (B, 1, 1)
        """
        B, N, C = x.shape
        
        # Reshape angle to (B, 1, 1) so it broadcasts across the N and C dimensions
        if angle.dim() == 1:
            angle = angle.view(B, 1, 1)
        elif angle.dim() == 2:
            angle = angle.view(B, 1, 1)
            
        # Multiply angle by learned frequencies. Shape: (B, 1, C // 2)
        # It will automatically broadcast across the N dimension during rotation
        theta = angle * self.freqs
        
        # Split features into pairs. 
        # Using ... handles the (B, N) dimensions automatically
        x1 = x[..., 0::2]  # Shape: (B, N, C // 2)
        x2 = x[..., 1::2]  # Shape: (B, N, C // 2)
        
        # Apply 2D rotation matrix to each pair
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        x1_rot = x1 * cos_theta - x2 * sin_theta
        x2_rot = x1 * sin_theta + x2 * cos_theta
        
        # Interleave the rotated pairs back together
        out = torch.empty_like(x)
        out[..., 0::2] = x1_rot
        out[..., 1::2] = x2_rot
        
        return out

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

        # if self.useap:
        #     self.pool_info_sat = nn.AdaptiveAvgPool2d((3, 3))
        # else:
        self.attnPooling = PoolingHead(self.vision_model.head, SiglipVisionConfig())
        self.pooler_rotator = Rotator(self.feature_dim)
        self.gate_fc = nn.Linear(proj_dim * 2, 1)
        # self.fuse_fc = nn.Linear(proj_dim, proj_dim)

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
        if angle is not None:
            angle_rad = torch.deg2rad(angle)
            anchor_feats_rotate = self.pooler_rotator(anchor_feats, angle_rad)
            anchor_pooler = self.vision_model.head(anchor_feats_rotate)
        else:
            anchor_pooler = anchor_output.pooler_output


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
        sat_feature_2d_pool = self.attnPooling(sat_feats)

        text_feats = None
        if input_ids is not None:
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
        proj_dim=PROJECTION_DIM,
        usesg=False,
        useap=True,
    ):
        super().__init__()

        try:
            vision_backbone = AutoModel.from_pretrained(
                model_name,
                cache_dir=CACHE_DIR,
                trust_remote_code=True,
            )
            self.vision_model = vision_backbone

            text_backbone = AutoModel.from_pretrained(
                text_model_name,
                cache_dir=CACHE_DIR,
                trust_remote_code=True,
            )
            self.text_model = getattr(text_backbone, "text_model", None)

            self.feature_dim = int(getattr(self.vision_model.config, "hidden_size", proj_dim))
            if self.text_model is not None:
                self.text_feature_dim = self.text_model.config.hidden_size
            else:
                self.text_feature_dim = proj_dim
        except Exception as e:
            print(f"Error loading C-RADIOv4-H model: {e}")
            raise

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

    def _vision_forward(self, pixel_values):
        try:
            return self.vision_model(pixel_values, interpolate_pos_encoding=True)
        except TypeError:
            try:
                return self.vision_model(
                    pixel_values=pixel_values,
                    interpolate_pos_encoding=True,
                )
            except TypeError:
                try:
                    return self.vision_model(pixel_values)
                except TypeError:
                    return self.vision_model(pixel_values=pixel_values)

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

    def forward(self, anchor_pixel_values, search_pixel_values, input_ids=None):
        B = anchor_pixel_values.shape[0]

        anchor_output = self._vision_forward(anchor_pixel_values)
        anchor_hidden = anchor_output.last_hidden_state
        anchor_h, anchor_w = anchor_pixel_values.shape[-2:]
        anchor_pooler_raw, anchor_patch_tokens = self._split_tokens(
            anchor_hidden, anchor_h, anchor_w
        )

        sat_output = self._vision_forward(search_pixel_values)
        sat_hidden = sat_output.last_hidden_state
        image_h, image_w = search_pixel_values.shape[-2:]
        _, sat_patch_tokens = self._split_tokens(sat_hidden, image_h, image_w)

        N = sat_patch_tokens.shape[1]
        H, W = _infer_patch_grid(N, image_h, image_w)
        sat_features_2d = sat_patch_tokens.permute(0, 2, 1).reshape(
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

        anchor_pooler = self.anchor_projector(anchor_pooler_raw)
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
        sat_hidden = sat_output.last_hidden_state
        image_h, image_w = search_pixel_values.shape[-2:]
        _, sat_patch_tokens = self._split_tokens(sat_hidden, image_h, image_w)

        B = sat_patch_tokens.shape[0]
        N = sat_patch_tokens.shape[1]
        H, W = _infer_patch_grid(N, image_h, image_w)
        sat_features_2d = sat_patch_tokens.permute(0, 2, 1).reshape(
            B, self.feature_dim, H, W
        )

        sat_feature_2d_pool = (
            self.pool_info_sat(sat_features_2d)
            .reshape(B, self.feature_dim, -1)
            .permute(0, 2, 1)
        )

        return sat_feature_2d_pool

if __name__ == "__main__":
    model = Encoder_gem()
    print(f"Model loaded successfully")
    print(f"Feature dim: {model.feature_dim}")
    print(f"Text feature dim: {model.text_feature_dim}")

    dummy_feat = torch.randn(2, 768, 20, 20)
    gem_out = model.pool_sat(dummy_feat)
    print(f"GeM output shape: {gem_out.shape}")

    text = torch.randn(2, 768)
    anchor = torch.randn(2, 768)
    fused = model.gated_fusion(text, anchor)
    print(f"Fused output shape: {fused.shape}")
