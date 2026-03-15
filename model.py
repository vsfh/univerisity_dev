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
CACHE_DIR = "/data/feihong/hf_cache"
PROJECTION_DIM = 768


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

class SiglipMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 9, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

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
        H = W = int(N**0.5)
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
        H = W = int(N**0.5)
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
        H = W = int(N**0.5)
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
        H = W = int(N**0.5)
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
        H = W = int(N**0.5)
        sat_features_2d = sat_feats.permute(0, 2, 1).reshape(B, self.feature_dim, H, W)

        sat_feature_2d_pool = (
            self.pool_info_sat(sat_features_2d)
            .reshape(B, self.feature_dim, -1)
            .permute(0, 2, 1)
        )

        return sat_feature_2d_pool
        

class ScoreHead(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 1. Projections (Only Query and Key needed for scoring)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        
        # 2. Score Fusion Network
        # Fuses the 8 different head scores into 1 final similarity score
        self.score_fusion = nn.Sequential(
            nn.Linear(num_heads, num_heads // 2),
            nn.GELU(),
            nn.Linear(num_heads // 2, 1)
        )

    def forward(self, query: torch.Tensor, search: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: Tensor of shape (B, 768)
            search: Tensor of shape (B * n, 768)
        Returns:
            scores: Tensor of shape (B, B * n)
        """
        B = query.size(0)
        N_total = search.size(0) # This is B * n
        
        # --- STEP 1: Linear Projection & Head Splitting ---
        # (B, 768) -> (B, num_heads, head_dim)
        q = self.q_proj(query).view(B, self.num_heads, self.head_dim)
        
        # (B*n, 768) -> (B*n, num_heads, head_dim)
        k = self.k_proj(search).view(N_total, self.num_heads, self.head_dim)
        
        # --- STEP 2: Prepare for Global Matrix Multiplication ---
        # We want num_heads as the batch dimension for torch.bmm
        # q becomes: (num_heads, B, head_dim)
        q = q.transpose(0, 1) 
        
        # k becomes: (num_heads, head_dim, B*n)
        k = k.permute(1, 2, 0) 
        
        
        
        # --- STEP 3: Compute All-to-All Attention Logits ---
        # (num_heads, B, head_dim) @ (num_heads, head_dim, B*n) -> (num_heads, B, B*n)
        attn_logits = torch.bmm(q, k) * self.scale
        
        # --- STEP 4: Fuse Multi-Head Scores ---
        # Rearrange to put num_heads last: (B, B*n, num_heads)
        attn_logits = attn_logits.permute(1, 2, 0)
        
        # Pass through MLP to get 1 score per query-document pair -> (B, B*n, 1)
        scores = self.score_fusion(attn_logits)
        
        # --- STEP 5: Final Output Shape ---
        # Remove the last dimension: (B, B*n, 1) -> (B, B*n)
        return scores.squeeze(-1)
    
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

        if self.useap:
            self.pool_info_sat = nn.AdaptiveAvgPool2d((3, 3))
        else:
            self.attnPooling = SiglipMultiheadAttentionPoolingHead(SiglipVisionConfig(hidden_size=self.feature_dim, num_attention_heads=12, layer_norm_eps=1e-6))

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
        self.scorer = ScoreHead(embed_dim=self.feature_dim, num_heads=12)

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

        with torch.no_grad():
            anchor_output = self.vision_model(anchor_pixel_values)
            anchor_feats = anchor_output.last_hidden_state
            anchor_pooler = anchor_output.pooler_output

            sat_output = self.vision_model(
                search_pixel_values, interpolate_pos_encoding=True
            )
            sat_feats = sat_output.last_hidden_state
            
        N = sat_feats.shape[1]
        H = W = int(N**0.5)
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
        pred_heading = torch.sigmoid(self.bbox_heading(fused_features))

        if self.useap:
            sat_feature_2d_pool = (
                self.pool_info_sat(sat_features_2d)
                .reshape(B, self.feature_dim, -1)
                .permute(0, 2, 1)
            )
        else:
            sat_feature_2d_pool = self.attnPooling(sat_feats)

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
        H = W = int(N**0.5)
        sat_features_2d = sat_feats.permute(0, 2, 1).reshape(B, self.feature_dim, H, W)

        sat_feature_2d_pool = (
            self.pool_info_sat(sat_features_2d)
            .reshape(B, self.feature_dim, -1)
            .permute(0, 2, 1)
        )

        return sat_feature_2d_pool
        
class Encoder_test(nn.Module):
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
        H = W = int(N**0.5)
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
        pred_heading = torch.sigmoid(self.bbox_heading(fused_features))

        if self.useap:
            sat_feature_2d_pool = (
                self.pool_info_sat(sat_features_2d)
                .reshape(B, self.feature_dim, -1)
                .permute(0, 2, 1)
            )
        else:
            sat_feature_2d_pool = sat_output.pooler_output

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
        H = W = int(N**0.5)
        sat_features_2d = sat_feats.permute(0, 2, 1).reshape(B, self.feature_dim, H, W)

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
