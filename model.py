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


class AngleConditionedFiLM(nn.Module):
    def __init__(self, angle_dim, feature_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(angle_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, features, angle_vec):
        cond_params = self.mlp(angle_vec.to(dtype=features.dtype))
        gamma, beta = cond_params.chunk(2, dim=1)

        if features.ndim == 4:
            gamma = gamma.view(-1, features.size(1), 1, 1)
            beta = beta.view(-1, features.size(1), 1, 1)
        elif features.ndim == 3:
            gamma = gamma.view(-1, 1, features.size(-1))
            beta = beta.view(-1, 1, features.size(-1))
        elif features.ndim == 2:
            gamma = gamma.view(-1, features.size(1))
            beta = beta.view(-1, features.size(1))
        else:
            raise ValueError(f"Unsupported feature shape for FiLM: {tuple(features.shape)}")

        return features * (1 + gamma) + beta
    
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

    def forward(
        self,
        anchor_pixel_values,
        search_pixel_values,
        input_ids=None,
        angle=None,
        attention_mask=None,
    ):
        B = anchor_pixel_values.shape[0]
        # with torch.no_grad():
        anchor_output = self.vision_model(anchor_pixel_values)
        anchor_feats = anchor_output.last_hidden_state
    
        anchor_pooler = anchor_output.pooler_output
        text_feats = None
        if input_ids is not None:
            text_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
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
        self.angle_film = AngleConditionedFiLM(angle_dim=3, feature_dim=self.feature_dim)

        nn.init.zeros_(self.bbox_adapter[-1].weight)
        nn.init.zeros_(self.bbox_adapter[-1].bias)
        self.text_weight = nn.Parameter(torch.tensor([0.693]))
        self.f_param = nn.Parameter(torch.tensor([0.693]))

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
            if idx == 1:
                hidden_states = hidden_states * (1+angle_feat_1) + angle_feat_2

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

    def forward(
        self,
        anchor_pixel_values,
        search_pixel_values,
        input_ids=None,
        angle=None,
        attention_mask=None,
    ):
        B = anchor_pixel_values.shape[0]
        # with torch.no_grad():

        anchor_output = self.vision_model(anchor_pixel_values)
        anchor_feats = anchor_output.last_hidden_state


        anchor_pooler = self.attnPooling(anchor_feats,1)[:,0,:]
        text_feats = torch.ones(B, self.feature_dim, device=anchor_feats.device, dtype=anchor_feats.dtype)
        
        if input_ids is not None:
            text_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
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

        anchor_feats_mod = anchor_feats
        if angle is not None:
            anchor_feats_mod = self.angle_film(anchor_feats.detach(), angle)

        if angle is not None:
            anchor_feats_rot = anchor_feats_mod
            anchor_context = (
                self.bbox_adapter(anchor_feats_rot) + anchor_feats_rot
            )

            #3d rotation in feature space
            # alpha_rad = -45 / 180 * math.pi
            # R_x = torch.tensor([
            #     [1., 0., 0.],
            #     [0., math.cos(alpha_rad), -math.sin(alpha_rad)],
            #     [0., math.sin(alpha_rad), math.cos(alpha_rad)]
            # ],device=anchor_feats.device, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)

            # cos_b = angle[:, 0]
            # sin_b = angle[:, 1]

            # R_z = torch.zeros((B, 3, 3), device=anchor_feats.device, dtype=torch.float32)
            # R_z[:, 0, 0] = cos_b
            # R_z[:, 0, 1] = -sin_b
            # R_z[:, 1, 0] = sin_b
            # R_z[:, 1, 1] = cos_b
            # R_z[:, 2, 2] = 1.0

            # R_rectify = torch.bmm(R_x, R_z)

            # K = torch.zeros((B, 3, 3), device=anchor_feats.device, dtype=torch.float32)
            # K[:, 0, 0] = 14*F.softplus(self.f_param)
            # K[:, 1, 1] = 14*F.softplus(self.f_param)
            # K[:, 0, 2] = 7
            # K[:, 1, 2] = 7
            # K[:, 2, 2] = 1.0
            # K_inv = torch.inverse(K)
            # trans = torch.bmm(K, torch.bmm(R_rectify, K_inv))
            # anchor_feats_2d = anchor_feats.detach().view(B, 14, 14, self.feature_dim).permute(0, 3, 1, 2)
            # anchor_feats_2d_rot = kornia.geometry.transform.warp_perspective(
            #             anchor_feats_2d, 
            #             trans, 
            #             dsize=(anchor_feats_2d.shape[2], anchor_feats_2d.shape[3]), 
            #             mode='bilinear', 
            #             padding_mode='zeros'
            #         )
            # anchor_feats_rot = anchor_feats_2d_rot.permute(0, 2, 3, 1).reshape(B, 196, self.feature_dim)
            # anchor_context = torch.cat(
            #     [self.bbox_adapter(anchor_feats_rot), anchor_feats_rot], dim=1
            # )

            # 2d rotation in feature space
            # cos_theta = angle[:, 0]
            # sin_theta = angle[:, 1]
            # theta = torch.zeros(B, 2, 3, device=anchor_feats.device, dtype=anchor_feats.dtype)
            # theta[:, 0, 0] = cos_theta
            # theta[:, 0, 1] = -sin_theta
            # theta[:, 1, 0] = sin_theta
            # theta[:, 1, 1] = cos_theta

            # anchor_feats_2d = anchor_feats.detach().view(B, 14, 14, self.feature_dim).permute(0, 3, 1, 2)
            # rotate_grid = F.affine_grid(theta, anchor_feats_2d.size(), align_corners=False)
            # anchor_feats_2d_rot = F.grid_sample(
            #     anchor_feats_2d,
            #     rotate_grid,
            #     mode="bilinear",
            #     padding_mode="zeros",
            #     align_corners=False,
            # )
            # anchor_feats_rot = anchor_feats_2d_rot.permute(0, 2, 3, 1).reshape(B, 196, self.feature_dim)
            # anchor_context = (
            #     self.bbox_adapter(anchor_feats_rot) + anchor_feats_rot
            # )
        else:
            anchor_feats_rot = anchor_feats_mod.detach()

            anchor_context = (
                self.bbox_adapter(anchor_feats_rot) + anchor_feats_rot
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


class Encoder_heat(Encoder_text_angle):
    def __init__(
        self,
        model_name=MODEL_NAME,
        proj_dim=PROJECTION_DIM,
        usesg=False,
        useap=False,
        heat_channels=128,
        heat_kernel_size=9,
    ):
        super().__init__(
            model_name=model_name,
            proj_dim=proj_dim,
            usesg=usesg,
            useap=useap,
        )
        del heat_channels
        self.heat_kernel_size = heat_kernel_size
        self.heat_fuse = nn.Conv2d(self.feature_dim + 1, self.feature_dim, kernel_size=1)
        self.bbox_pos_proj = nn.Conv2d(2, self.feature_dim, kernel_size=1)

        nn.init.zeros_(self.heat_fuse.weight)
        nn.init.zeros_(self.heat_fuse.bias)
        with torch.no_grad():
            for idx in range(self.feature_dim):
                self.heat_fuse.weight[idx, idx, 0, 0] = 1.0

        nn.init.zeros_(self.bbox_pos_proj.weight)
        nn.init.zeros_(self.bbox_pos_proj.bias)

    def _infer_anchor_grid(self, token_count):
        grid_h = int(token_count**0.5)
        while grid_h > 1 and token_count % grid_h != 0:
            grid_h -= 1
        grid_w = token_count // grid_h
        return grid_h, grid_w

    def _rotate_anchor_grid(self, anchor_grid, geo):
        B = anchor_grid.shape[0]
        if geo is None:
            return anchor_grid

        geo = geo.to(device=anchor_grid.device, dtype=anchor_grid.dtype)
        cos_theta = geo[:, 0]
        sin_theta = geo[:, 1]

        theta = torch.zeros(B, 2, 3, device=anchor_grid.device, dtype=anchor_grid.dtype)
        theta[:, 0, 0] = cos_theta
        theta[:, 0, 1] = sin_theta
        theta[:, 1, 0] = -sin_theta
        theta[:, 1, 1] = cos_theta

        rotate_grid = F.affine_grid(theta, anchor_grid.size(), align_corners=False)
        return F.grid_sample(
            anchor_grid,
            rotate_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

    def _dynamic_heatmap(self, anchor_feats, sat_features_2d, geo):
        B, C, H, W = sat_features_2d.shape
        anchor_h, anchor_w = self._infer_anchor_grid(anchor_feats.shape[1])
        anchor_grid = (
            anchor_feats.detach()
            .transpose(1, 2)
            .reshape(B, C, anchor_h, anchor_w)
            .contiguous()
        )

        rotated_anchor = self._rotate_anchor_grid(anchor_grid, geo)
        kernel = F.adaptive_avg_pool2d(
            rotated_anchor,
            (self.heat_kernel_size, self.heat_kernel_size),
        )
        kernel = kernel - kernel.mean(dim=(2, 3), keepdim=True)
        kernel = F.normalize(kernel.flatten(1), p=2, dim=1).view_as(kernel)

        sat_features = sat_features_2d.contiguous()
        sat_features = F.normalize(sat_features, p=2, dim=1)
        conv_input = sat_features.reshape(1, B * C, H, W)
        conv_kernel = kernel.reshape(
            B,
            C,
            self.heat_kernel_size,
            self.heat_kernel_size,
        )
        heatmap = F.conv2d(
            conv_input,
            conv_kernel,
            padding=self.heat_kernel_size // 2,
            groups=B,
        )
        return heatmap.view(B, 1, H, W)

    def _build_position_embedding(self, feature_map):
        B, _, H, W = feature_map.shape
        y_coords = torch.linspace(
            -1.0,
            1.0,
            H,
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
        x_coords = torch.linspace(
            -1.0,
            1.0,
            W,
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        return self.bbox_pos_proj(coords)

    def forward(
        self,
        anchor_pixel_values,
        search_pixel_values,
        input_ids=None,
        angle=None,
        attention_mask=None,
    ):
        B = anchor_pixel_values.shape[0]

        anchor_output = self.vision_model(anchor_pixel_values)
        anchor_feats = anchor_output.last_hidden_state

        anchor_pooler = self.attnPooling(anchor_feats, 1)[:, 0, :]
        text_feats = None
        if input_ids is not None:
            text_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            text_feats = F.normalize(
                self.text_projector(text_outputs.pooler_output),
                p=2,
                dim=1,
            )
            anchor_pooler = F.normalize(anchor_pooler, p=2, dim=1)
            text_weight = F.softplus(self.text_weight) / 20
            anchor_pooler = F.normalize(
                (1 - text_weight) * anchor_pooler + text_weight * text_feats,
                p=2,
                dim=1,
            )


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

        if angle is not None:
            if self.usesg:
                anchor_feats_mod = self.angle_film(anchor_feats.detach(), angle)
            else:
                anchor_feats_mod = self.angle_film(anchor_feats, angle)
            anchor_context = self.bbox_adapter(anchor_feats_mod) + anchor_feats_mod
        else:
            anchor_feats_detached = anchor_feats.detach()
            anchor_context = self.bbox_adapter(anchor_feats_detached) + anchor_feats_detached

        heatmap = self._dynamic_heatmap(anchor_feats, sat_features_2d, angle)
        if heatmap.shape[-2:] != sat_features_2d.shape[-2:]:
            heatmap = F.interpolate(
                heatmap,
                size=sat_features_2d.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        heat_gate = torch.sigmoid(heatmap)
        guided_sat_features = self.heat_fuse(
            torch.cat([sat_features_2d, heat_gate], dim=1)
        )
        guided_sat_features = guided_sat_features + self._build_position_embedding(
            guided_sat_features
        )

        fused_features = self.bbox_transformer(
            x=guided_sat_features, context=anchor_context
        )

        pred_anchor = self.bbox_fcn_out(fused_features)
        if heatmap.shape[-2:] != pred_anchor.shape[-2:]:
            heatmap = F.interpolate(
                heatmap,
                size=pred_anchor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return (
            pred_anchor,
            None,
            text_feats,
            anchor_pooler,
            sat_feature_2d_pool,
            fused_feats,
            heatmap,
        )


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

    def forward(
        self,
        anchor_pixel_values,
        search_pixel_values,
        input_ids=None,
        angle=None,
        attention_mask=None,
    ):
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
                text_outputs = self.text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
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
