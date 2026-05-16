from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers.models.siglip.modeling_siglip import SiglipVisionConfig

from bbox.yolo_utils import SpatialTransformer


# --- Configuration ---
MODEL_NAME = "google/siglip-base-patch16-224"
SIGLIP2_MODEL_NAME = "google/siglip2-base-patch16-224"
CACHE_DIR = "/media/data1/feihong/hf_cache"
PROJECTION_DIM = 768


def _infer_patch_grid(token_count: int, image_h: int, image_w: int) -> Tuple[int, int]:
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
        if h <= 1 or w <= 1:
            continue

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
            f"Cannot infer patch grid for token_count={token_count}, "
            f"image_shape={(image_h, image_w)}."
        )
    return best_h, best_w


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class AngleConditionedFiLM(nn.Module):
    def __init__(self, angle_dim: int, feature_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(angle_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, features: torch.Tensor, angle_vec: torch.Tensor) -> torch.Tensor:
        if angle_vec.ndim != 2 or angle_vec.shape[1] != 3:
            raise ValueError(f"Expected angle_vec shape (B, 3), got {tuple(angle_vec.shape)}.")
        if angle_vec.shape[0] != features.shape[0]:
            raise ValueError(
                f"Batch mismatch between features={features.shape[0]} and angle={angle_vec.shape[0]}."
            )

        cond_params = self.mlp(angle_vec.to(device=features.device, dtype=features.dtype))
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
            raise ValueError(f"Unsupported feature shape for FiLM: {tuple(features.shape)}.")

        return features * (1 + gamma) + beta


class PoolingHead(nn.Module):
    def __init__(self, siglip_pooling_head: nn.Module, config: SiglipVisionConfig):
        super().__init__()
        del config
        self.probe = siglip_pooling_head.probe
        self.head = siglip_pooling_head

    def forward(self, hidden_state: torch.Tensor, dim: int) -> torch.Tensor:
        if hidden_state.ndim != 3:
            raise ValueError(f"Expected hidden_state shape (B, N, C), got {tuple(hidden_state.shape)}.")
        if dim <= 0:
            raise ValueError(f"Pooling dim must be positive, got {dim}.")

        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, dim, 1)
        hidden_state = self.head.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.head.layernorm(hidden_state)
        hidden_state = residual + self.head.mlp(hidden_state)
        return hidden_state


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")

        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False

        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scaling = float(alpha) / float(rank)

        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


class Encoder_heat(nn.Module):
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        proj_dim: int = PROJECTION_DIM,
        usesg: bool = False,
        useap: bool = False,
        heat_channels: int = 128,
        heat_kernel_size: int = 9,
        heat_softmax_temperature: float = 1.0,
    ):
        super().__init__()
        del heat_channels

        model = AutoModel.from_pretrained(model_name, cache_dir=CACHE_DIR)
        self.vision_model = model.vision_model
        self.text_model = model.text_model

        self.feature_dim = int(self.vision_model.config.hidden_size)
        self.text_feature_dim = int(self.text_model.config.hidden_size)
        if proj_dim != self.feature_dim:
            raise ValueError(
                f"proj_dim={proj_dim} must match vision hidden size {self.feature_dim}."
            )

        self.usesg = usesg
        self.useap = useap
        self.heat_kernel_size = int(heat_kernel_size)
        self.heat_softmax_temperature = float(heat_softmax_temperature)
        self.roi_background_keep = 0.05
        self.roi_soft_tau = 4.0

        self.attnPooling = PoolingHead(self.vision_model.head, SiglipVisionConfig())
        if self.text_feature_dim != self.feature_dim:
            raise ValueError(
                f"Raw text pooler dim {self.text_feature_dim} must match "
                f"vision hidden size {self.feature_dim}."
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
        self.angle_film = AngleConditionedFiLM(angle_dim=3, feature_dim=self.feature_dim)
        self.heat_fuse = nn.Conv2d(self.feature_dim + 1, self.feature_dim, kernel_size=1)
        self.bbox_pos_proj = nn.Conv2d(2, self.feature_dim, kernel_size=1)

        nn.init.zeros_(self.bbox_adapter[-1].weight)
        nn.init.zeros_(self.bbox_adapter[-1].bias)
        nn.init.zeros_(self.heat_fuse.weight)
        nn.init.zeros_(self.heat_fuse.bias)
        with torch.no_grad():
            for idx in range(self.feature_dim):
                self.heat_fuse.weight[idx, idx, 0, 0] = 1.0
        nn.init.zeros_(self.bbox_pos_proj.weight)
        nn.init.zeros_(self.bbox_pos_proj.bias)

    def _vision_forward(
        self,
        pixel_values: torch.Tensor,
        image_size: Optional[Tuple[int, int]] = None,
        interpolate_pos_encoding: bool = False,
    ):
        del image_size
        if pixel_values.ndim != 4:
            raise ValueError(f"Expected image tensor shape (B, C, H, W), got {tuple(pixel_values.shape)}.")
        return self.vision_model(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

    def _feature_grid_shape(
        self,
        pixel_values: torch.Tensor,
        image_size: Optional[Tuple[int, int]],
        token_count: int,
    ) -> Tuple[int, int]:
        del image_size
        if pixel_values.ndim != 4:
            raise ValueError(f"Expected image tensor shape (B, C, H, W), got {tuple(pixel_values.shape)}.")
        return _infer_patch_grid(token_count, pixel_values.shape[-2], pixel_values.shape[-1])

    def _infer_anchor_grid(self, token_count: int) -> Tuple[int, int]:
        grid_h = int(token_count**0.5)
        while grid_h > 1 and token_count % grid_h != 0:
            grid_h -= 1
        grid_w = token_count // grid_h
        if grid_h * grid_w != token_count:
            raise ValueError(f"Cannot infer anchor grid for token_count={token_count}.")
        return grid_h, grid_w

    def _validate_geo(self, geo: Optional[torch.Tensor], batch_size: int) -> None:
        if geo is None:
            return
        if geo.ndim != 2 or geo.shape != (batch_size, 3):
            raise ValueError(f"Expected geo shape {(batch_size, 3)}, got {tuple(geo.shape)}.")

    def _rotate_anchor_grid(
        self,
        anchor_grid: torch.Tensor,
        geo: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B = anchor_grid.shape[0]
        self._validate_geo(geo, B)
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

    def _project_text(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if input_ids is None:
            return None

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return F.normalize(text_outputs.pooler_output, p=2, dim=1)

    def _text_dynamic_heatmap(
        self,
        text_feats: Optional[torch.Tensor],
        sat_features_2d: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if text_feats is None:
            return None

        B, C, H, W = sat_features_2d.shape
        if text_feats.shape != (B, C):
            raise ValueError(f"Expected text_feats shape {(B, C)}, got {tuple(text_feats.shape)}.")

        text_kernel = text_feats.to(
            device=sat_features_2d.device,
            dtype=sat_features_2d.dtype,
        )
        text_kernel = F.normalize(text_kernel, p=2, dim=1)
        sat_features = F.normalize(sat_features_2d.contiguous(), p=2, dim=1)

        conv_input = sat_features.reshape(1, B * C, H, W)
        conv_kernel = text_kernel.reshape(B, C, 1, 1)
        return F.conv2d(conv_input, conv_kernel, groups=B).view(B, 1, H, W)

    def _build_text_roi_mask(
        self,
        text_heatmap: Optional[torch.Tensor],
        geo: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if text_heatmap is None:
            return None

        B = text_heatmap.shape[0]
        self._validate_geo(geo, B)
        heatmap = text_heatmap.float()
        mean = heatmap.mean(dim=(-2, -1), keepdim=True)
        std = heatmap.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        score = (heatmap - mean) / std

        if geo is None:
            height_norm = torch.ones(B, 1, 1, 1, device=heatmap.device, dtype=heatmap.dtype)
        else:
            height_norm = geo[:, 2].to(device=heatmap.device, dtype=heatmap.dtype)
            height_norm = height_norm.view(B, 1, 1, 1).clamp(0.5, 1.0)

        low_h = torch.full_like(height_norm, 0.5)
        mid_h = torch.full_like(height_norm, 2.0 / 3.0)
        high_h = torch.full_like(height_norm, 5.0 / 6.0)
        top_h = torch.ones_like(height_norm)
        keep_150 = torch.full_like(height_norm, 0.08)
        keep_200 = torch.full_like(height_norm, 0.12)
        keep_250 = torch.full_like(height_norm, 0.18)
        keep_300 = torch.full_like(height_norm, 0.25)

        keep_ratio_150_200 = keep_150 + (height_norm - low_h) / (mid_h - low_h) * (keep_200 - keep_150)
        keep_ratio_200_250 = keep_200 + (height_norm - mid_h) / (high_h - mid_h) * (keep_250 - keep_200)
        keep_ratio_250_300 = keep_250 + (height_norm - high_h) / (top_h - high_h) * (keep_300 - keep_250)
        keep_ratio = torch.where(
            height_norm <= mid_h,
            keep_ratio_150_200,
            torch.where(height_norm <= high_h, keep_ratio_200_250, keep_ratio_250_300),
        ).clamp(0.08, 0.25)

        flat_score = score.flatten(1)
        keep_count = torch.ceil(keep_ratio.flatten() * flat_score.shape[1]).long()
        keep_count = keep_count.clamp(1, flat_score.shape[1])
        sorted_score = flat_score.sort(dim=1, descending=True).values
        threshold = sorted_score.gather(1, keep_count[:, None] - 1)
        threshold = threshold.view(B, 1, 1, 1)
        roi_mask = torch.sigmoid((score - threshold) * self.roi_soft_tau)
        return roi_mask.to(dtype=text_heatmap.dtype)

    def _dynamic_heatmap(
        self,
        anchor_feats: torch.Tensor,
        sat_features_2d: torch.Tensor,
        geo: Optional[torch.Tensor],
        text_feats: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if anchor_feats.ndim != 3:
            raise ValueError(f"Expected anchor_feats shape (B, N, C), got {tuple(anchor_feats.shape)}.")
        if sat_features_2d.ndim != 4:
            raise ValueError(f"Expected sat_features_2d shape (B, C, H, W), got {tuple(sat_features_2d.shape)}.")

        B, C, H, W = sat_features_2d.shape
        if anchor_feats.shape[0] != B or anchor_feats.shape[2] != C:
            raise ValueError(
                f"Anchor/satellite mismatch: anchor={tuple(anchor_feats.shape)}, "
                f"sat={tuple(sat_features_2d.shape)}."
            )

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

        text_heatmap = self._text_dynamic_heatmap(text_feats, sat_features_2d)
        roi_mask = self._build_text_roi_mask(text_heatmap, geo)

        sat_features = F.normalize(sat_features_2d.contiguous(), p=2, dim=1)
        if roi_mask is not None:
            roi_gate = self.roi_background_keep + (1.0 - self.roi_background_keep) * roi_mask
            sat_features = sat_features * roi_gate

        conv_input = sat_features.reshape(1, B * C, H, W)
        conv_kernel = kernel.reshape(B, C, self.heat_kernel_size, self.heat_kernel_size)
        heatmap = F.conv2d(
            conv_input,
            conv_kernel,
            padding=self.heat_kernel_size // 2,
            groups=B,
        ).view(B, 1, H, W)
        return heatmap, text_heatmap, roi_mask

    def _spatial_softmax(self, heatmap: torch.Tensor) -> torch.Tensor:
        if heatmap.ndim != 4:
            raise ValueError(f"Expected heatmap shape (B, C, H, W), got {tuple(heatmap.shape)}.")

        B, C, H, W = heatmap.shape
        temperature = max(self.heat_softmax_temperature, 1e-6)
        heatmap = heatmap.view(B, C, -1) / temperature
        heatmap = F.softmax(heatmap, dim=-1)
        return heatmap.view(B, C, H, W)

    def _build_position_embedding(self, feature_map: torch.Tensor) -> torch.Tensor:
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

    def _anchor_context(
        self,
        anchor_feats: torch.Tensor,
        geo: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if geo is not None:
            if self.usesg:
                anchor_feats = self.angle_film(anchor_feats.detach(), geo)
            else:
                anchor_feats = self.angle_film(anchor_feats, geo)
            return self.bbox_adapter(anchor_feats) + anchor_feats

        anchor_feats = anchor_feats.detach()
        return self.bbox_adapter(anchor_feats) + anchor_feats

    def _resize_heatmap_to_pred(
        self,
        heatmap: Optional[torch.Tensor],
        pred_anchor: torch.Tensor,
        apply_softmax: bool,
    ) -> Optional[torch.Tensor]:
        if heatmap is None:
            return None
        if heatmap.shape[-2:] != pred_anchor.shape[-2:]:
            heatmap = F.interpolate(
                heatmap,
                size=pred_anchor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if apply_softmax:
            return self._spatial_softmax(heatmap)
        return heatmap.clamp(0.0, 1.0)

    def forward(
        self,
        anchor_pixel_values: torch.Tensor,
        search_pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        angle: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        B = anchor_pixel_values.shape[0]
        self._validate_geo(angle, B)

        anchor_output = self._vision_forward(anchor_pixel_values)
        anchor_feats = anchor_output.last_hidden_state
        if anchor_feats.ndim != 3 or anchor_feats.shape[0] != B or anchor_feats.shape[2] != self.feature_dim:
            raise ValueError(f"Unexpected anchor feature shape: {tuple(anchor_feats.shape)}.")

        anchor_pooler = self.attnPooling(anchor_feats, 1)[:, 0, :]
        anchor_pooler_x = F.normalize(anchor_pooler, p=2, dim=1)
        text_feats = self._project_text(input_ids, attention_mask)

        sat_output = self._vision_forward(
            search_pixel_values,
            interpolate_pos_encoding=True,
        )
        sat_feats = sat_output.last_hidden_state
        if sat_feats.ndim != 3 or sat_feats.shape[0] != B or sat_feats.shape[2] != self.feature_dim:
            raise ValueError(f"Unexpected satellite feature shape: {tuple(sat_feats.shape)}.")

        sat_feature_2d_pool = self.attnPooling(sat_feats, 9)
        H, W = self._feature_grid_shape(
            search_pixel_values,
            image_size=None,
            token_count=sat_feats.shape[1],
        )
        sat_features_2d = sat_feats.permute(0, 2, 1).reshape(B, self.feature_dim, H, W)

        anchor_context = self._anchor_context(anchor_feats, angle)
        heatmap_logits, text_heatmap_logits, roi_mask = self._dynamic_heatmap(
            anchor_feats,
            sat_features_2d,
            angle,
            text_feats=text_feats,
        )
        heatmap = self._spatial_softmax(heatmap_logits)

        heat_gate = heatmap * heatmap.shape[-2] * heatmap.shape[-1] / 2
        guided_sat_features = self.heat_fuse(
            torch.cat([sat_features_2d, heat_gate], dim=1)
        )
        guided_sat_features = guided_sat_features + self._build_position_embedding(
            guided_sat_features
        )
        fused_features = self.bbox_transformer(
            x=guided_sat_features,
            context=anchor_context,
        )
        pred_anchor = self.bbox_fcn_out(fused_features)

        return (
            pred_anchor,
            None,
            text_feats,
            anchor_pooler_x,
            sat_feature_2d_pool,
            None,
            self._resize_heatmap_to_pred(heatmap_logits, pred_anchor, apply_softmax=True),
            self._resize_heatmap_to_pred(text_heatmap_logits, pred_anchor, apply_softmax=True),
            self._resize_heatmap_to_pred(roi_mask, pred_anchor, apply_softmax=False),
        )


class Encoder_sig(Encoder_heat):
    """Heat encoder using SigLIP2 weights.

    In the currently installed Transformers build, Google's SigLIP2 checkpoint
    is exposed through the regular SigLIP model class and image-tensor forward
    API. The strict check here is therefore on the requested checkpoint name,
    while the inherited forward path validates all tensor shapes.
    """

    def __init__(
        self,
        model_name: str = SIGLIP2_MODEL_NAME,
        proj_dim: int = PROJECTION_DIM,
        usesg: bool = False,
        useap: bool = False,
        heat_channels: int = 128,
        heat_kernel_size: int = 9,
        heat_softmax_temperature: float = 1.0,
        query_image_size: Tuple[int, int] = (256, 256),
        search_image_size: Tuple[int, int] = (432, 768),
    ):
        if "siglip2" not in model_name.lower():
            raise ValueError(f"Encoder_sig requires a SigLIP2 checkpoint name, got {model_name}.")

        super().__init__(
            model_name=model_name,
            proj_dim=proj_dim,
            usesg=usesg,
            useap=useap,
            heat_channels=heat_channels,
            heat_kernel_size=heat_kernel_size,
            heat_softmax_temperature=heat_softmax_temperature,
        )
        self.query_image_size = query_image_size
        self.search_image_size = search_image_size


class Encoder_lora(Encoder_sig):
    def __init__(
        self,
        model_name: str = SIGLIP2_MODEL_NAME,
        proj_dim: int = PROJECTION_DIM,
        usesg: bool = False,
        useap: bool = False,
        heat_channels: int = 128,
        heat_kernel_size: int = 9,
        heat_softmax_temperature: float = 1.0,
        query_image_size: Tuple[int, int] = (256, 256),
        search_image_size: Tuple[int, int] = (432, 768),
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ):
        super().__init__(
            model_name=model_name,
            proj_dim=proj_dim,
            usesg=usesg,
            useap=useap,
            heat_channels=heat_channels,
            heat_kernel_size=heat_kernel_size,
            heat_softmax_temperature=heat_softmax_temperature,
            query_image_size=query_image_size,
            search_image_size=search_image_size,
        )
        self._freeze_vision_encoder()
        self._freeze_text_encoder()
        replaced = self._inject_lora_into_vision(
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        if replaced <= 0:
            raise ValueError("No Linear layers were replaced by LoRA in vision_model.")

    def _freeze_vision_encoder(self) -> None:
        for param in self.vision_model.parameters():
            param.requires_grad = False

    def _freeze_text_encoder(self) -> None:
        for param in self.text_model.parameters():
            param.requires_grad = False

    def _inject_lora_into_vision(
        self,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> int:
        return self._replace_linear_with_lora(
            module=self.vision_model,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    def _replace_linear_with_lora(
        self,
        module: nn.Module,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> int:
        if isinstance(module, nn.MultiheadAttention):
            return 0

        replaced = 0
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                raise ValueError(f"Nested LoRA module is not expected at {child_name}.")
            if isinstance(child, nn.Linear):
                setattr(
                    module,
                    child_name,
                    LoRALinear(
                        base=child,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                    ),
                )
                replaced += 1
            else:
                replaced += self._replace_linear_with_lora(
                    module=child,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
        return replaced
