import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from bbox.yolo_utils import SpatialTransformer


# --- Configuration ---
MODEL_NAME = "google/siglip2-base-patch16-224"
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
    def __init__(self, siglip_pooling_head: nn.Module):
        super().__init__()
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
        self.dropout = nn.Dropout(float(dropout))
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
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        del heat_channels, useap
        if "siglip2" not in model_name.lower():
            raise ValueError(f"Encoder_heat now requires a SigLIP2 checkpoint, got {model_name}.")

        model = AutoModel.from_pretrained(model_name, cache_dir=CACHE_DIR)
        self.vision_model = model.vision_model
        self.text_model = model.text_model

        self.feature_dim = int(self.vision_model.config.hidden_size)
        self.text_feature_dim = int(self.text_model.config.hidden_size)
        if proj_dim != self.feature_dim:
            raise ValueError(f"proj_dim={proj_dim} must match vision hidden size {self.feature_dim}.")

        self.usesg = usesg
        self.heat_kernel_size = int(heat_kernel_size)
        self.heat_softmax_temperature = float(heat_softmax_temperature)

        self._freeze_backbone()
        replaced = self._inject_lora(
            module=self.vision_model,
            rank=int(lora_rank),
            alpha=float(lora_alpha),
            dropout=float(lora_dropout),
        )
        if replaced <= 0:
            raise ValueError("No vision Linear layers were replaced by LoRA.")

        self.attnPooling = PoolingHead(self.vision_model.head)
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

    def _freeze_backbone(self) -> None:
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.text_model.parameters():
            param.requires_grad = False

    def _inject_lora(
        self,
        module: nn.Module,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> int:
        replaced = 0
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.MultiheadAttention):
                continue
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
                continue
            replaced += self._inject_lora(
                module=child,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
        return replaced

    def _vision_forward(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = False,
    ):
        if pixel_values.ndim != 4:
            raise ValueError(f"Expected image tensor shape (B, C, H, W), got {tuple(pixel_values.shape)}.")
        return self.vision_model(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

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
        batch_size = anchor_grid.shape[0]
        self._validate_geo(geo, batch_size)
        if geo is None:
            return anchor_grid

        geo = geo.to(device=anchor_grid.device, dtype=anchor_grid.dtype)
        cos_theta = geo[:, 0]
        sin_theta = geo[:, 1]

        theta = torch.zeros(batch_size, 2, 3, device=anchor_grid.device, dtype=anchor_grid.dtype)
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

    def _dynamic_heatmap(
        self,
        anchor_feats: torch.Tensor,
        sat_features_2d: torch.Tensor,
        geo: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if anchor_feats.ndim != 3:
            raise ValueError(f"Expected anchor_feats shape (B, N, C), got {tuple(anchor_feats.shape)}.")
        if sat_features_2d.ndim != 4:
            raise ValueError(f"Expected sat_features_2d shape (B, C, H, W), got {tuple(sat_features_2d.shape)}.")

        batch_size, channels, height, width = sat_features_2d.shape
        if anchor_feats.shape[0] != batch_size or anchor_feats.shape[2] != channels:
            raise ValueError(
                f"Anchor/satellite mismatch: anchor={tuple(anchor_feats.shape)}, "
                f"sat={tuple(sat_features_2d.shape)}."
            )

        anchor_h, anchor_w = self._infer_anchor_grid(anchor_feats.shape[1])
        anchor_grid = (
            anchor_feats.detach()
            .transpose(1, 2)
            .reshape(batch_size, channels, anchor_h, anchor_w)
            .contiguous()
        )
        rotated_anchor = self._rotate_anchor_grid(anchor_grid, geo)
        kernel = F.adaptive_avg_pool2d(
            rotated_anchor,
            (self.heat_kernel_size, self.heat_kernel_size),
        )
        kernel = kernel - kernel.mean(dim=(2, 3), keepdim=True)
        kernel = F.normalize(kernel.flatten(1), p=2, dim=1).view_as(kernel)

        sat_features = F.normalize(sat_features_2d.contiguous(), p=2, dim=1)
        conv_input = sat_features.reshape(1, batch_size * channels, height, width)
        conv_kernel = kernel.reshape(batch_size, channels, self.heat_kernel_size, self.heat_kernel_size)
        heatmap = F.conv2d(
            conv_input,
            conv_kernel,
            padding=self.heat_kernel_size // 2,
            groups=batch_size,
        )
        return heatmap.view(batch_size, 1, height, width)

    def _spatial_softmax(self, heatmap: torch.Tensor) -> torch.Tensor:
        if heatmap.ndim != 4:
            raise ValueError(f"Expected heatmap shape (B, C, H, W), got {tuple(heatmap.shape)}.")
        batch_size, channels, height, width = heatmap.shape
        temperature = max(self.heat_softmax_temperature, 1e-6)
        heatmap = heatmap.view(batch_size, channels, -1) / temperature
        heatmap = F.softmax(heatmap, dim=-1)
        return heatmap.view(batch_size, channels, height, width)

    def _build_position_embedding(self, feature_map: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = feature_map.shape
        y_coords = torch.linspace(
            -1.0,
            1.0,
            height,
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
        x_coords = torch.linspace(
            -1.0,
            1.0,
            width,
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        return self.bbox_pos_proj(coords)

    def _anchor_context(
        self,
        anchor_feats: torch.Tensor,
        geo: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if geo is not None:
            anchor_feats = self.angle_film(anchor_feats.detach() if self.usesg else anchor_feats, geo)
        else:
            anchor_feats = anchor_feats.detach()
        return self.bbox_adapter(anchor_feats) + anchor_feats

    def forward(
        self,
        anchor_pixel_values: torch.Tensor,
        search_pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        angle: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        del input_ids, attention_mask

        batch_size = anchor_pixel_values.shape[0]
        self._validate_geo(angle, batch_size)

        anchor_output = self._vision_forward(anchor_pixel_values)
        anchor_feats = anchor_output.last_hidden_state
        if anchor_feats.ndim != 3 or anchor_feats.shape[0] != batch_size or anchor_feats.shape[2] != self.feature_dim:
            raise ValueError(f"Unexpected anchor feature shape: {tuple(anchor_feats.shape)}.")

        anchor_pooler = self.attnPooling(anchor_feats, 1)[:, 0, :]
        anchor_pooler = F.normalize(anchor_pooler, p=2, dim=1)
        text_feats = None
        fused_feats = None

        sat_output = self._vision_forward(
            search_pixel_values,
            interpolate_pos_encoding=True,
        )
        sat_feats = sat_output.last_hidden_state
        if sat_feats.ndim != 3 or sat_feats.shape[0] != batch_size or sat_feats.shape[2] != self.feature_dim:
            raise ValueError(f"Unexpected satellite feature shape: {tuple(sat_feats.shape)}.")

        sat_feature_2d_pool = self.attnPooling(sat_feats, 9)
        grid_h, grid_w = _infer_patch_grid(
            sat_feats.shape[1],
            search_pixel_values.shape[-2],
            search_pixel_values.shape[-1],
        )
        sat_features_2d = sat_feats.permute(0, 2, 1).reshape(batch_size, self.feature_dim, grid_h, grid_w)

        anchor_context = self._anchor_context(anchor_feats, angle)
        heatmap_logits = self._dynamic_heatmap(anchor_feats, sat_features_2d, angle)
        heatmap = self._spatial_softmax(heatmap_logits)

        heat_gate = heatmap * heatmap.shape[-2] * heatmap.shape[-1] / 2
        guided_sat_features = self.heat_fuse(
            torch.cat([sat_features_2d, heat_gate], dim=1)
        )
        guided_sat_features = guided_sat_features + self._build_position_embedding(guided_sat_features)

        fused_features = self.bbox_transformer(
            x=guided_sat_features,
            context=anchor_context,
        )
        pred_anchor = self.bbox_fcn_out(fused_features)

        heatmap_out = heatmap
        if heatmap_out.shape[-2:] != pred_anchor.shape[-2:]:
            heatmap_logits = F.interpolate(
                heatmap_logits,
                size=pred_anchor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            heatmap_out = self._spatial_softmax(heatmap_logits)

        return (
            pred_anchor,
            None,
            text_feats,
            anchor_pooler,
            sat_feature_2d_pool,
            fused_feats,
            heatmap_out,
        )


class Encoder_test(Encoder_heat):
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        proj_dim: int = PROJECTION_DIM,
        usesg: bool = False,
        useap: bool = False,
        heat_channels: int = 128,
        heat_kernel_size: int = 9,
        heat_softmax_temperature: float = 1.0,
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
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        if self.text_feature_dim != self.feature_dim:
            raise ValueError(
                f"Encoder_test requires text dim {self.text_feature_dim} "
                f"to match vision dim {self.feature_dim}."
            )
        text_replaced = self._inject_lora(
            module=self.text_model,
            rank=int(lora_rank),
            alpha=float(lora_alpha),
            dropout=float(lora_dropout),
        )
        if text_replaced <= 0:
            raise ValueError("No text Linear layers were replaced by LoRA.")

        self.text_grounding_adapter = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.text_roi_gate = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.text_adapter_residual_scale = 0.1
        self.max_text_roi_scale = 0.5
        nn.init.zeros_(self.text_grounding_adapter[-1].weight)
        nn.init.zeros_(self.text_grounding_adapter[-1].bias)
        nn.init.zeros_(self.text_roi_gate[-1].weight)
        nn.init.constant_(self.text_roi_gate[-1].bias, -4.0)

    def _bounded_text_scale(self, raw_scale: torch.Tensor, max_abs: float) -> torch.Tensor:
        return float(max_abs) * torch.tanh(raw_scale)

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

        batch_size, channels, height, width = sat_features_2d.shape
        if text_feats.shape != (batch_size, channels):
            raise ValueError(f"Expected text_feats shape {(batch_size, channels)}, got {tuple(text_feats.shape)}.")

        text_kernel = F.normalize(
            text_feats.to(device=sat_features_2d.device, dtype=sat_features_2d.dtype),
            p=2,
            dim=1,
        )
        sat_features = F.normalize(sat_features_2d.contiguous(), p=2, dim=1)
        conv_input = sat_features.reshape(1, batch_size * channels, height, width)
        conv_kernel = text_kernel.reshape(batch_size, channels, 1, 1)
        return F.conv2d(conv_input, conv_kernel, groups=batch_size).view(batch_size, 1, height, width)

    def _build_text_roi_gate(
        self,
        text_roi: torch.Tensor,
        visual_heatmap: torch.Tensor,
        geo: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if text_roi.shape != visual_heatmap.shape:
            raise ValueError(
                f"Expected text_roi and visual_heatmap to share shape, got "
                f"{tuple(text_roi.shape)} and {tuple(visual_heatmap.shape)}."
            )
        if text_roi.ndim != 4 or text_roi.shape[1] != 1:
            raise ValueError(f"Expected ROI shape (B, 1, H, W), got {tuple(text_roi.shape)}.")

        batch_size = text_roi.shape[0]
        text_prob = text_roi.detach().flatten(1).float()
        visual_prob = visual_heatmap.detach().flatten(1).float()
        text_prob = text_prob / text_prob.sum(dim=1, keepdim=True).clamp_min(1e-6)
        visual_prob = visual_prob / visual_prob.sum(dim=1, keepdim=True).clamp_min(1e-6)

        token_count = float(text_prob.shape[1])
        text_entropy = -(text_prob * text_prob.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        text_entropy = text_entropy / max(math.log(max(token_count, 2.0)), 1e-6)
        visual_peak = visual_prob.amax(dim=1, keepdim=True)
        overlap = (text_prob * visual_prob).sum(dim=1, keepdim=True) * token_count

        if geo is None:
            height = text_prob.new_zeros(batch_size, 1)
        else:
            if geo.ndim != 2 or geo.shape != (batch_size, 3):
                raise ValueError(f"Expected geo shape {(batch_size, 3)}, got {tuple(geo.shape)}.")
            height = geo[:, 2:3].detach().to(device=text_prob.device, dtype=text_prob.dtype)

        gate_input = torch.cat([text_entropy, visual_peak, overlap, height], dim=1)
        gate = torch.sigmoid(self.text_roi_gate(gate_input.to(device=text_roi.device, dtype=text_roi.dtype)))
        return gate.view(batch_size, 1, 1, 1) * float(self.max_text_roi_scale)

    def forward(
        self,
        anchor_pixel_values: torch.Tensor,
        search_pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        angle: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        batch_size = anchor_pixel_values.shape[0]
        self._validate_geo(angle, batch_size)

        anchor_output = self._vision_forward(anchor_pixel_values)
        anchor_feats = anchor_output.last_hidden_state
        if anchor_feats.ndim != 3 or anchor_feats.shape[0] != batch_size or anchor_feats.shape[2] != self.feature_dim:
            raise ValueError(f"Unexpected anchor feature shape: {tuple(anchor_feats.shape)}.")

        anchor_pooler = self.attnPooling(anchor_feats, 1)[:, 0, :]
        anchor_pooler = F.normalize(anchor_pooler, p=2, dim=1)
        text_feats = self._project_text(input_ids, attention_mask)
        text_grounding = (
            F.normalize(
                text_feats
                + self.text_adapter_residual_scale * self.text_grounding_adapter(text_feats),
                p=2,
                dim=1,
            )
            if text_feats is not None
            else None
        )
        fused_feats = None

        sat_output = self._vision_forward(
            search_pixel_values,
            interpolate_pos_encoding=True,
        )
        sat_feats = sat_output.last_hidden_state
        if sat_feats.ndim != 3 or sat_feats.shape[0] != batch_size or sat_feats.shape[2] != self.feature_dim:
            raise ValueError(f"Unexpected satellite feature shape: {tuple(sat_feats.shape)}.")

        sat_feature_2d_pool = self.attnPooling(sat_feats, 9)
        grid_h, grid_w = _infer_patch_grid(
            sat_feats.shape[1],
            search_pixel_values.shape[-2],
            search_pixel_values.shape[-1],
        )
        sat_features_2d = sat_feats.permute(0, 2, 1).reshape(batch_size, self.feature_dim, grid_h, grid_w)

        anchor_context = self._anchor_context(anchor_feats, angle)
        heatmap_logits = self._dynamic_heatmap(anchor_feats, sat_features_2d, angle)
        text_heatmap_logits = self._text_dynamic_heatmap(text_grounding, sat_features_2d)
        heatmap = self._spatial_softmax(heatmap_logits)

        heat_gate = heatmap * heatmap.shape[-2] * heatmap.shape[-1] / 2
        if text_heatmap_logits is not None:
            text_roi = self._spatial_softmax(text_heatmap_logits)
            text_roi = text_roi / text_roi.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            text_roi_gate = self._build_text_roi_gate(
                text_roi=text_roi,
                visual_heatmap=heatmap,
                geo=angle,
            )
            heat_gate = heat_gate * (1.0 + text_roi_gate.to(dtype=heat_gate.dtype) * text_roi.to(dtype=heat_gate.dtype))
        guided_sat_features = self.heat_fuse(
            torch.cat([sat_features_2d, heat_gate], dim=1)
        )
        guided_sat_features = guided_sat_features + self._build_position_embedding(guided_sat_features)

        fused_features = self.bbox_transformer(
            x=guided_sat_features,
            context=anchor_context,
        )
        pred_anchor = self.bbox_fcn_out(fused_features)

        heatmap_out = heatmap
        if heatmap_out.shape[-2:] != pred_anchor.shape[-2:]:
            heatmap_logits = F.interpolate(
                heatmap_logits,
                size=pred_anchor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            heatmap_out = self._spatial_softmax(heatmap_logits)

        return (
            pred_anchor,
            None,
            text_feats,
            anchor_pooler,
            sat_feature_2d_pool,
            fused_feats,
            heatmap_out,
        )
