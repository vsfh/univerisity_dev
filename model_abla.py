from typing import Optional, Tuple

import torch
from transformers import AutoModel

from hf_cache_utils import from_pretrained_prefer_local
from model import (
    CACHE_DIR,
    MODEL_NAME,
    PROJECTION_DIM,
    Encoder_test,
    PoolingHead,
    _infer_patch_grid,
)


def _checkpoint_state(ckpt_path: str):
    if not ckpt_path:
        raise ValueError("model_pre requires a non-empty ckpt_path.")

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload in {ckpt_path}: {type(state).__name__}.")

    state = {str(key).replace("module.", "", 1): value for key, value in state.items()}
    return {
        key: value
        for key, value in state.items()
        if not key.startswith("text_projector.")
    }


class model_pre(Encoder_test):
    """Encoder_test initialized from a previously trained model checkpoint."""

    def __init__(
        self,
        ckpt_path: str,
        model_name: str = MODEL_NAME,
        proj_dim: int = PROJECTION_DIM,
        usesg: bool = False,
        useap: bool = False,
        heat_channels: int = 128,
        heat_kernel_size: int = 9,
        heat_softmax_temperature: float = 1.0,
        use_heatmap: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        use_text_grounding_path: bool = False,
    ):
        super().__init__(
            model_name=model_name,
            proj_dim=proj_dim,
            usesg=usesg,
            useap=useap,
            heat_channels=heat_channels,
            heat_kernel_size=heat_kernel_size,
            heat_softmax_temperature=heat_softmax_temperature,
            use_heatmap=use_heatmap,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_text_grounding_path=use_text_grounding_path,
        )
        self.load_state_dict(_checkpoint_state(ckpt_path), strict=True)


class model_bi(Encoder_test):
    """Encoder_test with independent drone and satellite SigLIP vision models."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        proj_dim: int = PROJECTION_DIM,
        usesg: bool = False,
        useap: bool = False,
        heat_channels: int = 128,
        heat_kernel_size: int = 9,
        heat_softmax_temperature: float = 1.0,
        use_heatmap: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        use_text_grounding_path: bool = False,
    ):
        super().__init__(
            model_name=model_name,
            proj_dim=proj_dim,
            usesg=usesg,
            useap=useap,
            heat_channels=heat_channels,
            heat_kernel_size=heat_kernel_size,
            heat_softmax_temperature=heat_softmax_temperature,
            use_heatmap=use_heatmap,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_text_grounding_path=use_text_grounding_path,
        )

        satellite_model = from_pretrained_prefer_local(AutoModel, model_name, CACHE_DIR)
        self.satellite_vision_model = satellite_model.vision_model
        satellite_feature_dim = int(self.satellite_vision_model.config.hidden_size)
        if satellite_feature_dim != self.feature_dim:
            raise ValueError(
                f"Satellite hidden size {satellite_feature_dim} must match "
                f"drone hidden size {self.feature_dim}."
            )

        for param in self.satellite_vision_model.parameters():
            param.requires_grad = False
        replaced = self._inject_lora(
            module=self.satellite_vision_model,
            rank=int(lora_rank),
            alpha=float(lora_alpha),
            dropout=float(lora_dropout),
        )
        if replaced <= 0:
            raise ValueError("No satellite vision Linear layers were replaced by LoRA.")
        self.satellite_attn_pooling = PoolingHead(self.satellite_vision_model.head)

    def _satellite_vision_forward(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = True,
    ):
        if pixel_values.ndim != 4:
            raise ValueError(
                f"Expected satellite image tensor shape (B, C, H, W), "
                f"got {tuple(pixel_values.shape)}."
            )
        return self.satellite_vision_model(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

    def forward(
        self,
        anchor_pixel_values: torch.Tensor,
        search_pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        angle: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple:
        batch_size = anchor_pixel_values.shape[0]
        self._validate_geo(angle, batch_size)

        anchor_output = self._vision_forward(anchor_pixel_values)
        anchor_feats = anchor_output.last_hidden_state
        if (
            anchor_feats.ndim != 3
            or anchor_feats.shape[0] != batch_size
            or anchor_feats.shape[2] != self.feature_dim
        ):
            raise ValueError(f"Unexpected anchor feature shape: {tuple(anchor_feats.shape)}.")
        anchor_pooler = anchor_output.pooler_output

        text_pooler = None
        if input_ids is not None:
            _, text_pooler = self._encode_text_anchor(input_ids, attention_mask)

        sat_output = self._satellite_vision_forward(search_pixel_values)
        sat_feats = sat_output.last_hidden_state
        if (
            sat_feats.ndim != 3
            or sat_feats.shape[0] != batch_size
            or sat_feats.shape[2] != self.feature_dim
        ):
            raise ValueError(f"Unexpected satellite feature shape: {tuple(sat_feats.shape)}.")

        sat_feature_2d_pool = self.satellite_attn_pooling(sat_feats, 9)
        grid_h, grid_w = _infer_patch_grid(
            sat_feats.shape[1],
            search_pixel_values.shape[-2],
            search_pixel_values.shape[-1],
        )
        sat_features_2d = sat_feats.permute(0, 2, 1).reshape(
            batch_size,
            self.feature_dim,
            grid_h,
            grid_w,
        )

        pred_anchor, heatmap_out = self._bbox_forward_from_anchor_feats(
            anchor_feats,
            sat_features_2d,
            angle,
            detach_anchor=True,
        )
        return (
            pred_anchor,
            None,
            text_pooler,
            anchor_pooler,
            sat_feature_2d_pool,
            {},
            heatmap_out,
        )


__all__ = ["model_pre", "model_bi"]
