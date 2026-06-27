from dataclasses import dataclass
from typing import Any, Callable, Dict, Type

import torch.nn as nn

from grounding.adapters import BaseAdapter, DirectBboxAdapter, LegacyAnchorAdapter, SMGeoAdapter, SiglipTupleAdapter

DEFAULT_SAMPLE4GEO_CHECKPOINT = "/media/data1/feihong/remote/ckpt/Sample4Geo.pth"
DEFAULT_SMGEO_CHECKPOINT = "/media/data1/feihong/remote/ckpt/SMGeo.pth"


@dataclass
class ModelEntry:
    builder: Callable[[Dict[str, Any]], nn.Module]
    adapter_cls: Type[BaseAdapter]


def _model_bool(model_cfg: Dict[str, Any], key: str, default: bool) -> bool:
    value = model_cfg.get(key, default)
    if isinstance(value, str):
        return value.lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _build_siglip2_heat(cfg: Dict[str, Any]) -> nn.Module:
    from model import Encoder_heat

    return Encoder_heat(
        model_name=cfg["model"]["model_name"],
        usesg=True,
        useap=True,
    )


def _build_siglip2_test(cfg: Dict[str, Any]) -> nn.Module:
    from model import Encoder_test

    return Encoder_test(
        model_name=cfg["model"]["model_name"],
        usesg=True,
        useap=True,
        use_text_grounding_path=False,
    )


def _build_siglip_ground(cfg: Dict[str, Any]) -> nn.Module:
    from model_ground import Encoder_ground

    return Encoder_ground(
        model_name=cfg["model"]["model_name"],
        usesg=True,
        useap=True,
    )


def _build_lpn(cfg: Dict[str, Any]) -> nn.Module:
    from grounding.legacy.ground_cvos import LPNGeoLite

    model_cfg = cfg.get("model", {})
    return LPNGeoLite(pretrained=_model_bool(model_cfg, "pretrained", True))


def _build_sample4geo(cfg: Dict[str, Any]) -> nn.Module:
    from grounding.legacy.ground_cvos import SampleGeoLite, load_sample4geo_backbone

    model_cfg = cfg.get("model", {})
    checkpoint = model_cfg.get("checkpoint", DEFAULT_SAMPLE4GEO_CHECKPOINT)
    model = SampleGeoLite(pretrained=False if checkpoint else _model_bool(model_cfg, "pretrained", True))
    if checkpoint:
        load_info = load_sample4geo_backbone(model, str(checkpoint))
        print(f"Loaded Sample4Geo checkpoint from {checkpoint}: {load_info}")
    return model


def _build_smgeo(cfg: Dict[str, Any]) -> nn.Module:
    from grounding.legacy.train_sm import SMGeoLite, load_smgeo_pretrained

    model_cfg = cfg.get("model", {})
    model = SMGeoLite(
        num_experts=int(model_cfg.get("num_experts", 6)),
        top_k=int(model_cfg.get("top_k", 2)),
    )
    checkpoint = model_cfg.get("checkpoint", DEFAULT_SMGEO_CHECKPOINT)
    if checkpoint:
        load_info = load_smgeo_pretrained(model, str(checkpoint))
        print(f"Loaded SMGeo checkpoint from {checkpoint}: {load_info}")
    return model


def _build_ocg(cfg: Dict[str, Any]) -> nn.Module:
    from grounding.legacy.train_ocg import OCGNetLite

    model_cfg = cfg.get("model", {})
    pretrained = _model_bool(model_cfg, "pretrained_backbone", _model_bool(model_cfg, "pretrained", True))
    return OCGNetLite(pretrained_backbone=pretrained)


def _build_trogeolite(cfg: Dict[str, Any]) -> nn.Module:
    from grounding.legacy.ground_cvos import TROGeoLite

    return TROGeoLite()


def _build_det(cfg: Dict[str, Any]) -> nn.Module:
    from grounding.legacy.ground_cvos import DetGeoLite

    return DetGeoLite()


REGISTRY: Dict[str, ModelEntry] = {
    "siglip2_heat": ModelEntry(_build_siglip2_heat, SiglipTupleAdapter),
    "siglip2_test": ModelEntry(_build_siglip2_test, SiglipTupleAdapter),
    "siglip_ground": ModelEntry(_build_siglip_ground, SiglipTupleAdapter),
    "lpn": ModelEntry(_build_lpn, LegacyAnchorAdapter),
    "sample4geo": ModelEntry(_build_sample4geo, LegacyAnchorAdapter),
    "smgeo": ModelEntry(_build_smgeo, SMGeoAdapter),
    "ocg": ModelEntry(_build_ocg, LegacyAnchorAdapter),
    "trogeolite": ModelEntry(_build_trogeolite, LegacyAnchorAdapter),
    "det": ModelEntry(_build_det, LegacyAnchorAdapter),
}

MODEL_TYPES = tuple(REGISTRY.keys())


def get_model_entry(model_type: str) -> ModelEntry:
    return REGISTRY[model_type]


def build_model_and_adapter(cfg: Dict[str, Any]):
    entry = get_model_entry(cfg["model"]["type"])
    model = entry.builder(cfg)
    adapter = entry.adapter_cls(model, cfg)
    return model, adapter
