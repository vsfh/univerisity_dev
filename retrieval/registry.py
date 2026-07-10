from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import torch.nn as nn

from retrieval.adapters import BaseRetrievalAdapter, ForwardMethodAdapter, SampleRetrievalAdapter


@dataclass
class ModelEntry:
    canonical_type: str
    builder: Callable[[Dict[str, Any]], nn.Module]
    adapter_cls: type[BaseRetrievalAdapter]
    io_builder: Callable[[Dict[str, Any]], Tuple[Any, Any]]


def _build_clip(cfg: Dict[str, Any]) -> nn.Module:
    from retrieval.train_clip import Encoder

    model_cfg = cfg["model"]
    return Encoder(str(model_cfg["model_name"]), proj_dim=int(model_cfg.get("proj_dim", 768)))


def _build_siglip(cfg: Dict[str, Any]) -> nn.Module:
    from retrieval.train_siglip import Encoder

    model_cfg = cfg["model"]
    return Encoder(str(model_cfg["model_name"]), proj_dim=int(model_cfg.get("proj_dim", 768)))


def _build_openclip(cfg: Dict[str, Any]) -> nn.Module:
    from retrieval.train_openclip import Encoder

    model_cfg = cfg["model"]
    return Encoder(
        str(model_cfg["model_name"]),
        pretrained=str(model_cfg.get("pretrained") or ""),
        proj_dim=int(model_cfg.get("proj_dim", 768)),
    )


def _build_evaclip(cfg: Dict[str, Any]) -> nn.Module:
    from retrieval.train_evaclip import Encoder

    model_cfg = cfg["model"]
    return Encoder(str(model_cfg["model_name"]), proj_dim=int(model_cfg.get("proj_dim", 768)))


def _build_sample_retrieval(cfg: Dict[str, Any]) -> nn.Module:
    from grounding.train_sample_retrieval import SampleGeoLite

    model_cfg = cfg["model"]
    return SampleGeoLite(
        emb_size=int(model_cfg.get("emb_size", 1024)),
        pretrained=bool(model_cfg.get("pretrained_backbone", True)),
    )


def _set_size(processor: Any, size: Dict[str, int]) -> None:
    if hasattr(processor, "image_processor"):
        processor.image_processor.size = dict(size)
    elif hasattr(processor, "size"):
        processor.size = dict(size)


def _build_hf_io(cfg: Dict[str, Any]) -> Tuple[Any, Any]:
    from transformers import AutoImageProcessor, AutoTokenizer, CLIPProcessor

    model_cfg = cfg["model"]
    model_name = str(model_cfg["model_name"])
    cache_dir = str(model_cfg["cache_dir"])
    if str(model_cfg["type"]) == "clip":
        processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        processor_sat = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    else:
        processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        processor_sat = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    _set_size(processor_sat, cfg["data"]["sat_size"])
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    return (processor, processor_sat), tokenizer


def _build_openclip_io(cfg: Dict[str, Any]) -> Tuple[Any, Any]:
    import open_clip
    from retrieval.train_openclip import OpenClipImageProcessorWrapper, OpenClipTokenizerWrapper

    model_cfg = cfg["model"]
    _, _, preprocess = open_clip.create_model_and_transforms(
        str(model_cfg["model_name"]),
        pretrained=model_cfg.get("pretrained"),
        cache_dir=str(model_cfg["cache_dir"]),
    )
    processor = OpenClipImageProcessorWrapper(preprocess)
    tokenizer = OpenClipTokenizerWrapper(open_clip.get_tokenizer(str(model_cfg["model_name"])))
    return (processor, processor), tokenizer


def _build_evaclip_io(cfg: Dict[str, Any]) -> Tuple[Any, Any]:
    import open_clip
    from retrieval.train_evaclip import OpenClipImageProcessorWrapper, OpenClipTokenizerWrapper

    model_cfg = cfg["model"]
    _, _, preprocess = open_clip.create_model_and_transforms(
        str(model_cfg["model_name"]),
        cache_dir=str(model_cfg["cache_dir"]),
    )
    processor = OpenClipImageProcessorWrapper(preprocess)
    tokenizer = OpenClipTokenizerWrapper(open_clip.get_tokenizer(str(model_cfg["model_name"])))
    return (processor, processor), tokenizer


def _build_sample_io(cfg: Dict[str, Any]) -> Tuple[Any, Any]:
    from grounding.train_sample_retrieval import CVOGL_TRANSFORM, DummyTokenizer, TransformProcessorWrapper

    sat_size = cfg["data"]["sat_size"]
    drone_size = cfg["data"]["drone_size"]
    processor = TransformProcessorWrapper(
        CVOGL_TRANSFORM,
        (int(drone_size["width"]), int(drone_size["height"])),
    )
    processor_sat = TransformProcessorWrapper(
        CVOGL_TRANSFORM,
        (int(sat_size["width"]), int(sat_size["height"])),
    )
    return (processor, processor_sat), DummyTokenizer()


REGISTRY: Dict[str, ModelEntry] = {
    "clip": ModelEntry("clip", _build_clip, ForwardMethodAdapter, _build_hf_io),
    "siglip": ModelEntry("siglip", _build_siglip, ForwardMethodAdapter, _build_hf_io),
    "openclip": ModelEntry("openclip", _build_openclip, ForwardMethodAdapter, _build_openclip_io),
    "evaclip": ModelEntry("evaclip", _build_evaclip, ForwardMethodAdapter, _build_evaclip_io),
    "sample_retrieval": ModelEntry("sample_retrieval", _build_sample_retrieval, SampleRetrievalAdapter, _build_sample_io),
    "sample4geo": ModelEntry("sample_retrieval", _build_sample_retrieval, SampleRetrievalAdapter, _build_sample_io),
}
MODEL_TYPES = ("clip", "siglip", "openclip", "evaclip", "sample_retrieval")


def get_model_entry(model_type: str) -> ModelEntry:
    return REGISTRY[str(model_type).lower()]


def build_model_and_adapter(cfg: Dict[str, Any]):
    entry = get_model_entry(cfg["model"]["type"])
    model = entry.builder(cfg)
    return model, entry.adapter_cls(model, cfg)


def build_processors_and_tokenizer(cfg: Dict[str, Any]):
    return get_model_entry(cfg["model"]["type"]).io_builder(cfg)
